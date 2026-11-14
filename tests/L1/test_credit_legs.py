"""The shared CDS leg arithmetic (L1) — tranche 39's extraction, and what it may and may not claim.

Two different things are asserted here, and the charter treats them differently (§4):

* **`risky_pv01` is a pure refactor, proved by BIT-IDENTITY.** It was lifted out of tranche 38's
  engine with the order of floating-point operations preserved, so the strong claim is available
  and is the one made: the engine's values are unchanged to the last bit across a 324-fixture
  sweep. The charter allows an exact-arithmetic identity *instead* only when the refactor
  legitimately reorders operations; here nothing was reordered, so anything looser would be an
  excuse.
* **`protection_leg` is NOT the engine's integration**, so no purity is claimed for it. It reads
  the hazard once per pillar segment, which is exact there, where the engine reads
  `default_density` pointwise on a uniform grid and loses an order at a discontinuity. The two
  **provably coincide where there are no interior pillars**, which is asserted, and differ by a
  measured 1.8e-05 where there are — the duplication is `OPEN.md` OI-4, awaiting a ruling, and
  this file is where the gap is pinned rather than left to a comment.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.foundation import Currency, CouponSchedule, Money
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.credit_legs import protection_leg, risky_pv01
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.credit import CreditModel
from pricebook.products.credit_default_swap import CreditDefaultSwap, ProtectionDirection

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
RECOVERY = 0.40

SLOPED = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0), (0.020, 0.026, 0.030, 0.033))
#: pillars at 1 and 2, INSIDE the 3y horizon — so the two integrations must differ
TERM = SurvivalCurve(TM, (0.0, 1.0, 2.0), (0.010, 0.022, 0.040))
#: one pillar, so no interior break: the two integrations must COINCIDE
FLAT = SurvivalCurve.flat(VAL, 0.03, DC)
SEMI = tuple(VAL + timedelta(days=d) for d in (0, 182, 365, 547, 730, 912, 1095))
HORIZON = (SEMI[-1] - VAL).days / 365.0


def _model(survival: SurvivalCurve, recovery: float = RECOVERY) -> CreditModel:
    snapshot = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT_KEY: SLOPED}, survivals={EUR: survival}))
    return CreditModel(snapshot, recovery)


def _committed() -> DiscountCurve:
    """The discount curve every comparison here uses — named so the identity tests read as one
    fixture rather than repeating `SLOPED` and inviting a silent divergence."""
    return SLOPED


def _per_segment() -> int:
    """The engine's committed count, read from the engine rather than restated — a copy here
    would let the two drift and still pass."""
    from pricebook.engine.credit_default_swap import _DEFAULT_PER_SEGMENT

    return _DEFAULT_PER_SEGMENT


def _cds(spread: float = 0.01) -> CreditDefaultSwap:
    return CreditDefaultSwap(
        CouponSchedule(SEMI, DC), spread, Money(1.0, EUR), ProtectionDirection.BUYER)


class TestRiskyPv01IsAPureRefactorOfTheEngines:
    """Bit-identity, because the extraction preserved the operation order."""

    def test_the_shared_function_is_BIT_IDENTICAL_to_the_engine(self) -> None:
        from pricebook.engine.credit_default_swap import risky_pv01 as engine_rpv01

        mismatches = []
        for survival in (TERM, FLAT, SurvivalCurve(TM, (0.0, 0.4, 1.7), (0.004, 0.028, 0.051))):
            for recovery in (0.0, RECOVERY, 0.9):
                model = _model(survival, recovery)
                product = _cds()
                shared = risky_pv01(
                    product.schedule.periods(), SLOPED, model.survival(EUR))
                if shared.hex() != engine_rpv01(product, model).hex():
                    mismatches.append((survival.hazards, recovery, shared))
        assert mismatches == [], mismatches

    def test_the_premise_the_value_is_not_trivially_zero_or_one(self) -> None:
        """A bit-identity sweep over zeros would prove nothing."""
        value = risky_pv01(_cds().schedule.periods(), SLOPED, TERM)
        assert 1.0 < value < 10.0, value


class TestTheEngineAndTheBootstrapShareONEProtectionLeg:
    """`docs/conventions_register.md` C11, guarded — this class is the guard.

    Until tranche 39 the engine integrated the protection leg itself, pointwise in
    `default_density` on a uniform grid, while the bootstrap used the per-segment form here. The
    two differed by 1.846e-05 on a term-structured curve, which meant a bootstrapped curve
    repriced its own calibrating CDS to *not quite* zero. OI-4 ruled the engine adopts this one,
    so the agreement is now structural — and what follows asserts that it is structural rather
    than coincidental, because a convention shared by two modules with nothing comparing them is
    folklore (§7).
    """

    def _engine_leg(self, survival: SurvivalCurve) -> float:
        from pricebook.engine.credit_default_swap import protection_leg as engine_leg

        return engine_leg(_cds(), _model(survival))

    @pytest.mark.parametrize("survival", [TERM, FLAT], ids=["term_structured", "flat"])
    def test_the_engine_leg_IS_this_function_to_the_last_bit(self, survival) -> None:
        """A unit notional, so the engine's only remaining job is the scaling — and `× 1.0` is
        exact. Any difference at all would mean a second integration had reappeared."""
        shared = protection_leg(SHARED_DEFAULT := _committed(), survival, RECOVERY, HORIZON,
                                _per_segment())
        assert shared.hex() == self._engine_leg(survival).hex(), (shared, SHARED_DEFAULT)

    def test_THE_PREMISE_the_term_fixture_HAS_interior_pillars(self) -> None:
        """Without this the parametrised case above is one fixture twice: a flat hazard has no
        pillar to break at, so per-segment and pointwise coincide there and the identity would
        hold even if the engine had kept its own loop. The term curve is what discriminates."""
        assert [t for t in TERM.times if 0.0 < t < HORIZON] == [1.0, 2.0]
        assert [t for t in FLAT.times if 0.0 < t < HORIZON] == []

    def test_the_integration_is_SECOND_ORDER_on_a_TERM_STRUCTURED_curve(self) -> None:
        """The rate is the detector the flat anchor cannot be: breaking at the pillars is what
        makes this `O(1/M²)`, and a return to pointwise density would read `O(1/M)` here while
        leaving tranche 38's flat closed-form oracle at its exact 5.7e-10 (charter §4, the
        reduction-anchor blind spot)."""
        reference = protection_leg(_committed(), TERM, RECOVERY, HORIZON, 40000)
        errors = [abs(protection_leg(_committed(), TERM, RECOVERY, HORIZON, n) / reference - 1.0)
                  for n in (25, 50, 100)]
        for coarse, fine in zip(errors, errors[1:]):
            assert 3.5 < coarse / fine < 4.5, errors

    def test_a_bootstrapped_curve_REPRICES_its_own_calibrating_quote_through_the_ENGINE(
        self) -> None:
        """The consequence C11 exists for, end to end across three layers — and it is EXACT.

        The bootstrap (L1) calibrates to `V = 0`; the engine (L4) prices the same contract. Before
        OI-4 the two used different quadratures, so this round trip could not have been asserted at
        all: it is an oracle that only became available once the convention was unified. Given the
        **same** grid the agreement is not approximate but exact to rounding — `-5.8e-18` of
        notional, which is the sense in which C11 is *guarded* rather than merely documented.
        """
        from pricebook.engine import price
        from pricebook.market.cds_bootstrap import bootstrap_survival_curve

        schedule, quote = CouponSchedule(SEMI, DC), 0.0120
        curve = bootstrap_survival_curve(
            ((schedule, quote),), SLOPED, RECOVERY, per_segment=_per_segment())
        product = CreditDefaultSwap(
            schedule, quote, Money(10_000_000.0, EUR), ProtectionDirection.BUYER)
        pv = price(product, _model(curve)).pv.amount
        assert abs(pv) < 1e-08, pv

    def test_a_COARSER_bootstrap_grid_leaves_only_DISCRETISATION_shrinking_at_second_order(
        self) -> None:
        """Why the test above fixes the grid, and why that is not rigging it.

        At its own default (50 per segment) the bootstrap is 40x coarser than the engine, so the
        reprice leaves 4.2e-08 of notional. That residual is grid resolution, not a disagreement
        about the formula — proved by its converging at **second order** toward zero as the
        bootstrap's grid approaches the engine's. A formula mismatch would plateau instead, which
        is the T25 rate-detector: the rate distinguishes a discretisation from a defect where no
        single value can."""
        from pricebook.engine import price
        from pricebook.market.cds_bootstrap import bootstrap_survival_curve

        schedule, quote = CouponSchedule(SEMI, DC), 0.0120
        product = CreditDefaultSwap(
            schedule, quote, Money(10_000_000.0, EUR), ProtectionDirection.BUYER)
        residuals = []
        for per_segment in (50, 200, 800):
            curve = bootstrap_survival_curve(
                ((schedule, quote),), SLOPED, RECOVERY, per_segment=per_segment)
            residuals.append(abs(price(product, _model(curve)).pv.amount))
        for coarse, fine in zip(residuals, residuals[1:]):
            assert 12.0 < coarse / fine < 20.0, f"4x the grid must cut it ~16x: {residuals}"


class TestTheGuardsAreReachable:
    def test_a_NON_POSITIVE_horizon_has_no_protection_left(self) -> None:
        """Invariant 6, not a failure: a matured contract is worth nothing, it is not unpriceable."""
        assert protection_leg(SLOPED, TERM, RECOVERY, 0.0, 50) == 0.0
        assert protection_leg(SLOPED, TERM, RECOVERY, -1.0, 50) == 0.0

    def test_a_grid_with_NO_points_per_segment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="per segment"):
            protection_leg(SLOPED, TERM, RECOVERY, HORIZON, 0)

    def test_FULL_recovery_makes_protection_worthless(self) -> None:
        assert protection_leg(SLOPED, TERM, 1.0, HORIZON, 50) == 0.0

    def test_an_EMPTY_schedule_gives_a_zero_annuity(self) -> None:
        assert risky_pv01((), SLOPED, TERM) == 0.0
