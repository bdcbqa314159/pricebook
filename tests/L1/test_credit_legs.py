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


class TestProtectionLegVersusTheEngines:
    """No purity claim — the honest statement of where they agree and where they do not."""

    def _engine_leg(self, survival: SurvivalCurve) -> float:
        from pricebook.engine.credit_default_swap import protection_leg as engine_leg

        return engine_leg(_cds(), _model(survival))

    def test_they_COINCIDE_when_there_is_no_interior_pillar(self) -> None:
        """A flat hazard has nothing to break at, so per-segment and pointwise are one integral.

        This is the premise behind tranche 38's rung-1 closed-form oracle reading 5.7e-10 — and
        equally the reason that oracle is structurally blind to the term-structured error below."""
        assert [t for t in FLAT.times if 0.0 < t < HORIZON] == []
        shared = protection_leg(SLOPED, FLAT, RECOVERY, HORIZON, 2000)
        assert shared == pytest.approx(self._engine_leg(FLAT), rel=1e-12)

    def test_they_DIFFER_by_a_measured_amount_when_pillars_are_interior(self) -> None:
        """1.8e-05, the figure OI-4 is about. Asserted so the gap cannot drift unnoticed."""
        assert [t for t in TERM.times if 0.0 < t < HORIZON] == [1.0, 2.0]
        shared = protection_leg(SLOPED, TERM, RECOVERY, HORIZON, 2000)
        gap = shared / self._engine_leg(TERM) - 1.0
        assert gap == pytest.approx(1.846e-05, rel=0.05), gap

    def test_the_SHARED_one_is_the_more_accurate_side(self) -> None:
        """Against a far finer reference of each: the gap is the engine's error, not a difference
        of opinion. Refining the shared integration barely moves it; refining the engine's walks
        it toward the shared value."""
        reference = protection_leg(SLOPED, TERM, RECOVERY, HORIZON, 40000)
        shared = protection_leg(SLOPED, TERM, RECOVERY, HORIZON, 2000)
        assert abs(shared / reference - 1.0) < 1e-08
        assert abs(self._engine_leg(TERM) / reference - 1.0) > 1e-05

    def test_the_integration_is_SECOND_ORDER(self) -> None:
        reference = protection_leg(SLOPED, TERM, RECOVERY, HORIZON, 40000)
        errors = [abs(protection_leg(SLOPED, TERM, RECOVERY, HORIZON, n) / reference - 1.0)
                  for n in (25, 50, 100)]
        for coarse, fine in zip(errors, errors[1:]):
            assert 3.5 < coarse / fine < 4.5, errors


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
