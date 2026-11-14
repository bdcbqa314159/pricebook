"""The credit default swap (L2/L3/L4) — tranche 38, project five's first unit. **Pre-registered.**

O'Kane ch 6. Two legs on the **reused** `SurvivalCurve` (T25) and discount curve (T1/T8):

    Premium leg  = S₀ · RPV01,    RPV01 = ½ Σₙ Δₙ · Z(tₙ) · (Q(tₙ₋₁) + Q(tₙ))        (6.2/6.4)
    Protection   = (1−R) ∫₀ᵀ Z(t) λ(t) Q(t) dt
    V(buyer)     = Protection − Premium,   S_par = Protection / RPV01

**The protection leg IS the T25 CVA integral.** `(1−R)∫Z·λQ dt` is `LGD·∫DF·λS dt` with the exposure
flat at 1 — literally what `risk/credit_valuation_adjustment.py` computes. So this tranche's
correctness anchor is a **cross-arc** oracle: project five's protection leg reconciled against
project three's CVA integral, different code written from different sources. No within-CDS check can
replace it.

**The credit triangle is the exact reduction.** Under a flat hazard, flat rates and the continuous
limit, `S_par = λ(1−R)` exactly (O'Kane §3) — because `Protection → λ(1−R)∫ZQ` and `RPV01 → ∫ZQ`, so
the ratio is `λ(1−R)` whatever the curve. It is a *limit*, so it is asserted by convergence as the
premium periods refine, not at one coarse schedule.

**The `τ = 1` history is designed out from the first line, not patched in later.** Three tranches
(T3, T8, T9) shipped oracles blind to an accrual because their fixtures were annual on ACT/365F.
This file's companion fixture therefore carries **non-unit, unequal accruals**, a **term-structured**
hazard read beyond its last pillar, and a **non-flat** discount curve — each asserted as a premise,
so the `½(Qₙ₋₁+Qₙ) → Qₙ` mutation (dropping the accrual-on-default half) has somewhere to show.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation import Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.credit import CreditModel
from pricebook.products.credit_default_swap import CreditDefaultSwap, ProtectionDirection
from pricebook.risk.credit_valuation_adjustment import ExposureProfile, unilateral_cva

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
DISCOUNT = CurveKey(CurveRole.DISCOUNT, EUR)
NOTIONAL = Money(10_000_000.0, EUR)
RECOVERY = 0.40
SPREAD = 0.01

#: NON-UNIT and UNEQUAL accruals (182/183-day periods), and a NON-FLAT discount curve: the τ=1
#: history designed out from the start rather than remediated at a later re-audit.
SEMI_DATES = tuple(VAL + timedelta(days=d) for d in (0, 182, 365, 547, 730, 912, 1095))
ANNUAL_DATES = tuple(VAL + timedelta(days=365 * k) for k in range(6))
#: TERM-STRUCTURED hazard, read beyond its last pillar at the 3y maturity.
TERM_HAZARD = SurvivalCurve(TM, (0.0, 1.0, 2.0), (0.010, 0.022, 0.040))
SLOPED = DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0), (0.020, 0.026, 0.030, 0.033))
FLAT_RATE = 0.025


def _market(curve: DiscountCurve, survival: SurvivalCurve) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT: curve}),
        survivals={EUR: survival},
    )


def _model(curve: DiscountCurve = SLOPED, survival: SurvivalCurve = TERM_HAZARD,
           recovery: float = RECOVERY) -> CreditModel:
    return CreditModel(_market(curve, survival), recovery)


def _cds(dates: tuple = SEMI_DATES, spread: float = SPREAD,
         direction: ProtectionDirection = ProtectionDirection.BUYER) -> CreditDefaultSwap:
    from pricebook.foundation import CouponSchedule

    return CreditDefaultSwap(CouponSchedule(dates, DC), spread, NOTIONAL, direction)


def _flat(hazard: float) -> SurvivalCurve:
    return SurvivalCurve(TM, (0.0,), (hazard,))


class TestTheFixtureIsNotDegenerate:
    """Every premise the mutation table needs, asserted — the τ=1 lesson applied pre-emptively."""

    def test_the_premium_ACCRUALS_are_non_unit_and_unequal(self) -> None:
        from pricebook.foundation import CouponSchedule

        accruals = [psi for _, _, psi in CouponSchedule(SEMI_DATES, DC).periods()]
        assert all(abs(a - 1.0) > 0.4 for a in accruals), accruals
        assert len({round(a, 6) for a in accruals}) > 1, f"must differ from each other: {accruals}"

    def test_the_hazard_is_TERM_STRUCTURED_and_read_beyond_its_last_pillar(self) -> None:
        assert max(TERM_HAZARD.hazards) - min(TERM_HAZARD.hazards) > 0.02
        horizon = TM.year_fraction(SEMI_DATES[-1])
        assert horizon > TERM_HAZARD.times[-1]
        assert TERM_HAZARD.survival(horizon) != pytest.approx(
            math.exp(-TERM_HAZARD.hazards[0] * horizon), rel=1e-3
        )

    def test_the_discount_curve_is_NON_FLAT(self) -> None:
        assert abs(SLOPED.zero_rates[-1] - SLOPED.zero_rates[0]) > 0.01


class TestTheCreditTriangleTheExactReduction:
    """`S_par = λ(1−R)` under a flat hazard, flat rate and the continuous limit (O'Kane §3)."""

    @pytest.mark.parametrize("hazard", [0.005, 0.02, 0.06])
    @pytest.mark.parametrize("recovery", [0.0, 0.4, 0.75])
    def test_the_par_spread_CONVERGES_to_lambda_times_one_minus_R(
        self, hazard: float, recovery: float
    ) -> None:
        """A limit, so asserted by refinement rather than at one coarse schedule."""
        flat_curve = DiscountCurve.flat(VAL, FLAT_RATE, DC)
        model = _model(flat_curve, _flat(hazard), recovery)
        target = hazard * (1.0 - recovery)

        def par(periods_per_year: int) -> float:
            step = round(365 / periods_per_year)
            dates = tuple(VAL + timedelta(days=step * k) for k in range(3 * periods_per_year + 1))
            return par_spread(_cds(dates), model)

        from pricebook.engine.credit_default_swap import par_spread

        errors = [abs(par(n) / target - 1.0) for n in (2, 12, 73)]
        assert all(b < a for a, b in zip(errors, errors[1:])), errors
        assert errors[-1] < 5e-3, f"must approach the triangle: {errors}"

    def test_THE_COMPANION_a_term_structured_hazard_breaks_the_triangle(self) -> None:
        """The reduction is satisfied by an implementation ignoring the schedule entirely, so the
        non-flat case must differ materially — asserted, not assumed."""
        from pricebook.engine.credit_default_swap import par_spread

        flat_equivalent = TERM_HAZARD.hazard(0.0) * (1.0 - RECOVERY)
        assert abs(par_spread(_cds(), _model()) / flat_equivalent - 1.0) > 0.3


class TestTheProtectionLegIsTheCvaIntegral:
    """The cross-ARC oracle: project five's leg against project three's integral."""

    def test_the_protection_leg_equals_TRANCHE_25s_cva_on_a_FLAT_UNIT_exposure(self) -> None:
        """`(1−R)∫Z·λQ dt` is `LGD·∫DF·EPE·λS dt` with `EPE ≡ 1` — the same integral, reached by
        different code written from a different source in a different arc."""
        from pricebook.engine.credit_default_swap import protection_leg

        horizon = TM.year_fraction(SEMI_DATES[-1])
        grid = tuple(horizon * k / 600 for k in range(1, 601))
        profile = ExposureProfile(grid, tuple(1.0 for _ in grid))
        through_t25 = unilateral_cva(profile, TERM_HAZARD, SLOPED, 1.0 - RECOVERY)
        assert protection_leg(_cds(), _model()) / NOTIONAL.amount == pytest.approx(
            through_t25, rel=2e-3
        )

    def test_REFINING_both_sides_CONVERGES_so_they_are_the_SAME_integral(self) -> None:
        """Convergence rather than a point match — the T31 lesson: two close numbers are not
        evidence of one quantity."""
        from pricebook.engine.credit_default_swap import protection_leg

        horizon = TM.year_fraction(SEMI_DATES[-1])

        def gap(points: int) -> float:
            grid = tuple(horizon * k / points for k in range(1, points + 1))
            profile = ExposureProfile(grid, tuple(1.0 for _ in grid))
            t25 = unilateral_cva(profile, TERM_HAZARD, SLOPED, 1.0 - RECOVERY)
            return abs(protection_leg(_cds(), _model(), steps=8000) / NOTIONAL.amount - t25)

        gaps = [gap(n) for n in (40, 160, 640)]
        assert all(b < a for a, b in zip(gaps, gaps[1:])), gaps


class TestTheRequirementOracles:
    def test_RPV01_is_positive_and_DECREASING_in_the_hazard(self) -> None:
        """More default risk shortens the expected annuity."""
        from pricebook.engine.credit_default_swap import risky_pv01

        values = [risky_pv01(_cds(), _model(SLOPED, _flat(h))) for h in (0.005, 0.03, 0.10)]
        assert all(v > 0.0 for v in values)
        assert all(b < a for a, b in zip(values, values[1:])), values

    def test_the_protection_leg_is_INCREASING_in_the_hazard(self) -> None:
        from pricebook.engine.credit_default_swap import protection_leg

        values = [protection_leg(_cds(), _model(SLOPED, _flat(h))) for h in (0.005, 0.03, 0.10)]
        assert all(b > a for a, b in zip(values, values[1:])), values

    def test_a_CDS_struck_AT_the_par_spread_is_worth_ZERO(self) -> None:
        """The par-reprice: a telescoping self-consistency guard, kept as one and not the anchor."""
        from pricebook.engine.credit_default_swap import par_spread

        model = _model()
        out = price(_cds(spread=par_spread(_cds(), model)), model)
        assert isinstance(out, PricingResult), out
        assert abs(out.pv.amount) < 1e-6 * NOTIONAL.amount

    def test_the_value_is_LINEAR_and_decreasing_in_the_contractual_spread(self) -> None:
        """`∂V/∂S₀ = −RPV01` for the buyer — exact, since `V = Protection − S₀·RPV01`."""
        from pricebook.engine.credit_default_swap import risky_pv01

        model = _model()
        rpv01 = risky_pv01(_cds(), model)

        def value(spread: float) -> float:
            out = price(_cds(spread=spread), model)
            assert isinstance(out, PricingResult)
            return out.pv.amount

        slope = (value(0.02) - value(0.01)) / 0.01
        assert slope == pytest.approx(-rpv01, rel=1e-12)

    def test_the_SELLER_is_the_exact_negative_of_the_BUYER(self) -> None:
        model = _model()
        buyer = price(_cds(), model)
        seller = price(_cds(direction=ProtectionDirection.SELLER), model)
        assert isinstance(buyer, PricingResult) and isinstance(seller, PricingResult)
        assert buyer.pv.amount == -seller.pv.amount

    def test_a_ZERO_hazard_makes_protection_worthless_and_the_buyer_pay_the_premium(self) -> None:
        from pricebook.engine.credit_default_swap import protection_leg, risky_pv01

        model = _model(SLOPED, _flat(0.0))
        assert protection_leg(_cds(), model) == 0.0
        out = price(_cds(), model)
        assert isinstance(out, PricingResult)
        assert out.pv.amount == pytest.approx(-SPREAD * risky_pv01(_cds(), model), rel=1e-12)

    def test_FULL_recovery_makes_protection_worthless(self) -> None:
        from pricebook.engine.credit_default_swap import protection_leg

        assert protection_leg(_cds(), _model(recovery=1.0)) == 0.0


class TestTheDiscretisationIsMeasuredNotToleranced:
    """O'Kane's protection leg is an integral; the trapezoid's error is measured, and bracketed."""

    def test_the_trapezoid_sits_INSIDE_the_Riemann_BRACKET(self) -> None:
        """A requirement for all parameters: the integrand is monotone decreasing in `t` on this
        fixture, so the left sum bounds above and the right sum below."""
        from pricebook.engine.credit_default_swap import protection_leg, riemann_bracket

        lower, upper = riemann_bracket(_cds(), _model(), steps=24)
        value = protection_leg(_cds(), _model(), steps=24)
        assert lower <= value <= upper, f"{value} outside [{lower}, {upper}]"
        assert upper > lower, "the bracket must be non-degenerate at this step count"

    def test_the_bracket_TIGHTENS_and_the_error_falls_at_the_stated_rate(self) -> None:
        """`O(1/M)` for the bracket width; the trapezoid itself converges faster. Asserting the
        *rate* rather than a flat band — the T25 convergence-plateau lesson."""
        from pricebook.engine.credit_default_swap import protection_leg, riemann_bracket

        widths = []
        for steps in (24, 96, 384):
            lower, upper = riemann_bracket(_cds(), _model(), steps=steps)
            widths.append(upper - lower)
        ratios = [a / b for a, b in zip(widths, widths[1:])]
        assert all(3.0 < r < 5.0 for r in ratios), f"expected ~4x per 4x refinement: {ratios}"

    def test_the_committed_step_count_RESOLVES_its_own_error(self) -> None:
        """State what a green rules out: the committed `M` against a very fine reference."""
        from pricebook.engine.credit_default_swap import protection_leg

        reference = protection_leg(_cds(), _model(), steps=20_000)
        committed = protection_leg(_cds(), _model())
        assert abs(committed / reference - 1.0) < 1e-6


class TestTheGuardsAreReachable:
    def test_a_RECOVERY_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="recovery"):
            CreditModel(_market(SLOPED, TERM_HAZARD), 1.5)

    def test_a_survival_curve_on_a_DIFFERENT_clock_is_refused(self) -> None:
        elsewhere = SurvivalCurve(TimeMeasure(date(2027, 6, 1), DC), (0.0,), (0.02,))
        with pytest.raises(ValueError, match="time measure"):
            CreditModel(_market(SLOPED, elsewhere), RECOVERY)

    def test_a_MATURED_cds_is_PV_zero_invariant_6(self) -> None:
        past = tuple(VAL - timedelta(days=365 * (3 - k)) for k in range(4))
        out = price(_cds(dates=past), _model())
        assert isinstance(out, PricingResult), out
        assert out.pv.amount == 0.0

    def test_a_snapshot_with_NO_survival_curve_is_a_PricingFailure(self) -> None:
        bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT: SLOPED}))
        with pytest.raises(ValueError, match="survival"):
            CreditModel(bare, RECOVERY)
