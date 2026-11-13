"""Equity-option CVA under independent default (L5) — tranche 36, layer 1. **Pre-registered.**

The capstone's first half: project three (XVA, T25–T29) meets project four (equity, T30–T35).

    CVA = (1 − R)·(1 − p₀(T))·V₀,     V₀ = P₀(T)·BSCall(F₀(T), K, σ̂, T)

**Why the option case collapses to a closed form while the book's *forward* case needs an integral.**
Brockhaus 14.2.1 gives the general form `CVA = (1−R)N₀·E[(1/N_τ)1{0<τ≤T}·V_τ]`, with `V_τ` the
*remaining* option value at the default time. Under independence of `τ` and `(N,S)`, the discounted
option value is a **martingale**, so `E[V_t/N_t] = V₀/N₀` for every fixed `t` — the expectation does
not depend on when default happens. Integrating against the default density therefore gives
`(V₀/N₀)·P[0<τ≤T] = (V₀/N₀)(1−p₀(T))`, and the integral collapses.

The book's forward contract does **not** collapse, and the reason is instructive: its exposure is
`(F_t−K)⁺`, which is *not* the contract's value (a forward can be worth less than nothing), so the
martingale argument does not apply to it and the maturity-`t` option inside the integral is a
genuinely different object at each `t`. The distinction is the whole reason this tranche's layer 1
is a closed form at all.

**The headline oracle is a cross-ARC reduction.** The same number must come out of tranche 25's
`unilateral_cva` — a completely different composition, built in a different arc from a different
source, integrating `DF(t)·EPE(t)·dPD(t)` over a time grid. With deterministic rates
`EPE(t) = E[V_t] = V₀/DF(t)`, so T25's integrand is `DF(t)·(V₀/DF(t))·dPD = V₀·dPD` and its integral
is `V₀·(1−p(T))`. Two arcs, different code, one number.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation import Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import DividendSchedule, EquityAsset, EquityUnderlying, dividend_forward
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.equity_black import EquityBlackModel
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption
from pricebook.risk.credit_valuation_adjustment import ExposureProfile, unilateral_cva
from pricebook.risk.equity_cva import equity_option_cva

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
ACME = EquityAsset("ACME")
DISCOUNT = CurveKey(CurveRole.DISCOUNT, EUR)

SPOT, RATE, YIELD, VOL = 100.0, 0.03, 0.01, 0.28
EXPIRY = VAL + timedelta(days=1095)
YEARS = 3.0
STRIKE = 105.0
NOTIONAL = Money(1_000_000.0, EUR)
RECOVERY = 0.4

#: NON-FLAT and read beyond its last pillar — the T25 degeneracy, inherited.
SURVIVAL = SurvivalCurve(TM, (0.0, 1.0, 2.0), (0.012, 0.035, 0.065))
CURVE = DiscountCurve.flat(VAL, RATE, DC)
UNDERLYING = EquityUnderlying(SPOT, 0.0, DividendSchedule(continuous_yield=YIELD))
MARKET = MarketSnapshot(
    valuation_date=VAL, curves=CurveSet({DISCOUNT: CURVE}), equities={ACME: UNDERLYING}
)
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)
DF = CURVE.df(EXPIRY)
OPTION = EuropeanEquityOption(ACME, STRIKE, EXPIRY, NOTIONAL, EquityOptionKind.CALL)


def _option_value() -> float:
    out = price(OPTION, EquityBlackModel(MARKET, VOL))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _cva(survival: SurvivalCurve = SURVIVAL, recovery: float = RECOVERY) -> float:
    return equity_option_cva(OPTION, EquityBlackModel(MARKET, VOL), survival, recovery)


class TestTheFixtureIsNotDegenerate:
    def test_the_default_probability_is_MATERIAL(self) -> None:
        assert 1.0 - SURVIVAL.survival(YEARS) > 0.08, SURVIVAL.survival(YEARS)

    def test_the_survival_curve_is_NON_FLAT_and_read_beyond_its_last_pillar(self) -> None:
        assert max(SURVIVAL.hazards) - min(SURVIVAL.hazards) > 0.04
        assert YEARS > SURVIVAL.times[-1]
        assert SURVIVAL.survival(YEARS) != pytest.approx(
            math.exp(-SURVIVAL.hazards[0] * YEARS), rel=1e-3
        )

    def test_the_RECOVERY_is_neither_zero_nor_one(self) -> None:
        """At `R = 0` the `(1−R)` factor is invisible; at `R = 1` the whole CVA is."""
        assert 0.1 < RECOVERY < 0.9


class TestTheCrossArcReduction:
    """The headline: tranche 25's CVA, built in a different arc from a different source, must
    produce the same number by a completely different composition."""

    def test_the_closed_form_equals_TRANCHE_25s_unilateral_cva(self) -> None:
        """T25 integrates `DF(t)·EPE(t)·dPD(t)` on a time grid; this is a closed form in `p(T)`.

        With deterministic rates `EPE(t) = E[V_t] = V₀/DF(t)`, so T25's integrand is `V₀·dPD(t)` and
        its integral is `V₀·(1−p(T))`. The agreement is therefore exact in the continuum, and the
        gap here is T25's trapezoid discretisation alone.
        """
        value = _option_value()
        grid = tuple(YEARS * k / 400 for k in range(1, 401))
        profile = ExposureProfile(grid, tuple(value / CURVE.df_at(t) for t in grid))
        through_t25 = unilateral_cva(profile, SURVIVAL, CURVE, 1.0 - RECOVERY)
        assert _cva() == pytest.approx(through_t25, rel=2e-3)

    def test_REFINING_the_T25_grid_CONVERGES_to_the_closed_form(self) -> None:
        """The rate is what proves they are the same quantity rather than two close numbers — the
        T31 lesson. A wrong closed form converges to the wrong limit and the ratio flattens."""
        value = _option_value()

        def gap(points: int) -> float:
            grid = tuple(YEARS * k / points for k in range(1, points + 1))
            profile = ExposureProfile(grid, tuple(value / CURVE.df_at(t) for t in grid))
            return abs(unilateral_cva(profile, SURVIVAL, CURVE, 1.0 - RECOVERY) - _cva())

        gaps = [gap(n) for n in (50, 200, 800)]
        assert all(b < a for a, b in zip(gaps, gaps[1:])), gaps
        assert gaps[-1] / gaps[0] < 0.1


class TestTheClosedForm:
    def test_the_cva_is_the_loss_rate_times_the_default_probability_times_the_option(self) -> None:
        expected = (1.0 - RECOVERY) * (1.0 - SURVIVAL.survival(YEARS)) * _option_value()
        assert _cva() == pytest.approx(expected, rel=1e-15)

    def test_a_ZERO_hazard_gives_a_ZERO_cva(self) -> None:
        assert _cva(SurvivalCurve(TM, (0.0,), (0.0,))) == 0.0

    def test_FULL_recovery_gives_a_ZERO_cva(self) -> None:
        assert _cva(recovery=1.0) == 0.0

    def test_the_cva_is_BOUNDED_by_the_full_loss_value(self) -> None:
        """`p → 0` is the worst case: the whole option value, times the loss rate."""
        certain = SurvivalCurve(TM, (0.0,), (40.0,))
        assert _cva(certain) == pytest.approx((1.0 - RECOVERY) * _option_value(), rel=1e-9)
        assert _cva() < (1.0 - RECOVERY) * _option_value()

    def test_the_cva_RISES_with_the_hazard_and_FALLS_with_the_recovery(self) -> None:
        hazards = [_cva(SurvivalCurve(TM, (0.0,), (h,))) for h in (0.01, 0.05, 0.2)]
        assert all(b > a for a, b in zip(hazards, hazards[1:])), hazards
        recoveries = [_cva(recovery=r) for r in (0.1, 0.4, 0.8)]
        assert all(b < a for a, b in zip(recoveries, recoveries[1:])), recoveries


class TestTheGuardsAreReachable:
    def test_a_RECOVERY_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="recovery"):
            _cva(recovery=1.4)

    def test_a_survival_curve_on_a_DIFFERENT_clock_is_refused(self) -> None:
        elsewhere = SurvivalCurve(TimeMeasure(date(2027, 6, 1), DC), SURVIVAL.times, SURVIVAL.hazards)
        with pytest.raises(ValueError, match="time measure"):
            _cva(elsewhere)

    def test_an_UNPRICEABLE_option_surfaces_the_failure_rather_than_a_number(self) -> None:
        bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT: CURVE}))
        with pytest.raises(ValueError, match="cannot price"):
            equity_option_cva(OPTION, EquityBlackModel(bare, VOL), SURVIVAL, RECOVERY)
