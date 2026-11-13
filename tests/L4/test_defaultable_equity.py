"""Defaultable equity under independent default (L3/L4) — tranche 35. **Pre-registered.**

Brockhaus 13.1.1: a stock that can jump to zero, `S_t = S̃_t·1{τ>t}`, with `S̃` lognormal and the
default time `τ` **independent** of the diffusion. Writing `p(t)` for the survival probability:

    F̃(t) = F(t)/p(t)                                                            (13.3)
    E[(S_t − K)⁺] = p·BSCall(F/p, K, σ̃, t)  =  BSCall(F, p·K, σ̃, t)
    E[(K − S_t)⁺] = (1−p)K + p·BSPut(F/p, K, σ̃, t)  =  (1−p)K + BSPut(F, p·K, σ̃, t)

**The book prints both forms of each**, which hands the tranche a source-level identity: Black is
degree-1 homogeneous in `(forward, strike)`, so `p·BSCall(F/p, K) = BSCall(F, pK)` is not a
rearrangement someone invented but the text's own pair. Asserting it checks the strike-scaling
transcription against the scaled-forward one — two renderings of the same object, in the source
itself.

**The reduction is exact and vacuous alone**, which is the whole reason for the companion. At `h → 0`
every defaultable term disappears — `p = 1`, `p·K → K`, `(1−p)K → 0` — so an implementation that
ignored the survival curve entirely would still pass it. The companion therefore needs `p(t) < 1`
**materially**, and a **non-flat** survival curve read **beyond its first segment**, so a mutant
reading `hazard(0)` and exponentiating is separated from one integrating the hazard properly (the
tranche-25 survival-curve degeneracy, which this fixture inherits rather than rediscovers).
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.defaultable_equity import (
    defaultable_call,
    defaultable_put,
    surviving_forward,
)
from pricebook.foundation import Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import DividendSchedule, EquityAsset, EquityUnderlying, dividend_forward
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.black import black_price
from pricebook.models.defaultable_equity import DefaultableEquityModel
from pricebook.models.equity_black import EquityBlackModel
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
ACME = EquityAsset("ACME")
DISCOUNT = CurveKey(CurveRole.DISCOUNT, EUR)

SPOT, RATE, YIELD = 100.0, 0.03, 0.01
VOL = 0.30
EXPIRY = VAL + timedelta(days=1460)   # 4 years — past the curve's last pillar, deliberately
YEARS = 4.0
STRIKE = 95.0
NOTIONAL = Money(1.0, EUR)

#: NON-FLAT and read beyond its first segment — the T25 degeneracy, inherited on purpose.
SURVIVAL = SurvivalCurve(TM, (0.0, 1.0, 2.5), (0.015, 0.045, 0.080))
CURVE = DiscountCurve.flat(VAL, RATE, DC)
UNDERLYING = EquityUnderlying(SPOT, 0.0, DividendSchedule(continuous_yield=YIELD))
MARKET = MarketSnapshot(
    valuation_date=VAL, curves=CurveSet({DISCOUNT: CURVE}), equities={ACME: UNDERLYING}
)
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)
DF = CURVE.df(EXPIRY)
SURVIVAL_PROB = SURVIVAL.survival(YEARS)


def _model(survival: SurvivalCurve = SURVIVAL, vol: float = VOL) -> DefaultableEquityModel:
    return DefaultableEquityModel(MARKET, vol, survival)


def _flat(hazard: float) -> SurvivalCurve:
    return SurvivalCurve(TM, (0.0,), (hazard,))


def _option(kind: EquityOptionKind = EquityOptionKind.CALL, strike: float = STRIKE):
    return EuropeanEquityOption(ACME, strike, EXPIRY, NOTIONAL, kind)


def _pv(option, model) -> float:
    out = price(option, model)
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _black(forward: float, strike: float, kind: EquityOptionKind) -> float:
    return black_price(forward, strike, VOL * VOL * YEARS, kind)


class TestTheFixtureIsNotDegenerate:
    def test_the_survival_probability_is_MATERIALLY_below_one(self) -> None:
        """Else every defaultable term is invisible and the reduction is the only thing tested."""
        assert SURVIVAL_PROB < 0.85, f"p(t) = {SURVIVAL_PROB}"

    def test_the_survival_curve_is_NON_FLAT_and_read_BEYOND_its_first_segment(self) -> None:
        """The T25 degeneracy: a flat hazard makes `p = e^{−h t}`, so reading `hazard(0)` and
        exponentiating agrees everywhere and a mis-integrated hazard is invisible."""
        assert max(SURVIVAL.hazards) - min(SURVIVAL.hazards) > 0.05
        assert YEARS > SURVIVAL.times[-1], "the horizon must sit past the last pillar"
        assert SURVIVAL_PROB != pytest.approx(math.exp(-SURVIVAL.hazards[0] * YEARS), rel=1e-3)

    def test_the_strike_is_chosen_so_the_SCALING_moves_the_price_materially(self) -> None:
        plain = _black(FORWARD, STRIKE, EquityOptionKind.CALL)
        scaled = _black(FORWARD, SURVIVAL_PROB * STRIKE, EquityOptionKind.CALL)
        assert abs(scaled / plain - 1.0) > 0.05, f"{scaled} vs {plain}"


class TestTheSourcesOwnTwoFormsAgree:
    """`p·BSCall(F/p, K) = BSCall(F, pK)` — Black's degree-1 homogeneity, and the book prints both.

    An identity in exact arithmetic, so the two agree to a few ULP rather than bit-identically: the
    operations are reordered (one scales the forward, the other the strike), which is §4's
    exact-arithmetic case rather than its bit-identity case.
    """

    @pytest.mark.parametrize("strike", [60.0, 95.0, 140.0, 200.0])
    def test_the_SCALED_STRIKE_and_SCALED_FORWARD_calls_agree(self, strike: float) -> None:
        scaled_forward = SURVIVAL_PROB * _black(
            FORWARD / SURVIVAL_PROB, strike, EquityOptionKind.CALL
        )
        scaled_strike = _black(FORWARD, SURVIVAL_PROB * strike, EquityOptionKind.CALL)
        assert scaled_forward == pytest.approx(scaled_strike, rel=1e-14)

    @pytest.mark.parametrize("strike", [60.0, 95.0, 140.0])
    def test_the_two_PUT_forms_agree(self, strike: float) -> None:
        scaled_forward = (1.0 - SURVIVAL_PROB) * strike + SURVIVAL_PROB * _black(
            FORWARD / SURVIVAL_PROB, strike, EquityOptionKind.PUT
        )
        scaled_strike = (1.0 - SURVIVAL_PROB) * strike + _black(
            FORWARD, SURVIVAL_PROB * strike, EquityOptionKind.PUT
        )
        assert scaled_forward == pytest.approx(scaled_strike, rel=1e-14)

    def test_the_engine_uses_the_SCALED_STRIKE_form(self) -> None:
        """Named so the choice is visible: it needs no division by `p`, which vanishes as the hazard
        grows, and is exactly the default-free formula at `p = 1`."""
        assert defaultable_call(FORWARD, STRIKE, SURVIVAL_PROB, VOL, YEARS) == pytest.approx(
            _black(FORWARD, SURVIVAL_PROB * STRIKE, EquityOptionKind.CALL), rel=1e-15
        )


class TestTheSurvivingForwardIdentity:
    def test_the_surviving_forward_is_F_over_p(self) -> None:
        """Eq 13.3 — and it is what makes `E[S_t] = p·F̃ = F` hold."""
        assert surviving_forward(FORWARD, SURVIVAL_PROB) == pytest.approx(
            FORWARD / SURVIVAL_PROB, rel=1e-15
        )

    def test_the_DEFAULT_RISKY_forward_is_still_the_plain_forward(self) -> None:
        """`E[S_t] = p(t)·F̃(t) = F(t)` — the requirement that pins the scaling independently of the
        volatility, and it holds for every survival probability."""
        for probability in (0.3, 0.6, 0.95, 1.0):
            assert SURVIVAL_PROB * 0 + probability * surviving_forward(
                FORWARD, probability
            ) == pytest.approx(FORWARD, rel=1e-14)


class TestTheExactReductionToDefaultFreeEquity:
    def test_a_ZERO_hazard_is_BIT_IDENTICAL_to_tranche_30s_black(self) -> None:
        certain = _flat(0.0)
        assert _pv(_option(), _model(certain)) == _pv(_option(), EquityBlackModel(MARKET, VOL))

    def test_the_PUT_reduction_is_bit_identical_too(self) -> None:
        certain = _flat(0.0)
        assert _pv(_option(EquityOptionKind.PUT), _model(certain)) == _pv(
            _option(EquityOptionKind.PUT), EquityBlackModel(MARKET, VOL)
        )

    def test_THE_COMPANION_a_live_hazard_moves_BOTH_sides_in_the_right_DIRECTION(self) -> None:
        """A call on a stock that can become worthless is worth **less**; the put is worth **more**
        (it collects `K` on default). Magnitudes asserted, not just signs."""
        plain_call = _pv(_option(), EquityBlackModel(MARKET, VOL))
        plain_put = _pv(_option(EquityOptionKind.PUT), EquityBlackModel(MARKET, VOL))
        risky_call = _pv(_option(), _model())
        risky_put = _pv(_option(EquityOptionKind.PUT), _model())
        assert risky_call < plain_call and abs(risky_call / plain_call - 1.0) > 0.05
        assert risky_put > plain_put and abs(risky_put / plain_put - 1.0) > 0.05


class TestTheModelFreeRequirements:
    def test_PUT_CALL_PARITY_holds_with_the_DEFAULT_TERMS_CANCELLING(self) -> None:
        """`C − P = df·(F − K)` — the survival scaling and the default lump cancel **exactly**:
        `BSCall(F,pK) − (1−p)K − BSPut(F,pK) = (F − pK) − (1−p)K = F − K`. It holds for every `p`,
        so it is blind to a transcription slip in either formula and catches one in only one.
        """
        for strike in (60.0, 95.0, 140.0):
            call = _pv(_option(strike=strike), _model())
            put = _pv(_option(EquityOptionKind.PUT, strike=strike), _model())
            assert call - put == pytest.approx(DF * (FORWARD - strike), rel=1e-12)

    def test_the_call_FALLS_and_the_put_RISES_as_the_hazard_grows(self) -> None:
        calls = [_pv(_option(), _model(_flat(h))) for h in (0.0, 0.05, 0.15, 0.4)]
        puts = [_pv(_option(EquityOptionKind.PUT), _model(_flat(h))) for h in (0.0, 0.05, 0.15, 0.4)]
        assert all(b < a for a, b in zip(calls, calls[1:])), calls
        assert all(b > a for a, b in zip(puts, puts[1:])), puts

    def test_the_call_respects_the_NO_ARBITRAGE_bounds(self) -> None:
        for strike in (50.0, 95.0, 180.0):
            call = _pv(_option(strike=strike), _model())
            assert 0.0 <= call <= DF * FORWARD + 1e-12

    def test_a_CERTAIN_default_makes_the_call_worthless_and_the_put_worth_the_strike(self) -> None:
        """`p → 0`: nothing survives, so the call is zero and the put is the discounted strike."""
        doomed = SurvivalCurve(TM, (0.0,), (50.0,))   # p(4y) ~ 1e-87
        assert _pv(_option(), _model(doomed)) == pytest.approx(0.0, abs=1e-9)
        assert _pv(_option(EquityOptionKind.PUT), _model(doomed)) == pytest.approx(
            DF * STRIKE, rel=1e-9
        )


class TestTheGuardsAreReachable:
    def test_a_NEGATIVE_volatility_is_refused(self) -> None:
        with pytest.raises(ValueError, match="volatilit"):
            DefaultableEquityModel(MARKET, -0.1, SURVIVAL)

    def test_an_EXPIRED_option_is_PV_zero_invariant_6(self) -> None:
        expired = EuropeanEquityOption(ACME, STRIKE, VAL, NOTIONAL, EquityOptionKind.CALL)
        assert _pv(expired, _model()) == 0.0

    def test_an_unknown_asset_is_a_PricingFailure(self) -> None:
        bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT: CURVE}))
        assert not isinstance(price(_option(), DefaultableEquityModel(bare, VOL, SURVIVAL)), PricingResult)

    def test_a_survival_curve_on_a_DIFFERENT_clock_is_refused(self) -> None:
        """C12's family: the option's `t` is read on both the discount curve and the survival curve,
        so two anchors would make one `t` mean two dates."""
        elsewhere = SurvivalCurve(TimeMeasure(date(2027, 6, 1), DC), SURVIVAL.times, SURVIVAL.hazards)
        with pytest.raises(ValueError, match="time measure"):
            DefaultableEquityModel(MARKET, VOL, elsewhere)
