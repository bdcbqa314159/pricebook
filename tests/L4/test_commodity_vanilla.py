"""The commodity forward and the European commodity option — tranche 60, PRE-REGISTERED.
`CLARK-2014` ch 2.1.2, 2.2.3, 2.2.6.

**The new object is the FORWARD, not the option formula.** Black-76 has existed since tranche 5 —
`models/black.py` *is* Black-76 on a forward — and `engine/equity_vanilla.py` set the precedent of a
product engine reusing it. So this tranche builds the commodity **carry** and reuses the rest, which
is both the simpler design and the stronger oracle: the reduction to an existing verified function
is a check, where a second transcription of Black would only be another chance to be wrong.

**The source, transcribed from both renderings and reconciled (§4).** Clark ch 2.2.6, p.37:

    F_{0,T} = E^d[S_T]
            = S_0 · E^d[exp(σ W^d_T + (r^d − r^f − ½σ²)T)]        the pre-substitution step
            = S_0 · e^{(r^d − r^f)T}                               (2.55)
    F_{t,T} = S_t · e^{(r^d − r^f)(T−t)}                           (2.56)

`r^d` is the domestic risk-free rate and **`r^f` is the convenience yield** — Clark's own notation,
by explicit analogy with FX: §2.2.3 introduces a "foreign commodity bond" `B^f` with `dB^f = r^f B^f dt`,
and the resulting PDE (2.19) carries `(r^d − r^f)S ∂V/∂S` in the convection term and `r^d` alone in
the forcing term. That asymmetry *is* the convenience yield: it changes the drift, not the discounting.

**The two renderings disagree in presentation and agree in content**, which is what reconciliation
means: `-layout` prints `F0,T = S0 e^{(rd−rf)T}` with the superscripts lifted clear, while `-raw`
flattens them to `= S0e(rd−rf )T` and splits `W^d_T` across three lines. Neither is ambiguous about
which symbols are in the exponent.

**And the source derives it in steps, which is stronger than two renderings of one line (§4).** The
pre-substitution form above is an *independent* check on (2.55): it is the expectation before the
martingale `E[exp(σW_T − ½σ²T)] = 1` is applied. A misprint would have to corrupt both steps
identically to survive. `TestTheDerivationStepChecksTheFinalForm` evaluates that earlier step by
**Gauss-Hermite quadrature** — a different numerical path, not a second transcription — and requires
it to reproduce the closed form.

**Why the convenience yield is a carried field that is genuinely constitutive (§6).** Two commodity
underlyings identical but for `δ` are different instruments — they imply different forwards at every
maturity, which is the whole content of contango vs backwardation (Clark p.19). So it is carried and
exercised by the round-trip, not "the pricer might read it later".

**The kinship with `dividend_forward` is real and deliberately not merged.** A commodity forward
`S·e^{−δT}/df(T)` is arithmetically what `market/equity.py`'s `dividend_forward` computes for a stock
whose only convention is a continuous yield. They are **not** merged: an `EquityUnderlying` carries a
cash-dividend schedule, proportional dividends and a repo rate that a commodity does not have, so
sharing the type would hang carried-but-unread fields on every commodity (§6), and the charter's DRY
rule is explicit that two blocks representing *different ideas* must not be merged because it couples
things that will diverge — which these are about to, at T64's mean-reverting Schwartz model.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.quadrature import standard_normal_expectation
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market.commodity import (
    CommodityAsset,
    CommodityUnderlying,
    commodity_forward,
)
from pricebook.market.curve import DiscountCurve
from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black import black_price
from pricebook.models.commodity_black import CommodityBlackModel
from pricebook.products.commodity_option import CommodityOption, CommodityOptionKind

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
USD = Currency("USD")
WTI = CommodityAsset("WTI")

#: **Deliberately not `T = 1`** — at one year `(r − δ)·T` hides the time factor entirely, so a
#: mutation dropping `T` would survive. 912 days is 2.4986 years.
EXPIRY = VAL + timedelta(days=912)
YEARS = 912 / 365.0

#: `r ≠ δ`, **both non-zero and of different magnitude**: at `r = δ` the forward collapses to the
#: spot and the whole carry is invisible, and a mutation swapping them would survive.
RATE, CONVENIENCE = 0.042, 0.015
SPOT, VOL = 78.5, 0.31

FLAT = DiscountCurve.flat(VAL, RATE, DC)
#: A **sloped** curve, so the constructor is seen to read the curve rather than a scalar rate.
SLOPED = DiscountCurve(
    TimeMeasure(VAL, DC), (0.0, 1.0, 2.5, 5.0), (0.030, 0.038, 0.045, 0.051)
)


def _underlying(convenience: float = CONVENIENCE) -> CommodityUnderlying:
    return CommodityUnderlying(asset=WTI, spot=SPOT, convenience_yield=convenience)


def _market(curve: DiscountCurve = FLAT, convenience: float = CONVENIENCE) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): curve}),
        commodities={WTI: _underlying(convenience)},
    )


def _option(
    strike: float = 80.0, kind: CommodityOptionKind = CommodityOptionKind.CALL
) -> CommodityOption:
    return CommodityOption(
        asset=WTI, strike=strike, expiry=EXPIRY, notional=Money(1_000.0, USD), kind=kind
    )


def _pv(option: CommodityOption, market: MarketSnapshot | None = None) -> float:
    out = price(option, CommodityBlackModel(market or _market(), VOL))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


class TestThePremisesTheOraclesRestOn:
    """Asserted, not believed — the third bite of the tenth shape is that a premise only *believed*
    written is not written."""

    def test_the_horizon_is_NOT_one_year(self) -> None:
        assert abs(YEARS - 1.0) > 1.4, YEARS

    def test_the_rate_and_the_CONVENIENCE_YIELD_differ_and_neither_is_zero(self) -> None:
        """`r = δ` makes `F = S` and hides the entire carry; either being zero hides one of them;
        and equal magnitudes would let a mutation swapping `r` and `δ` survive."""
        assert RATE > 0.0 and CONVENIENCE > 0.0
        assert abs(RATE - CONVENIENCE) > 0.02, (RATE, CONVENIENCE)

    def test_the_carry_MOVES_the_forward_away_from_spot_materially(self) -> None:
        """The premise the whole tranche rests on: if the forward sat on the spot, every oracle
        below would pass against an implementation that ignored the carry entirely."""
        forward = commodity_forward(_underlying(), EXPIRY, FLAT)
        assert abs(forward / SPOT - 1.0) > 0.05, forward


class TestTheForwardPointwise:
    """§4's "every load-bearing formula is transcribed pointwise at least once". This is the one
    genuinely-new formula in the tranche, so it gets its pointwise oracle — the reference computed
    here from eq (2.56), never read back from the code."""

    @pytest.mark.parametrize("convenience", [0.0, 0.015, 0.04, 0.085])
    @pytest.mark.parametrize("days", [30, 365, 912, 1825])
    def test_the_forward_matches_eq_2_56(self, convenience: float, days: int) -> None:
        expiry = VAL + timedelta(days=days)
        years = days / 365.0
        expected = SPOT * math.exp((RATE - convenience) * years)
        assert commodity_forward(_underlying(convenience), expiry, FLAT) == pytest.approx(
            expected, rel=1e-13
        )

    def test_the_forward_reads_the_CURVE_not_a_scalar_rate(self) -> None:
        """On a sloped curve the carry is the curve's own zero rate at `T`, not a single `r`. The
        companion that stops the flat-curve oracle above being the only witness: a flat fixture
        cannot tell `1/df(T)` from `e^{rT}` for any single `r`."""
        expected = SPOT * math.exp(-CONVENIENCE * YEARS) / SLOPED.df_at(YEARS)
        assert commodity_forward(_underlying(), EXPIRY, SLOPED) == pytest.approx(
            expected, rel=1e-13
        )
        assert SLOPED.df_at(YEARS) != pytest.approx(FLAT.df_at(YEARS), rel=1e-6)

    def test_CONTANGO_and_BACKWARDATION_are_both_reachable(self) -> None:
        """Clark p.19 — the sign of `r − δ` is the whole economics, and the two cases must both be
        expressible. A mutation clamping the carry to one side dies here."""
        contango = commodity_forward(_underlying(0.0), EXPIRY, FLAT)
        backwardation = commodity_forward(_underlying(0.12), EXPIRY, FLAT)
        assert contango > SPOT, contango
        assert backwardation < SPOT, backwardation


class TestTheDerivationStepChecksTheFinalForm:
    """§4: where the source derives a formula in steps, an **earlier step is an independent check**
    on the final one — stronger than two renderings of one line, because a misprint would have to
    corrupt both steps identically.

    Clark p.37 prints `F_{0,T} = S_0 E^d[exp(σW^d_T + (r^d − r^f − ½σ²)T)]` *before* applying
    `E[exp(σW_T − ½σ²T)] = 1`. That expectation is evaluated here by **Gauss-Hermite quadrature** —
    a different numerical path, not a second transcription of the same line."""

    @pytest.mark.parametrize("convenience", [0.0, 0.015, 0.04])
    def test_the_PRE_SUBSTITUTION_expectation_reproduces_the_closed_form(
        self, convenience: float
    ) -> None:
        drift = (RATE - convenience - 0.5 * VOL * VOL) * YEARS
        diffusion = VOL * math.sqrt(YEARS)
        by_quadrature = SPOT * standard_normal_expectation(
            lambda z: math.exp(drift + diffusion * z)
        )
        assert by_quadrature == pytest.approx(
            commodity_forward(_underlying(convenience), EXPIRY, FLAT), rel=1e-10
        )

    def test_the_quadrature_reference_is_NOT_the_closed_form_in_disguise(self) -> None:
        """The premise that makes the check above independent: the quadrature genuinely integrates
        a volatility-dependent integrand, so it would disagree if the `−½σ²` convexity correction
        in the drift were wrong. Dropping it moves the answer by `e^{½σ²T}` ≈ 12.7%."""
        drift_without_convexity = (RATE - CONVENIENCE) * YEARS
        diffusion = VOL * math.sqrt(YEARS)
        wrong = SPOT * standard_normal_expectation(
            lambda z: math.exp(drift_without_convexity + diffusion * z)
        )
        correct = commodity_forward(_underlying(), EXPIRY, FLAT)
        assert wrong / correct - 1.0 > 0.10, (wrong, correct)


class TestTheCostOfCarryReduction:
    """The anchor and its mandatory companion (§4, T39)."""

    def test_ZERO_convenience_yield_is_PURE_cost_of_carry(self) -> None:
        """`δ = 0 ⇒ F = S·e^{rT}`, exact by construction rather than as a numerical limit."""
        assert commodity_forward(_underlying(0.0), EXPIRY, FLAT) == pytest.approx(
            SPOT * math.exp(RATE * YEARS), rel=1e-14
        )

    def test_the_companion_is_NON_DEGENERATE_in_the_very_parameter_the_reduction_KILLS(
        self,
    ) -> None:
        """**T39, stated as the charter demands it.** The reduction above is exact *because* it
        zeroes `δ` — which is precisely the parameter a convenience-yield defect lives in, so the
        anchor is structurally blind to that whole error class. The companion exercises a
        materially non-zero `δ`, and this test measures that it is material: 3.8% here."""
        with_yield = commodity_forward(_underlying(), EXPIRY, FLAT)
        without = commodity_forward(_underlying(0.0), EXPIRY, FLAT)
        assert abs(without / with_yield - 1.0) > 0.03, (with_yield, without)


class TestTheEngineIsAFaithfulBlackWrapper:
    def test_the_price_is_EXACTLY_discounted_black_on_the_same_forward(self) -> None:
        """The reduction that proves this tranche adds carry and **does not perturb Black**: the
        engine's output must equal a direct `black_price` call on the same forward, to the last
        bit. Anything less would mean the engine re-derived something already verified."""
        forward = commodity_forward(_underlying(), EXPIRY, FLAT)
        direct = black_price(forward, 80.0, VOL * VOL * YEARS, CommodityOptionKind.CALL)
        assert _pv(_option()) == 1_000.0 * FLAT.df(EXPIRY) * direct

    def test_the_total_variance_carries_its_TIME_factor(self) -> None:
        """Black-76 takes `σ²T`, not `σ`. The premise: at `T ≈ 2.5` the two differ by a factor of
        2.5 in variance, so a mutation dropping `T` cannot hide — which `T = 1` would have let it."""
        forward = commodity_forward(_underlying(), EXPIRY, FLAT)
        without_time = black_price(forward, 80.0, VOL * VOL, CommodityOptionKind.CALL)
        assert _pv(_option()) != pytest.approx(
            1_000.0 * FLAT.df(EXPIRY) * without_time, rel=1e-6
        )


class TestPutCallParityTheRequirementOracle:
    """Model-free, so it survives a transcription error the source-value oracles would share (§4).
    Swept **off the money**: at `K = F` a call and a put are equal and the identity degenerates to
    `0 = 0`, which every wrong implementation also satisfies."""

    @pytest.mark.parametrize("strike", [45.0, 65.0, 95.0, 130.0])
    def test_call_minus_put_is_the_discounted_forward_less_strike(self, strike: float) -> None:
        call = _pv(_option(strike, CommodityOptionKind.CALL))
        put = _pv(_option(strike, CommodityOptionKind.PUT))
        forward = commodity_forward(_underlying(), EXPIRY, FLAT)
        expected = 1_000.0 * FLAT.df(EXPIRY) * (forward - strike)
        assert call - put == pytest.approx(expected, rel=1e-12)

    def test_the_SWEEP_is_genuinely_off_the_money(self) -> None:
        """The premise: every strike above must sit away from the forward, or parity is `0 = 0`."""
        forward = commodity_forward(_underlying(), EXPIRY, FLAT)
        for strike in (45.0, 65.0, 95.0, 130.0):
            assert abs(strike / forward - 1.0) > 0.1, (strike, forward)


class TestTheCrossSourceAgreementIsAConsistencyGuardNOTIndependence:
    """**Named for what it is, because the T17 lesson was exactly this.** `CLARK-2014` §2.2.4 and
    `BERHAN-2012` (our `black.py`) render the **same** Black-76 — one object, two printings — so
    their agreement is guaranteed by construction and proves nothing about either transcription.

    It is kept as a consistency guard and labelled as one. What it would catch is a future drift
    between the commodity engine and `black_price`; what it cannot do is supply the cross-source
    independence it superficially resembles."""

    def test_the_commodity_engine_and_black_py_agree_BY_CONSTRUCTION(self) -> None:
        forward = commodity_forward(_underlying(), EXPIRY, FLAT)
        for strike in (60.0, 80.0, 110.0):
            engine = _pv(_option(strike)) / (1_000.0 * FLAT.df(EXPIRY))
            assert engine == black_price(
                forward, strike, VOL * VOL * YEARS, CommodityOptionKind.CALL
            )


class TestTheContractAndItsGuards:
    def test_an_EXPIRED_option_is_PV_ZERO_and_needs_no_commodity_data(self) -> None:
        """Invariant 6, checked before any market lookup — so a settled trade prices without the
        snapshot carrying the commodity at all."""
        expired = CommodityOption(
            asset=WTI,
            strike=80.0,
            expiry=VAL - timedelta(days=1),
            notional=Money(1_000.0, USD),
            kind=CommodityOptionKind.CALL,
        )
        bare = MarketSnapshot(
            valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT})
        )
        out = price(expired, CommodityBlackModel(bare, VOL))
        assert isinstance(out, PricingResult) and out.pv.amount == 0.0

    def test_an_expired_IN_THE_MONEY_option_is_still_PV_ZERO(self) -> None:
        """F-4.1's sub-shape: an *out-of-the-money* expired option has intrinsic `0`, which is what
        a dropped invariant-6 guard would also return — so the guard stays invisible unless the
        fixture is **in the money**. Here `F > K` by a wide margin."""
        deep = CommodityOption(
            asset=WTI,
            strike=5.0,
            expiry=VAL - timedelta(days=1),
            notional=Money(1_000.0, USD),
            kind=CommodityOptionKind.CALL,
        )
        assert _pv(deep) == 0.0

    def test_a_MISSING_commodity_is_a_PricingFailure_not_a_crash(self) -> None:
        bare = MarketSnapshot(
            valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT})
        )
        out = price(_option(), CommodityBlackModel(bare, VOL))
        assert isinstance(out, PricingFailure) and "WTI" in out.reason

    def test_a_MISSING_discount_curve_is_a_PricingFailure(self) -> None:
        out = price(
            _option(),
            CommodityBlackModel(
                MarketSnapshot(valuation_date=VAL, curves=CurveSet({}), commodities={WTI: _underlying()}),
                VOL,
            ),
        )
        assert isinstance(out, PricingFailure)

    def test_a_NON_POSITIVE_strike_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="strike"):
            _option(strike=-1.0)

    def test_a_NEGATIVE_volatility_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="volatilit"):
            CommodityBlackModel(_market(), -0.1)

    def test_a_NON_POSITIVE_spot_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="spot"):
            CommodityUnderlying(asset=WTI, spot=0.0, convenience_yield=CONVENIENCE)

    def test_the_CONVENIENCE_YIELD_is_constitutive_and_survives_the_round_trip(self) -> None:
        """§6: a carried field is allowed when it is constitutive of the instrument's identity, and
        then it is exercised by the round-trip. Two underlyings differing only in `δ` are different
        instruments — they imply different forwards — so it is carried, and equality sees it."""
        assert _underlying(0.02) != _underlying(0.03)
        assert _underlying(0.02) == _underlying(0.02)

    def test_the_option_round_trips_and_refuses_an_unknown_version(self) -> None:
        option = _option()
        assert CommodityOption.from_dict(option.to_dict()) == option
        bad = option.to_dict()
        bad["schema_version"] = 99
        with pytest.raises(ValueError, match="reads up to"):
            CommodityOption.from_dict(bad)
