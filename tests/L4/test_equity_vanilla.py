"""The equity European option and its dividend forward (L1/L2/L4) — tranche 30.

First unit of the equity & hybrids arc, and the first **equity** vertical in the tree. The option
itself is not new work: `black_price` has been verified since tranche 5, so the vanilla is
`black_price` on an equity forward and the tranche's real content is **the forward**.

    F(T) = (S0 - sum_{t_i <= T} D_i*df(t_i)) * prod_{t_i <= T}(1 - d_i) * exp(-(q+b)T) / df(T)

three dividend conventions at once (discrete cash `D_i`, proportional `d_i`, continuous yield `q`)
plus a repo/borrow rate `b`. Brockhaus ch 5.1-5.3.

**The anchor is the exact reduction to the verified core**: zero dividends and zero repo give
`F = S0/df(T)` and the engine must be **bit-identically** `black_price` on that forward. Tranche 30
must not perturb Black. Its companion is a *material* dividend stream where the forward sits visibly
below `S0/df(T)` — a reduction a no-op also satisfies is vacuous (§4).

**The non-obvious degeneracy here is a DATE boundary, not a value.** Every oracle below would pass
if the `t_i <= T` restriction were dropped, *provided every dividend in the fixture falls before
expiry*. So the fixture carries a dividend **after** expiry, and a test asserts it cannot move
`F(T)` — the boundary the sum must get right, in the same family as invariant 6's `pay >
valuation_date` and C10's 41% mispricing.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation import Currency, Money, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import (
    DividendSchedule,
    EquityAsset,
    EquityUnderlying,
    dividend_forward,
)
from pricebook.models.black import black_price
from pricebook.models.equity_black import EquityBlackModel
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
ACME = EquityAsset("ACME")
DISCOUNT = CurveKey(CurveRole.DISCOUNT, EUR)

SPOT = 100.0
RATE = 0.04
VOL = 0.25
EXPIRY = VAL + timedelta(days=730)
STRIKE = 105.0
NOTIONAL = Money(1.0, EUR)

#: MATERIAL dividends, and one of them lands AFTER expiry on purpose (see the module docstring).
CASH = ((VAL + timedelta(days=180), 2.5), (VAL + timedelta(days=545), 2.5))
PROPORTIONAL = ((VAL + timedelta(days=300), 0.015),)
POST_EXPIRY = (VAL + timedelta(days=900), 9.0)
#: The same boundary for the PROPORTIONAL branch. Needed because the `t_i <= T` restriction is a
#: SEPARATE line of code per convention: with only a post-expiry *cash* dividend in the fixture, a
#: mutant dropping the proportional restriction survived the whole suite. The trap this module's
#: docstring names, half-applied by its own author and caught by the sweep.
POST_EXPIRY_PROPORTIONAL = (VAL + timedelta(days=900), 0.08)


def _curve(rate: float = RATE) -> DiscountCurve:
    return DiscountCurve.flat(VAL, rate, DC)


def _underlying(
    cash: tuple[tuple[date, float], ...] = CASH,
    proportional: tuple[tuple[date, float], ...] = PROPORTIONAL,
    continuous: float = 0.01,
    repo: float = 0.005,
) -> EquityUnderlying:
    return EquityUnderlying(SPOT, repo, DividendSchedule(cash, proportional, continuous))


def _market(underlying: EquityUnderlying | None = None, rate: float = RATE) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT: _curve(rate)}),
        equities={ACME: underlying if underlying is not None else _underlying()},
    )


def _option(kind: EquityOptionKind = EquityOptionKind.CALL, strike: float = STRIKE) -> EuropeanEquityOption:
    return EuropeanEquityOption(ACME, strike, EXPIRY, NOTIONAL, kind)


def _pv(option: EuropeanEquityOption, market: MarketSnapshot, vol: float = VOL) -> float:
    out = price(option, EquityBlackModel(market, vol))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def _years(d: date) -> float:
    return (d - VAL).days / 365.0


class TestTheFixtureIsNotDegenerate:
    def test_the_dividend_stream_is_MATERIAL(self) -> None:
        """Else `F` equals the no-dividend forward and every bit of equity content is invisible."""
        plain = dividend_forward(EquityUnderlying(SPOT, 0.0, DividendSchedule()), EXPIRY, _curve())
        rich = dividend_forward(_underlying(), EXPIRY, _curve())
        assert plain / rich - 1.0 > 0.05, f"dividends must bite: {plain} vs {rich}"

    def test_all_THREE_dividend_conventions_are_exercised_at_once(self) -> None:
        schedule = _underlying().dividends
        assert schedule.cash and schedule.proportional and schedule.continuous_yield > 0.0

    def test_the_fixture_carries_a_post_expiry_dividend_of_EVERY_discrete_kind(self) -> None:
        """The boundary is enforced by a separate line per convention, so each needs its own
        post-expiry entry. Guarding only the cash branch let the proportional mutant live."""
        assert POST_EXPIRY[0] > EXPIRY
        assert POST_EXPIRY_PROPORTIONAL[0] > EXPIRY


class TestTheDividendForwardIsTranscribedPointwise:
    """The test computes its own reference from ch 5.1-5.3, never reading it back from the code."""

    def test_the_forward_matches_an_independently_composed_reference(self) -> None:
        curve = _curve()
        pv_cash = math.fsum(amount * curve.df(when) for when, amount in CASH if when <= EXPIRY)
        growth = math.prod(1.0 - rate for when, rate in PROPORTIONAL if when <= EXPIRY)
        carry = math.exp(-(0.01 + 0.005) * _years(EXPIRY))
        expected = (SPOT - pv_cash) * growth * carry / curve.df(EXPIRY)
        assert dividend_forward(_underlying(), EXPIRY, curve) == pytest.approx(expected, rel=1e-14)

    def test_a_CONTINUOUS_yield_alone_is_exactly_the_textbook_form(self) -> None:
        """`F = S0 * exp((r - q)T)` — the one form with a closed expression to check against."""
        underlying = EquityUnderlying(SPOT, 0.0, DividendSchedule(continuous_yield=0.03))
        got = dividend_forward(underlying, EXPIRY, _curve())
        assert got == pytest.approx(SPOT * math.exp((RATE - 0.03) * _years(EXPIRY)), rel=1e-14)

    def test_DISCRETE_cash_dividends_are_PV_discounted_not_summed_raw(self) -> None:
        """The commonest slip: `sum(D_i)` instead of `sum(D_i * df(t_i))`."""
        curve = _curve()
        underlying = EquityUnderlying(SPOT, 0.0, DividendSchedule(cash=CASH))
        discounted = math.fsum(a * curve.df(w) for w, a in CASH)
        raw = math.fsum(a for _, a in CASH)
        assert discounted < raw, "the premise: discounting must matter on this fixture"
        got = dividend_forward(underlying, EXPIRY, curve)
        assert got == pytest.approx((SPOT - discounted) / curve.df(EXPIRY), rel=1e-14)
        assert got != pytest.approx((SPOT - raw) / curve.df(EXPIRY), rel=1e-6)


class TestTheDateBoundary:
    def test_a_dividend_AFTER_expiry_cannot_move_the_forward(self) -> None:
        """A shareholder at `T` never receives it, so `F(T)` cannot know about it."""
        without = dividend_forward(_underlying(), EXPIRY, _curve())
        with_late = dividend_forward(_underlying(cash=CASH + (POST_EXPIRY,)), EXPIRY, _curve())
        assert with_late == without

    def test_a_PROPORTIONAL_dividend_after_expiry_cannot_move_the_forward_either(self) -> None:
        """The same boundary, the other branch — the one the sweep found unguarded."""
        without = dividend_forward(_underlying(), EXPIRY, _curve())
        with_late = dividend_forward(
            _underlying(proportional=PROPORTIONAL + (POST_EXPIRY_PROPORTIONAL,)), EXPIRY, _curve()
        )
        assert with_late == without

    def test_the_late_PROPORTIONAL_dividend_is_material_at_a_later_expiry(self) -> None:
        """The companion: without it the test above would pass on a negligible dividend."""
        later = POST_EXPIRY_PROPORTIONAL[0] + timedelta(days=30)
        without = dividend_forward(_underlying(), later, _curve())
        with_late = dividend_forward(
            _underlying(proportional=PROPORTIONAL + (POST_EXPIRY_PROPORTIONAL,)), later, _curve()
        )
        assert with_late < 0.95 * without

    def test_but_it_DOES_move_a_LATER_expiry(self) -> None:
        """The companion: the post-expiry dividend is material, so the test above is not vacuous."""
        later = POST_EXPIRY[0] + timedelta(days=30)
        without = dividend_forward(_underlying(), later, _curve())
        with_late = dividend_forward(_underlying(cash=CASH + (POST_EXPIRY,)), later, _curve())
        assert with_late < 0.95 * without


class TestTheCrossFormConsistency:
    """A strip of small proportional dividends must converge to the continuous yield — different
    formulas for the same object, so a shared transcription error shows up here."""

    def test_a_FINE_strip_of_proportional_dividends_converges_to_the_continuous_yield(self) -> None:
        q, years = 0.03, _years(EXPIRY)
        target = dividend_forward(
            EquityUnderlying(SPOT, 0.0, DividendSchedule(continuous_yield=q)), EXPIRY, _curve()
        )

        def strip_error(steps: int) -> float:
            dt = years / steps
            strip = tuple(
                (VAL + timedelta(days=int(round(365 * dt * (k + 1)))), q * dt) for k in range(steps)
            )
            got = dividend_forward(
                EquityUnderlying(SPOT, 0.0, DividendSchedule(proportional=strip)), EXPIRY, _curve()
            )
            return abs(got / target - 1.0)

        coarse, fine = strip_error(8), strip_error(128)
        assert fine < coarse, f"must converge: {coarse} -> {fine}"
        assert fine < 1e-4, f"a fine strip must reach the continuous form: {fine}"


class TestTheReductionToTheVerifiedBlackCore:
    def test_with_NO_dividends_and_NO_repo_the_option_is_BIT_IDENTICALLY_black(self) -> None:
        """Tranche 30 must not perturb Black, verified since tranche 5."""
        plain = EquityUnderlying(SPOT, 0.0, DividendSchedule())
        market = _market(plain)
        curve = _curve()
        years = _years(EXPIRY)
        forward = SPOT / curve.df(EXPIRY)
        expected = curve.df(EXPIRY) * black_price(forward, STRIKE, VOL * VOL * years, EquityOptionKind.CALL)
        assert _pv(_option(), market) == expected

    def test_the_COMPANION_a_material_dividend_stream_moves_the_price(self) -> None:
        """A reduction a no-op also satisfies is vacuous — so assert the general case differs."""
        plain = _pv(_option(), _market(EquityUnderlying(SPOT, 0.0, DividendSchedule())))
        rich = _pv(_option(), _market())
        assert rich < 0.85 * plain, f"dividends must lower a call materially: {rich} vs {plain}"


class TestTheRequirementOraclesTheFinanceForces:
    def test_PUT_CALL_PARITY_holds_on_the_equity_option(self) -> None:
        """`C - P = df*(F - K)` — model-free, so it survives a transcription error in the forward."""
        market = _market()
        call = _pv(_option(EquityOptionKind.CALL), market)
        put = _pv(_option(EquityOptionKind.PUT), market)
        curve = _curve()
        forward = dividend_forward(_underlying(), EXPIRY, curve)
        assert call - put == pytest.approx(curve.df(EXPIRY) * (forward - STRIKE), rel=1e-12)

    def test_the_forward_FALLS_with_every_dividend_and_with_the_repo(self) -> None:
        base = dividend_forward(_underlying(), EXPIRY, _curve())
        assert dividend_forward(_underlying(cash=((CASH[0][0], 6.0), CASH[1])), EXPIRY, _curve()) < base
        assert dividend_forward(_underlying(proportional=((PROPORTIONAL[0][0], 0.05),)), EXPIRY, _curve()) < base
        assert dividend_forward(_underlying(continuous=0.04), EXPIRY, _curve()) < base
        assert dividend_forward(_underlying(repo=0.03), EXPIRY, _curve()) < base

    def test_the_forward_tends_to_the_no_dividend_case_as_dividends_vanish(self) -> None:
        plain = SPOT / _curve().df(EXPIRY)
        for scale in (1.0, 0.1, 0.001):
            got = dividend_forward(
                _underlying(
                    cash=tuple((w, a * scale) for w, a in CASH),
                    proportional=tuple((w, r * scale) for w, r in PROPORTIONAL),
                    continuous=0.01 * scale,
                    repo=0.005 * scale,
                ),
                EXPIRY,
                _curve(),
            )
            assert abs(got / plain - 1.0) < 0.1 * scale + 1e-12

    def test_a_call_RISES_with_volatility_and_respects_the_no_arbitrage_bounds(self) -> None:
        market = _market()
        prices = [_pv(_option(), market, vol=v) for v in (0.05, 0.2, 0.5)]
        assert all(b > a for a, b in zip(prices, prices[1:]))
        curve = _curve()
        forward = dividend_forward(_underlying(), EXPIRY, curve)
        for price_at in prices:
            assert curve.df(EXPIRY) * max(forward - STRIKE, 0.0) <= price_at + 1e-12
            assert price_at <= curve.df(EXPIRY) * forward + 1e-12


class TestTheGuardsAreReachable:
    def test_an_unknown_asset_is_a_PricingFailure_not_a_raise(self) -> None:
        market = MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT: _curve()}), equities={})
        out = price(_option(), EquityBlackModel(market, VOL))
        assert not isinstance(out, PricingResult), out

    def test_an_EXPIRED_option_is_PV_zero_invariant_6(self) -> None:
        expired = EuropeanEquityOption(ACME, STRIKE, VAL, NOTIONAL, EquityOptionKind.CALL)
        assert _pv(expired, _market()) == 0.0

    def test_a_NEGATIVE_spot_is_refused(self) -> None:
        with pytest.raises(ValueError, match="spot"):
            EquityUnderlying(-1.0, 0.0, DividendSchedule())

    def test_a_NEGATIVE_volatility_is_refused(self) -> None:
        with pytest.raises(ValueError, match="volatilit"):
            EquityBlackModel(_market(), -0.1)
