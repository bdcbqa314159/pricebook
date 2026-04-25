"""XI5: Equity Forward → Option → Implied Vol integration chain.

Spot + discrete dividends → equity forward → option via div_yield →
implied vol → round-trip. Verify forward matches (S - PV(divs)) / df(T).

Bug hotspots:
- equity_option_price takes continuous div_yield, forward uses discrete dividends
- conversion between discrete and continuous must be consistent
- implied vol solver must round-trip: price → iv → price
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.black76 import OptionType, black76_price
from pricebook.discount_curve import DiscountCurve
from pricebook.dividend_model import (
    Dividend, pv_dividends, dividend_adjusted_forward, equity_option_discrete_divs,
)
from pricebook.equity_forward import EquityForward
from pricebook.equity_option import equity_option_price
from pricebook.implied_vol import implied_vol_black76


# ---- Helpers ----

REF = date(2026, 4, 25)
SPOT = 100.0
VOL = 0.25
RATE = 0.04


def _flat_curve(ref: date, rate: float = RATE) -> DiscountCurve:
    return DiscountCurve.flat(ref, rate)


def _dividends(ref: date) -> list[Dividend]:
    """Quarterly dividends of $1."""
    return [
        Dividend(ref + timedelta(days=91), 1.0),
        Dividend(ref + timedelta(days=182), 1.0),
        Dividend(ref + timedelta(days=273), 1.0),
        Dividend(ref + timedelta(days=364), 1.0),
    ]


# ---- R1: Forward price consistency ----

class TestXI5R1ForwardConsistency:
    """Equity forward with discrete dividends matches (S - PV(divs)) / df(T)."""

    def test_forward_no_divs(self):
        """Without dividends, F = S × exp(r × T)."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=365)
        T = 1.0

        fwd_obj = EquityForward(SPOT, mat, REF, div_yield=0.0)
        fwd = fwd_obj.forward_price(curve)

        expected = SPOT * math.exp(RATE * T)
        assert fwd == pytest.approx(expected, rel=1e-4)

    def test_forward_with_div_yield(self):
        """With continuous div yield, F = S × exp((r - q) × T)."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=365)
        T = 1.0
        q = 0.02

        fwd_obj = EquityForward(SPOT, mat, REF, div_yield=q)
        fwd = fwd_obj.forward_price(curve)

        expected = SPOT * math.exp((RATE - q) * T)
        assert fwd == pytest.approx(expected, rel=1e-4)

    def test_forward_discrete_divs_formula(self):
        """F = (S - PV(divs)) / df(T) with discrete dividends."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=365)
        divs = _dividends(REF)

        fwd_obj = EquityForward(SPOT, mat, REF, dividends=divs)
        fwd = fwd_obj.forward_price(curve)

        pv_divs = pv_dividends(divs, curve, mat)
        df = curve.df(mat)
        expected = (SPOT - pv_divs) / df
        assert fwd == pytest.approx(expected, rel=1e-4)

    def test_dividend_adjusted_forward_matches(self):
        """dividend_adjusted_forward must match EquityForward."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=365)
        divs = _dividends(REF)

        fwd1 = EquityForward(SPOT, mat, REF, dividends=divs).forward_price(curve)
        fwd2 = dividend_adjusted_forward(SPOT, divs, curve, mat)
        assert fwd1 == pytest.approx(fwd2, rel=1e-6)

    def test_dividends_lower_forward(self):
        """Dividends should lower the forward price."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=365)

        fwd_no_div = EquityForward(SPOT, mat, REF).forward_price(curve)
        fwd_with_div = EquityForward(SPOT, mat, REF,
                                      dividends=_dividends(REF)).forward_price(curve)
        assert fwd_with_div < fwd_no_div


# ---- R2: Option pricing consistency ----

class TestXI5R2OptionPricing:
    """equity_option_price (continuous q) vs equity_option_discrete_divs."""

    def test_option_price_no_divs_matches_black76(self):
        """No dividends: equity_option_price = Black-76 on forward."""
        T = 1.0
        K = 100.0
        fwd = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)

        eq_price = equity_option_price(SPOT, K, RATE, VOL, T, OptionType.CALL)
        b76_price = black76_price(fwd, K, VOL, T, df, OptionType.CALL)
        assert eq_price == pytest.approx(b76_price, rel=1e-6)

    def test_discrete_divs_option_lower_than_no_divs(self):
        """Call with dividends should be cheaper (lower forward)."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=365)
        K = 100.0

        call_no_div = equity_option_price(SPOT, K, RATE, VOL, 1.0, OptionType.CALL)
        call_div = equity_option_discrete_divs(SPOT, K, _dividends(REF), curve,
                                                VOL, mat, OptionType.CALL)
        assert call_div < call_no_div

    def test_put_call_parity_continuous(self):
        """C - P = S×exp(-qT) - K×exp(-rT)."""
        T = 1.0
        K = 105.0
        q = 0.02

        call = equity_option_price(SPOT, K, RATE, VOL, T, OptionType.CALL, div_yield=q)
        put = equity_option_price(SPOT, K, RATE, VOL, T, OptionType.PUT, div_yield=q)

        lhs = call - put
        rhs = SPOT * math.exp(-q * T) - K * math.exp(-RATE * T)
        assert lhs == pytest.approx(rhs, rel=1e-6)

    def test_put_call_parity_discrete_divs(self):
        """With discrete divs: C - P = (S - PV(divs)) - K×df."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=365)
        K = 100.0
        divs = _dividends(REF)

        call = equity_option_discrete_divs(SPOT, K, divs, curve, VOL, mat, OptionType.CALL)
        put = equity_option_discrete_divs(SPOT, K, divs, curve, VOL, mat, OptionType.PUT)

        pv_divs = pv_dividends(divs, curve, mat)
        df = curve.df(mat)

        lhs = call - put
        rhs = (SPOT - pv_divs) - K * df
        assert lhs == pytest.approx(rhs, rel=0.01)


# ---- R3: Implied vol round-trip ----

class TestXI5R3ImpliedVolRoundTrip:
    """price → implied_vol → price must round-trip."""

    def test_iv_round_trip_atm_call(self):
        T = 1.0
        K = 100.0
        fwd = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)

        price = black76_price(fwd, K, VOL, T, df, OptionType.CALL)
        iv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
        assert iv == pytest.approx(VOL, abs=1e-6)

    def test_iv_round_trip_otm_call(self):
        T = 1.0
        K = 120.0
        fwd = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)

        price = black76_price(fwd, K, VOL, T, df, OptionType.CALL)
        iv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
        assert iv == pytest.approx(VOL, abs=1e-6)

    def test_iv_round_trip_put(self):
        T = 1.0
        K = 95.0
        fwd = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)

        price = black76_price(fwd, K, VOL, T, df, OptionType.PUT)
        iv = implied_vol_black76(price, fwd, K, T, df, OptionType.PUT)
        assert iv == pytest.approx(VOL, abs=1e-6)

    def test_iv_round_trip_low_vol(self):
        T = 1.0
        K = 100.0
        low_vol = 0.05
        fwd = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)

        price = black76_price(fwd, K, low_vol, T, df, OptionType.CALL)
        iv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
        assert iv == pytest.approx(low_vol, abs=1e-6)

    def test_iv_round_trip_high_vol(self):
        T = 1.0
        K = 100.0
        high_vol = 0.80
        fwd = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)

        price = black76_price(fwd, K, high_vol, T, df, OptionType.CALL)
        iv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
        assert iv == pytest.approx(high_vol, abs=1e-4)

    def test_iv_round_trip_discrete_divs(self):
        """Full chain: discrete divs → forward → option price → IV → reprice."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=365)
        K = 100.0
        divs = _dividends(REF)
        T = 1.0

        # Price with discrete divs
        price = equity_option_discrete_divs(SPOT, K, divs, curve, VOL, mat,
                                             OptionType.CALL)

        # Extract IV using the dividend-adjusted forward
        fwd = dividend_adjusted_forward(SPOT, divs, curve, mat)
        df = curve.df(mat)
        iv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)

        # Reprice with recovered IV
        repriced = black76_price(fwd, K, iv, T, df, OptionType.CALL)
        assert repriced == pytest.approx(price, rel=1e-6)


# ---- R4: Edge cases ----

class TestXI5R4EdgeCases:
    """Edge cases for equity integration."""

    def test_deep_itm_call_near_intrinsic(self):
        """Deep ITM call at low vol ≈ intrinsic."""
        T = 1.0
        K = 60.0  # deep ITM
        q = 0.02
        fwd = SPOT * math.exp((RATE - q) * T)
        df = math.exp(-RATE * T)

        price = equity_option_price(SPOT, K, RATE, 1e-6, T, OptionType.CALL, div_yield=q)
        intrinsic = df * max(fwd - K, 0)
        assert price == pytest.approx(intrinsic, rel=0.01)

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be very cheap."""
        T = 0.25
        K = 200.0
        price = equity_option_price(SPOT, K, RATE, VOL, T, OptionType.CALL)
        assert price < 0.01

    def test_short_expiry_iv(self):
        """IV round-trip with 1-week expiry."""
        T = 7 / 365.0
        K = 100.0
        fwd = SPOT * math.exp(RATE * T)
        df = math.exp(-RATE * T)

        price = black76_price(fwd, K, VOL, T, df, OptionType.CALL)
        iv = implied_vol_black76(price, fwd, K, T, df, OptionType.CALL)
        assert iv == pytest.approx(VOL, abs=1e-4)

    def test_zero_dividend_pv(self):
        """No dividends before maturity → PV(divs) = 0."""
        curve = _flat_curve(REF)
        mat = REF + timedelta(days=30)  # before first div
        divs = _dividends(REF)  # first div at +91 days

        pv = pv_dividends(divs, curve, mat)
        assert pv == pytest.approx(0.0, abs=1e-10)

    def test_higher_vol_higher_option_price(self):
        """Higher vol → higher call and put prices."""
        T = 1.0
        K = 100.0
        low = equity_option_price(SPOT, K, RATE, 0.10, T, OptionType.CALL)
        high = equity_option_price(SPOT, K, RATE, 0.40, T, OptionType.CALL)
        assert high > low
