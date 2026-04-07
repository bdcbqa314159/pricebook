"""Tests for curve trading strategies."""

import pytest
from datetime import date

from pricebook.curve_trading import (
    swap_dv01, spread_trade, butterfly_trade,
    swap_carry, breakeven_rate_move,
)
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)


def _curve(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _upward_curve():
    """Upward sloping curve: short rates lower than long rates."""
    # Pillars at 1Y, 2Y, 5Y, 10Y, 30Y
    pillar_dates = [
        date(2025, 1, 15), date(2026, 1, 15), date(2029, 1, 15),
        date(2034, 1, 15), date(2054, 1, 15),
    ]
    # Rising zero rates
    import math
    rates = [0.03, 0.035, 0.04, 0.045, 0.05]
    dfs = [math.exp(-r * ((d - REF).days / 365.0)) for r, d in zip(rates, pillar_dates)]
    return DiscountCurve(REF, pillar_dates, dfs)


def _swap(end_year=2034, rate=0.05, direction=SwapDirection.PAYER, notional=1_000_000):
    return InterestRateSwap(
        start=REF, end=date(end_year, 1, 15),
        fixed_rate=rate, direction=direction, notional=notional,
    )


# ---- DV01 ----

class TestSwapDV01:
    def test_nonzero(self):
        swap = _swap()
        dv01 = swap_dv01(swap, _curve())
        assert dv01 != 0.0

    def test_payer_receiver_opposite(self):
        curve = _curve()
        payer = _swap(direction=SwapDirection.PAYER)
        receiver = _swap(direction=SwapDirection.RECEIVER)
        assert swap_dv01(payer, curve) == pytest.approx(-swap_dv01(receiver, curve))

    def test_scales_with_notional(self):
        curve = _curve()
        s1 = _swap(notional=1_000_000)
        s2 = _swap(notional=2_000_000)
        assert swap_dv01(s2, curve) == pytest.approx(2 * swap_dv01(s1, curve), rel=1e-6)


# ---- Spread trade ----

class TestSpreadTrade:
    def test_dv01_neutral(self):
        curve = _curve()
        result = spread_trade(curve, REF, 2, 10)
        assert abs(result["net_dv01"]) < 1.0  # near zero

    def test_steepener(self):
        curve = _curve()
        result = spread_trade(curve, REF, 2, 10, direction="steepener")
        assert result["direction"] == "steepener"
        assert result["short_tenor"] == 2
        assert result["long_tenor"] == 10

    def test_flattener(self):
        curve = _curve()
        result = spread_trade(curve, REF, 2, 10, direction="flattener")
        assert result["direction"] == "flattener"

    def test_at_par_pv_near_zero(self):
        curve = _curve()
        result = spread_trade(curve, REF, 2, 10)
        # Built at par, so PV ≈ 0
        assert abs(result["pv"]) < 100  # small relative to notional

    def test_short_notional_larger(self):
        """Short-end needs more notional to match long-end DV01."""
        curve = _curve()
        result = spread_trade(curve, REF, 2, 10, notional_long=1_000_000)
        assert result["notional_short"] > result["notional_long"]

    def test_different_tenors(self):
        curve = _curve()
        r1 = spread_trade(curve, REF, 2, 10)
        r2 = spread_trade(curve, REF, 5, 30)
        # Both should be DV01 neutral
        assert abs(r1["net_dv01"]) < 1.0
        assert abs(r2["net_dv01"]) < 1.0


# ---- Butterfly trade ----

class TestButterflyTrade:
    def test_dv01_neutral(self):
        curve = _curve()
        result = butterfly_trade(curve, REF, 2, 5, 10)
        assert abs(result["net_dv01"]) < 1.0

    def test_three_legs(self):
        curve = _curve()
        result = butterfly_trade(curve, REF, 2, 5, 10)
        assert result["short_tenor"] == 2
        assert result["belly_tenor"] == 5
        assert result["long_tenor"] == 10

    def test_at_par_pv_near_zero(self):
        curve = _curve()
        result = butterfly_trade(curve, REF, 2, 5, 10)
        assert abs(result["pv"]) < 100

    def test_wing_notionals(self):
        """Wings should have notionals on either side of belly."""
        curve = _curve()
        result = butterfly_trade(curve, REF, 2, 5, 10)
        # Short wing has more notional, long wing has less
        assert result["notional_short"] > result["notional_long"]

    def test_different_fly(self):
        curve = _curve()
        result = butterfly_trade(curve, REF, 2, 5, 10)
        result2 = butterfly_trade(curve, REF, 5, 10, 30)
        assert abs(result["net_dv01"]) < 1.0
        assert abs(result2["net_dv01"]) < 1.0


# ---- Carry and roll-down ----

class TestCarryRolldown:
    def test_at_par_receiver_rolldown_upward_curve(self):
        """At-par receiver on upward curve: positive roll-down.

        As time passes, the swap rolls down the curve into lower rates,
        making the above-market fixed rate more valuable.
        """
        curve = _upward_curve()
        par = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=0.05,
            direction=SwapDirection.PAYER, notional=10_000_000,
        ).par_rate(curve)
        receiver = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=par,
            direction=SwapDirection.RECEIVER, notional=10_000_000,
        )
        carry = swap_carry(receiver, curve, horizon_days=30)
        assert carry > 0

    def test_carry_changes_sign_with_direction(self):
        curve = _upward_curve()
        par = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=0.05,
            direction=SwapDirection.PAYER, notional=10_000_000,
        ).par_rate(curve)
        payer = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=par,
            direction=SwapDirection.PAYER, notional=10_000_000,
        )
        receiver = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=par,
            direction=SwapDirection.RECEIVER, notional=10_000_000,
        )
        carry_p = swap_carry(payer, curve, horizon_days=30)
        carry_r = swap_carry(receiver, curve, horizon_days=30)
        assert carry_p * carry_r < 0

    def test_flat_curve_small_carry(self):
        """Flat curve at par → carry is very small."""
        curve = _curve(0.05)
        swap = _swap(end_year=2034, rate=0.05)
        carry = swap_carry(swap, curve, horizon_days=1)
        assert abs(carry) < 500

    def test_breakeven_positive(self):
        curve = _upward_curve()
        par = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=0.05,
            direction=SwapDirection.PAYER, notional=10_000_000,
        ).par_rate(curve)
        swap = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=par,
            direction=SwapDirection.RECEIVER, notional=10_000_000,
        )
        be = breakeven_rate_move(swap, curve, horizon_days=30)
        assert be > 0

    def test_breakeven_larger_horizon(self):
        """Longer horizon → more roll-down → larger breakeven."""
        curve = _upward_curve()
        par = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=0.05,
            direction=SwapDirection.PAYER, notional=10_000_000,
        ).par_rate(curve)
        swap = InterestRateSwap(
            REF, date(2029, 1, 15), fixed_rate=par,
            direction=SwapDirection.RECEIVER, notional=10_000_000,
        )
        be_1d = breakeven_rate_move(swap, curve, horizon_days=1)
        be_30d = breakeven_rate_move(swap, curve, horizon_days=30)
        assert be_30d > be_1d
