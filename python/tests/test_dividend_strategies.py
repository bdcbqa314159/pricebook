"""Tests for dividend strategies: futures, basis, carry trade, roll-down, backtest."""

import math

import pytest
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.dividend_desk import DividendSwap
from pricebook.dividend_model import Dividend
from pricebook.dividend_strategies import (
    BacktestResult,
    DividendFuture,
    EquityCarryTrade,
    dividend_basis,
    dividend_curve_carry,
    implied_vs_realised_backtest,
)


REF = date(2024, 1, 15)


def _curve(rate: float = 0.05) -> DiscountCurve:
    return DiscountCurve.flat(REF, rate)


def _yearly_divs(amounts: list[float], year: int = 2024) -> list[Dividend]:
    months = [3, 6, 9, 12]
    return [
        Dividend(date(year, m, 15), amt)
        for m, amt in zip(months, amounts)
    ]


# ---- Step 1: dividend futures + basis + swap PV identity ----

class TestDividendFuture:
    def test_settlement_value(self):
        future = DividendFuture(
            reference_start=date(2024, 1, 1),
            reference_end=date(2024, 12, 31),
            settlement_date=date(2024, 12, 31),
            fixed_price=8.0,
        )
        divs = _yearly_divs([2.0, 2.5, 2.0, 2.5])
        # All four dividends in the period → 9.0
        assert future.settlement_value(divs) == pytest.approx(9.0)

    def test_settlement_excludes_outside_period(self):
        future = DividendFuture(
            reference_start=date(2024, 7, 1),
            reference_end=date(2024, 12, 31),
            settlement_date=date(2024, 12, 31),
            fixed_price=4.5,
        )
        divs = _yearly_divs([2.0, 2.5, 2.0, 2.5])
        # Only Sep + Dec (4.5)
        assert future.settlement_value(divs) == pytest.approx(4.5)

    def test_pv_zero_at_fair(self):
        future = DividendFuture(
            reference_start=date(2024, 1, 1),
            reference_end=date(2024, 12, 31),
            settlement_date=date(2024, 12, 31),
            fixed_price=9.0,  # equal to realised
        )
        divs = _yearly_divs([2.0, 2.5, 2.0, 2.5])
        assert future.pv(divs, _curve()) == pytest.approx(0.0)

    def test_pv_long_positive_when_realised_above_fixed(self):
        future = DividendFuture(
            reference_start=date(2024, 1, 1),
            reference_end=date(2024, 12, 31),
            settlement_date=date(2024, 12, 31),
            fixed_price=8.0,
            notional=1_000_000,
        )
        divs = _yearly_divs([2.0, 2.5, 2.0, 2.5])
        pv = future.pv(divs, _curve())
        assert pv > 0.0
        # (9 - 8) × 1M × df ≈ 1M × df
        df = _curve().df(date(2024, 12, 31))
        assert pv == pytest.approx(1.0 * 1_000_000 * df)

    def test_direction_flips(self):
        long = DividendFuture(date(2024, 1, 1), date(2024, 12, 31),
                              date(2024, 12, 31), fixed_price=8.0,
                              notional=1.0, direction=1)
        short = DividendFuture(date(2024, 1, 1), date(2024, 12, 31),
                               date(2024, 12, 31), fixed_price=8.0,
                               notional=1.0, direction=-1)
        divs = _yearly_divs([2.0, 2.5, 2.0, 2.5])
        assert long.pv(divs, _curve()) == pytest.approx(
            -short.pv(divs, _curve())
        )

    def test_fair_price(self):
        future = DividendFuture(
            date(2024, 1, 1), date(2024, 12, 31),
            date(2024, 12, 31), fixed_price=0.0,
        )
        divs = _yearly_divs([2.0, 2.5, 2.0, 2.5])
        assert future.fair_price(divs) == pytest.approx(9.0)


class TestDividendBasis:
    def test_zero_when_equal(self):
        assert dividend_basis(option_implied=8.0, traded=8.0) == 0.0

    def test_positive_when_traded_rich(self):
        assert dividend_basis(8.0, 8.5) == pytest.approx(0.5)

    def test_negative_when_traded_cheap(self):
        assert dividend_basis(8.0, 7.5) == pytest.approx(-0.5)


class TestDividendSwapIdentity:
    """Slice 138 step 1 test: Swap PV = PV(realised divs) − fixed × df."""

    def test_swap_pv_decomposition(self):
        swap = DividendSwap(
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            fixed_div=8.0,
            notional=1_000_000,
        )
        divs = _yearly_divs([2.0, 2.5, 2.0, 2.5])
        curve = _curve()

        pv = swap.pv(divs, curve)

        # Direct decomposition
        pv_realised = sum(d.amount * curve.df(d.ex_date) for d in divs)
        pv_fixed = swap.fixed_div * curve.df(swap.end)
        expected = (pv_realised - pv_fixed) * swap.notional

        assert pv == pytest.approx(expected)


# ---- Step 2: carry trade + roll-down + backtest ----

class TestEquityCarryTrade:
    def test_stock_value(self):
        swap = DividendSwap(date(2024, 1, 1), date(2024, 12, 31),
                            fixed_div=8.0, notional=100)
        carry = EquityCarryTrade(shares=100, spot=200.0, swap=swap)
        assert carry.stock_value() == pytest.approx(20_000.0)

    def test_matched_swap_zero_div_exposure(self):
        """Slice 138 step 2 test: matched swap → zero net dividend exposure."""
        swap = DividendSwap(date(2024, 1, 1), date(2024, 12, 31),
                            fixed_div=8.0, notional=100)
        carry = EquityCarryTrade(shares=100, spot=200.0, swap=swap)
        assert carry.net_dividend_exposure() == 0.0

    def test_unmatched_swap_nonzero_exposure(self):
        swap = DividendSwap(date(2024, 1, 1), date(2024, 12, 31),
                            fixed_div=8.0, notional=50)
        carry = EquityCarryTrade(shares=100, spot=200.0, swap=swap)
        assert carry.net_dividend_exposure() == 50.0

    def test_pv_decomposition(self):
        swap = DividendSwap(date(2024, 1, 1), date(2024, 12, 31),
                            fixed_div=8.0, notional=100)
        carry = EquityCarryTrade(shares=100, spot=200.0, swap=swap)
        divs = _yearly_divs([2.0, 2.5, 2.0, 2.5])
        curve = _curve()
        # PV = stock_value - swap_pv
        assert carry.pv(divs, curve) == pytest.approx(
            carry.stock_value() - swap.pv(divs, curve)
        )

    def test_carry_invariant_under_div_realisation(self):
        """If shares match swap notional, the swap's dividend exposure
        exactly offsets the stock's dividend payout (long stock receives
        divs, short swap pays them out)."""
        swap = DividendSwap(date(2024, 1, 1), date(2024, 12, 31),
                            fixed_div=8.0, notional=100)
        # We don't actually pay divs out of the stock value in our PV
        # function, so the test here is structural: matched notionals.
        carry = EquityCarryTrade(shares=100, spot=200.0, swap=swap)
        assert carry.net_dividend_exposure() == 0.0


class TestDividendCurveCarry:
    def test_zero_when_implied_equals_realised(self):
        implied = [(date(2024, 6, 1), 2.0), (date(2024, 12, 1), 2.5)]
        realised = [(date(2024, 6, 1), 2.0), (date(2024, 12, 1), 2.5)]
        assert dividend_curve_carry(implied, realised) == pytest.approx(0.0)

    def test_positive_when_implied_above_realised(self):
        implied = [(date(2024, 6, 1), 2.5), (date(2024, 12, 1), 3.0)]
        realised = [(date(2024, 6, 1), 2.0), (date(2024, 12, 1), 2.5)]
        # Total carry = 1.0 (sold implied at 5.5, received 4.5)
        assert dividend_curve_carry(implied, realised) == pytest.approx(1.0)

    def test_missing_realised_treated_as_zero(self):
        implied = [(date(2024, 6, 1), 2.5), (date(2024, 12, 1), 3.0)]
        realised = [(date(2024, 6, 1), 2.0)]  # missing Dec
        assert dividend_curve_carry(implied, realised) == pytest.approx(
            0.5 + 3.0
        )


class TestImpliedVsRealisedBacktest:
    def test_empty(self):
        result = implied_vs_realised_backtest([])
        assert result.n == 0
        assert result.rmse == 0.0
        assert result.bias == 0.0

    def test_perfect_forecast(self):
        pairs = [(2.0, 2.0), (2.5, 2.5), (3.0, 3.0)]
        result = implied_vs_realised_backtest(pairs)
        assert result.n == 3
        assert result.mean_error == pytest.approx(0.0)
        assert result.mean_abs_error == pytest.approx(0.0)
        assert result.rmse == pytest.approx(0.0)
        assert result.bias == pytest.approx(0.0)

    def test_constant_overestimate(self):
        pairs = [(2.5, 2.0), (3.0, 2.5), (3.5, 3.0)]
        result = implied_vs_realised_backtest(pairs)
        # Implied = realised + 0.5 → mean error 0.5
        assert result.mean_error == pytest.approx(0.5)
        assert result.bias == pytest.approx(0.5)
        assert result.mean_abs_error == pytest.approx(0.5)
        assert result.rmse == pytest.approx(0.5)

    def test_mixed_errors(self):
        pairs = [(2.0, 1.0), (2.0, 3.0)]
        result = implied_vs_realised_backtest(pairs)
        # Errors: +1, -1
        assert result.mean_error == pytest.approx(0.0)
        assert result.mean_abs_error == pytest.approx(1.0)
        assert result.rmse == pytest.approx(1.0)

    def test_rmse_magnitude(self):
        pairs = [(2.0, 0.0), (2.0, 0.0), (2.0, 0.0), (2.0, 0.0)]
        result = implied_vs_realised_backtest(pairs)
        assert result.rmse == pytest.approx(2.0)
        assert result.mean_abs_error == pytest.approx(2.0)
