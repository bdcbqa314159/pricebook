"""Tests for money market deposit."""

import pytest
from datetime import date

from pricebook.day_count import DayCountConvention
from pricebook.deposit import Deposit


class TestDeposit:
    """Core deposit behaviour."""

    def test_year_fraction(self):
        # 3M deposit: Jan 15 to Apr 15 = 91 days, ACT/360
        dep = Deposit(date(2024, 1, 15), date(2024, 4, 15), rate=0.05)
        assert dep.year_fraction == pytest.approx(91 / 360.0)

    def test_discount_factor(self):
        # df = 1 / (1 + r * yf)
        dep = Deposit(date(2024, 1, 15), date(2024, 4, 15), rate=0.05)
        yf = 91 / 360.0
        expected_df = 1.0 / (1.0 + 0.05 * yf)
        assert dep.discount_factor == pytest.approx(expected_df)

    def test_cashflow(self):
        dep = Deposit(date(2024, 1, 15), date(2024, 4, 15), rate=0.05, notional=1_000_000)
        yf = 91 / 360.0
        expected = 1_000_000 * (1.0 + 0.05 * yf)
        assert dep.cashflow == pytest.approx(expected)

    def test_pv_at_own_discount_factor(self):
        # PV using the deposit's own implied df should be zero (par)
        dep = Deposit(date(2024, 1, 15), date(2024, 4, 15), rate=0.05, notional=1_000_000)
        assert dep.pv(dep.discount_factor) == pytest.approx(0.0)

    def test_pv_with_different_df(self):
        dep = Deposit(date(2024, 1, 15), date(2024, 4, 15), rate=0.05, notional=1_000_000)
        # A lower discount factor means higher rates -> deposit is worth less
        df_higher_rate = dep.discount_factor * 0.99
        assert dep.pv(df_higher_rate) < 0.0

    def test_unit_notional_default(self):
        dep = Deposit(date(2024, 1, 15), date(2024, 4, 15), rate=0.05)
        assert dep.notional == 1.0

    def test_act_365_convention(self):
        dep = Deposit(
            date(2024, 1, 15), date(2024, 4, 15), rate=0.05,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        assert dep.year_fraction == pytest.approx(91 / 365.0)


class TestDepositValidation:
    """Input validation."""

    def test_start_equals_end_raises(self):
        with pytest.raises(ValueError):
            Deposit(date(2024, 1, 15), date(2024, 1, 15), rate=0.05)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            Deposit(date(2024, 4, 15), date(2024, 1, 15), rate=0.05)

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            Deposit(date(2024, 1, 15), date(2024, 4, 15), rate=0.05, notional=-100)

    def test_zero_notional_raises(self):
        with pytest.raises(ValueError):
            Deposit(date(2024, 1, 15), date(2024, 4, 15), rate=0.05, notional=0)
