"""Tests for loan settlement: LSTA conventions, trade economics, loan-bond basis."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.loan_settlement import (
    LoanSettlement, delayed_compensation, break_funding,
    failed_settlement_penalty, LoanTradeEcon,
    loan_bond_basis, basis_z_score,
    SETTLEMENT_DAYS,
)


# ---- Settlement mechanics ----

class TestLoanSettlement:

    def test_settlement_days(self):
        s = LoanSettlement(
            trade_date=date(2024, 1, 10),
            settle_date=date(2024, 1, 17),
            price=98.5, par_amount=10_000_000,
        )
        assert s.settlement_days == 7

    def test_total_consideration(self):
        s = LoanSettlement(
            trade_date=date(2024, 1, 10),
            settle_date=date(2024, 1, 17),
            price=98.5, par_amount=10_000_000,
            accrued=25_000,
        )
        expected = 10_000_000 * 98.5 / 100.0 + 25_000
        assert s.total_consideration == pytest.approx(expected)

    def test_par_trade(self):
        s = LoanSettlement(
            trade_date=date(2024, 1, 10),
            settle_date=date(2024, 1, 17),
            price=100.0, par_amount=5_000_000,
        )
        assert s.total_consideration == pytest.approx(5_000_000)

    def test_discount_trade(self):
        s = LoanSettlement(
            trade_date=date(2024, 1, 10),
            settle_date=date(2024, 1, 17),
            price=95.0, par_amount=10_000_000,
        )
        assert s.total_consideration == pytest.approx(9_500_000)

    def test_trade_type_default(self):
        s = LoanSettlement(
            trade_date=date(2024, 1, 10),
            settle_date=date(2024, 1, 17),
            price=98.0, par_amount=1_000_000,
        )
        assert s.trade_type == "assignment"

    def test_to_dict(self):
        s = LoanSettlement(
            trade_date=date(2024, 1, 10),
            settle_date=date(2024, 1, 17),
            price=98.5, par_amount=10_000_000,
            accrued=25_000, trade_type="participation",
        )
        d = s.to_dict()
        assert d["trade_date"] == "2024-01-10"
        assert d["settle_date"] == "2024-01-17"
        assert d["price"] == 98.5
        assert d["trade_type"] == "participation"

    def test_standard_settlement_days(self):
        assert SETTLEMENT_DAYS["assignment"] == 7
        assert SETTLEMENT_DAYS["participation"] == 10
        assert SETTLEMENT_DAYS["distressed"] == 20


# ---- Delayed compensation ----

class TestDelayedCompensation:

    def test_basic(self):
        # rate=5%, 10 days, 10M notional
        comp = delayed_compensation(0.05, 10, 10_000_000)
        expected = 0.05 * 10 / 360.0 * 10_000_000
        assert comp == pytest.approx(expected)

    def test_zero_days(self):
        assert delayed_compensation(0.05, 0, 10_000_000) == 0.0

    def test_zero_rate(self):
        assert delayed_compensation(0.0, 10, 10_000_000) == 0.0

    def test_proportional_to_days(self):
        c1 = delayed_compensation(0.05, 5, 1_000_000)
        c2 = delayed_compensation(0.05, 10, 1_000_000)
        assert c2 == pytest.approx(2 * c1)

    def test_proportional_to_notional(self):
        c1 = delayed_compensation(0.05, 7, 1_000_000)
        c2 = delayed_compensation(0.05, 7, 2_000_000)
        assert c2 == pytest.approx(2 * c1)


# ---- Break funding ----

class TestBreakFunding:

    def test_positive_when_rate_drops(self):
        # Old rate higher: seller is compensated
        cost = break_funding(0.06, 0.04, 90, 10_000_000)
        assert cost > 0

    def test_negative_when_rate_rises(self):
        # New rate higher: buyer benefits
        cost = break_funding(0.04, 0.06, 90, 10_000_000)
        assert cost < 0

    def test_zero_when_same_rate(self):
        assert break_funding(0.05, 0.05, 90, 10_000_000) == 0.0

    def test_formula(self):
        cost = break_funding(0.06, 0.04, 90, 10_000_000)
        expected = (0.06 - 0.04) * 90 / 360.0 * 10_000_000
        assert cost == pytest.approx(expected)


# ---- Failed settlement penalty ----

class TestFailedSettlement:

    def test_positive(self):
        p = failed_settlement_penalty(5, 10_000_000)
        assert p > 0

    def test_formula(self):
        p = failed_settlement_penalty(10, 10_000_000, base_rate=0.05, penalty_spread=0.02)
        expected = (0.05 + 0.02) * 10 / 360.0 * 10_000_000
        assert p == pytest.approx(expected)

    def test_zero_days(self):
        assert failed_settlement_penalty(0, 10_000_000) == 0.0

    def test_proportional_to_days(self):
        p1 = failed_settlement_penalty(5, 10_000_000)
        p2 = failed_settlement_penalty(10, 10_000_000)
        assert p2 == pytest.approx(2 * p1)


# ---- Trade economics ----

class TestLoanTradeEcon:

    def test_price_pnl_gain(self):
        econ = LoanTradeEcon(buy_price=97.0, sell_price=99.0,
                             par_amount=10_000_000, hold_days=90)
        assert econ.price_pnl == pytest.approx(200_000)

    def test_price_pnl_loss(self):
        econ = LoanTradeEcon(buy_price=99.0, sell_price=97.0,
                             par_amount=10_000_000, hold_days=90)
        assert econ.price_pnl == pytest.approx(-200_000)

    def test_carry(self):
        econ = LoanTradeEcon(buy_price=98.0, sell_price=98.0,
                             par_amount=10_000_000, hold_days=90,
                             coupon_income=120_000, funding_cost=80_000)
        assert econ.carry == pytest.approx(40_000)

    def test_total_return(self):
        econ = LoanTradeEcon(buy_price=97.0, sell_price=99.0,
                             par_amount=10_000_000, hold_days=180,
                             coupon_income=250_000, funding_cost=150_000)
        assert econ.total_return == pytest.approx(200_000 + 100_000)

    def test_annualised_return(self):
        econ = LoanTradeEcon(buy_price=97.0, sell_price=99.0,
                             par_amount=10_000_000, hold_days=365,
                             coupon_income=250_000, funding_cost=150_000)
        invested = 9_700_000
        total_ret = 300_000
        expected = total_ret / invested  # 1 year → no annualisation adjustment
        assert econ.annualised_return == pytest.approx(expected)

    def test_annualised_return_zero_hold(self):
        econ = LoanTradeEcon(buy_price=97.0, sell_price=99.0,
                             par_amount=10_000_000, hold_days=0)
        assert econ.annualised_return == 0.0

    def test_breakeven_hold_with_loss(self):
        econ = LoanTradeEcon(buy_price=99.0, sell_price=97.0,
                             par_amount=10_000_000, hold_days=90,
                             coupon_income=90_000, funding_cost=45_000)
        # daily carry = 45000/90 = 500/day
        # loss = 200_000
        # breakeven = ceil(200_000 / 500) = 400 days
        be = econ.breakeven_hold()
        assert be == 400

    def test_breakeven_hold_no_loss(self):
        econ = LoanTradeEcon(buy_price=97.0, sell_price=99.0,
                             par_amount=10_000_000, hold_days=90,
                             coupon_income=90_000, funding_cost=45_000)
        # No price loss → breakeven = 1 (already profitable)
        assert econ.breakeven_hold() == 1

    def test_breakeven_hold_zero_carry(self):
        econ = LoanTradeEcon(buy_price=99.0, sell_price=97.0,
                             par_amount=10_000_000, hold_days=90,
                             coupon_income=0, funding_cost=0)
        assert econ.breakeven_hold() == 999999

    def test_breakeven_hold_explicit_daily_carry(self):
        econ = LoanTradeEcon(buy_price=99.0, sell_price=97.0,
                             par_amount=10_000_000, hold_days=90)
        be = econ.breakeven_hold(daily_carry=1000)
        assert be == math.ceil(200_000 / 1000)

    def test_to_dict(self):
        econ = LoanTradeEcon(buy_price=97.0, sell_price=99.0,
                             par_amount=10_000_000, hold_days=90,
                             coupon_income=100_000, funding_cost=50_000)
        d = econ.to_dict()
        assert "price_pnl" in d
        assert "carry" in d
        assert "total_return" in d
        assert "annualised_return" in d
        assert d["hold_days"] == 90


# ---- Loan-bond basis ----

class TestLoanBondBasis:

    def test_positive_basis(self):
        # Loan DM wider than bond ASW: typical
        basis = loan_bond_basis(0.035, 0.025)
        assert basis == pytest.approx(100)  # 100bp

    def test_negative_basis(self):
        # Unusual: bond wider than loan
        basis = loan_bond_basis(0.025, 0.035)
        assert basis == pytest.approx(-100)

    def test_zero_basis(self):
        assert loan_bond_basis(0.03, 0.03) == pytest.approx(0)

    def test_typical_range(self):
        # IG issuer: 50-150bp
        basis = loan_bond_basis(0.035, 0.028)
        assert 50 < basis < 150


# ---- Basis z-score ----

class TestBasisZScore:

    def test_at_mean(self):
        assert basis_z_score(100, 100, 20) == pytest.approx(0.0)

    def test_one_sigma_wide(self):
        assert basis_z_score(120, 100, 20) == pytest.approx(1.0)

    def test_one_sigma_tight(self):
        assert basis_z_score(80, 100, 20) == pytest.approx(-1.0)

    def test_zero_std(self):
        assert basis_z_score(100, 100, 0) == 0.0

    def test_negative_std(self):
        assert basis_z_score(100, 100, -5) == 0.0

    def test_extreme_wide(self):
        z = basis_z_score(200, 100, 20)
        assert z > 3  # very wide
