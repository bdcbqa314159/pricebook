"""Tests for unified TotalReturnSwap (equity, bond, loan, multi-period)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.bond import FixedRateBond
from pricebook.bootstrap import bootstrap
from pricebook.discount_curve import DiscountCurve
from pricebook.dividend_model import Dividend
from pricebook.pricing_context import PricingContext
from pricebook.schedule import Frequency
from pricebook.trade import Trade, Portfolio
from pricebook.trs import TotalReturnSwap, FundingLegSpec, TRSResult


REF = date(2026, 4, 27)


def _curve(ref):
    deposits = [(ref + timedelta(days=91), 0.04)]
    swaps = [(ref + timedelta(days=365), 0.038),
             (ref + timedelta(days=1825), 0.035)]
    return bootstrap(ref, deposits, swaps)


# ---- Equity TRS ----

class TestEquityTRS:
    def test_basic_price(self):
        curve = _curve(REF)
        trs = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                               start=REF, end=REF + timedelta(days=365))
        result = trs.price(curve)
        assert isinstance(result, TRSResult)
        assert math.isfinite(result.value)

    def test_with_repo_spread(self):
        curve = _curve(REF)
        trs = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                               start=REF, end=REF + timedelta(days=365),
                               repo_spread=0.02)
        result = trs.price(curve)
        assert result.fva > 0
        assert result.repo_factor > 1.0

    def test_with_discrete_dividends(self):
        curve = _curve(REF)
        divs = [Dividend(REF + timedelta(days=180), 2.0)]
        trs = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                               start=REF, end=REF + timedelta(days=365),
                               dividends=divs)
        result = trs.price(curve)
        assert math.isfinite(result.value)

    def test_pv_ctx(self):
        curve = _curve(REF)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        trs = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                               start=REF, end=REF + timedelta(days=365))
        trade = Trade(trs, trade_id="EQ_TRS")
        pv = trade.pv(ctx)
        assert math.isfinite(pv)

    def test_greeks(self):
        curve = _curve(REF)
        trs = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                               start=REF, end=REF + timedelta(days=365),
                               repo_spread=0.01)
        g = trs.greeks(curve)
        assert "delta" in g
        assert "repo_sensitivity" in g
        assert math.isfinite(g["delta"])

    def test_result_protocol(self):
        curve = _curve(REF)
        trs = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                               start=REF, end=REF + timedelta(days=365))
        result = trs.price(curve)
        assert hasattr(result, "price")
        assert isinstance(result.to_dict(), dict)
        assert result.mtm == result.value  # backward compat


# ---- Bond TRS ----

class TestBondTRS:
    def test_basic_price(self):
        curve = _curve(REF)
        bond = FixedRateBond(REF, REF + timedelta(days=3650), coupon_rate=0.03)
        trs = TotalReturnSwap(underlying=bond, notional=10_000_000,
                               start=REF, end=REF + timedelta(days=365))
        result = trs.price(curve)
        assert math.isfinite(result.value)

    def test_with_haircut(self):
        curve = _curve(REF)
        bond = FixedRateBond(REF, REF + timedelta(days=3650), coupon_rate=0.03)
        trs = TotalReturnSwap(underlying=bond, notional=10_000_000,
                               start=REF, end=REF + timedelta(days=365),
                               repo_spread=0.01, haircut=0.05)
        result = trs.price(curve)
        assert result.fva > 0

    def test_pv_ctx(self):
        curve = _curve(REF)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        bond = FixedRateBond(REF, REF + timedelta(days=3650), coupon_rate=0.03)
        trs = TotalReturnSwap(underlying=bond, notional=10_000_000,
                               start=REF, end=REF + timedelta(days=365))
        trade = Trade(trs, trade_id="BOND_TRS")
        pv = trade.pv(ctx)
        assert math.isfinite(pv)


# ---- Loan TRS ----

class TestLoanTRS:
    def test_basic_price(self):
        curve = _curve(REF)
        from pricebook.loan import TermLoan
        loan = TermLoan(REF, REF + timedelta(days=1825), spread=0.03,
                        notional=10_000_000)
        trs = TotalReturnSwap(underlying=loan, notional=10_000_000,
                               start=REF, end=REF + timedelta(days=365),
                               funding=FundingLegSpec(spread=0.01))
        result = trs.price(curve)
        assert math.isfinite(result.value)

    def test_pv_ctx(self):
        curve = _curve(REF)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        from pricebook.loan import TermLoan
        loan = TermLoan(REF, REF + timedelta(days=1825), spread=0.03,
                        notional=10_000_000)
        trs = TotalReturnSwap(underlying=loan, notional=10_000_000,
                               start=REF, end=REF + timedelta(days=365))
        trade = Trade(trs, trade_id="LOAN_TRS")
        pv = trade.pv(ctx)
        assert math.isfinite(pv)


# ---- Multi-period ----

class TestMultiPeriod:
    def test_quarterly_resets(self):
        curve = _curve(REF)
        resets = [REF + timedelta(days=91), REF + timedelta(days=182),
                  REF + timedelta(days=273)]
        trs = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                               start=REF, end=REF + timedelta(days=365),
                               reset_dates=resets)
        result = trs.price(curve)
        assert math.isfinite(result.value)

    def test_mtm_reset(self):
        curve = _curve(REF)
        resets = [REF + timedelta(days=182)]
        trs_fixed = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                                     start=REF, end=REF + timedelta(days=365),
                                     reset_dates=resets, mtm_reset=False)
        trs_mtm = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                                   start=REF, end=REF + timedelta(days=365),
                                   reset_dates=resets, mtm_reset=True)
        r_fixed = trs_fixed.price(curve)
        r_mtm = trs_mtm.price(curve)
        assert math.isfinite(r_fixed.value)
        assert math.isfinite(r_mtm.value)


# ---- Portfolio integration ----

class TestPortfolioIntegration:
    def test_all_types_in_portfolio(self):
        curve = _curve(REF)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)

        bond = FixedRateBond(REF, REF + timedelta(days=3650), coupon_rate=0.03)
        from pricebook.loan import TermLoan
        loan = TermLoan(REF, REF + timedelta(days=1825), spread=0.03,
                        notional=5_000_000)

        eq_trs = TotalReturnSwap(underlying=100.0, notional=1_000_000,
                                  start=REF, end=REF + timedelta(days=365))
        bond_trs = TotalReturnSwap(underlying=bond, notional=5_000_000,
                                    start=REF, end=REF + timedelta(days=365))
        loan_trs = TotalReturnSwap(underlying=loan, notional=5_000_000,
                                    start=REF, end=REF + timedelta(days=365))

        port = Portfolio(name="trs_book")
        port.add(Trade(eq_trs, trade_id="EQ_TRS"))
        port.add(Trade(bond_trs, trade_id="BOND_TRS"))
        port.add(Trade(loan_trs, trade_id="LOAN_TRS"))

        total_pv = port.pv(ctx)
        assert math.isfinite(total_pv)
        assert len(port) == 3

        by_trade = port.pv_by_trade(ctx)
        for tid, pv in by_trade:
            assert math.isfinite(pv), f"{tid} has non-finite PV"
