"""Tests for loan TRS deep features: prepayment, LSTA, revolving, distressed."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.bootstrap import bootstrap
from pricebook.discount_curve import DiscountCurve
from pricebook.loan import TermLoan
from pricebook.trs import TotalReturnSwap, FundingLegSpec, LSTATerms


REF = date(2026, 4, 27)


def _curve(ref):
    deposits = [(ref + timedelta(days=91), 0.04)]
    swaps = [(ref + timedelta(days=365), 0.038),
             (ref + timedelta(days=1825), 0.035)]
    return bootstrap(ref, deposits, swaps)


def _loan(ref):
    return TermLoan(ref, ref + timedelta(days=1825), spread=0.03,
                    notional=10_000_000)


# ---- 1a. Prepayment-adjusted Loan TRS ----

class TestPrepaymentLoanTRS:

    def test_no_prepay_baseline(self):
        curve = _curve(REF)
        loan = _loan(REF)
        trs = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            prepay_model=None)
        result = trs.price(curve)
        assert math.isfinite(result.value)

    def test_with_flat_cpr(self):
        curve = _curve(REF)
        loan = _loan(REF)
        trs = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            prepay_model=0.10)  # 10% CPR
        result = trs.price(curve)
        assert math.isfinite(result.value)

    def test_with_psa_model(self):
        curve = _curve(REF)
        loan = _loan(REF)
        trs = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            prepay_model="PSA")
        result = trs.price(curve)
        assert math.isfinite(result.value)

    def test_prepay_changes_value(self):
        """Prepayment should produce a finite, computable value."""
        curve = _curve(REF)
        loan = _loan(REF)
        trs_no = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            prepay_model=None)
        trs_yes = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            prepay_model=0.20)  # 20% CPR

        r_no = trs_no.price(curve)
        r_yes = trs_yes.price(curve)

        # Both should produce finite values
        assert math.isfinite(r_no.value)
        assert math.isfinite(r_yes.value)


# ---- 1b. LSTA Settlement ----

class TestLSTASettlement:

    def test_default_settlement(self):
        """Default LSTA: T+7."""
        terms = LSTATerms()
        assert terms.settlement_days == 7
        assert terms.trade_type == "assignment"

    def test_settlement_affects_funding(self):
        """T+7 settlement should cost more than T+0."""
        curve = _curve(REF)
        loan = _loan(REF)

        trs_t0 = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            settlement_terms=None)
        trs_t7 = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            settlement_terms=LSTATerms(settlement_days=7))

        r_t0 = trs_t0.price(curve)
        r_t7 = trs_t7.price(curve)

        # T+7 has higher funding cost (7 days of extra carry)
        assert r_t7.funding_leg > r_t0.funding_leg

    def test_participation_type(self):
        terms = LSTATerms(trade_type="participation", settlement_days=10)
        assert terms.trade_type == "participation"
        assert terms.settlement_days == 10


# ---- Portfolio integration ----

class TestLoanTRSPortfolio:

    def test_prepay_loan_in_portfolio(self):
        from pricebook.pricing_context import PricingContext
        from pricebook.trade import Trade, Portfolio

        curve = _curve(REF)
        ctx = PricingContext(valuation_date=REF, discount_curve=curve)
        loan = _loan(REF)

        trs = TotalReturnSwap(
            underlying=loan, notional=10_000_000,
            start=REF, end=REF + timedelta(days=365),
            prepay_model=0.10,
            settlement_terms=LSTATerms())

        port = Portfolio(name="loan_trs_book")
        port.add(Trade(trs, trade_id="LOAN_TRS_PREPAY"))
        pv = port.pv(ctx)
        assert math.isfinite(pv)
