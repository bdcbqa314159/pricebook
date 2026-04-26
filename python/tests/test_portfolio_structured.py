"""Integration test: all 4 structured products in a Portfolio via Trade/pv_ctx."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.bond import FixedRateBond
from pricebook.bootstrap import bootstrap
from pricebook.cmasw import CMASWInstrument
from pricebook.cmt import CMTInstrument
from pricebook.discount_curve import DiscountCurve
from pricebook.index_linked_hybrid import IndexLinkedHybridInstrument
from pricebook.instrument_result import InstrumentResult
from pricebook.pricing_context import PricingContext
from pricebook.schedule import Frequency
from pricebook.trade import Trade, Portfolio
from pricebook.treasury_lock import TreasuryLock


REF = date(2026, 4, 26)


def _curve(ref: date) -> DiscountCurve:
    deposits = [(ref + timedelta(days=91), 0.04), (ref + timedelta(days=182), 0.039)]
    swaps = [
        (ref + timedelta(days=365), 0.038),
        (ref + timedelta(days=730), 0.037),
        (ref + timedelta(days=1825), 0.035),
        (ref + timedelta(days=3650), 0.034),
    ]
    return bootstrap(ref, deposits, swaps)


def _ctx(ref: date) -> PricingContext:
    curve = _curve(ref)
    return PricingContext(valuation_date=ref, discount_curve=curve)


class TestStructuredPortfolio:
    """All 4 products in a Portfolio via Trade framework."""

    def test_tlock_pv_ctx(self):
        ctx = _ctx(REF)
        bond = FixedRateBond(REF, REF + timedelta(days=3650), coupon_rate=0.03)
        tlock = TreasuryLock(bond, locked_yield=0.03, expiry=REF + timedelta(days=182),
                             repo_rate=0.02)
        trade = Trade(tlock, trade_id="TLOCK_10Y")
        pv = trade.pv(ctx)
        assert math.isfinite(pv)

    def test_cmasw_pv_ctx(self):
        ctx = _ctx(REF)
        cmasw = CMASWInstrument(
            fixing_date=REF + timedelta(days=1825),
            payment_date=REF + timedelta(days=2007),
            swap_tenor=5, bond_price=0.95,
            sigma_swp=0.30, sigma_asw=0.25, rho=0.5)
        trade = Trade(cmasw, trade_id="CMASW_5Y")
        pv = trade.pv(ctx)
        assert math.isfinite(pv)

    def test_cmt_pv_ctx(self):
        ctx = _ctx(REF)
        cmt = CMTInstrument(
            fixing_date=REF + timedelta(days=1825),
            payment_date=REF + timedelta(days=2190),
            bond_tenor=10, sigma=0.20, hazard_rate=0.01)
        trade = Trade(cmt, trade_id="CMT_10Y")
        pv = trade.pv(ctx)
        assert math.isfinite(pv)

    def test_hybrid_pv_ctx(self):
        ctx = _ctx(REF)
        hybrid = IndexLinkedHybridInstrument(
            expiry=REF + timedelta(days=1825),
            swap_tenor=5, index_forward=0.04,
            sigma_F=0.30, sigma_U=0.25, rho=0.3,
            n_paths=5_000, n_steps=20)
        trade = Trade(hybrid, trade_id="HYBRID_5Y")
        pv = trade.pv(ctx)
        assert math.isfinite(pv)

    def test_all_four_in_portfolio(self):
        """All 4 structured products aggregate in a single Portfolio."""
        ctx = _ctx(REF)

        bond = FixedRateBond(REF, REF + timedelta(days=3650), coupon_rate=0.03)
        tlock = TreasuryLock(bond, locked_yield=0.03,
                             expiry=REF + timedelta(days=182), repo_rate=0.02)
        cmasw = CMASWInstrument(
            REF + timedelta(days=1825), REF + timedelta(days=2007),
            swap_tenor=5, bond_price=0.95)
        cmt = CMTInstrument(
            REF + timedelta(days=1825), REF + timedelta(days=2190),
            bond_tenor=10, sigma=0.20, hazard_rate=0.01)
        hybrid = IndexLinkedHybridInstrument(
            REF + timedelta(days=1825), swap_tenor=5, index_forward=0.04,
            n_paths=5_000, n_steps=20)

        port = Portfolio(name="structured_book")
        port.add(Trade(tlock, trade_id="TLOCK"))
        port.add(Trade(cmasw, trade_id="CMASW"))
        port.add(Trade(cmt, trade_id="CMT"))
        port.add(Trade(hybrid, trade_id="HYBRID"))

        assert len(port) == 4

        total_pv = port.pv(ctx)
        assert math.isfinite(total_pv)

        by_trade = port.pv_by_trade(ctx)
        assert len(by_trade) == 4
        for tid, pv in by_trade:
            assert math.isfinite(pv), f"{tid} has non-finite PV"

        sum_pv = sum(pv for _, pv in by_trade)
        assert total_pv == pytest.approx(sum_pv, rel=1e-10)

    def test_result_protocol_compliance(self):
        """All instrument results implement InstrumentResult protocol."""
        ctx = _ctx(REF)

        bond = FixedRateBond(REF, REF + timedelta(days=3650), coupon_rate=0.03)
        tlock = TreasuryLock(bond, locked_yield=0.03,
                             expiry=REF + timedelta(days=182))
        r1 = tlock.price(ctx.discount_curve)
        assert isinstance(r1, InstrumentResult)
        assert math.isfinite(r1.price)
        assert isinstance(r1.to_dict(), dict)
