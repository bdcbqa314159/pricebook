"""Tests for futures trading desk: multi-asset book, margin, carry, stress, lifecycle."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.futures_desk import (
    BondFuture, FXFuture, FuturesAssetClass,
    futures_risk_metrics, FuturesRiskMetrics,
    FuturesBook, FuturesBookEntry,
    futures_daily_settlement, futures_margin_check,
    futures_carry_roll,
    futures_dashboard, FuturesDashboard,
    futures_stress_suite, FuturesStressResult,
    futures_hedge_recommendations,
    FuturesLifecycle,
)
from pricebook.futures import EquityFuture
from pricebook.ir_futures import IRFuture, FuturesType
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


def _bond_future():
    return BondFuture(
        trade_price=110.5, market_price=111.25,
        expiry=REF + relativedelta(months=3),
        multiplier=1000, ctd_dv01=0.085, ctd_cf=0.82,
    )


def _ir_future():
    return IRFuture(
        REF, REF + relativedelta(months=3),
        futures_type=FuturesType.SOFR_3M,
    )


def _equity_future():
    return EquityFuture(
        spot=5400, expiry=REF + relativedelta(months=3),
        rate=0.04, div_yield=0.015, notional_per_point=50,
    )


# ── BondFuture ──

class TestBondFuture:

    def test_pv_positive_when_market_above_trade(self):
        bf = _bond_future()
        assert bf.pv(contracts=10) > 0  # 111.25 - 110.5 = 0.75 × 1000 × 10

    def test_l11_pv_hand_calc(self):
        """PV = (market - trade) × multiplier × contracts."""
        bf = _bond_future()
        expected = (111.25 - 110.5) * 1000 * 5
        assert abs(bf.pv(5) - expected) < 0.01

    def test_dv01(self):
        bf = _bond_future()
        dv01 = bf.dv01(10)
        # DV01 = 0.085 / 0.82 × 1000/100 × 10 ≈ 10.37
        expected = 0.085 / 0.82 * 1000 / 100 * 10
        assert abs(dv01 - expected) < 0.01

    def test_pv_ctx(self):
        bf = _bond_future()
        assert math.isfinite(bf.pv_ctx(None))


# ── FXFuture ──

class TestFXFuture:

    def test_fair_price(self):
        fx = FXFuture("EUR", "USD", spot=1.08,
                      expiry=REF + relativedelta(months=3))
        usd = make_flat_curve(REF, 0.05)
        eur = make_flat_curve(REF, 0.03)
        fwd = fx.fair_price(usd, eur, REF)
        # EUR rates < USD → forward > spot (USD depreciates)
        assert fwd > 1.08

    def test_pv(self):
        fx = FXFuture("EUR", "USD", spot=1.08,
                      expiry=REF + relativedelta(months=3))
        pv = fx.pv(trade_price=1.08, market_price=1.09, contracts=5)
        assert pv == (1.09 - 1.08) * 125_000 * 5


# ── Risk metrics ──

class TestFuturesRiskMetrics:

    def test_bond_future_metrics(self):
        rm = futures_risk_metrics(_bond_future(), contracts=10, margin_per_contract=5000)
        assert rm.pv > 0
        assert rm.dv01 > 0
        assert rm.margin_required == 50_000

    def test_ir_future_metrics(self):
        rm = futures_risk_metrics(_ir_future(), contracts=20)
        assert rm.asset_class == "ir"


# ── Book ──

class TestFuturesBook:

    def test_add_and_count(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 10,
                                   FuturesAssetClass.BOND, exchange="CME"))
        book.add(FuturesBookEntry("IR1", _ir_future(), 20,
                                   FuturesAssetClass.IR, exchange="CME"))
        assert len(book) == 2

    def test_by_asset_class(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 10, FuturesAssetClass.BOND))
        book.add(FuturesBookEntry("EQ1", _equity_future(), 5, FuturesAssetClass.EQUITY))
        bac = book.by_asset_class()
        assert "bond" in bac
        assert "equity" in bac

    def test_total_margin(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 10,
                                   margin_per_contract=5000))
        book.add(FuturesBookEntry("IR1", _ir_future(), 20,
                                   margin_per_contract=1000))
        assert book.total_margin() == 70_000

    def test_aggregate_risk(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 10,
                                   margin_per_contract=5000))
        risk = book.aggregate_risk()
        assert risk["n_positions"] == 1
        assert risk["total_margin"] == 50_000


# ── Margin ──

class TestMargin:

    def test_daily_settlement(self):
        pnl = futures_daily_settlement(110.5, 111.0, 1000, 10)
        assert pnl == 0.5 * 1000 * 10

    def test_margin_check_ok(self):
        ms = futures_margin_check(5000, 50_000, 30_000)
        assert ms.margin_call == 0.0

    def test_margin_call_triggered(self):
        ms = futures_margin_check(-40_000, 50_000, 30_000)
        assert ms.margin_call > 0


# ── Carry ──

class TestCarryRoll:

    def test_carry_finite(self):
        cr = futures_carry_roll(_bond_future(), contracts=10, days=30,
                                margin_per_contract=5000)
        assert math.isfinite(cr.net_carry)

    def test_financing_cost_positive(self):
        cr = futures_carry_roll(_bond_future(), contracts=10, days=30,
                                margin_per_contract=5000, financing_rate=0.04)
        assert cr.financing_cost > 0

    def test_to_dict(self):
        cr = futures_carry_roll(_bond_future())
        d = cr.to_dict()
        assert "basis_carry" in d
        assert "net" in d


# ── Dashboard ──

class TestDashboard:

    def test_dashboard_fields(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 10,
                                   FuturesAssetClass.BOND, exchange="CME",
                                   margin_per_contract=5000))
        db = futures_dashboard(book, REF)
        assert db.n_positions == 1
        assert db.total_contracts == 10
        assert db.total_margin == 50_000
        assert "bond" in db.by_asset_class

    def test_to_dict(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 10))
        db = futures_dashboard(book, REF)
        d = db.to_dict()
        assert "dv01" in d
        assert "margin" in d


# ── Stress ──

class TestStress:

    def test_five_scenarios(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 10, FuturesAssetClass.BOND))
        results = futures_stress_suite(book)
        assert len(results) == 5

    def test_rates_up_has_pnl(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 10, FuturesAssetClass.BOND))
        results = futures_stress_suite(book)
        up = [r for r in results if r.scenario == "rates_up_100"][0]
        assert up.pnl != 0


# ── Hedge ──

class TestHedge:

    def test_no_recs_within_limits(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 1))
        recs = futures_hedge_recommendations(book, dv01_limit=1e12)
        assert len(recs) == 0

    def test_recs_when_breached(self):
        book = FuturesBook()
        book.add(FuturesBookEntry("BF1", _bond_future(), 100,
                                   margin_per_contract=100_000))
        recs = futures_hedge_recommendations(book, margin_limit=1_000)
        assert len(recs) >= 1


# ── Lifecycle ──

class TestLifecycle:

    def test_expiry_alert(self):
        bf = BondFuture(110, 111, REF + relativedelta(days=5), 1000)
        lc = FuturesLifecycle(bf, "BF1", 10)
        alert = lc.expiry_alert(REF, alert_days=10)
        assert alert is not None
        assert alert["days_remaining"] == 5

    def test_no_alert_far(self):
        bf = _bond_future()
        lc = FuturesLifecycle(bf, "BF1")
        alert = lc.expiry_alert(REF, alert_days=5)
        assert alert is None

    def test_record_roll(self):
        lc = FuturesLifecycle(_bond_future(), "BF1")
        ev = lc.record_roll(REF, REF + relativedelta(months=6), roll_cost=250)
        assert ev["roll_cost"] == 250
        assert len(lc.history) == 1

    def test_record_delivery(self):
        lc = FuturesLifecycle(_bond_future(), "BF1", 10)
        ev = lc.record_delivery(REF, 111.5)
        assert ev["contracts"] == 10
