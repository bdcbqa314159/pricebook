"""Tests for convertible_bond_desk.py — 9-component desk protocol."""

from datetime import date
import pytest

from pricebook.convertible_bond import ConvertibleBond
from pricebook.desks.convertible_bond_desk import (
    cb_risk_metrics, CBRiskMetrics,
    CBBook, CBBookEntry,
    cb_carry_decomposition, CBCarryDecomposition,
    cb_daily_pnl, CBDailyPnL,
    cb_dashboard, CBDashboard,
    cb_stress_suite, CBStressResult,
    cb_capital, CBCapitalResult,
    cb_hedge_recommendations, CBHedgeRecommendation,
    CBLifecycle,
)

N_PATHS = 3_000
SEED = 42


def _sample_cb():
    return ConvertibleBond(notional=100, coupon_rate=0.02,
                           maturity_years=5, conversion_ratio=1.0)


def _sample_book():
    book = CBBook("test")
    cb1 = _sample_cb()
    cb2 = ConvertibleBond(100, 0.03, 3, 1.2)
    book.add(CBBookEntry("cb1", cb1, 100, 0.04, 0.25, 0.02, 0.0, "ACME", "ACME"))
    book.add(CBBookEntry("cb2", cb2, 80, 0.04, 0.30, 0.03, 0.01, "BETA", "BETA"))
    return book


# ---- Risk Metrics ----

class TestCBRiskMetrics:
    def test_basic(self):
        rm = cb_risk_metrics(_sample_cb(), 100, 0.04, 0.25, 0.02,
                             n_paths=N_PATHS, seed=SEED)
        assert rm.pv > 0
        assert rm.bond_floor > 0
        assert rm.conversion_value == 100.0
        assert rm.notional == 100

    def test_delta_positive(self):
        rm = cb_risk_metrics(_sample_cb(), 100, 0.04, 0.25, 0.02,
                             n_paths=N_PATHS, seed=SEED)
        assert rm.equity_delta > 0  # CB price increases with stock

    def test_cs01_nonzero(self):
        rm = cb_risk_metrics(_sample_cb(), 100, 0.04, 0.25, 0.02,
                             n_paths=N_PATHS, seed=SEED)
        assert rm.credit_cs01 != 0  # spread sensitivity exists

    def test_rate_dv01_nonzero(self):
        rm = cb_risk_metrics(_sample_cb(), 100, 0.04, 0.25, 0.02,
                             n_paths=N_PATHS, seed=SEED)
        assert rm.rate_dv01 != 0  # rate sensitivity exists

    def test_otm_cs01_negative(self):
        """OTM CB (bond-like) should have negative CS01."""
        cb = ConvertibleBond(100, 0.02, 5, 0.5)  # conv price = 200
        rm = cb_risk_metrics(cb, 80, 0.04, 0.25, 0.02,
                             n_paths=5_000, seed=SEED)
        assert rm.credit_cs01 < 0

    def test_to_dict(self):
        rm = cb_risk_metrics(_sample_cb(), 100, 0.04, 0.25, 0.02,
                             n_paths=N_PATHS, seed=SEED)
        d = rm.to_dict()
        assert "pv" in d
        assert "equity_delta" in d
        assert "vega" in d


# ---- Book ----

class TestCBBook:
    def test_book_basic(self):
        book = _sample_book()
        assert len(book) == 2
        assert book.total_notional() == 200

    def test_by_issuer(self):
        book = _sample_book()
        by_iss = book.by_issuer()
        assert "ACME" in by_iss
        assert "BETA" in by_iss

    def test_aggregate_risk(self):
        book = _sample_book()
        agg = book.aggregate_risk(n_paths=N_PATHS, seed=SEED)
        assert agg["n_positions"] == 2
        assert agg["total_notional"] == 200
        assert "total_delta" in agg
        assert "total_gamma" in agg


# ---- Carry ----

class TestCBCarry:
    def test_carry_basic(self):
        carry = cb_carry_decomposition(
            _sample_cb(), 100, 0.04, 0.25, 0.02,
            n_paths=N_PATHS, seed=SEED)
        assert carry.coupon_carry > 0
        assert carry.funding_cost > 0
        assert carry.horizon_days == 1

    def test_carry_to_dict(self):
        carry = cb_carry_decomposition(
            _sample_cb(), 100, 0.04, 0.25, 0.02,
            n_paths=N_PATHS, seed=SEED)
        d = carry.to_dict()
        assert "coupon" in d
        assert "gamma" in d


# ---- Daily PnL ----

class TestCBDailyPnL:
    def test_pnl_basic(self):
        pnl = cb_daily_pnl(
            _sample_cb(),
            spot_t0=100, spot_t1=102,
            rate_t0=0.04, rate_t1=0.04,
            vol_t0=0.25, vol_t1=0.25,
            cs_t0=0.02, cs_t1=0.02,
            pnl_date=date(2024, 7, 16),
            n_paths=N_PATHS, seed=SEED,
        )
        assert pnl.total > 0  # stock went up, CB should gain
        assert pnl.delta_pnl > 0

    def test_pnl_to_dict(self):
        pnl = cb_daily_pnl(
            _sample_cb(), 100, 102, 0.04, 0.04, 0.25, 0.25,
            0.02, 0.02, date(2024, 7, 16), n_paths=N_PATHS, seed=SEED)
        d = pnl.to_dict()
        assert "delta" in d
        assert "gamma" in d
        assert "vega" in d


# ---- Dashboard ----

class TestCBDashboard:
    def test_dashboard(self):
        book = _sample_book()
        dash = cb_dashboard(book, date(2024, 7, 15), n_paths=N_PATHS, seed=SEED)
        assert dash.n_positions == 2
        assert dash.total_notional == 200
        assert dash.avg_conversion_premium > 0

    def test_dashboard_to_dict(self):
        book = _sample_book()
        dash = cb_dashboard(book, date(2024, 7, 15), n_paths=N_PATHS, seed=SEED)
        d = dash.to_dict()
        assert "avg_premium" in d


# ---- Stress ----

class TestCBStress:
    def test_stress_suite(self):
        book = _sample_book()
        results = cb_stress_suite(book, n_paths=N_PATHS, seed=SEED)
        assert len(results) == 9
        assert all(isinstance(r, CBStressResult) for r in results)

    def test_stress_to_dict(self):
        book = _sample_book()
        results = cb_stress_suite(book, n_paths=N_PATHS, seed=SEED)
        d = results[0].to_dict()
        assert "scenario" in d
        assert "pnl" in d


# ---- Capital ----

class TestCBCapital:
    def test_capital(self):
        cap = cb_capital(_sample_cb(), 100, 0.04, 0.25, 0.02,
                         n_paths=N_PATHS, seed=SEED)
        assert cap.ead > 0
        assert cap.rwa > 0
        assert cap.capital > 0
        assert cap.eq_charge > 0

    def test_capital_to_dict(self):
        cap = cb_capital(_sample_cb(), 100, 0.04, 0.25, 0.02,
                         n_paths=N_PATHS, seed=SEED)
        d = cap.to_dict()
        assert "girr" in d
        assert "csr" in d
        assert "eq" in d


# ---- Hedge Recommendations ----

class TestCBHedgeRecs:
    def test_recs_with_breach(self):
        book = _sample_book()
        recs = cb_hedge_recommendations(book, delta_limit=50,
                                        n_paths=N_PATHS, seed=SEED)
        assert any(r.risk_type == "equity_delta" for r in recs)

    def test_recs_no_breach(self):
        book = _sample_book()
        recs = cb_hedge_recommendations(book, delta_limit=1_000_000,
                                        gamma_limit=1_000_000,
                                        vega_limit=1_000_000,
                                        cs01_limit=1_000_000,
                                        n_paths=N_PATHS, seed=SEED)
        # Only concentration recs
        assert all(r.risk_type == "concentration" for r in recs)


# ---- Lifecycle ----

class TestCBLifecycle:
    def test_maturity_alert(self):
        lc = CBLifecycle(_sample_cb(), "cb1")
        alert = lc.maturity_alert(0.3)
        assert alert is not None
        assert alert["type"] == "maturity"

    def test_no_maturity_alert(self):
        lc = CBLifecycle(_sample_cb(), "cb1")
        assert lc.maturity_alert(2.0) is None

    def test_parity_alert_itm(self):
        lc = CBLifecycle(_sample_cb(), "cb1")
        alert = lc.parity_alert(150)
        assert alert is not None
        assert "ITM" in alert["message"]

    def test_parity_alert_otm(self):
        lc = CBLifecycle(_sample_cb(), "cb1")
        alert = lc.parity_alert(50)
        assert alert is not None
        assert "OTM" in alert["message"]

    def test_record_event(self):
        lc = CBLifecycle(_sample_cb(), "cb1")
        ev = lc.record_event(date(2024, 7, 15), "coupon", {"amount": 1.0})
        assert ev["type"] == "coupon"
        assert len(lc.history) == 1
