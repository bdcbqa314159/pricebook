"""Tests for bond rich/cheap analysis and spread strategies."""

import pytest

from pricebook.bond_rv import (
    CrossMarketRV,
    CrossoverSignal,
    CreditSpreadCurveTrade,
    FittedCurveRV,
    NewIssuePremium,
    SpreadMonitorResult,
    asw_spread_monitor,
    build_credit_curve_trade,
    cross_market_rv,
    crossover_monitor,
    fitted_curve_rv,
    new_issue_premium,
    zspread_monitor,
)


# ---- Step 1: fitted curve RV ----

class TestFittedCurveRV:
    def test_cheap_signal(self):
        """Step 1 test: extreme z-score triggers signal."""
        history = [5.0, 6.0, 7.0, 4.0, 5.5] * 4
        rv = fitted_curve_rv("UST_10Y", 0.0450, 0.0400,
                             history_bps=history, threshold=2.0)
        # 50bp spread vs avg ~5.5bp → very cheap
        assert rv.signal == "rich"  # positive z = bond yields more = "rich" in spread terms
        assert rv.spread_bps == pytest.approx(50.0)

    def test_no_history_fair(self):
        rv = fitted_curve_rv("UST_10Y", 0.04, 0.039)
        assert rv.signal == "fair"
        assert rv.z_score is None

    def test_spread_formula(self):
        rv = fitted_curve_rv("X", 0.0500, 0.0480)
        assert rv.spread_bps == pytest.approx(20.0)


class TestCrossMarketRV:
    def test_basic_spread(self):
        rv = cross_market_rv("UST", 0.04, "Bund", 0.025)
        assert rv.spread_bps == pytest.approx(150.0)
        assert rv.fx_adjusted_spread_bps == pytest.approx(150.0)

    def test_fx_hedge_cost(self):
        rv = cross_market_rv("UST", 0.04, "Bund", 0.025,
                             fx_hedge_cost_bps=30.0)
        assert rv.fx_adjusted_spread_bps == pytest.approx(120.0)

    def test_z_score_with_history(self):
        history = [100, 110, 120, 130, 140] * 4
        rv = cross_market_rv("UST", 0.04, "Bund", 0.025,
                             history_bps=history)
        assert rv.z_score is not None


class TestASWMonitor:
    def test_wide_signal(self):
        history = [50, 55, 60, 45, 52] * 4
        result = asw_spread_monitor("AAPL", 120.0, history, threshold=2.0)
        assert result.signal == "rich"
        assert result.spread_type == "ASW"

    def test_fair(self):
        history = [50, 55, 60, 45, 52] * 4
        result = asw_spread_monitor("AAPL", 53.0, history)
        assert result.signal == "fair"


class TestZSpreadMonitor:
    def test_tight(self):
        history = [100, 105, 110, 95, 102] * 4
        result = zspread_monitor("X", 50.0, history, threshold=2.0)
        assert result.signal == "cheap"
        assert result.spread_type == "Z-spread"


# ---- Step 2: spread strategies ----

class TestCreditSpreadCurveTrade:
    def test_dv01_neutral(self):
        """Step 2 test: spread strategy is DV01-neutral."""
        trade = build_credit_curve_trade(
            short_tenor="2Y", long_tenor="10Y",
            short_spread_bps=80, long_spread_bps=120,
            short_dv01_per_million=20.0, long_dv01_per_million=85.0,
            notional=10_000_000,
        )
        assert trade.is_dv01_neutral

    def test_long_face_computed(self):
        trade = build_credit_curve_trade(
            "2Y", "10Y", 80, 120,
            short_dv01_per_million=20.0, long_dv01_per_million=80.0,
            notional=10_000_000,
        )
        # long_face = 10M × 20/80 = 2.5M
        assert trade.long_face == pytest.approx(2_500_000)

    def test_curve_spread(self):
        trade = build_credit_curve_trade("2Y", "10Y", 80, 120, 20, 80)
        assert trade.curve_spread_bps == pytest.approx(40.0)


class TestCrossoverMonitor:
    def test_wide_signal(self):
        history = [200, 210, 220, 190, 205] * 4
        result = crossover_monitor(120.0, 500.0, history, threshold=2.0)
        # crossover = 500 - 120 = 380 vs avg ~205 → wide
        assert result.signal == "rich"
        assert result.crossover_spread_bps == pytest.approx(380.0)

    def test_fair(self):
        history = [200, 210, 220, 190, 205] * 4
        result = crossover_monitor(100.0, 305.0, history)
        assert result.signal == "fair"

    def test_no_history(self):
        result = crossover_monitor(100.0, 300.0)
        assert result.signal == "fair"


class TestNewIssuePremium:
    def test_positive_premium(self):
        nip = new_issue_premium("ACME", 130.0, 120.0)
        assert nip.premium_bps == pytest.approx(10.0)
        assert nip.premium_bps > 0

    def test_zero_premium(self):
        nip = new_issue_premium("ACME", 120.0, 120.0)
        assert nip.premium_bps == pytest.approx(0.0)

    def test_negative_premium(self):
        nip = new_issue_premium("ACME", 115.0, 120.0)
        assert nip.premium_bps < 0
