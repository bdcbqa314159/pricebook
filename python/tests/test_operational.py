"""Tests for operational layers: statistics, market data tools, EOD."""

import math
from datetime import date

import numpy as np
import pytest

from pricebook.statistics import (
    cointegration_test, regime_detect, bootstrap_ci, rolling_stats,
)
from pricebook.market_data_tools import (
    synthetic_market, parse_json_quotes, MarketSnapshot,
)
from pricebook.eod import (
    eod_mtm, eod_pnl, attribute_pnl, eod_risk_report, check_limits,
)


# ---- Statistics ----

class TestCointegration:
    def test_cointegrated(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(500))
        y = 0.5 * x + rng.standard_normal(500) * 0.5
        result = cointegration_test(y, x)
        assert result.hedge_ratio == pytest.approx(0.5, rel=0.2)

    def test_not_cointegrated(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(500))
        y = np.cumsum(rng.standard_normal(500))
        result = cointegration_test(y, x)
        # Two independent random walks — likely not cointegrated
        assert isinstance(result.is_cointegrated, bool)


class TestRegimeDetect:
    def test_two_regimes(self):
        ret = np.concatenate([
            np.random.default_rng(42).standard_normal(200) * 0.005,
            np.random.default_rng(43).standard_normal(200) * 0.02,
        ])
        result = regime_detect(ret, n_regimes=2)
        assert result.n_regimes == 2
        assert len(result.regime_labels) == 400

    def test_momentum_method(self):
        ret = np.random.default_rng(42).standard_normal(200) * 0.01
        result = regime_detect(ret, method="momentum")
        assert result.current_regime in [0, 1]


class TestBootstrap:
    def test_mean_ci(self):
        data = np.random.default_rng(42).standard_normal(500)
        ci = bootstrap_ci(data, "mean", 0.95, n_bootstrap=1000)
        assert ci.lower < ci.estimate < ci.upper

    def test_sharpe_ci(self):
        data = np.random.default_rng(42).standard_normal(500) * 0.01 + 0.0003
        ci = bootstrap_ci(data, "sharpe", 0.95, n_bootstrap=1000)
        assert isinstance(ci.estimate, float)


class TestRollingStats:
    def test_basic(self):
        ret = np.random.default_rng(42).standard_normal(200) * 0.01
        stats = rolling_stats(ret, window=30)
        assert len(stats.rolling_mean) == 200
        assert not np.isnan(stats.rolling_sharpe[50])


# ---- Market Data Tools ----

class TestSyntheticMarket:
    def test_generate(self):
        mkt = synthetic_market(date(2026, 4, 21))
        assert len(mkt.deposits) >= 3
        assert len(mkt.swaps) >= 5
        assert "EUR/USD" in mkt.fx_spots

    def test_json_quotes(self):
        quotes = parse_json_quotes({"USD_5Y": 0.04, "EUR_5Y": 0.03})
        assert len(quotes) == 2
        assert quotes[0].value == 0.04


class TestMarketSnapshot:
    def test_snapshot(self):
        snap = MarketSnapshot(date(2026, 4, 21), {"USD_5Y": 0.04})
        assert snap.quotes["USD_5Y"] == 0.04


# ---- EOD ----

class TestEODMTM:
    def test_basic(self):
        mtm = eod_mtm({"swap": 15000, "bond": -3000})
        assert len(mtm) == 2


class TestEODPnL:
    def test_basic(self):
        today = {"swap": 16000, "bond": -2500}
        yesterday = {"swap": 15000, "bond": -3000}
        result = eod_pnl(today, yesterday)
        assert result.total_pnl == pytest.approx(1500)
        assert result.market_move_pnl == pytest.approx(1500)

    def test_new_trades(self):
        result = eod_pnl({"swap": 100}, {"swap": 100}, new_trades={"new": 500})
        assert result.new_trade_pnl == 500

    def test_attribution(self):
        attrib = attribute_pnl(1000, delta=0.5, spot_move=1500, gamma=0.01)
        assert attrib.delta_pnl == pytest.approx(750)
        assert attrib.total_pnl == 1000


class TestEODRisk:
    def test_report(self):
        report = eod_risk_report({"a": 1e6, "b": -5e5},
                                  sensitivities={"a": {"dv01": 100}, "b": {"dv01": -50}})
        assert report.total_pv == pytest.approx(500_000)
        assert report.total_dv01 == pytest.approx(50)
        assert report.n_trades == 2


class TestLimits:
    def test_breach(self):
        breaches = check_limits(total_dv01=150_000, dv01_limit=100_000)
        assert len(breaches) == 1
        assert breaches[0].limit_type == "DV01"

    def test_no_breach(self):
        breaches = check_limits(total_dv01=50_000, dv01_limit=100_000)
        assert len(breaches) == 0

    def test_multiple(self):
        breaches = check_limits(total_dv01=150_000, total_vega=80_000,
                                dv01_limit=100_000, vega_limit=50_000)
        assert len(breaches) == 2
