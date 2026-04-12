"""Tests for cross-asset vol correlation."""

import pytest

from pricebook.vol_correlation import (
    CorrelationSignal,
    CorrelationTrade,
    build_correlation_trade,
    correlation_monitor,
    correlation_sensitivity,
    is_valid_correlation_matrix,
    vol_correlation_matrix,
)


# ---- Step 1: correlation matrix ----

class TestVolCorrelationMatrix:
    def test_symmetric_diagonal_one(self):
        """Step 1 test: correlation matrix is symmetric, diagonal = 1."""
        series = {
            "equity": [0.20, 0.22, 0.19, 0.21, 0.23],
            "fx": [0.08, 0.09, 0.07, 0.085, 0.095],
            "ir": [0.005, 0.006, 0.004, 0.0055, 0.0065],
        }
        matrix = vol_correlation_matrix(series)
        assert is_valid_correlation_matrix(matrix)
        assert matrix[("equity", "equity")] == pytest.approx(1.0)
        assert matrix[("fx", "fx")] == pytest.approx(1.0)
        assert matrix[("equity", "fx")] == pytest.approx(matrix[("fx", "equity")])

    def test_perfect_correlation(self):
        series = {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]}
        matrix = vol_correlation_matrix(series)
        assert matrix[("a", "b")] == pytest.approx(1.0)

    def test_anti_correlation(self):
        series = {"a": [1, 2, 3, 4, 5], "b": [10, 8, 6, 4, 2]}
        matrix = vol_correlation_matrix(series)
        assert matrix[("a", "b")] == pytest.approx(-1.0)

    def test_single_series(self):
        series = {"a": [1, 2, 3]}
        matrix = vol_correlation_matrix(series)
        assert matrix[("a", "a")] == 1.0

    def test_short_series(self):
        series = {"a": [1.0], "b": [2.0]}
        matrix = vol_correlation_matrix(series)
        assert matrix[("a", "b")] == 0.0


class TestCorrelationMonitor:
    def test_high_signal(self):
        history = [0.5, 0.55, 0.6, 0.45, 0.52] * 4
        sig = correlation_monitor(("equity", "fx"), 0.9, history, threshold=2.0)
        assert sig.signal == "high"

    def test_low_signal(self):
        history = [0.5, 0.55, 0.6, 0.45, 0.52] * 4
        sig = correlation_monitor(("equity", "fx"), 0.1, history, threshold=2.0)
        assert sig.signal == "low"

    def test_fair(self):
        history = [0.5, 0.55, 0.6, 0.45, 0.52] * 4
        sig = correlation_monitor(("equity", "fx"), 0.53, history)
        assert sig.signal == "fair"

    def test_no_history(self):
        sig = correlation_monitor(("a", "b"), 0.5, [])
        assert sig.z_score is None


# ---- Step 2: correlation trading ----

class TestCorrelationTrade:
    def test_zero_net_vega(self):
        """Step 2 test: correlation trade has zero net vega."""
        trade = build_correlation_trade(
            "equity", "fx", long_quantity=100,
            long_vega_per_unit=50.0, short_vega_per_unit=80.0,
        )
        assert trade.net_vega == pytest.approx(0.0)

    def test_pnl_when_correlation_breaks(self):
        trade = build_correlation_trade(
            "equity", "fx", 100, 50.0, 80.0,
        )
        # Equity vol up 5pts, FX vol flat → profit
        pnl = trade.pnl(0.05, 0.0)
        assert pnl > 0

    def test_pnl_when_correlated_move(self):
        trade = build_correlation_trade(
            "equity", "fx", 100, 50.0, 80.0,
        )
        # Both up by same amount → net zero (vega neutral)
        pnl = trade.pnl(0.05, 0.05)
        assert pnl == pytest.approx(0.0)

    def test_short_qty_computed(self):
        trade = build_correlation_trade("a", "b", 100, 50.0, 25.0)
        # short_qty = 100 × 50/25 = 200
        assert trade.short_quantity == pytest.approx(200.0)


class TestCorrelationSensitivity:
    def test_positive(self):
        sens = correlation_sensitivity(1000, 500, 0.20, 0.08)
        assert sens == pytest.approx(1000 * 500 * 0.20 * 0.08)

    def test_zero_when_no_vega(self):
        assert correlation_sensitivity(0, 500, 0.20, 0.08) == 0.0
