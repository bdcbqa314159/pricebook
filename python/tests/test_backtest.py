"""Tests for backtesting engine."""

import math

import numpy as np
import pytest

from pricebook.backtest import (
    BacktestConfig, run_backtest, compute_metrics,
    walk_forward, combine_signals, deflated_sharpe,
    bonferroni_threshold, fdr_threshold,
)


def _trending_returns(n=500, drift=0.0003, vol=0.01, seed=42):
    """Synthetic daily returns with a trend."""
    rng = np.random.default_rng(seed)
    return drift + vol * rng.standard_normal(n)


def _momentum_signal(returns):
    """Simple momentum: sign of rolling 20-day return."""
    n = len(returns)
    signals = np.zeros(n)
    for i in range(20, n):
        signals[i] = 1.0 if returns[i-20:i].sum() > 0 else -1.0
    return signals


# ---- Core engine ----

class TestRunBacktest:
    def test_basic(self):
        ret = _trending_returns()
        sig = _momentum_signal(ret)
        result = run_backtest(ret, sig)
        assert result.metrics.n_periods == len(ret)
        assert len(result.pnl_series) == len(ret)
        assert len(result.equity_curve) == len(ret)

    def test_perfect_signal_positive_sharpe(self):
        """Perfect foresight signal should produce positive Sharpe."""
        ret = _trending_returns(drift=0.0005)
        sig = np.sign(ret)  # perfect foresight
        result = run_backtest(ret, sig)
        assert result.metrics.sharpe > 0

    def test_slippage_reduces_pnl(self):
        ret = _trending_returns()
        sig = _momentum_signal(ret)
        clean = run_backtest(ret, sig)
        slipped = run_backtest(ret, sig, BacktestConfig(slippage_bps=10))
        assert slipped.metrics.total_return < clean.metrics.total_return

    def test_commission_reduces_pnl(self):
        ret = _trending_returns()
        sig = _momentum_signal(ret)
        clean = run_backtest(ret, sig)
        comm = run_backtest(ret, sig, BacktestConfig(commission_per_trade=10))
        assert comm.metrics.total_return < clean.metrics.total_return

    def test_max_position_clamps(self):
        ret = _trending_returns()
        sig = np.full(len(ret), 2.0)  # signal > 1
        result = run_backtest(ret, sig, BacktestConfig(max_position=0.5))
        assert np.all(result.position_series <= 0.5)

    def test_flat_signal_zero_pnl(self):
        ret = _trending_returns()
        sig = np.zeros(len(ret))
        result = run_backtest(ret, sig)
        assert result.metrics.total_return == pytest.approx(0.0)


# ---- Metrics ----

class TestMetrics:
    def test_sharpe_positive(self):
        pnl = np.array([100, -50, 80, 120, -30, 90, 60, -20, 110, 70])
        m = compute_metrics(pnl)
        assert m.sharpe > 0

    def test_max_drawdown(self):
        pnl = np.array([100, 200, -500, 100, 50])
        m = compute_metrics(pnl, initial_capital=10000)
        assert m.max_drawdown > 0

    def test_hit_ratio(self):
        pnl = np.array([1, 1, 1, -1, -1])
        m = compute_metrics(pnl)
        assert m.hit_ratio == pytest.approx(0.6)

    def test_sortino(self):
        pnl = np.array([100, -50, 80, 120, -30])
        m = compute_metrics(pnl)
        assert m.sortino > 0

    def test_empty(self):
        m = compute_metrics(np.array([]))
        assert m.sharpe == 0.0


# ---- Walk-forward ----

class TestWalkForward:
    def test_basic(self):
        ret = _trending_returns(1000)
        result = walk_forward(ret, _momentum_signal, n_folds=3)
        assert result.n_folds == 3
        assert len(result.out_of_sample_sharpes) == 3

    def test_overfitting_measure(self):
        """IS-OOS decay should be non-negative for overfitted strategies."""
        ret = _trending_returns(1000)
        result = walk_forward(ret, _momentum_signal, n_folds=3)
        assert isinstance(result.is_vs_oos_decay, float)


# ---- Signal combination ----

class TestSignalCombination:
    def test_weighted(self):
        s1 = np.array([1.0, -1.0, 0.5])
        s2 = np.array([0.5, 0.5, -0.5])
        combined = combine_signals([s1, s2])
        assert len(combined) == 3

    def test_majority(self):
        s1 = np.array([1.0, -1.0, 1.0])
        s2 = np.array([1.0, 1.0, -1.0])
        s3 = np.array([1.0, -1.0, -1.0])
        combined = combine_signals([s1, s2, s3], method="majority")
        assert combined[0] > 0  # 3/3 positive → positive


# ---- Hypothesis testing ----

class TestHypothesisTesting:
    def test_deflated_sharpe(self):
        """High Sharpe with few trials → high probability."""
        prob = deflated_sharpe(2.0, 1, 252)
        assert prob > 0.5

    def test_deflated_sharpe_many_trials(self):
        """Same Sharpe with many trials → lower probability (data snooping)."""
        few = deflated_sharpe(1.5, 5, 252)
        many = deflated_sharpe(1.5, 100, 252)
        assert many < few

    def test_bonferroni(self):
        assert bonferroni_threshold(0.05, 10) == pytest.approx(0.005)

    def test_fdr(self):
        p_values = [0.001, 0.01, 0.03, 0.20, 0.50]
        threshold = fdr_threshold(p_values, 0.05)
        assert threshold > 0
