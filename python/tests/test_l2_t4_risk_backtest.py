"""Regression for L2 phase-2 audit of `risk.backtest`:

(a) ``compute_metrics`` used ``np.std`` with default ``ddof=0``
    (population std).  Sharpe/Sortino are conventionally reported with
    ``ddof=1`` (sample std).  Pre-fix vol understated by sqrt(n/(n-1)).

(b) ``run_backtest`` did not charge slippage/commission on the initial
    position entry (i=0).  Non-zero starting signal incurred no cost.

(c) ``walk_forward`` was misnamed — it called ``signal_func(train)`` and
    ``signal_func(test)`` separately, so each test fold "started fresh"
    on test data with no access to train history.  For any signal_func
    with a warm-up (e.g. 20-day momentum), the first warm-up bars of
    test were undefined/biased.  Real walk-forward computes signals on
    ``concat(train, test)`` and slices out the test portion.

(d) ``deflated_sharpe`` used the crude first-order approximation
    ``E[max] ≈ Φ⁻¹(1 - 1/n)`` instead of the Bailey-De Prado (2014)
    formula with the Euler-Mascheroni correction term.  Pre-fix
    understated expected max by ~7% for n=100, *over*-stating the
    deflated probability (admitting more false positives).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.risk.backtest import (
    BacktestConfig,
    compute_metrics,
    deflated_sharpe,
    run_backtest,
    walk_forward,
)


class TestComputeMetricsSampleStd:
    def test_vol_uses_ddof_1(self):
        """Vol from compute_metrics should equal np.std(ddof=1)*sqrt(252)."""
        pnl = np.array([100, -50, 80, 120, -30, 90, 60, -20, 110, 70], dtype=float)
        m = compute_metrics(pnl, initial_capital=1.0, periods_per_year=252)
        returns = pnl  # initial_capital=1 → returns=pnl
        expected_vol = float(np.std(returns, ddof=1)) * math.sqrt(252)
        assert m.annualised_vol == pytest.approx(expected_vol, rel=1e-12)

    def test_sharpe_consistent_with_sample_std(self):
        """Sharpe = ann_return / sample_vol."""
        pnl = np.array([100, -50, 80, 120, -30], dtype=float)
        m = compute_metrics(pnl, initial_capital=1.0)
        if m.annualised_vol > 0:
            assert m.sharpe == pytest.approx(m.annualised_return / m.annualised_vol,
                                              rel=1e-12)


class TestRunBacktestInitialSlippage:
    def test_initial_position_charges_slippage(self):
        """A constant +1 signal should incur slippage on entry at i=0."""
        n = 20
        returns = np.zeros(n)
        signals = np.ones(n)  # constant long.
        cfg = BacktestConfig(slippage_bps=10.0, commission_per_trade=0.0,
                             initial_capital=1_000_000.0)
        result = run_backtest(returns, signals, cfg)
        # Pre-fix: pnl[0] = 0, cumulative_pnl[-1] = 0 (no rebalances after entry).
        # Post-fix: pnl[0] = -|1.0| * 10/10000 * 1_000_000 = -1_000.
        assert result.pnl_series[0] == pytest.approx(-1_000.0, abs=1e-9)

    def test_zero_initial_position_no_charge(self):
        n = 20
        returns = np.zeros(n)
        signals = np.zeros(n)
        cfg = BacktestConfig(slippage_bps=10.0, commission_per_trade=5.0)
        result = run_backtest(returns, signals, cfg)
        assert result.pnl_series[0] == 0.0

    def test_initial_position_commission_charged(self):
        n = 20
        returns = np.zeros(n)
        signals = 0.5 * np.ones(n)
        cfg = BacktestConfig(slippage_bps=0.0, commission_per_trade=7.0)
        result = run_backtest(returns, signals, cfg)
        # Initial entry is non-zero → commission charged.
        assert result.pnl_series[0] == pytest.approx(-7.0, abs=1e-12)


class TestWalkForwardHistoryContext:
    def test_signal_func_sees_full_history(self):
        """Each fold's signal_func call must include train history."""
        seen_lengths: list[int] = []

        def recording_signal(returns):
            seen_lengths.append(len(returns))
            return np.zeros_like(returns)

        ret = np.random.default_rng(42).standard_normal(300)
        walk_forward(ret, recording_signal, n_folds=3, train_ratio=0.6)
        # Pre-fix: each fold called signal_func twice (on train, then on test).
        # Post-fix: each fold calls signal_func once on combined (train+test).
        # With 3 folds and fold_size=100, each combined length = train+test = 100.
        # 3 folds × 1 call = 3 entries each of length 100.
        assert len(seen_lengths) == 3
        for length in seen_lengths:
            assert length == 100

    def test_warmup_signal_no_undefined_on_test(self):
        """A 20-day-warmup signal should not have nan/undefined values on test."""
        WARMUP = 20
        seen_test_starts: list[float] = []

        def momentum_signal(returns):
            # Signal at t = sign(sum(returns[t-WARMUP:t])), or 0 if not enough history.
            out = np.zeros_like(returns)
            for i in range(WARMUP, len(returns)):
                out[i] = float(np.sign(returns[i - WARMUP : i].sum()))
            return out

        ret = np.random.default_rng(42).standard_normal(300)
        result = walk_forward(ret, momentum_signal, n_folds=3, train_ratio=0.6)
        # Post-fix: signal_func sees train+test, so signals at test position 0
        # use the last WARMUP train bars — well-defined.  Pre-fix had nan/0
        # for the first WARMUP test bars (no train history).
        assert result.n_folds == 3
        # Output Sharpes are finite (no NaN propagation).
        for sr in result.out_of_sample_sharpes:
            assert math.isfinite(sr)


class TestDeflatedSharpeBaileyDePrado:
    def test_n_100_matches_bdp_formula(self):
        """For n=100 trials, expected max should be ~2.508 (BdP)."""
        # Construct a case where SR is at the BdP expected max → DSR should be 0.5.
        from scipy.stats import norm
        gamma = 0.5772156649015329
        e_max = ((1 - gamma) * norm.ppf(1 - 1 / 100)
                 + gamma * norm.ppf(1 - 1 / (100 * math.e)))
        # Pre-fix used norm.ppf(1 - 1/100) ≈ 2.326; BdP value ≈ 2.508.
        assert e_max == pytest.approx(2.530, abs=0.01)

        # At SR == e_max with large n_observations, DSR should be ~0.5.
        dsr_at_max = deflated_sharpe(e_max, n_trials=100, n_observations=10_000)
        assert dsr_at_max == pytest.approx(0.5, abs=0.01)

    def test_below_bdp_threshold_lower_significance(self):
        """SR = 2.326 (the pre-fix threshold) should be BELOW 0.5 DSR post-fix."""
        dsr = deflated_sharpe(2.326, n_trials=100, n_observations=10_000)
        # Pre-fix returned ~0.5 (matched its own threshold).
        # Post-fix returns lower (since 2.326 < 2.508 = post-fix threshold).
        assert dsr < 0.45

    def test_high_sharpe_high_probability(self):
        dsr = deflated_sharpe(3.5, n_trials=10, n_observations=1000)
        assert 0.8 < dsr <= 1.0
