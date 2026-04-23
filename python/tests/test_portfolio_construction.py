"""Tests for portfolio construction."""

import numpy as np
import pytest

from pricebook.portfolio_construction import (
    mean_variance, black_litterman, risk_parity, rebalance,
)


def _cov():
    return np.array([[0.04, 0.01, 0.005],
                     [0.01, 0.02, 0.003],
                     [0.005, 0.003, 0.01]])


class TestMeanVariance:
    def test_weights_sum_to_one(self):
        mu = np.array([0.08, 0.05, 0.03])
        result = mean_variance(mu, _cov(), names=["EQ", "BD", "CM"])
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_long_only(self):
        mu = np.array([0.08, -0.02, 0.03])
        result = mean_variance(mu, _cov(), long_only=True)
        assert np.all(result.weights >= -1e-10)

    def test_sharpe_positive(self):
        mu = np.array([0.08, 0.05, 0.03])
        result = mean_variance(mu, _cov())
        assert result.sharpe > 0

    def test_higher_return_higher_allocation(self):
        mu = np.array([0.10, 0.02, 0.02])
        result = mean_variance(mu, _cov(), long_only=True)
        assert result.weights[0] > result.weights[1]


class TestBlackLitterman:
    def test_basic(self):
        cov = _cov()
        mkt_w = np.array([0.5, 0.3, 0.2])
        Q = np.array([0.05])           # view: asset 0 returns 5%
        omega = np.array([0.001])      # high confidence
        P = np.array([[1, 0, 0]])      # view on asset 0 only
        result = black_litterman(mkt_w, cov, Q, omega, P)
        assert len(result.optimal_weights) == 3
        assert result.posterior_returns is not None

    def test_posterior_shifts_toward_view(self):
        cov = _cov()
        mkt_w = np.array([0.33, 0.33, 0.34])
        pi = 2.5 * cov @ mkt_w  # equilibrium
        # Strong view: asset 0 returns much more
        Q = np.array([0.15])
        omega = np.array([0.0001])  # very confident
        P = np.array([[1, 0, 0]])
        result = black_litterman(mkt_w, cov, Q, omega, P)
        # Posterior for asset 0 should be higher than equilibrium
        assert result.posterior_returns[0] > pi[0]


class TestRiskParity:
    def test_weights_sum_to_one(self):
        result = risk_parity(_cov(), names=["A", "B", "C"])
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_positive(self):
        result = risk_parity(_cov())
        assert np.all(result.weights > 0)

    def test_lower_vol_higher_weight(self):
        """Lower-vol asset gets higher weight in risk parity."""
        result = risk_parity(_cov(), names=["EQ", "BD", "CM"])
        # CM (0.01 var) should have highest weight
        assert result.weights[2] > result.weights[0]

    def test_custom_budgets(self):
        budgets = np.array([0.5, 0.3, 0.2])
        result = risk_parity(_cov(), risk_budgets=budgets)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestRebalance:
    def test_within_threshold(self):
        current = np.array([0.50, 0.30, 0.20])
        target = np.array([0.51, 0.29, 0.20])
        result = rebalance(current, target, threshold=0.05)
        assert not result.should_rebalance

    def test_exceeds_threshold(self):
        current = np.array([0.60, 0.25, 0.15])
        target = np.array([0.50, 0.30, 0.20])
        result = rebalance(current, target, threshold=0.05)
        assert result.should_rebalance

    def test_turnover(self):
        current = np.array([0.50, 0.50])
        target = np.array([0.60, 0.40])
        result = rebalance(current, target)
        assert result.turnover == pytest.approx(0.20)

    def test_min_trade_filters(self):
        current = np.array([0.500, 0.500])
        target = np.array([0.505, 0.495])
        result = rebalance(current, target, min_trade=0.01)
        assert result.trades[0] == 0.0  # below min_trade
