"""Tests for Kalman filter, dynamic beta, trend extraction."""
import pytest
import numpy as np
from pricebook.statistics.kalman import (
    KalmanFilter, dynamic_beta, dynamic_hedge_ratio, trend_extraction,
)


class TestKalmanFilter:
    def test_constant_signal(self):
        obs = np.ones(50) * 3.0
        kf = KalmanFilter(F=np.array([[1.0]]), H=np.array([[1.0]]),
                          Q=np.array([[0.01]]), R=np.array([[0.1]]))
        result = kf.filter(obs)
        assert abs(result.filtered_states[-1][0] - 3.0) < 0.5

    def test_filter_length(self):
        obs = np.random.default_rng(42).normal(0, 1, 100)
        kf = KalmanFilter(F=np.array([[1.0]]), H=np.array([[1.0]]),
                          Q=np.array([[0.01]]), R=np.array([[1.0]]))
        result = kf.filter(obs)
        assert len(result.filtered_states) == 100


class TestDynamicBeta:
    def test_beta_near_one(self):
        rng = np.random.default_rng(42)
        market = rng.normal(0, 0.01, 200)
        stock = market + rng.normal(0, 0.001, 200)
        betas = dynamic_beta(stock, market)
        assert abs(betas[-1] - 1.0) < 0.5

    def test_length(self):
        rng = np.random.default_rng(42)
        betas = dynamic_beta(rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        assert len(betas) == 100


class TestTrend:
    def test_smoother_than_original(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(0, 1, 100)) + rng.normal(0, 0.5, 100)
        trend = trend_extraction(x)
        assert np.std(np.diff(trend)) < np.std(np.diff(x))
