"""Tests for multi-factor hybrid MC."""
import math, numpy as np, pytest
from pricebook.hybrid_mc import HybridFactor, HybridMCEngine, hybrid_payoff_evaluate

class TestHybridMCEngine:
    def test_basic(self):
        factors = [
            HybridFactor("equity", 100, 0.20, 0, 0, "gbm"),
            HybridFactor("rate", 0.04, 0.01, 0.1, 0.04, "ou"),
            HybridFactor("fx", 1.10, 0.08, 0, 0, "gbm"),
        ]
        corr = np.array([[1, 0.3, -0.2], [0.3, 1, 0.1], [-0.2, 0.1, 1]])
        engine = HybridMCEngine(factors, corr)
        r = engine.simulate(1.0, n_paths=500, seed=42)
        assert r.n_factors == 3
        assert "equity" in r.paths and "rate" in r.paths and "fx" in r.paths

    def test_paths_shape(self):
        factors = [HybridFactor("S", 100, 0.20, 0, 0, "gbm")]
        engine = HybridMCEngine(factors, np.eye(1))
        r = engine.simulate(1.0, n_paths=200, n_steps=30, seed=42)
        assert r.paths["S"].shape == (200, 31)

    def test_gbm_positive(self):
        factors = [HybridFactor("S", 100, 0.30, 0, 0, "gbm")]
        engine = HybridMCEngine(factors, np.eye(1))
        r = engine.simulate(1.0, n_paths=500, seed=42)
        assert np.all(r.paths["S"] > 0)

class TestPayoffEvaluate:
    def test_basic(self):
        factors = [HybridFactor("S", 100, 0.20, 0, 0, "gbm")]
        engine = HybridMCEngine(factors, np.eye(1))
        r = engine.simulate(1.0, n_paths=1000, seed=42)
        payoff_fn = lambda p: np.maximum(p["S"][:, -1] - 100, 0)
        result = hybrid_payoff_evaluate(r, payoff_fn, rate=0.03)
        assert result.price > 0
        assert result.std_error > 0

    def test_multi_factor_payoff(self):
        factors = [HybridFactor("eq", 100, 0.20, 0, 0, "gbm"),
                    HybridFactor("fx", 1.0, 0.10, 0, 0, "gbm")]
        corr = np.array([[1, 0.3], [0.3, 1]])
        engine = HybridMCEngine(factors, corr)
        r = engine.simulate(1.0, n_paths=1000, seed=42)
        # Quanto payoff: equity performance in FX
        payoff_fn = lambda p: np.maximum(p["eq"][:, -1] * p["fx"][:, -1] / 100 - 100, 0)
        result = hybrid_payoff_evaluate(r, payoff_fn, rate=0.03)
        assert result.price >= 0
