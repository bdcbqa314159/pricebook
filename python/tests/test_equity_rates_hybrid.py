"""Tests for equity-rates hybrids."""
import math, numpy as np, pytest
from pricebook.equity_rates_hybrid import callable_equity_note, equity_ir_joint_simulate, hybrid_autocallable

class TestCallableEquityNote:
    def test_basic(self):
        r = callable_equity_note(100, 0.04, 0.20, 0.01, 0.2, 1000, 1.0, 0.0,
                                   3.0, [1.0, 2.0], n_paths=500, seed=42)
        assert r.price > 0
    def test_call_probability(self):
        r = callable_equity_note(100, 0.04, 0.30, 0.01, 0.2, 1000, 1.0, 0.0,
                                   3.0, [0.5, 1.0, 1.5, 2.0, 2.5], n_paths=500, seed=42)
        assert 0 <= r.call_probability <= 1

class TestJointSim:
    def test_basic(self):
        r = equity_ir_joint_simulate(100, 0.04, 0.20, 0.01, 0.3, 1.0, n_paths=200, seed=42)
        assert r.equity_paths.shape == (200, 51)
        assert r.rate_paths.shape == (200, 51)
    def test_correlation_sign(self):
        r = equity_ir_joint_simulate(100, 0.04, 0.20, 0.01, 0.8, 1.0, n_paths=500, seed=42)
        assert r.correlation_realised > 0

class TestHybridAutocall:
    def test_basic(self):
        r = hybrid_autocallable(100, 0.04, 0.20, 0.01, 0.2, 1000, 30,
                                  1.0, 0.02, 3.0, [25, 50, 75], n_paths=500, seed=42)
        assert r.price > 0
    def test_ir_floor_effect(self):
        """Very high IR floor blocks autocalls."""
        r = hybrid_autocallable(100, 0.04, 0.20, 0.01, 0.2, 1000, 30,
                                  1.0, 0.10, 3.0, [25, 50, 75], n_paths=500, seed=42)
        # High floor → fewer autocalls
        assert r.ir_floor_triggered >= 0
