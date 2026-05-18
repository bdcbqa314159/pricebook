"""Tests for numerical._mc."""
import pytest, numpy as np
from pricebook.numerical._mc import qe_heston_step, multilevel_mc

class TestQEHeston:
    def test_positive_variance(self):
        rng = np.random.default_rng(42)
        v = np.array([0.04])
        v_new = qe_heston_step(v, kappa=2.0, theta=0.04, xi=0.3, dt=0.01, rng=rng)
        assert v_new[0] > 0

class TestMLMC:
    def test_callable(self):
        assert callable(multilevel_mc)
