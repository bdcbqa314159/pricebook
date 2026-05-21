"""Tests for correlated recovery model."""

import pytest
import numpy as np

from pricebook.credit.correlated_recovery import (
    CorrelatedRecoveryModel, systematic_recovery,
)


@pytest.fixture
def model():
    return CorrelatedRecoveryModel(base_recovery=0.40, recovery_vol=0.25, systematic_factor_loading=0.30)


class TestConditionalRecovery:
    def test_neutral(self, model):
        """M=0 → base recovery."""
        r = model.conditional_recovery(0.0)
        assert abs(r - 0.40) < 1e-10

    def test_stress_lower(self, model):
        """Recession (M<0) → lower recovery."""
        r = model.conditional_recovery(-2.0)
        assert r < 0.40

    def test_expansion_higher(self, model):
        """Expansion (M>0) → higher recovery."""
        r = model.conditional_recovery(2.0)
        assert r > 0.40

    def test_bounded(self, model):
        """Recovery always in [0.01, 0.95]."""
        assert model.conditional_recovery(-10.0) >= 0.01
        assert model.conditional_recovery(10.0) <= 0.95

    def test_stress_shortcut(self, model):
        assert model.stress_recovery(2.0) == model.conditional_recovery(-2.0)

    def test_expansion_shortcut(self, model):
        assert model.expansion_recovery(1.0) == model.conditional_recovery(1.0)


class TestSampling:
    def test_sample_shape(self, model):
        samples = model.sample_recoveries(1000, 0.0, rng=np.random.default_rng(42))
        assert samples.shape == (1000,)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_sample_mean_near_conditional(self, model):
        """Samples should have mean near conditional recovery."""
        for M in [-2, 0, 2]:
            samples = model.sample_recoveries(10000, M, rng=np.random.default_rng(42))
            expected = model.conditional_recovery(M)
            assert abs(np.mean(samples) - expected) < 0.03

    def test_stress_samples_lower(self, model):
        """Stress samples should have lower mean than expansion."""
        s_stress = model.sample_recoveries(5000, -2.0, rng=np.random.default_rng(42))
        s_exp = model.sample_recoveries(5000, 2.0, rng=np.random.default_rng(42))
        assert np.mean(s_stress) < np.mean(s_exp)


class TestDistribution:
    def test_scenarios(self, model):
        dist = model.recovery_distribution([-3, -1, 0, 1, 3])
        assert len(dist) == 5
        assert dist[0]["scenario"] == "stress"
        assert dist[2]["scenario"] == "normal"
        assert dist[4]["scenario"] == "expansion"

    def test_monotonic(self, model):
        """Higher M → higher recovery."""
        dist = model.recovery_distribution([-2, -1, 0, 1, 2])
        recoveries = [d["recovery"] for d in dist]
        for i in range(1, len(recoveries)):
            assert recoveries[i] >= recoveries[i - 1]


class TestSystematicRecovery:
    def test_low_default_rate(self):
        """Low default rate → high recovery."""
        r = systematic_recovery(0.40, 0.01)
        assert r > 0.40

    def test_high_default_rate(self):
        """Very high default rate → low recovery."""
        r = systematic_recovery(0.40, 0.80)  # 80% default rate = deep recession
        assert r < 0.40

    def test_bounded(self):
        assert systematic_recovery(0.40, 0.001) <= 0.95
        assert systematic_recovery(0.40, 0.50) >= 0.01


class TestSerialization:
    def test_to_dict(self, model):
        d = model.to_dict()
        assert d["base_recovery"] == 0.40
        assert "stress_2sigma" in d
        assert d["stress_2sigma"] < 0.40
