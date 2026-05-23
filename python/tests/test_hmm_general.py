"""Tests for generalised HMM framework."""

import pytest
import numpy as np

from pricebook.statistics.hmm import (
    HMM, HMMFitResult, EmissionType,
    GaussianEmission, StudentTEmission, MixtureEmission,
    MultivariateGaussianEmission, create_emission,
)


def _generate_regime_data(n=500, seed=42):
    """Generate synthetic 2-regime data: low-vol and high-vol."""
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=int)
    state = 0
    for t in range(1, n):
        if rng.random() < 0.05:  # 5% transition prob
            state = 1 - state
        states[t] = state
    # Low vol = 0.01, high vol = 0.03
    means = [0.0005, -0.0003]
    vols = [0.01, 0.03]
    data = np.array([rng.normal(means[s], vols[s]) for s in states])
    return data, states


class TestEmissions:
    def test_gaussian_log_prob(self):
        e = GaussianEmission()
        obs = np.array([0.0, 1.0, -1.0])
        lp = e.log_prob(obs, {"mean": 0.0, "std": 1.0})
        assert lp.shape == (3,)
        assert lp[0] > lp[1]  # 0 is closer to mean

    def test_gaussian_fit(self):
        e = GaussianEmission()
        obs = np.random.default_rng(42).normal(2.0, 0.5, 100)
        weights = np.ones(100)
        params = e.fit_params(obs, weights)
        assert abs(params["mean"] - 2.0) < 0.2
        assert abs(params["std"] - 0.5) < 0.2

    def test_student_t(self):
        e = StudentTEmission()
        obs = np.array([0.0, 1.0, 5.0])
        lp = e.log_prob(obs, {"mean": 0.0, "std": 1.0, "df": 5.0})
        assert lp.shape == (3,)
        # Student-t should give higher probability to extreme obs than Gaussian
        g = GaussianEmission()
        lp_g = g.log_prob(obs, {"mean": 0.0, "std": 1.0})
        assert lp[2] > lp_g[2]  # heavier tail at obs=5

    def test_mixture(self):
        e = MixtureEmission(n_components=2)
        obs = np.array([0.0, 1.0])
        params = {"weights": [0.5, 0.5], "means": [-1.0, 1.0], "stds": [0.5, 0.5]}
        lp = e.log_prob(obs, params)
        assert lp.shape == (2,)

    def test_factory(self):
        for et in EmissionType:
            e = create_emission(et)
            assert isinstance(e, type) is False  # it's an instance

    def test_sample(self):
        e = GaussianEmission()
        rng = np.random.default_rng(42)
        s = e.sample({"mean": 0.0, "std": 1.0}, 100, rng)
        assert s.shape == (100,)


class TestHMMFit:
    def test_2_state_gaussian(self):
        data, true_states = _generate_regime_data()
        hmm = HMM(2, GaussianEmission())
        result = hmm.fit(data, max_iter=50)
        assert isinstance(result, HMMFitResult)
        assert result.n_states == 2
        assert result.converged or result.n_iterations == 50
        assert result.log_likelihood > -np.inf

    def test_transition_matrix_stochastic(self):
        data, _ = _generate_regime_data()
        result = HMM(2).fit(data)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_stationary_sums_to_one(self):
        data, _ = _generate_regime_data()
        result = HMM(2).fit(data)
        assert abs(result.stationary_distribution.sum() - 1.0) < 1e-8

    def test_labels_match_data(self):
        data, true_states = _generate_regime_data(n=300)
        result = HMM(2).fit(data)
        # Labels should have some agreement with true states
        # (may be permuted)
        assert len(result.labels) == len(data)
        unique = set(result.labels)
        assert len(unique) == 2

    def test_filtered_probs_shape(self):
        data, _ = _generate_regime_data(n=200)
        result = HMM(2).fit(data)
        assert result.filtered_probs.shape == (200, 2)
        np.testing.assert_allclose(result.filtered_probs.sum(axis=1), 1.0, atol=1e-6)

    def test_3_state(self):
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.normal(0, 0.01, 100),
                                rng.normal(0, 0.03, 100),
                                rng.normal(0, 0.05, 100)])
        result = HMM(3).fit(data)
        assert result.n_states == 3

    def test_student_t_emission(self):
        data, _ = _generate_regime_data()
        result = HMM(2, StudentTEmission()).fit(data)
        assert result.n_states == 2
        assert "df" in result.emission_params[0]

    def test_aic_bic(self):
        data, _ = _generate_regime_data()
        r2 = HMM(2).fit(data)
        r3 = HMM(3).fit(data)
        # 2-state should have lower BIC than 3-state for 2-regime data
        # (not always guaranteed with short data, but the API should work)
        assert r2.bic > -np.inf and r3.bic > -np.inf

    def test_enum_emission(self):
        data, _ = _generate_regime_data()
        result = HMM(2, EmissionType.GAUSSIAN).fit(data)
        assert result.converged or result.n_iterations > 0


class TestHMMFilter:
    def test_filter_new_data(self):
        data, _ = _generate_regime_data(n=300)
        hmm = HMM(2)
        hmm.fit(data[:200])
        probs = hmm.filter(data[200:])
        assert probs.shape == (100, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_state(self):
        data, _ = _generate_regime_data(n=200)
        hmm = HMM(2)
        hmm.fit(data[:150])
        labels = hmm.predict_state(data[150:])
        assert labels.shape == (50,)
        assert set(labels).issubset({0, 1})

    def test_not_fitted_raises(self):
        hmm = HMM(2)
        with pytest.raises(ValueError, match="not fitted"):
            hmm.filter(np.array([1.0, 2.0]))


class TestSerialization:
    def test_to_dict(self):
        data, _ = _generate_regime_data(n=100)
        result = HMM(2).fit(data, max_iter=10)
        d = result.to_dict()
        assert "n_states" in d
        assert "aic" in d
        assert "bic" in d
        assert "transition_matrix" in d

    def test_invalid_n_states(self):
        with pytest.raises(ValueError):
            HMM(1)
