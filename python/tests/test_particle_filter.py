"""Tests for particle filter."""

import pytest
import numpy as np

from pricebook.statistics.particle_filter import (
    ParticleFilter, ParticleFilterResult,
)


def _make_linear_gaussian(process_std=0.1, obs_std=0.5):
    """Linear-Gaussian model: x_t = x_{t-1} + w, y_t = x_t + v."""
    def transition(particles, rng):
        return particles + rng.normal(0, process_std, len(particles))

    def obs_log_lik(y, particles):
        return -0.5 * ((y - particles) / obs_std)**2

    return transition, obs_log_lik


def _generate_data(n=100, process_std=0.1, obs_std=0.5, seed=42):
    """Generate data from linear-Gaussian model."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] + rng.normal(0, process_std)
    y = x + rng.normal(0, obs_std, n)
    return x, y


class TestBasicFiltering:
    def test_produces_result(self):
        trans, obs = _make_linear_gaussian()
        pf = ParticleFilter(500, trans, obs)
        _, y = _generate_data(50)
        result = pf.filter(y)
        assert isinstance(result, ParticleFilterResult)
        assert len(result.filtered_means) == 50

    def test_tracks_signal(self):
        """Filtered means should track the true state."""
        trans, obs = _make_linear_gaussian(0.1, 0.3)
        pf = ParticleFilter(1000, trans, obs)
        x_true, y = _generate_data(100, 0.1, 0.3)
        result = pf.filter(y)
        # Correlation between filtered and true should be high
        corr = np.corrcoef(result.filtered_means, x_true)[0, 1]
        assert corr > 0.5

    def test_reduces_noise(self):
        """Filtered std should be less than observation noise."""
        trans, obs = _make_linear_gaussian(0.05, 1.0)
        pf = ParticleFilter(500, trans, obs)
        x_true, y = _generate_data(100, 0.05, 1.0)
        result = pf.filter(y)
        obs_mse = np.mean((y - x_true)**2)
        filter_mse = np.mean((result.filtered_means - x_true)**2)
        assert filter_mse < obs_mse

    def test_stds_positive(self):
        trans, obs = _make_linear_gaussian()
        pf = ParticleFilter(200, trans, obs)
        _, y = _generate_data(30)
        result = pf.filter(y)
        assert np.all(result.filtered_stds >= 0)


class TestESS:
    def test_ess_bounded(self):
        trans, obs = _make_linear_gaussian()
        pf = ParticleFilter(500, trans, obs)
        _, y = _generate_data(50)
        result = pf.filter(y)
        assert np.all(result.effective_sample_sizes > 0)
        assert np.all(result.effective_sample_sizes <= 500)

    def test_resampling_occurs(self):
        """With tight observations, resampling should happen."""
        trans, obs = _make_linear_gaussian(0.1, 0.01)  # very precise obs
        pf = ParticleFilter(200, trans, obs, ess_threshold=0.5)
        _, y = _generate_data(50, 0.1, 0.01)
        result = pf.filter(y)
        assert result.n_resamplings > 0


class TestNonLinear:
    def test_nonlinear_model(self):
        """Particle filter should handle non-linear dynamics."""
        def transition(x, rng):
            return 0.5 * x + 25 * x / (1 + x**2) + rng.normal(0, 1, len(x))

        def obs_log_lik(y, x):
            return -0.5 * ((y - x**2 / 20) / 1.0)**2

        pf = ParticleFilter(500, transition, obs_log_lik,
                             initial_distribution=lambda n, rng: rng.normal(0, 1, n))
        rng = np.random.default_rng(42)
        x = np.zeros(50)
        y = np.zeros(50)
        for t in range(1, 50):
            x[t] = 0.5 * x[t-1] + 25 * x[t-1] / (1 + x[t-1]**2) + rng.normal(0, 1)
            y[t] = x[t]**2 / 20 + rng.normal(0, 1)

        result = pf.filter(y)
        assert result.log_likelihood > -np.inf


class TestEdgeCases:
    def test_few_particles_raises(self):
        with pytest.raises(ValueError, match="n_particles"):
            ParticleFilter(5, lambda x, r: x, lambda y, x: np.zeros(len(x)))

    def test_single_observation(self):
        trans, obs = _make_linear_gaussian()
        pf = ParticleFilter(100, trans, obs)
        result = pf.filter(np.array([1.0]))
        assert len(result.filtered_means) == 1

    def test_to_dict(self):
        trans, obs = _make_linear_gaussian()
        pf = ParticleFilter(100, trans, obs)
        _, y = _generate_data(20)
        d = pf.filter(y).to_dict()
        assert "log_likelihood" in d
        assert "avg_ess" in d
