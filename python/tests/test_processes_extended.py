"""Tests for extended stochastic processes."""

import math

import numpy as np
import pytest

from pricebook.processes_extended import (
    bates_paths,
    cev_paths,
    hawkes_paths,
    kou_paths,
    three_halves_paths,
    vg_full_paths,
)
from pricebook.numerical_safety import martingale_test


# ---- CEV ----

class TestCEV:
    def test_beta_one_is_gbm(self):
        """CEV β=1 reduces to GBM."""
        result = cev_paths(100, 0.05, 0.20, beta=1.0, T=1.0,
                           n_steps=100, n_paths=20_000, seed=42)
        mt = martingale_test(result.paths[:, -1], 100, 0.05, 1.0)
        assert mt.passed

    def test_non_negative(self):
        result = cev_paths(100, 0.05, 0.30, beta=0.5, T=1.0,
                           n_steps=100, n_paths=5_000, seed=42)
        assert np.all(result.paths >= 0)

    def test_lower_beta_tighter_distribution(self):
        """Lower β with same vol param → less diffusion at S=100."""
        gbm = cev_paths(100, 0.05, 0.20, beta=1.0, T=1.0,
                        n_steps=100, n_paths=20_000, seed=42)
        cev_low = cev_paths(100, 0.05, 0.20, beta=0.5, T=1.0,
                            n_steps=100, n_paths=20_000, seed=42)
        # σS^0.5 < σS at S=100, so std is lower for β=0.5
        assert cev_low.paths[:, -1].std() < gbm.paths[:, -1].std()


# ---- 3/2 model ----

class TestThreeHalves:
    def test_mean_reverts(self):
        result = three_halves_paths(0.10, kappa=2.0, theta=0.04,
                                     epsilon=0.5, T=5.0,
                                     n_steps=500, n_paths=10_000, seed=42)
        terminal = result.paths[:, -1].mean()
        assert terminal == pytest.approx(0.04, rel=0.80)

    def test_non_negative(self):
        result = three_halves_paths(0.04, 2.0, 0.04, 0.3, T=1.0,
                                     n_steps=100, n_paths=5_000, seed=42)
        assert np.all(result.paths >= 0)


# ---- Kou ----

class TestKou:
    def test_positive_paths(self):
        result = kou_paths(100, 0.05, 0.15, lam=5.0, p=0.6,
                           eta1=10.0, eta2=8.0, T=1.0,
                           n_steps=250, n_paths=5_000, seed=42)
        assert np.all(result.paths[:, -1] > 0)

    def test_jumps_occur(self):
        result = kou_paths(100, 0.05, 0.15, lam=10.0, p=0.5,
                           eta1=5.0, eta2=5.0, T=1.0,
                           n_steps=100, n_paths=1_000, seed=42)
        assert result.n_jumps.mean() > 0

    def test_asymmetric_jumps_skew(self):
        """Upward-biased jumps (p=0.9) → positive skew; downward (p=0.1) → negative."""
        up = kou_paths(100, 0.05, 0.15, lam=5.0, p=0.9,
                       eta1=10.0, eta2=10.0, T=1.0,
                       n_steps=100, n_paths=10_000, seed=42)
        down = kou_paths(100, 0.05, 0.15, lam=5.0, p=0.1,
                         eta1=10.0, eta2=10.0, T=1.0,
                         n_steps=100, n_paths=10_000, seed=42)
        log_up = np.log(up.paths[:, -1] / 100)
        log_down = np.log(down.paths[:, -1] / 100)
        skew_up = np.mean(log_up**3) / np.mean(log_up**2)**1.5
        skew_down = np.mean(log_down**3) / np.mean(log_down**2)**1.5
        assert skew_up > skew_down


# ---- Bates ----

class TestBates:
    def test_reduces_to_heston_without_jumps(self):
        """Bates with λ=0 should behave like Heston."""
        result = bates_paths(100, 0.04, 0.05, kappa=2.0, theta=0.04,
                             xi=0.3, rho=-0.7, lam=0.0, mu_j=0.0,
                             sigma_j=0.0, T=1.0, n_steps=100,
                             n_paths=20_000, seed=42)
        mt = martingale_test(result.S_paths[:, -1], 100, 0.05, 1.0)
        assert mt.passed

    def test_jumps_increase_tails(self):
        """With jumps, tails should be fatter than pure Heston."""
        heston = bates_paths(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7,
                             lam=0.0, mu_j=0.0, sigma_j=0.0, T=1.0,
                             n_steps=100, n_paths=20_000, seed=42)
        bates = bates_paths(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7,
                            lam=3.0, mu_j=-0.05, sigma_j=0.10, T=1.0,
                            n_steps=100, n_paths=20_000, seed=42)
        # Bates should have higher kurtosis
        kurt_h = np.mean((heston.S_paths[:, -1] - heston.S_paths[:, -1].mean())**4)
        kurt_b = np.mean((bates.S_paths[:, -1] - bates.S_paths[:, -1].mean())**4)
        assert kurt_b > kurt_h

    def test_variance_non_negative(self):
        result = bates_paths(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7,
                             lam=3.0, mu_j=0.0, sigma_j=0.1, T=1.0,
                             n_steps=100, n_paths=5_000, seed=42)
        assert np.all(result.v_paths >= 0)


# ---- Hawkes ----

class TestHawkes:
    def test_events_occur(self):
        result = hawkes_paths(mu=1.0, alpha=0.5, beta=2.0, T=10.0,
                              n_paths=100, seed=42)
        n_events = [len(e) for e in result.event_times]
        assert np.mean(n_events) > 0

    def test_intensity_increases_after_event(self):
        """Self-exciting: intensity should spike after events."""
        result = hawkes_paths(mu=0.5, alpha=1.0, beta=3.0, T=5.0,
                              n_paths=1, n_grid=500, seed=42)
        events = result.event_times[0]
        if len(events) > 0:
            # Intensity at event time should be > baseline
            t_event = events[0]
            idx = np.searchsorted(result.times, t_event)
            if idx < len(result.times) - 1:
                assert result.intensities[0, idx + 1] > 0.5  # > baseline μ

    def test_higher_alpha_more_events(self):
        low = hawkes_paths(mu=1.0, alpha=0.1, beta=2.0, T=10.0,
                           n_paths=50, seed=42)
        high = hawkes_paths(mu=1.0, alpha=0.8, beta=2.0, T=10.0,
                            n_paths=50, seed=42)
        n_low = np.mean([len(e) for e in low.event_times])
        n_high = np.mean([len(e) for e in high.event_times])
        assert n_high > n_low


# ---- VG full paths ----

class TestVGFullPaths:
    def test_shape(self):
        result = vg_full_paths(100, 0.05, 0.20, theta_vg=-0.1,
                               nu=0.2, T=1.0, n_steps=50,
                               n_paths=1_000, seed=42)
        assert result.paths.shape == (1_000, 51)

    def test_positive_paths(self):
        result = vg_full_paths(100, 0.05, 0.20, -0.1, 0.2, T=1.0,
                               n_steps=100, n_paths=5_000, seed=42)
        assert np.all(result.paths[:, -1] > 0)

    def test_negative_theta_left_skew(self):
        """Negative θ_VG → left-skewed returns."""
        result = vg_full_paths(100, 0.05, 0.20, theta_vg=-0.3,
                               nu=0.2, T=1.0, n_steps=100,
                               n_paths=20_000, seed=42)
        log_returns = np.log(result.paths[:, -1] / 100)
        skew = np.mean(log_returns**3) / np.mean(log_returns**2)**1.5
        assert skew < 0
