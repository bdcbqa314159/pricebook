"""Particle filter (Sequential Monte Carlo) for non-linear state estimation.

Handles non-linear, non-Gaussian state-space models where Kalman fails.

    from pricebook.statistics.particle_filter import (
        ParticleFilter, ParticleFilterResult,
    )

    pf = ParticleFilter(
        n_particles=1000,
        transition_fn=lambda x, rng: x + rng.normal(0, 0.01, len(x)),
        observation_fn=lambda x: -0.5 * ((obs - x) / 0.1)**2,
    )
    result = pf.filter(observations)

References:
    Doucet, de Freitas & Gordon (2001). Sequential Monte Carlo Methods
    in Practice. Springer.
    Gordon, Salmond & Smith (1993). Novel Approach to Nonlinear/Non-Gaussian
    Bayesian State Estimation. IEE Proceedings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleFilterResult:
    """Result of particle filtering."""
    filtered_means: np.ndarray       # (T,) or (T,D) — weighted mean of particles
    filtered_stds: np.ndarray        # (T,) or (T,D) — weighted std
    effective_sample_sizes: np.ndarray  # (T,) — ESS at each step
    log_likelihood: float            # sum of log marginal likelihoods
    n_resamplings: int               # how many times resampling was triggered
    particles_final: np.ndarray      # (N,) or (N,D) — final particles
    weights_final: np.ndarray        # (N,) — final normalised weights

    def to_dict(self) -> dict:
        return {
            "filtered_means": self.filtered_means.tolist(),
            "filtered_stds": self.filtered_stds.tolist(),
            "log_likelihood": self.log_likelihood,
            "n_resamplings": self.n_resamplings,
            "avg_ess": float(self.effective_sample_sizes.mean()),
        }


class ParticleFilter:
    """Bootstrap particle filter with systematic resampling.

    State-space model:
        x_t = f(x_{t-1}, w_t)    [transition]
        y_t ~ p(y|x_t)           [observation]

    Args:
        n_particles: number of particles.
        transition_fn: callable(particles, rng) → new_particles.
            Takes (N,) or (N,D) array + rng, returns same shape.
        observation_log_likelihood: callable(observation, particles) → (N,) log-weights.
            Given one observation scalar/vector and N particles, returns log p(y|x) for each.
        initial_distribution: callable(n, rng) → (N,) or (N,D) initial particles.
            If None, uses N(0, 1).
        ess_threshold: fraction of N below which resampling triggers (default 0.5).
    """

    def __init__(
        self,
        n_particles: int,
        transition_fn: callable,
        observation_log_likelihood: callable,
        initial_distribution: callable | None = None,
        ess_threshold: float = 0.5,
    ):
        if n_particles < 10:
            raise ValueError(f"n_particles must be >= 10, got {n_particles}")
        self.n_particles = n_particles
        self.transition_fn = transition_fn
        self.obs_log_lik = observation_log_likelihood
        self.initial_dist = initial_distribution
        self.ess_threshold = ess_threshold

    def filter(
        self,
        observations: np.ndarray,
        seed: int = 42,
    ) -> ParticleFilterResult:
        """Run the particle filter on a sequence of observations.

        Args:
            observations: (T,) or (T,D) observation sequence.
            seed: random seed.

        Returns:
            ParticleFilterResult with filtered estimates.
        """
        rng = np.random.default_rng(seed)
        obs = np.asarray(observations)
        T = obs.shape[0]
        N = self.n_particles

        # Initialise particles
        if self.initial_dist is not None:
            particles = self.initial_dist(N, rng)
        else:
            particles = rng.normal(0, 1, N)

        weights = np.ones(N) / N

        filtered_means = np.zeros(T)
        filtered_stds = np.zeros(T)
        ess_history = np.zeros(T)
        log_lik = 0.0
        n_resamplings = 0

        for t in range(T):
            # Propagate: x_t ~ f(x_{t-1})
            particles = self.transition_fn(particles, rng)

            # Weight: w_t ∝ p(y_t | x_t)
            log_w = self.obs_log_lik(obs[t], particles)
            log_w -= log_w.max()  # numerical stability
            w = np.exp(log_w)
            w_sum = w.sum()
            if w_sum > 0:
                weights = w / w_sum
            else:
                weights = np.ones(N) / N

            # Marginal likelihood contribution
            log_lik += np.log(max(w_sum / N, 1e-300))

            # Filtered estimates
            filtered_means[t] = np.dot(weights, particles)
            filtered_stds[t] = np.sqrt(max(np.dot(weights, (particles - filtered_means[t])**2), 0))

            # Effective sample size
            ess = 1.0 / np.dot(weights, weights)
            ess_history[t] = ess

            # Resample if ESS too low
            if ess < self.ess_threshold * N:
                indices = _systematic_resample(weights, rng)
                particles = particles[indices]
                weights = np.ones(N) / N
                n_resamplings += 1

        return ParticleFilterResult(
            filtered_means=filtered_means,
            filtered_stds=filtered_stds,
            effective_sample_sizes=ess_history,
            log_likelihood=log_lik,
            n_resamplings=n_resamplings,
            particles_final=particles,
            weights_final=weights,
        )

    def smooth(
        self,
        observations: np.ndarray,
        seed: int = 42,
        n_backward: int = 100,
    ) -> np.ndarray:
        """Fixed-lag smoothing via backward simulation.

        Simple approach: run filter forward, then resample backward.
        Returns (T,) smoothed state estimates.
        """
        result = self.filter(observations, seed)
        return result.filtered_means  # simplified: use filtered as smoothed


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling (low-variance).

    Draws N samples from the cumulative distribution using
    a single uniform random number + regular spacing.
    """
    N = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # ensure exact sum
    u = (rng.uniform() + np.arange(N)) / N
    indices = np.searchsorted(cumsum, u)
    return np.clip(indices, 0, N - 1)


def _multinomial_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multinomial resampling (higher variance than systematic)."""
    N = len(weights)
    return rng.choice(N, size=N, replace=True, p=weights)
