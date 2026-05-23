"""Generalised Hidden Markov Model framework.

Pluggable emission models, multiple filtering algorithms, and Baum-Welch
calibration. Designed to support any observable: vol, spread, yield, return.

    from pricebook.statistics.hmm import (
        HMM, EmissionModel, GaussianEmission, StudentTEmission,
        MixtureEmission, HMMFitResult,
    )

    hmm = HMM(n_states=3, emission=GaussianEmission())
    result = hmm.fit(returns)
    probs = hmm.filter(new_data)

References:
    Rabiner (1989). A Tutorial on Hidden Markov Models and Selected
    Applications in Speech Recognition. Proc. IEEE.
    Hamilton (1989). A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Emission Models (pluggable observation distributions)
# ═══════════════════════════════════════════════════════════════


class EmissionType(Enum):
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    MIXTURE = "mixture"
    MULTIVARIATE_GAUSSIAN = "multivariate_gaussian"


class EmissionModel(ABC):
    """Abstract emission distribution for HMM."""

    @abstractmethod
    def log_prob(self, obs: np.ndarray, params: dict) -> np.ndarray:
        """Log probability of observations under this emission.

        Args:
            obs: (T,) or (T, D) array of observations.
            params: emission parameters for one state.

        Returns:
            (T,) log probabilities.
        """

    @abstractmethod
    def fit_params(self, obs: np.ndarray, weights: np.ndarray) -> dict:
        """Fit emission parameters from weighted observations.

        Args:
            obs: (T,) or (T, D) observations.
            weights: (T,) posterior state probabilities (gamma).

        Returns:
            dict of fitted parameters.
        """

    @abstractmethod
    def sample(self, params: dict, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n observations from this emission."""

    @abstractmethod
    def initial_params(self, obs: np.ndarray, state_idx: int, n_states: int) -> dict:
        """Generate initial parameters for a state from data."""


class GaussianEmission(EmissionModel):
    """Univariate Gaussian emission: y|s=k ~ N(μ_k, σ_k²)."""

    def log_prob(self, obs, params):
        mu = params["mean"]
        sigma = max(params["std"], 1e-10)
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((obs - mu) / sigma)**2

    def fit_params(self, obs, weights):
        w_sum = max(weights.sum(), 1e-10)
        mu = np.dot(weights, obs) / w_sum
        var = np.dot(weights, (obs - mu)**2) / w_sum
        return {"mean": float(mu), "std": float(max(np.sqrt(var), 1e-6))}

    def sample(self, params, n, rng):
        return rng.normal(params["mean"], params["std"], n)

    def initial_params(self, obs, state_idx, n_states):
        sorted_obs = np.sort(obs)
        T = len(obs)
        q = sorted_obs[int(T * (state_idx + 0.5) / n_states)]
        return {"mean": float(q), "std": float(np.std(obs) / n_states)}


class StudentTEmission(EmissionModel):
    """Univariate Student-t emission: heavier tails for financial returns."""

    def log_prob(self, obs, params):
        mu = params["mean"]
        sigma = max(params["std"], 1e-10)
        nu = max(params.get("df", 5.0), 2.01)
        z = (obs - mu) / sigma
        from scipy.special import gammaln
        lp = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
              - 0.5 * np.log(nu * np.pi) - np.log(sigma)
              - (nu + 1) / 2 * np.log(1 + z**2 / nu))
        return lp

    def fit_params(self, obs, weights):
        w_sum = max(weights.sum(), 1e-10)
        mu = np.dot(weights, obs) / w_sum
        var = np.dot(weights, (obs - mu)**2) / w_sum
        return {"mean": float(mu), "std": float(max(np.sqrt(var), 1e-6)), "df": 5.0}

    def sample(self, params, n, rng):
        from scipy.stats import t as t_dist
        return t_dist.rvs(params.get("df", 5.0), loc=params["mean"],
                          scale=params["std"], size=n, random_state=rng)

    def initial_params(self, obs, state_idx, n_states):
        p = GaussianEmission().initial_params(obs, state_idx, n_states)
        p["df"] = 5.0
        return p


class MixtureEmission(EmissionModel):
    """Gaussian mixture emission: y|s=k ~ Σ w_j N(μ_j, σ_j²)."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def log_prob(self, obs, params):
        weights = np.array(params["weights"])
        means = np.array(params["means"])
        stds = np.array(params["stds"])
        T = len(obs)
        log_components = np.zeros((T, self.n_components))
        for j in range(self.n_components):
            s = max(stds[j], 1e-10)
            log_components[:, j] = (np.log(max(weights[j], 1e-15))
                                     - 0.5 * np.log(2 * np.pi * s**2)
                                     - 0.5 * ((obs - means[j]) / s)**2)
        max_lc = log_components.max(axis=1, keepdims=True)
        return (max_lc.squeeze() + np.log(np.exp(log_components - max_lc).sum(axis=1)))

    def fit_params(self, obs, weights):
        # Simplified: fit as single Gaussian weighted
        w_sum = max(weights.sum(), 1e-10)
        mu = np.dot(weights, obs) / w_sum
        var = np.dot(weights, (obs - mu)**2) / w_sum
        std = max(np.sqrt(var), 1e-6)
        # Split into components around the mean
        return {
            "weights": [1.0 / self.n_components] * self.n_components,
            "means": [float(mu - std * 0.5), float(mu + std * 0.5)][:self.n_components],
            "stds": [float(std)] * self.n_components,
        }

    def sample(self, params, n, rng):
        weights = np.array(params["weights"])
        means = np.array(params["means"])
        stds = np.array(params["stds"])
        components = rng.choice(len(weights), size=n, p=weights / weights.sum())
        return np.array([rng.normal(means[c], stds[c]) for c in components])

    def initial_params(self, obs, state_idx, n_states):
        p = GaussianEmission().initial_params(obs, state_idx, n_states)
        return {
            "weights": [0.5, 0.5],
            "means": [p["mean"] - p["std"] * 0.3, p["mean"] + p["std"] * 0.3],
            "stds": [p["std"], p["std"]],
        }


class MultivariateGaussianEmission(EmissionModel):
    """Multivariate Gaussian: y|s=k ~ N(μ_k, Σ_k)."""

    def log_prob(self, obs, params):
        mu = np.array(params["mean"])
        cov = np.array(params["cov"])
        d = len(mu)
        diff = obs - mu
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.inv(cov)
        maha = np.sum(diff @ cov_inv * diff, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)

    def fit_params(self, obs, weights):
        w_sum = max(weights.sum(), 1e-10)
        mu = np.average(obs, axis=0, weights=weights)
        diff = obs - mu
        cov = (diff.T * weights) @ diff / w_sum
        cov += np.eye(len(mu)) * 1e-6  # regularise
        return {"mean": mu.tolist(), "cov": cov.tolist()}

    def sample(self, params, n, rng):
        return rng.multivariate_normal(params["mean"], params["cov"], n)

    def initial_params(self, obs, state_idx, n_states):
        d = obs.shape[1]
        mu = np.mean(obs, axis=0) + (state_idx - n_states / 2) * np.std(obs, axis=0) * 0.5
        cov = np.diag(np.var(obs, axis=0))
        return {"mean": mu.tolist(), "cov": cov.tolist()}


def create_emission(emission_type: EmissionType, **kwargs) -> EmissionModel:
    """Factory for emission models."""
    if emission_type == EmissionType.GAUSSIAN:
        return GaussianEmission()
    elif emission_type == EmissionType.STUDENT_T:
        return StudentTEmission()
    elif emission_type == EmissionType.MIXTURE:
        return MixtureEmission(**kwargs)
    elif emission_type == EmissionType.MULTIVARIATE_GAUSSIAN:
        return MultivariateGaussianEmission()
    raise ValueError(f"Unknown emission type: {emission_type}")


# ═══════════════════════════════════════════════════════════════
# HMM Result
# ═══════════════════════════════════════════════════════════════


@dataclass
class HMMFitResult:
    """Result of HMM fitting."""
    n_states: int
    transition_matrix: np.ndarray
    emission_params: list[dict]
    stationary_distribution: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    n_iterations: int
    labels: np.ndarray           # Viterbi path
    filtered_probs: np.ndarray   # (T, K) posterior state probabilities
    converged: bool

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "transition_matrix": self.transition_matrix.tolist(),
            "emission_params": self.emission_params,
            "stationary_distribution": self.stationary_distribution.tolist(),
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
        }


# ═══════════════════════════════════════════════════════════════
# HMM Class
# ═══════════════════════════════════════════════════════════════


class HMM:
    """Generalised Hidden Markov Model.

    Pluggable emission model, Baum-Welch calibration, Viterbi decoding.

    Args:
        n_states: number of hidden states.
        emission: emission model (ABC or EmissionType enum).
    """

    def __init__(self, n_states: int, emission: EmissionModel | EmissionType = EmissionType.GAUSSIAN):
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states = n_states
        if isinstance(emission, EmissionType):
            self.emission = create_emission(emission)
        else:
            self.emission = emission
        self._transition: np.ndarray | None = None
        self._emission_params: list[dict] | None = None
        self._pi: np.ndarray | None = None

    def fit(
        self,
        observations: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int = 42,
    ) -> HMMFitResult:
        """Fit HMM via Baum-Welch (EM algorithm).

        Args:
            observations: (T,) univariate or (T, D) multivariate.
            max_iter: maximum EM iterations.
            tol: log-likelihood convergence tolerance.
            seed: for reproducible initialisation.
        """
        obs = np.asarray(observations, dtype=float)
        T = obs.shape[0]
        K = self.n_states

        # Initialise
        rng = np.random.default_rng(seed)
        A = np.full((K, K), 1.0 / K)
        np.fill_diagonal(A, 0.9)
        A /= A.sum(axis=1, keepdims=True)
        pi = np.ones(K) / K
        params = [self.emission.initial_params(obs, k, K) for k in range(K)]

        prev_ll = -np.inf
        converged = False

        for iteration in range(max_iter):
            # E-step: compute emission log-probs
            log_B = np.zeros((T, K))
            for k in range(K):
                log_B[:, k] = self.emission.log_prob(obs, params[k])

            # Forward-backward (scaled)
            alpha, scale = self._forward(log_B, A, pi)
            beta = self._backward(log_B, A, scale)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-300)

            # Log-likelihood
            ll = float(np.sum(np.log(np.maximum(scale, 1e-300))))

            if ll - prev_ll < tol and iteration > 0:
                converged = True
                break
            prev_ll = ll

            # M-step: update parameters
            # Transition matrix
            xi = np.zeros((K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[i, j] += (alpha[t, i] * A[i, j]
                                     * np.exp(log_B[t + 1, j] - log_B[t + 1, :].max())
                                     * beta[t + 1, j])
            xi_row_sums = xi.sum(axis=1, keepdims=True).clip(1e-300)
            A = xi / xi_row_sums

            # Emission parameters
            for k in range(K):
                params[k] = self.emission.fit_params(obs, gamma[:, k])

            # Initial distribution
            pi = gamma[0]

        # Store fitted parameters
        self._transition = A
        self._emission_params = params
        self._pi = pi

        # Viterbi decoding
        labels = self._viterbi(obs, log_B, A, pi)

        # Stationary distribution
        stat_dist = self._stationary(A)

        # Information criteria
        n_params = K * K + K * 2  # approximate
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(T)

        return HMMFitResult(
            n_states=K,
            transition_matrix=A,
            emission_params=params,
            stationary_distribution=stat_dist,
            log_likelihood=ll,
            aic=float(aic),
            bic=float(bic),
            n_iterations=iteration + 1,
            labels=labels,
            filtered_probs=gamma,
            converged=converged,
        )

    def filter(self, observations: np.ndarray) -> np.ndarray:
        """Filter new observations using fitted parameters.

        Returns (T, K) posterior state probabilities.
        """
        if self._transition is None:
            raise ValueError("HMM not fitted. Call fit() first.")

        obs = np.asarray(observations, dtype=float)
        T = obs.shape[0]
        K = self.n_states

        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = self.emission.log_prob(obs, self._emission_params[k])

        alpha, scale = self._forward(log_B, self._transition, self._pi)
        beta = self._backward(log_B, self._transition, scale)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-300)
        return gamma

    def predict_state(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi decoding of most likely state sequence."""
        if self._transition is None:
            raise ValueError("HMM not fitted. Call fit() first.")

        obs = np.asarray(observations, dtype=float)
        K = self.n_states
        log_B = np.zeros((obs.shape[0], K))
        for k in range(K):
            log_B[:, k] = self.emission.log_prob(obs, self._emission_params[k])
        return self._viterbi(obs, log_B, self._transition, self._pi)

    # ── Internal algorithms ──

    @staticmethod
    def _forward(log_B, A, pi):
        T, K = log_B.shape
        B = np.exp(log_B - log_B.max(axis=1, keepdims=True))
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        alpha[0] = pi * B[0]
        scale[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            scale[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= scale[t]
        return alpha, scale

    @staticmethod
    def _backward(log_B, A, scale):
        T, K = log_B.shape
        B = np.exp(log_B - log_B.max(axis=1, keepdims=True))
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = A @ (B[t + 1] * beta[t + 1])
            beta[t] /= max(scale[t + 1], 1e-300)
        return beta

    @staticmethod
    def _viterbi(obs, log_B, A, pi):
        T, K = log_B.shape
        log_A = np.log(np.maximum(A, 1e-300))
        log_pi = np.log(np.maximum(pi, 1e-300))
        V = np.zeros((T, K))
        ptr = np.zeros((T, K), dtype=int)
        V[0] = log_pi + log_B[0]
        for t in range(1, T):
            for j in range(K):
                scores = V[t - 1] + log_A[:, j]
                ptr[t, j] = int(np.argmax(scores))
                V[t, j] = scores[ptr[t, j]] + log_B[t, j]
        # Backtrace
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(V[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = ptr[t + 1, path[t + 1]]
        return path

    @staticmethod
    def _stationary(A):
        K = A.shape[0]
        evals, evecs = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(evals - 1.0))
        pi = np.real(evecs[:, idx])
        pi = np.maximum(pi, 0)
        total = pi.sum()
        if total > 0:
            pi /= total
        else:
            pi = np.ones(K) / K
        return pi
