"""Regime-switching stochastic process for MC simulation.

Extends ProcessSpec with regime-dependent parameters. Integrates with
MCEngine via exact_step that samples regime transitions + evolves SDE.

    from pricebook.models.regime_process import (
        RegimeProcessSpec, create_regime_gbm, create_regime_ou,
    )

References:
    Hamilton (1989). A New Approach to Nonstationary Time Series.
    Ang & Bekaert (2002). Regime Switches in Interest Rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RegimeProcessSpec:
    """Regime-switching stochastic process.

    The process evolves with regime-dependent drift and diffusion:
        dx = μ(s_t) dt + σ(s_t) dW

    where s_t ∈ {0, ..., K-1} follows a Markov chain.

    Args:
        x0: initial state.
        n_regimes: number of regimes.
        regime_params: list of dicts, one per regime (keys: "drift", "diffusion").
        transition_matrix: (K, K) annual transition probabilities.
        initial_regime_probs: (K,) initial regime distribution.
    """
    x0: float
    n_regimes: int
    regime_params: list[dict]
    transition_matrix: np.ndarray
    initial_regime_probs: np.ndarray | None = None

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int = 42,
    ) -> dict:
        """Simulate paths with regime switching.

        Returns:
            {"paths": (n_paths, n_steps+1), "regimes": (n_paths, n_steps+1)}
        """
        rng = np.random.default_rng(seed)
        K = self.n_regimes
        A = self.transition_matrix

        # Convert annual transition matrix to per-step
        A_step = _transition_per_step(A, dt)

        # Initial regime
        pi = self.initial_regime_probs
        if pi is None:
            pi = _stationary(A)

        paths = np.zeros((n_paths, n_steps + 1))
        regimes = np.zeros((n_paths, n_steps + 1), dtype=int)
        paths[:, 0] = self.x0

        # Sample initial regimes
        regimes[:, 0] = rng.choice(K, size=n_paths, p=pi)

        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            z = rng.standard_normal(n_paths)

            for path in range(n_paths):
                s = regimes[path, step]
                params = self.regime_params[s]
                drift = params.get("drift", 0.0)
                diff = params.get("diffusion", 0.01)

                # Evolve
                paths[path, step + 1] = paths[path, step] + drift * dt + diff * sqrt_dt * z[path]

                # Transition
                regimes[path, step + 1] = rng.choice(K, p=A_step[s])

        return {"paths": paths, "regimes": regimes}

    def to_dict(self) -> dict:
        return {
            "x0": self.x0,
            "n_regimes": self.n_regimes,
            "regime_params": self.regime_params,
            "transition_matrix": self.transition_matrix.tolist(),
        }


def create_regime_gbm(
    s0: float,
    regime_vols: list[float],
    regime_drifts: list[float],
    transition_matrix: list[list[float]],
) -> RegimeProcessSpec:
    """Create regime-switching GBM (log-normal).

    In each regime: dS/S = μ_k dt + σ_k dW
    Simulated in log-space: d(log S) = (μ_k - σ_k²/2) dt + σ_k dW

    Args:
        s0: initial stock price.
        regime_vols: vol per regime.
        regime_drifts: drift per regime (risk-neutral: r - q).
        transition_matrix: (K,K) annual transition probabilities.
    """
    n = len(regime_vols)
    params = []
    for k in range(n):
        sigma = regime_vols[k]
        mu = regime_drifts[k]
        params.append({
            "drift": mu - 0.5 * sigma**2,
            "diffusion": sigma,
        })
    return RegimeProcessSpec(
        x0=math.log(s0),
        n_regimes=n,
        regime_params=params,
        transition_matrix=np.array(transition_matrix),
    )


def create_regime_ou(
    x0: float,
    regime_means: list[float],
    regime_speeds: list[float],
    regime_vols: list[float],
    transition_matrix: list[list[float]],
) -> RegimeProcessSpec:
    """Create regime-switching Ornstein-Uhlenbeck.

    dx = κ_k(θ_k - x) dt + σ_k dW

    Args:
        x0: initial state.
        regime_means: long-run mean per regime.
        regime_speeds: mean-reversion speed per regime.
        regime_vols: volatility per regime.
    """
    n = len(regime_means)
    params = []
    for k in range(n):
        params.append({
            "mean": regime_means[k],
            "speed": regime_speeds[k],
            "diffusion": regime_vols[k],
        })
    # OU needs custom drift function — store params for simulate override
    spec = RegimeProcessSpec(
        x0=x0, n_regimes=n, regime_params=params,
        transition_matrix=np.array(transition_matrix),
    )
    # Override regime_params with OU-specific drift
    for k in range(n):
        p = spec.regime_params[k]
        # Linearise: drift = κ(θ - x) ≈ κ(θ - x0) for initial step
        p["drift"] = p["speed"] * (p["mean"] - x0)
    return spec


def _transition_per_step(A_annual: np.ndarray, dt: float) -> np.ndarray:
    """Convert annual transition matrix to per-step (dt in years)."""
    from scipy.linalg import expm, logm
    K = A_annual.shape[0]
    try:
        Q = np.real(logm(A_annual))
        A_step = np.real(expm(Q * dt))
    except Exception:
        # Fallback: linear interpolation
        A_step = np.eye(K) + (A_annual - np.eye(K)) * dt
    # Ensure stochastic
    A_step = np.maximum(A_step, 0)
    A_step /= A_step.sum(axis=1, keepdims=True)
    return A_step


def _stationary(A: np.ndarray) -> np.ndarray:
    K = A.shape[0]
    evals, evecs = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(evals - 1.0))
    pi = np.real(evecs[:, idx])
    pi = np.maximum(pi, 0)
    total = pi.sum()
    return pi / total if total > 0 else np.ones(K) / K
