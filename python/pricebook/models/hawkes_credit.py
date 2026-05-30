"""Fractional Hawkes processes for credit derivatives.

Self-exciting point processes with configurable memory kernels:
- Exponential: φ(t) = α exp(-βt)               [Markov, short memory]
- Power-law: φ(t) = α (t+ε)^(H-3/2)            [fractional, long memory]
- Mittag-Leffler: φ(t) = α E_γ(-βt^γ)          [interpolates exp ↔ power-law]
- Sum-of-exponentials: Σ wₖ exp(-βₖt)           [Markov approximation of power-law]

Supports univariate (single name) and multivariate (N names with
cross-excitation matrix) for credit contagion modelling.

    from pricebook.models.hawkes_credit import (
        FractionalHawkesProcess, MultivariateHawkesProcess,
        HawkesKernel, HawkesCreditResult,
    )

References:
    Hawkes (1971). Spectra of Some Self-Exciting Point Processes. Biometrika.
    Bacry, Mastromatteo & Muzy (2015). Hawkes Processes in Finance. MPC.
    Ait-Sahalia, Cacho-Diaz & Laeven (2015). Modeling Financial Contagion.
    Jaisson & Rosenbaum (2015). Limit Theorems for Nearly Unstable Hawkes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class HawkesKernel(Enum):
    """Memory kernel type for the Hawkes process."""
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    MITTAG_LEFFLER = "mittag_leffler"
    SUM_EXPONENTIAL = "sum_exponential"


@dataclass
class HawkesCreditResult:
    """Result of Hawkes process simulation."""
    event_times: list[list[float]]
    intensities: np.ndarray       # (n_paths, n_grid)
    times: np.ndarray
    n_events_mean: float
    n_events_std: float
    branching_ratio: float
    kernel: str

    def to_dict(self) -> dict:
        return {
            "n_events_mean": self.n_events_mean,
            "n_events_std": self.n_events_std,
            "branching_ratio": self.branching_ratio,
            "kernel": self.kernel,
            "n_paths": len(self.event_times),
        }


@dataclass
class MultivariateHawkesResult:
    """Result of multivariate Hawkes simulation."""
    event_times: list[list[list[float]]]   # [path][name][events]
    intensities: np.ndarray                 # (n_paths, n_names, n_grid)
    times: np.ndarray
    default_times: np.ndarray               # (n_paths, n_names) — first event per name
    n_names: int
    kernel: str

    def to_dict(self) -> dict:
        return {
            "n_names": self.n_names,
            "kernel": self.kernel,
            "n_paths": len(self.event_times),
        }


# ═══════════════════════════════════════════════════════════════
# Kernel functions
# ═══════════════════════════════════════════════════════════════

def _kernel_exponential(t: float, alpha: float, beta: float) -> float:
    """φ(t) = α exp(-βt)."""
    return alpha * math.exp(-beta * t)


def _kernel_power_law(t: float, alpha: float, H: float, eps: float = 1e-6) -> float:
    """φ(t) = α (t+ε)^(H-3/2).

    H ∈ (0, 0.5): rough, long memory. H = 0.5 recovers Brownian scaling.
    ε prevents singularity at t=0.
    """
    return alpha * (t + eps) ** (H - 1.5)


def _kernel_mittag_leffler(t: float, alpha: float, beta: float, gamma: float) -> float:
    """φ(t) = α E_γ(-βt^γ) where E_γ is the Mittag-Leffler function.

    γ = 1: reduces to exponential.
    γ ∈ (0, 1): power-law tail, interpolates exp ↔ pure power-law.

    Uses series expansion: E_γ(z) = Σ z^k / Γ(γk + 1).
    """
    z = -beta * t ** gamma
    result = 0.0
    z_power = 1.0
    for k in range(50):
        result += z_power / math.gamma(gamma * k + 1)
        z_power *= z
        if abs(z_power) < 1e-15:
            break
    return alpha * result


def _kernel_sum_exponential(t: float, weights: list[float], betas: list[float]) -> float:
    """φ(t) = Σ wₖ exp(-βₖ t).

    Approximates power-law kernel via Bochner representation.
    """
    return sum(w * math.exp(-b * t) for w, b in zip(weights, betas))


def evaluate_kernel(
    t: float,
    kernel: HawkesKernel,
    params: dict,
) -> float:
    """Evaluate kernel function at time t."""
    if kernel == HawkesKernel.EXPONENTIAL:
        return _kernel_exponential(t, params["alpha"], params["beta"])
    elif kernel == HawkesKernel.POWER_LAW:
        return _kernel_power_law(t, params["alpha"], params["H"],
                                  params.get("eps", 1e-6))
    elif kernel == HawkesKernel.MITTAG_LEFFLER:
        return _kernel_mittag_leffler(t, params["alpha"], params["beta"],
                                      params["gamma"])
    elif kernel == HawkesKernel.SUM_EXPONENTIAL:
        return _kernel_sum_exponential(t, params["weights"], params["betas"])
    raise ValueError(f"Unknown kernel: {kernel}")


def branching_ratio(kernel: HawkesKernel, params: dict, T_max: float = 100.0,
                    n_quad: int = 1000) -> float:
    """Compute branching ratio n* = ∫₀^∞ φ(t) dt.

    Must be < 1 for stationarity.
    """
    if kernel == HawkesKernel.EXPONENTIAL:
        return params["alpha"] / params["beta"]

    # Numerical integration for non-exponential kernels
    dt = T_max / n_quad
    total = sum(evaluate_kernel(i * dt, kernel, params) * dt for i in range(1, n_quad + 1))
    return total


# ═══════════════════════════════════════════════════════════════
# Power-law kernel: sum-of-exponentials approximation
# ═══════════════════════════════════════════════════════════════

def approximate_power_law(H: float, alpha: float, K: int = 10,
                          beta_min: float = 0.01, beta_max: float = 100.0) -> dict:
    """Approximate power-law kernel t^(H-3/2) by K exponentials.

    Uses log-spaced grid of decay rates. Returns params for SUM_EXPONENTIAL kernel.

    Args:
        H: Hurst parameter ∈ (0, 0.5).
        alpha: overall scaling.
        K: number of exponential terms.
        beta_min: slowest decay rate (longest memory).
        beta_max: fastest decay rate (shortest memory).
    """
    betas = np.logspace(np.log10(beta_min), np.log10(beta_max), K).tolist()
    # Weights from matching the power-law at log-spaced points
    # w_k ∝ β_k^(H-1/2) × Δ(log β)
    log_spacing = (np.log(beta_max) - np.log(beta_min)) / K
    weights = [alpha * b ** (H - 0.5) * log_spacing / math.gamma(1.5 - H)
               for b in betas]
    return {"weights": weights, "betas": betas}


# ═══════════════════════════════════════════════════════════════
# Univariate fractional Hawkes
# ═══════════════════════════════════════════════════════════════

class FractionalHawkesProcess:
    """Self-exciting point process with configurable memory kernel.

    λ(t) = μ + ∫₀ᵗ φ(t-s) dN(s) = μ + Σ_{tᵢ < t} φ(t - tᵢ)

    Args:
        mu: baseline intensity (events per unit time).
        kernel: HawkesKernel enum.
        kernel_params: dict of kernel parameters.
    """

    def __init__(
        self,
        mu: float,
        kernel: HawkesKernel = HawkesKernel.EXPONENTIAL,
        kernel_params: dict | None = None,
    ):
        if mu < 0:
            raise ValueError(f"mu must be non-negative, got {mu}")
        self.mu = mu
        self.kernel = kernel
        self.kernel_params = kernel_params or {"alpha": 0.5, "beta": 1.0}

        self._branching_ratio = branching_ratio(kernel, self.kernel_params)
        if self._branching_ratio >= 1.0:
            import warnings
            warnings.warn(
                f"Branching ratio {self._branching_ratio:.3f} >= 1: process is non-stationary",
                RuntimeWarning, stacklevel=2,
            )

    @property
    def branching_ratio_value(self) -> float:
        return self._branching_ratio

    def intensity(self, t: float, event_times: list[float]) -> float:
        """Compute intensity λ(t) given past events."""
        lam = self.mu
        for ti in event_times:
            if ti < t:
                lam += evaluate_kernel(t - ti, self.kernel, self.kernel_params)
        return max(lam, 0.0)

    def simulate(
        self,
        T: float,
        n_paths: int = 1000,
        n_grid: int = 200,
        seed: int | None = 42,
    ) -> HawkesCreditResult:
        """Simulate via Ogata thinning, adapted for non-exponential kernels.

        For power-law/Mittag-Leffler kernels, the intensity upper bound is
        tracked dynamically since the kernel doesn't have a simple closed form.
        """
        rng = np.random.default_rng(seed)
        grid = np.linspace(0, T, n_grid)

        all_events: list[list[float]] = []
        all_intensity = np.zeros((n_paths, n_grid))

        for p in range(n_paths):
            events: list[float] = []
            t = 0.0

            while t < T:
                # Compute current intensity for upper bound
                lam_t = self.intensity(t, events)
                lam_star = lam_t * 1.5 + self.mu + 0.01  # safety margin

                # Next candidate via exponential with rate lam_star
                u = rng.random()
                if u < 1e-15:
                    u = 1e-15
                dt_cand = -math.log(u) / lam_star
                t += dt_cand

                if t >= T:
                    break

                # Accept/reject
                lam_t = self.intensity(t, events)
                if rng.random() * lam_star <= lam_t:
                    events.append(t)

            all_events.append(events)

            # Sample intensity on grid
            for i, tg in enumerate(grid):
                all_intensity[p, i] = self.intensity(tg, events)

        n_events = [len(e) for e in all_events]

        return HawkesCreditResult(
            event_times=all_events,
            intensities=all_intensity,
            times=grid,
            n_events_mean=float(np.mean(n_events)),
            n_events_std=float(np.std(n_events)),
            branching_ratio=self._branching_ratio,
            kernel=self.kernel.value,
        )


# ═══════════════════════════════════════════════════════════════
# Multivariate Hawkes (N names with cross-excitation)
# ═══════════════════════════════════════════════════════════════

class MultivariateHawkesProcess:
    """N-dimensional Hawkes with cross-excitation matrix.

    λᵢ(t) = μᵢ + Σⱼ Σ_{tⱼₖ < t} φᵢⱼ(t - tⱼₖ)

    The cross-excitation matrix Φ = [φᵢⱼ] controls contagion:
    - φᵢᵢ: self-excitation (own default raises own intensity)
    - φᵢⱼ: cross-excitation (j's default raises i's intensity)

    Args:
        mu: (N,) baseline intensities.
        alpha_matrix: (N, N) excitation magnitudes.
        kernel: shared kernel type.
        kernel_params: shared kernel parameters (except alpha replaced per pair).
    """

    def __init__(
        self,
        mu: np.ndarray,
        alpha_matrix: np.ndarray,
        kernel: HawkesKernel = HawkesKernel.EXPONENTIAL,
        kernel_params: dict | None = None,
    ):
        self.mu = np.asarray(mu, dtype=float)
        self.alpha_matrix = np.asarray(alpha_matrix, dtype=float)
        self.n_names = len(self.mu)
        self.kernel = kernel
        self.kernel_params = kernel_params or {"beta": 1.0}

        if self.alpha_matrix.shape != (self.n_names, self.n_names):
            raise ValueError(f"alpha_matrix shape {self.alpha_matrix.shape} != ({self.n_names}, {self.n_names})")

    def intensity(self, t: float, name: int,
                  event_times: list[list[float]]) -> float:
        """Compute λᵢ(t) given all names' past events."""
        lam = self.mu[name]
        for j in range(self.n_names):
            alpha_ij = self.alpha_matrix[name, j]
            if alpha_ij == 0:
                continue
            params = {**self.kernel_params, "alpha": alpha_ij}
            for tk in event_times[j]:
                if tk < t:
                    lam += evaluate_kernel(t - tk, self.kernel, params)
        return max(lam, 0.0)

    def simulate(
        self,
        T: float,
        n_paths: int = 1000,
        n_grid: int = 200,
        seed: int | None = 42,
    ) -> MultivariateHawkesResult:
        """Simulate multivariate Hawkes via sequential thinning."""
        rng = np.random.default_rng(seed)
        grid = np.linspace(0, T, n_grid)
        N = self.n_names

        all_events: list[list[list[float]]] = []
        all_intensity = np.zeros((n_paths, N, n_grid))
        all_default_times = np.full((n_paths, N), np.inf)

        for p in range(n_paths):
            events: list[list[float]] = [[] for _ in range(N)]
            t = 0.0

            while t < T:
                # Total intensity upper bound
                total_lam = sum(self.intensity(t, i, events) for i in range(N))
                lam_star = total_lam * 1.5 + sum(self.mu) + 0.01

                u = rng.random()
                if u < 1e-15:
                    u = 1e-15
                dt_cand = -math.log(u) / lam_star
                t += dt_cand

                if t >= T:
                    break

                # Compute component intensities
                lam_components = [self.intensity(t, i, events) for i in range(N)]
                total_lam = sum(lam_components)

                if rng.random() * lam_star <= total_lam:
                    # Accept — assign to a name proportional to intensity
                    cumsum = np.cumsum(lam_components)
                    u2 = rng.random() * total_lam
                    name_idx = int(np.searchsorted(cumsum, u2))
                    name_idx = min(name_idx, N - 1)
                    events[name_idx].append(t)

                    # Record first event as default time
                    if all_default_times[p, name_idx] == np.inf:
                        all_default_times[p, name_idx] = t

            all_events.append(events)

            # Sample intensity on grid
            for i in range(N):
                for k, tg in enumerate(grid):
                    all_intensity[p, i, k] = self.intensity(tg, i, events)

        return MultivariateHawkesResult(
            event_times=all_events,
            intensities=all_intensity,
            times=grid,
            default_times=all_default_times,
            n_names=N,
            kernel=self.kernel.value,
        )


# ═══════════════════════════════════════════════════════════════
# MLE calibration (exponential kernel)
# ═══════════════════════════════════════════════════════════════

def hawkes_mle_exponential(
    event_times: list[float],
    T: float,
) -> dict:
    """Maximum likelihood estimation for exponential Hawkes.

    log L = Σ log λ(tᵢ) - ∫₀ᵀ λ(t) dt

    Uses scipy.optimize for (μ, α, β).
    """
    from scipy.optimize import minimize as _minimize

    events = sorted(event_times)
    n = len(events)
    if n < 3:
        return {"mu": n / T, "alpha": 0.0, "beta": 1.0, "log_likelihood": float('-inf')}

    def neg_log_likelihood(params):
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
            return 1e10

        # Recursive computation of A(i) = Σ_{j<i} exp(-β(tᵢ - tⱼ))
        A = 0.0
        ll = 0.0
        for i, ti in enumerate(events):
            lam_i = mu + alpha * A
            if lam_i <= 0:
                return 1e10
            ll += math.log(lam_i)
            if i + 1 < n:
                A = (A + 1) * math.exp(-beta * (events[i + 1] - ti))

        # Compensator: ∫₀ᵀ λ(t) dt
        compensator = mu * T
        for ti in events:
            compensator += (alpha / beta) * (1 - math.exp(-beta * (T - ti)))

        return -(ll - compensator)

    # Initial guess
    mu0 = n / T * 0.5
    result = _minimize(neg_log_likelihood, [mu0, 0.3, 1.0],
                       method="Nelder-Mead", options={"maxiter": 1000})

    mu, alpha, beta = result.x
    return {
        "mu": max(mu, 1e-10),
        "alpha": max(alpha, 0.0),
        "beta": max(beta, 1e-10),
        "branching_ratio": alpha / max(beta, 1e-10),
        "log_likelihood": -result.fun,
        "converged": result.success,
    }
