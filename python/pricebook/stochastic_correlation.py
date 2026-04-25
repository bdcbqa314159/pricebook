"""Stochastic correlation: CIR-driven ρ, Wishart covariance, calibration.

* :class:`CIRCorrelation` — mean-reverting correlation via CIR dynamics.
* :func:`simulate_two_asset_stoch_corr` — MC with stochastic ρ.
* :class:`WishartCovariance` — Wishart matrix-valued process.
* :func:`calibrate_stoch_corr_to_dispersion` — fit ρ dynamics to dispersion.

References:
    Teng, Ehrhardt & Günther, *The Modelling of Stochastic Correlation*,
    J. Math. in Industry, 2016.
    Da Fonseca, Grasselli & Tebaldi, *Option Pricing When Correlations Are
    Stochastic: An Analytical Framework*, RFS, 2007.
    Bru, *Wishart Processes*, J. Theoretical Probability, 1991.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ---- CIR-driven correlation ----

@dataclass
class CIRCorrelationResult:
    """CIR correlation simulation result."""
    rho_paths: np.ndarray          # (n_paths, n_steps+1)
    mean_terminal_rho: float
    std_terminal_rho: float
    min_rho: float
    max_rho: float


class CIRCorrelation:
    """Mean-reverting correlation via transformed CIR process.

    Map ρ ∈ (−1, 1) to X ∈ (0, ∞) via X = (1 + ρ) / (1 − ρ).
    X follows CIR: dX = κ(θ_X − X) dt + σ_X √X dW.
    Ensures ρ stays in (−1, 1).

    Args:
        rho0: initial correlation.
        kappa: mean reversion speed.
        theta: long-run correlation.
        sigma: vol of correlation (in ρ-space, mapped internally).
    """

    def __init__(self, rho0: float, kappa: float, theta: float, sigma: float):
        self.rho0 = max(-0.999, min(0.999, rho0))
        self.kappa = kappa
        self.theta = max(-0.999, min(0.999, theta))
        self.sigma = sigma

    def _rho_to_x(self, rho: float) -> float:
        rho = max(-0.999, min(0.999, rho))
        return (1 + rho) / (1 - rho)

    def _x_to_rho(self, x: float) -> float:
        x = max(1e-10, x)
        return (x - 1) / (x + 1)

    def simulate(
        self,
        T: float,
        n_paths: int = 5_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> CIRCorrelationResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        x0 = self._rho_to_x(self.rho0)
        theta_x = self._rho_to_x(self.theta)
        # Map σ_ρ to σ_X via chain rule: σ_X ≈ σ_ρ × 2 / (1 − θ)²
        sigma_x = self.sigma * 2 / max((1 - self.theta)**2, 0.01)
        # Enforce Feller condition: 2κθ_X ≥ σ_X² to prevent divergence
        feller_max = math.sqrt(2 * self.kappa * theta_x) if theta_x > 0 else 0.0
        if feller_max > 0:
            sigma_x = min(sigma_x, 0.999 * feller_max)

        X = np.full((n_paths, n_steps + 1), x0)

        for step in range(n_steps):
            dW = rng.standard_normal(n_paths) * sqrt_dt
            x_pos = np.maximum(X[:, step], 1e-10)
            X[:, step + 1] = np.maximum(
                x_pos + self.kappa * (theta_x - x_pos) * dt
                + sigma_x * np.sqrt(x_pos) * dW,
                1e-10,
            )

        rho_paths = (X - 1) / (X + 1)

        return CIRCorrelationResult(
            rho_paths=rho_paths,
            mean_terminal_rho=float(rho_paths[:, -1].mean()),
            std_terminal_rho=float(rho_paths[:, -1].std()),
            min_rho=float(rho_paths.min()),
            max_rho=float(rho_paths.max()),
        )


# ---- Two-asset MC with stochastic ρ ----

@dataclass
class StochCorrPricingResult:
    """Multi-asset pricing with stochastic correlation."""
    price: float
    mean_correlation: float
    spot1_paths: np.ndarray
    spot2_paths: np.ndarray
    rho_paths: np.ndarray


def simulate_two_asset_stoch_corr(
    spot1: float,
    spot2: float,
    rate: float,
    div1: float,
    div2: float,
    vol1: float,
    vol2: float,
    corr_model: CIRCorrelation,
    T: float,
    n_paths: int = 5_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> StochCorrPricingResult:
    """Simulate two assets with CIR stochastic correlation.

    At each step, the instantaneous correlation is drawn from the CIR process.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    # First simulate ρ paths
    rho_result = corr_model.simulate(T, n_paths, n_steps, seed)
    rho = rho_result.rho_paths

    S1 = np.full((n_paths, n_steps + 1), float(spot1))
    S2 = np.full((n_paths, n_steps + 1), float(spot2))

    # Use a different seed for asset Brownians
    rng2 = np.random.default_rng(seed + 1 if seed else 1)

    for step in range(n_steps):
        z1 = rng2.standard_normal(n_paths)
        z_indep = rng2.standard_normal(n_paths)
        rho_t = rho[:, step]
        z2 = rho_t * z1 + np.sqrt(np.maximum(1 - rho_t**2, 0)) * z_indep

        drift1 = (rate - div1 - 0.5 * vol1**2) * dt
        drift2 = (rate - div2 - 0.5 * vol2**2) * dt

        S1[:, step + 1] = S1[:, step] * np.exp(drift1 + vol1 * sqrt_dt * z1)
        S2[:, step + 1] = S2[:, step] * np.exp(drift2 + vol2 * sqrt_dt * z2)

    mean_rho = float(rho.mean())

    return StochCorrPricingResult(
        price=0.0,  # caller computes payoff
        mean_correlation=mean_rho,
        spot1_paths=S1,
        spot2_paths=S2,
        rho_paths=rho,
    )


# ---- Wishart covariance ----

@dataclass
class WishartResult:
    """Wishart covariance simulation result."""
    covariance_paths: np.ndarray    # (n_paths, n_steps+1, n, n)
    correlation_paths: np.ndarray   # (n_paths, n_steps+1)  — only (0,1) element
    mean_terminal_corr: float
    is_pd: bool                     # all paths remain positive-definite


class WishartCovariance:
    """Wishart covariance matrix process.

    dΣ = (M Σ + Σ M' + Q'Q) dt + √Σ dW Q + Q' dW' √Σ

    Simplified 2×2 version:
    - Σ = [[v1, c], [c, v2]] with c = ρ √(v1 v2).
    - Simulate v1, v2, ρ jointly via CIR + correlation dynamics.

    Args:
        Sigma0: initial 2×2 covariance matrix.
        kappa: mean reversion speed (scalar, applied to all elements).
        theta: long-run covariance matrix.
        sigma: vol of vol (scalar).
    """

    def __init__(self, Sigma0: np.ndarray, kappa: float,
                 theta: np.ndarray, sigma: float):
        self.Sigma0 = np.array(Sigma0, dtype=float)
        self.kappa = kappa
        self.theta = np.array(theta, dtype=float)
        self.sigma = sigma
        self.n = self.Sigma0.shape[0]

    def simulate(
        self,
        T: float,
        n_paths: int = 1_000,
        n_steps: int = 50,
        seed: int | None = 42,
    ) -> WishartResult:
        """Simplified Wishart: simulate diagonal (variances) via CIR,
        off-diagonal (covariance) via OU."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        n = self.n

        # Store only the (0,1) correlation for output
        v1 = np.full((n_paths, n_steps + 1), self.Sigma0[0, 0])
        v2 = np.full((n_paths, n_steps + 1), self.Sigma0[1, 1])
        cov = np.full((n_paths, n_steps + 1), self.Sigma0[0, 1])

        theta_v1 = self.theta[0, 0]
        theta_v2 = self.theta[1, 1]
        theta_c = self.theta[0, 1]

        for step in range(n_steps):
            z1 = rng.standard_normal(n_paths) * sqrt_dt
            z2 = rng.standard_normal(n_paths) * sqrt_dt
            z3 = rng.standard_normal(n_paths) * sqrt_dt

            v1_pos = np.maximum(v1[:, step], 1e-10)
            v2_pos = np.maximum(v2[:, step], 1e-10)

            v1[:, step + 1] = np.maximum(
                v1_pos + self.kappa * (theta_v1 - v1_pos) * dt
                + self.sigma * np.sqrt(v1_pos) * z1, 1e-10)
            v2[:, step + 1] = np.maximum(
                v2_pos + self.kappa * (theta_v2 - v2_pos) * dt
                + self.sigma * np.sqrt(v2_pos) * z2, 1e-10)
            cov[:, step + 1] = (
                cov[:, step] + self.kappa * (theta_c - cov[:, step]) * dt
                + self.sigma * 0.5 * z3)  # z3 already contains √dt

            # Clamp covariance to valid range
            max_cov = np.sqrt(v1[:, step + 1] * v2[:, step + 1])
            cov[:, step + 1] = np.clip(cov[:, step + 1], -max_cov * 0.999, max_cov * 0.999)

        # Correlation paths
        corr = cov / np.sqrt(np.maximum(v1 * v2, 1e-20))

        # Check PD (all correlations in (-1, 1))
        is_pd = bool(np.all(np.abs(corr) < 1))

        # Build full covariance tensor (light: only store correlation)
        cov_paths = np.zeros((n_paths, n_steps + 1, 2, 2))
        cov_paths[:, :, 0, 0] = v1
        cov_paths[:, :, 1, 1] = v2
        cov_paths[:, :, 0, 1] = cov
        cov_paths[:, :, 1, 0] = cov

        return WishartResult(
            covariance_paths=cov_paths,
            correlation_paths=corr,
            mean_terminal_corr=float(corr[:, -1].mean()),
            is_pd=is_pd,
        )


# ---- Calibration to dispersion ----

@dataclass
class DispersionCalibrationResult:
    """Stochastic correlation calibration to dispersion result."""
    kappa: float
    theta: float
    sigma: float
    residual: float
    index_variance_model: float
    index_variance_target: float


def calibrate_stoch_corr_to_dispersion(
    component_vols: list[float],
    weights: list[float],
    index_variance_target: float,
    rho0: float = 0.5,
    T: float = 1.0,
) -> DispersionCalibrationResult:
    """Calibrate CIR correlation to match index variance.

    Index variance under stochastic ρ:
        σ²_index ≈ Σ wᵢ² σᵢ² + E[ρ] × Σ_{i≠j} wᵢ wⱼ σᵢ σⱼ

    Fit E[ρ] (long-run θ) to match target index variance, then estimate
    κ and σ from dispersion dynamics.

    Simplified: only calibrate θ to match ATM level.
    """
    w = np.array(weights)
    sig = np.array(component_vols)

    diag = float(np.sum(w**2 * sig**2))
    cross = float((np.sum(w * sig))**2 - np.sum(w**2 * sig**2))

    def model_index_var(theta_rho):
        return diag + theta_rho * cross

    # Solve: diag + θ × cross = target
    if abs(cross) > 1e-10:
        theta_opt = (index_variance_target - diag) / cross
        theta_opt = max(-0.999, min(0.999, theta_opt))
    else:
        theta_opt = rho0

    model_var = model_index_var(theta_opt)
    residual = abs(model_var - index_variance_target)

    # Heuristic for κ and σ: not identifiable from ATM alone
    kappa_est = 2.0
    sigma_est = 0.3

    return DispersionCalibrationResult(
        kappa=kappa_est,
        theta=float(theta_opt),
        sigma=sigma_est,
        residual=float(residual),
        index_variance_model=float(model_var),
        index_variance_target=float(index_variance_target),
    )
