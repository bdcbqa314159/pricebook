"""LMM deepening: swaption calibration, SABR-LMM, predictor-corrector, Greeks.

Extends :mod:`pricebook.lmm` with production-grade features.

* :func:`lmm_cascade_calibration` — column-by-column calibration to swaption matrix.
* :func:`lmm_global_calibration` — global fit to full swaption matrix.
* :class:`SABRLMM` — stochastic vol on each forward rate.
* :func:`lmm_predictor_corrector` — improved drift approximation.
* :func:`lmm_pathwise_greeks` — sensitivities to forward rates.

References:
    Rebonato, *Modern Pricing of Interest-Rate Derivatives*, Princeton, 2002.
    Rebonato, McKay & White, *The SABR/LIBOR Market Model*, Wiley, 2009.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 6-7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pricebook.black76 import black76_price, OptionType


# ---- LMM swaption calibration ----

@dataclass
class LMMCalibrationResult:
    """Result of LMM calibration to swaption matrix."""
    vols: np.ndarray          # calibrated instantaneous vols per forward
    residual: float
    n_swaptions: int
    method: str


def _rebonato_swaption_vol(
    inst_vols: np.ndarray,
    forward_rates: np.ndarray,
    expiry_idx: int,
    tenor_idx: int,
    dt: float,
) -> float:
    """Rebonato (2002) approximation for swaption vol from LMM vols.

    σ_swap² × T ≈ Σ_{i,j} w_i w_j σ_i σ_j ρ_{ij} T

    Uses annuity weights: w_i = τ × F_i × P(0, T_{i+1}) / (S × A)
    where P(0, T_k) = Π 1/(1+τF_j), A = annuity, S = par swap rate.
    Assumes unit correlation between forwards.
    """
    n = len(inst_vols)
    start = expiry_idx
    end = min(start + tenor_idx, n)
    if end <= start:
        return 0.0

    n_fwd = end - start

    # Compute discount factors from forwards: P(0, T_k) = Π 1/(1+τF_j)
    # Relative to P(0, T_start) — only ratios matter for weights
    disc = np.ones(n_fwd + 1)
    for k in range(n_fwd):
        idx = start + k
        if idx < n:
            disc[k + 1] = disc[k] / (1 + dt * forward_rates[idx])

    # Annuity: A = Σ τ × P(0, T_{start+k+1})
    annuity = sum(dt * disc[k + 1] for k in range(n_fwd))
    if annuity < 1e-15:
        return 0.0

    # Annuity weights: w_i = τ × F_i × P(0, T_{i+1}) / (S × A)
    # where S = par swap rate. Since S × A = Σ τ × F_i × P(0, T_{i+1}) × (something)...
    # Actually the standard Rebonato weight is: w_i = τ × P(T_{i+1}) / A × F_i / S
    # But S = (1 - P(T_end)) / A, so w_i = τ × P(T_{i+1}) × F_i / (1 - P(T_end))
    denom = 1.0 - disc[n_fwd]
    if abs(denom) < 1e-15:
        return 0.0

    weights = np.zeros(n_fwd)
    for k in range(n_fwd):
        idx = start + k
        if idx < n:
            weights[k] = dt * disc[k + 1] * forward_rates[idx] / denom

    var = 0.0
    for i in range(n_fwd):
        for j in range(n_fwd):
            idx_i = start + i
            idx_j = start + j
            if idx_i < n and idx_j < n:
                var += weights[i] * weights[j] * inst_vols[idx_i] * inst_vols[idx_j]

    T = (expiry_idx + 1) * dt
    if T > 0 and var > 0:
        return math.sqrt(var / T)
    return 0.0


def lmm_cascade_calibration(
    market_vols: dict[tuple[int, int], float],
    forward_rates: np.ndarray | list[float],
    dt: float = 0.5,
) -> LMMCalibrationResult:
    """Cascade (column-by-column) calibration of LMM to swaption matrix.

    Calibrates one column at a time (fixed tenor, varying expiry),
    fitting each forward rate's instantaneous vol sequentially.

    Args:
        market_vols: {(expiry_idx, tenor_idx) → swaption_vol}.
        forward_rates: initial forward rate curve.
        dt: tenor spacing.

    Reference:
        Brigo & Mercurio, Ch. 6.8.
    """
    fwd = np.asarray(forward_rates, dtype=float)
    n = len(fwd)
    inst_vols = np.full(n, 0.20)  # initial guess

    # Sort by expiry
    sorted_keys = sorted(market_vols.keys())

    for exp_idx, ten_idx in sorted_keys:
        if exp_idx >= n:
            continue

        target = market_vols[(exp_idx, ten_idx)]

        def objective(v):
            inst_vols[exp_idx] = v[0]
            model_vol = _rebonato_swaption_vol(inst_vols, fwd, exp_idx, ten_idx, dt)
            return (model_vol - target) ** 2

        result = minimize(objective, [inst_vols[exp_idx]], method='Nelder-Mead')
        inst_vols[exp_idx] = result.x[0]

    # Compute residual
    total_err = 0.0
    for (e, t), target in market_vols.items():
        if e < n:
            model = _rebonato_swaption_vol(inst_vols, fwd, e, t, dt)
            total_err += (model - target) ** 2

    return LMMCalibrationResult(
        inst_vols, math.sqrt(total_err / max(len(market_vols), 1)),
        len(market_vols), "cascade",
    )


def lmm_global_calibration(
    market_vols: dict[tuple[int, int], float],
    forward_rates: np.ndarray | list[float],
    dt: float = 0.5,
) -> LMMCalibrationResult:
    """Global calibration: fit all instantaneous vols simultaneously.

    Minimises Σ (model_vol − market_vol)² over all swaptions.
    """
    fwd = np.asarray(forward_rates, dtype=float)
    n = len(fwd)

    def objective(vols):
        total = 0.0
        for (e, t), target in market_vols.items():
            if e < n:
                model = _rebonato_swaption_vol(vols, fwd, e, t, dt)
                total += (model - target) ** 2
        return total

    x0 = np.full(n, 0.20)
    result = minimize(objective, x0, method='L-BFGS-B',
                      bounds=[(0.01, 1.0)] * n)

    residual = math.sqrt(result.fun / max(len(market_vols), 1))

    return LMMCalibrationResult(result.x, residual, len(market_vols), "global")


# ---- SABR-LMM ----

@dataclass
class SABRLMMResult:
    """SABR-LMM simulation result."""
    forward_paths: np.ndarray   # (n_paths, n_steps+1, n_forwards)
    vol_paths: np.ndarray       # (n_paths, n_steps+1, n_forwards)
    swaption_price: float


class SABRLMM:
    """SABR-LMM: stochastic vol on each forward rate.

    dF_i / F_i^β = σ_i dW_i
    dσ_i = α_i σ_i dZ_i
    corr(dW_i, dZ_i) = ρ_i

    Each forward rate has its own SABR parameters.

    Args:
        forward_rates: initial forward rate curve.
        betas: SABR β per forward (default 1.0 = lognormal).
        alphas: SABR α (vol-of-vol) per forward.
        rhos: SABR ρ (correlation) per forward.
        init_vols: initial σ per forward.
        dt: tenor spacing.
    """

    def __init__(
        self,
        forward_rates: list[float],
        betas: list[float] | None = None,
        alphas: list[float] | None = None,
        rhos: list[float] | None = None,
        init_vols: list[float] | None = None,
        dt: float = 0.5,
    ):
        self.n = len(forward_rates)
        self.fwd = np.array(forward_rates)
        self.betas = np.array(betas or [1.0] * self.n)
        self.alphas = np.array(alphas or [0.3] * self.n)
        self.rhos = np.array(rhos or [-0.3] * self.n)
        self.init_vols = np.array(init_vols or [0.20] * self.n)
        self.dt = dt

    def simulate(
        self,
        T: float,
        n_steps: int = 100,
        n_paths: int = 10_000,
        seed: int | None = None,
    ) -> SABRLMMResult:
        """Simulate SABR-LMM forward rate + vol paths."""
        rng = np.random.default_rng(seed)
        dt_sim = T / n_steps
        sqrt_dt = math.sqrt(dt_sim)

        F = np.zeros((n_paths, n_steps + 1, self.n))
        S = np.zeros((n_paths, n_steps + 1, self.n))
        F[:, 0, :] = self.fwd
        S[:, 0, :] = self.init_vols

        for step in range(n_steps):
            for i in range(self.n):
                z1 = rng.standard_normal(n_paths)
                z2 = rng.standard_normal(n_paths)
                dW = z1 * sqrt_dt
                dZ = (self.rhos[i] * z1 + math.sqrt(1 - self.rhos[i]**2) * z2) * sqrt_dt

                f = np.maximum(F[:, step, i], 1e-10)
                s = np.maximum(S[:, step, i], 1e-10)

                F[:, step + 1, i] = f + s * np.power(f, self.betas[i]) * dW
                F[:, step + 1, i] = np.maximum(F[:, step + 1, i], 0.0)

                S[:, step + 1, i] = s + self.alphas[i] * s * dZ
                S[:, step + 1, i] = np.maximum(S[:, step + 1, i], 1e-10)

        # Price a simple caplet on the first forward
        df = math.exp(-self.fwd[0] * self.dt)
        payoff = np.maximum(F[:, -1, 0] - self.fwd[0], 0.0) * self.dt
        price = float(df * payoff.mean())

        return SABRLMMResult(F, S, price)


# ---- Predictor-corrector ----

@dataclass
class PredictorCorrectorResult:
    """LMM with predictor-corrector drift."""
    forward_paths: np.ndarray
    caplet_price: float


def lmm_predictor_corrector(
    forward_rates: list[float],
    inst_vols: list[float],
    T: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    dt_tenor: float = 0.5,
    seed: int | None = None,
) -> PredictorCorrectorResult:
    """LMM simulation with predictor-corrector drift.

    Standard Euler uses the drift at t_n. Predictor-corrector:
    1. Predict: F* = F_n + μ(F_n)dt + σ dW  (Euler step)
    2. Correct: F_{n+1} = F_n + 0.5(μ(F_n) + μ(F*))dt + σ dW

    This gives O(dt) weak convergence improvement.
    """
    rng = np.random.default_rng(seed)
    fwd = np.array(forward_rates)
    vols = np.array(inst_vols)
    n_fwd = len(fwd)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    F = np.zeros((n_paths, n_steps + 1, n_fwd))
    F[:, 0, :] = fwd

    def drift(L, j):
        """LMM drift for forward j under terminal measure."""
        mu = 0.0
        for k in range(j + 1, n_fwd):
            tau = dt_tenor
            mu -= vols[k] * vols[j] * tau * L[k] / (1 + tau * L[k])
        return mu

    for step in range(n_steps):
        dW = rng.standard_normal((n_paths, n_fwd)) * sqrt_dt

        for j in range(n_fwd):
            L = np.maximum(F[:, step, :], 1e-10)

            # Predictor (Euler)
            mu_n = np.array([drift(L[p], j) for p in range(n_paths)])
            F_star = L[:, j] + mu_n * L[:, j] * dt + vols[j] * L[:, j] * dW[:, j]
            F_star = np.maximum(F_star, 0.0)

            # Corrector
            L_star = L.copy()
            L_star[:, j] = F_star
            mu_star = np.array([drift(L_star[p], j) for p in range(n_paths)])
            F[:, step + 1, j] = L[:, j] + 0.5 * (mu_n + mu_star) * L[:, j] * dt + vols[j] * L[:, j] * dW[:, j]
            F[:, step + 1, j] = np.maximum(F[:, step + 1, j], 0.0)

    # Price caplet on first forward
    df = math.exp(-fwd[0] * dt_tenor)
    payoff = np.maximum(F[:, -1, 0] - fwd[0], 0.0) * dt_tenor
    price = float(df * payoff.mean())

    return PredictorCorrectorResult(F, price)


# ---- LMM pathwise Greeks ----

@dataclass
class LMMGreeksResult:
    """LMM pathwise sensitivities."""
    deltas: np.ndarray   # ∂V/∂F_i for each forward rate
    total_delta: float


def lmm_pathwise_greeks(
    forward_rates: list[float],
    inst_vols: list[float],
    strike: float,
    expiry_idx: int,
    dt_tenor: float = 0.5,
    n_paths: int = 50_000,
    n_steps: int = 50,
    seed: int | None = None,
) -> LMMGreeksResult:
    """Pathwise (IPA) Greeks for a caplet under LMM.

    ∂V/∂F_i = E[df × 1_{F>K} × ∂F(T)/∂F_i(0)]

    For lognormal LMM: ∂F_i(T)/∂F_i(0) = F_i(T)/F_i(0).
    """
    rng = np.random.default_rng(seed)
    fwd = np.array(forward_rates)
    vols = np.array(inst_vols)
    n_fwd = len(fwd)
    T = (expiry_idx + 1) * dt_tenor
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    F = np.full((n_paths, n_fwd), fwd, dtype=float)

    for step in range(n_steps):
        dW = rng.standard_normal((n_paths, n_fwd)) * sqrt_dt
        for j in range(n_fwd):
            F[:, j] = F[:, j] * np.exp(
                -0.5 * vols[j]**2 * dt + vols[j] * dW[:, j]
            )

    # Caplet on forward expiry_idx
    idx = min(expiry_idx, n_fwd - 1)
    df = math.exp(-fwd[0] * T)
    itm = F[:, idx] > strike
    payoff = np.maximum(F[:, idx] - strike, 0.0) * dt_tenor

    # Pathwise delta: ∂payoff/∂F_i(0) = 1_{ITM} × F(T)/F(0) × dt_tenor
    deltas = np.zeros(n_fwd)
    for i in range(n_fwd):
        if i == idx:
            pathwise = df * itm * F[:, idx] / fwd[idx] * dt_tenor
            deltas[i] = float(pathwise.mean())

    return LMMGreeksResult(deltas, float(deltas.sum()))
