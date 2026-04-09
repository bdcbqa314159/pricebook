"""LMM calibration to swaption matrix and multi-factor SABR.

Calibrate LMM volatility functions to an ATM swaption grid using
Rebonato's approximation. Multi-factor SABR calibrates SABR params
across multiple expiries jointly.

    from pricebook.lmm_calibration import (
        rebonato_swaption_vol, calibrate_lmm_vols,
        MultiFactorSABR, calibrate_multi_factor_sabr,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.sabr import sabr_implied_vol, sabr_calibrate


# ---- Rebonato's swaption vol approximation ----

def rebonato_swaption_vol(
    forward_rates: list[float],
    vols: list[float],
    tau: float,
    expiry_idx: int,
    swap_length: int,
    correlation: np.ndarray | None = None,
) -> float:
    """Rebonato's approximation for swaption vol from LMM parameters.

    σ_swaption² ≈ (1/T) Σ_ij w_i w_j ρ_ij σ_i σ_j L_i L_j / S²

    where w_i are swap rate weights, S is the forward swap rate.

    Args:
        forward_rates: L_i(0) forward rates.
        vols: σ_i instantaneous vols.
        tau: accrual period.
        expiry_idx: swaption expiry (first rate that contributes).
        swap_length: number of forward rates in the underlying swap.
        correlation: correlation matrix (if None, identity).

    Returns:
        Approximate swaption Black vol.
    """
    n = len(forward_rates)
    start = expiry_idx
    end = min(start + swap_length, n)
    if start >= end:
        return 0.0

    L = np.array(forward_rates[start:end])
    sigma = np.array(vols[start:end])
    m = len(L)

    if correlation is None:
        rho = np.eye(m)
    else:
        rho = correlation[start:end, start:end]

    # Swap rate and weights
    # S = Σ w_i L_i where w_i = tau × P(T_i) / annuity
    # Simplified: S ≈ (Σ L_i τ P_i) / (Σ τ P_i)
    # For quick approximation, use uniform weights
    S = L.mean()
    if S <= 0:
        return 0.0
    w = L / (m * S)  # approximate weights

    T = (expiry_idx + 1) * tau
    if T <= 0:
        return 0.0

    total = 0.0
    for i in range(m):
        for j in range(m):
            total += w[i] * w[j] * rho[i, j] * sigma[i] * sigma[j] * L[i] * L[j]

    var = total / (S ** 2 * T)
    return math.sqrt(max(var, 0.0))


# ---- LMM vol calibration ----

def exponential_correlation(n: int, beta: float = 0.1) -> np.ndarray:
    """Exponential decay correlation: ρ_ij = exp(-β|i-j|)."""
    rho = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho[i, j] = math.exp(-beta * abs(i - j))
    return rho


@dataclass
class LMMCalibrationResult:
    """Result of LMM calibration."""
    calibrated_vols: list[float]
    target_swaption_vols: dict[tuple[int, int], float]
    fitted_swaption_vols: dict[tuple[int, int], float]
    rmse: float


def calibrate_lmm_vols(
    forward_rates: list[float],
    target_swaption_vols: dict[tuple[int, int], float],
    tau: float = 0.25,
    correlation_beta: float = 0.1,
    max_iter: int = 100,
) -> LMMCalibrationResult:
    """Calibrate LMM instantaneous vols to a swaption grid.

    Uses iterative scaling: adjust σ_i so that Rebonato formula
    matches target swaption vols.

    Args:
        forward_rates: initial forward rates L_i(0).
        target_swaption_vols: (expiry_idx, swap_length) -> target vol.
        tau: accrual period.
        correlation_beta: exponential correlation decay.
        max_iter: maximum calibration iterations.

    Returns:
        LMMCalibrationResult with calibrated vols and fit quality.
    """
    n = len(forward_rates)
    rho = exponential_correlation(n, correlation_beta)
    vols = np.full(n, 0.20)  # initial guess

    for iteration in range(max_iter):
        # For each target, compute model vol and adjust
        for (exp_idx, swap_len), target in target_swaption_vols.items():
            model = rebonato_swaption_vol(
                forward_rates, vols.tolist(), tau, exp_idx, swap_len, rho,
            )
            if model > 1e-10:
                ratio = target / model
                # Scale vols contributing to this swaption
                start = exp_idx
                end = min(start + swap_len, n)
                for k in range(start, end):
                    vols[k] *= ratio ** (1.0 / (end - start))

        vols = np.clip(vols, 0.01, 2.0)

    # Compute fitted vols
    fitted = {}
    errors = []
    for (exp_idx, swap_len), target in target_swaption_vols.items():
        model = rebonato_swaption_vol(
            forward_rates, vols.tolist(), tau, exp_idx, swap_len, rho,
        )
        fitted[(exp_idx, swap_len)] = model
        errors.append((model - target) ** 2)

    rmse = math.sqrt(sum(errors) / len(errors)) if errors else 0.0

    return LMMCalibrationResult(
        calibrated_vols=vols.tolist(),
        target_swaption_vols=target_swaption_vols,
        fitted_swaption_vols=fitted,
        rmse=rmse,
    )


# ---- Multi-factor SABR ----

@dataclass
class SABRSlice:
    """SABR parameters at one expiry."""
    expiry: float
    alpha: float
    beta: float
    rho: float
    nu: float


class MultiFactorSABR:
    """SABR with term structure of parameters.

    Each expiry has its own (alpha, rho, nu) with shared beta.

    Args:
        slices: list of SABRSlice, one per expiry.
    """

    def __init__(self, slices: list[SABRSlice]):
        self.slices = sorted(slices, key=lambda s: s.expiry)

    def vol(self, forward: float, strike: float, T: float) -> float:
        """Interpolated SABR vol at arbitrary expiry."""
        if len(self.slices) == 1:
            s = self.slices[0]
            return sabr_implied_vol(forward, strike, T, s.alpha, s.beta, s.rho, s.nu)

        # Find bracketing slices
        if T <= self.slices[0].expiry:
            s = self.slices[0]
            return sabr_implied_vol(forward, strike, T, s.alpha, s.beta, s.rho, s.nu)
        if T >= self.slices[-1].expiry:
            s = self.slices[-1]
            return sabr_implied_vol(forward, strike, T, s.alpha, s.beta, s.rho, s.nu)

        for i in range(len(self.slices) - 1):
            if self.slices[i].expiry <= T <= self.slices[i + 1].expiry:
                s1 = self.slices[i]
                s2 = self.slices[i + 1]
                w = (T - s1.expiry) / (s2.expiry - s1.expiry)
                # Interpolate params linearly
                alpha = s1.alpha * (1 - w) + s2.alpha * w
                rho = s1.rho * (1 - w) + s2.rho * w
                nu = s1.nu * (1 - w) + s2.nu * w
                beta = s1.beta  # shared
                return sabr_implied_vol(forward, strike, T, alpha, beta, rho, nu)

        s = self.slices[-1]
        return sabr_implied_vol(forward, strike, T, s.alpha, s.beta, s.rho, s.nu)

    @property
    def expiries(self) -> list[float]:
        return [s.expiry for s in self.slices]


def calibrate_multi_factor_sabr(
    forwards: list[float],
    expiries: list[float],
    strikes_per_expiry: list[list[float]],
    vols_per_expiry: list[list[float]],
    beta: float = 0.5,
) -> MultiFactorSABR:
    """Calibrate SABR at each expiry independently, then combine.

    Args:
        forwards: forward rate at each expiry.
        expiries: time to each expiry.
        strikes_per_expiry: strikes for each expiry's smile.
        vols_per_expiry: market vols at each strike/expiry.
        beta: shared CEV parameter.

    Returns:
        MultiFactorSABR with calibrated slices.
    """
    slices = []
    for fwd, T, strikes, mkt_vols in zip(forwards, expiries, strikes_per_expiry, vols_per_expiry):
        result = sabr_calibrate(fwd, strikes, mkt_vols, T, beta=beta)
        slices.append(SABRSlice(
            expiry=T,
            alpha=result["alpha"],
            beta=beta,
            rho=result["rho"],
            nu=result["nu"],
        ))

    return MultiFactorSABR(slices)
