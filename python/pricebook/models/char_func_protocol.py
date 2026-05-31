"""Characteristic function protocol and standalone factory functions.

Provides a unified interface for Fourier-based option pricing (COS, FFT).
Any model with a ``char_func(T)`` method satisfies the protocol structurally.

    from pricebook.models.char_func_protocol import (
        CharFuncModel, validate_char_func,
        merton_char_func, vg_char_func, kou_char_func, bates_char_func,
    )

    # Protocol: any class with char_func(T) -> Callable[[complex], complex]
    phi = model.char_func(T=1.0)
    price = cos_price(phi, spot, strike, rate, T)

    # Standalone: factory functions for quick use
    phi = merton_char_func(rate=0.05, sigma=0.2, lam=1.0, mu_j=-0.1, sigma_j=0.15, T=1.0)
    price = cos_price(phi, spot, strike, rate, T)

References:
    Fang & Oosterlee (2008). A Novel Pricing Method for European Options
        Based on Fourier-Cosine Series Expansions.
    Schoutens (2003). Lévy Processes in Finance.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


# ═══════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════

@runtime_checkable
class CharFuncModel(Protocol):
    """Protocol for models providing a characteristic function.

    Any class with a ``char_func(T) -> Callable`` method satisfies
    this protocol structurally (no inheritance required).
    """

    def char_func(self, T: float) -> Callable[[complex], complex]:
        """Return φ(u) = E[exp(iu · log(S_T/S_0))] for maturity T."""
        ...


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

def validate_char_func(
    phi: Callable[[complex], complex],
    T: float = 1.0,
    tol: float = 1e-8,
) -> dict:
    """Validate basic properties of a characteristic function.

    Checks:
    1. φ(0) = 1 (normalisation).
    2. |φ(u)| ≤ 1 for real u (boundedness).
    3. φ(-u) = conj(φ(u)) for real u (Hermitian symmetry).

    Returns dict with 'valid' bool and per-check results.
    """
    results = {}

    # φ(0) = 1
    phi_0 = phi(0.0)
    results["phi_0"] = phi_0
    results["phi_0_ok"] = abs(phi_0 - 1.0) < tol

    # Boundedness at several points
    test_u = [1.0, 5.0, 10.0, 20.0]
    bounded = True
    for u in test_u:
        if abs(phi(u)) > 1.0 + tol:
            bounded = False
            break
    results["bounded"] = bounded

    # Hermitian symmetry: φ(-u) = conj(φ(u))
    symmetric = True
    for u in test_u:
        diff = abs(phi(-u) - phi(u).conjugate())
        if diff > tol:
            symmetric = False
            break
    results["hermitian"] = symmetric

    results["valid"] = results["phi_0_ok"] and results["bounded"] and results["hermitian"]
    return results


# ═══════════════════════════════════════════════════════════════
# Cumulant extraction
# ═══════════════════════════════════════════════════════════════

@dataclass
class CumulantInfo:
    """Cumulants extracted from a characteristic function."""
    c1: float       # mean (first cumulant)
    c2: float       # variance (second cumulant)
    c3: float       # third cumulant
    c4: float       # fourth cumulant
    skewness: float
    excess_kurtosis: float

    def to_dict(self) -> dict:
        return vars(self)


def extract_cumulants(
    phi: Callable[[complex], complex],
    eps: float = 1e-4,
) -> CumulantInfo:
    """Extract first 4 cumulants from characteristic function numerically.

    Uses finite differences on log(φ(u)) at u=0.
    """
    ln = lambda u: cmath.log(phi(u)) if abs(phi(u)) > 1e-30 else 0

    ln0 = ln(0.0)
    ln_p = ln(eps)
    ln_m = ln(-eps)
    ln_2p = ln(2 * eps)
    ln_2m = ln(-2 * eps)

    # c1 = Im[d(ln φ)/du] at u=0
    c1 = float((ln_p - ln_m).imag / (2 * eps))

    # c2 = -Re[d²(ln φ)/du²] at u=0
    c2 = max(float(-(ln_p + ln_m - 2 * ln0).real / eps**2), 1e-10)

    # c3 = -Im[d³(ln φ)/du³]
    c3 = float(-(ln_2p - 2 * ln_p + 2 * ln_m - ln_2m).imag / (2 * eps**3))

    # c4 = Re[d⁴(ln φ)/du⁴]
    c4 = float((ln_2p - 4 * ln_p + 6 * ln0 - 4 * ln_m + ln_2m).real / eps**4)

    std = math.sqrt(c2) if c2 > 0 else 1e-10
    skew = c3 / std**3 if std > 1e-10 else 0.0
    kurt = c4 / std**4 if std > 1e-10 else 0.0

    return CumulantInfo(c1, c2, c3, c4, skew, kurt)


# ═══════════════════════════════════════════════════════════════
# Standalone characteristic function factories
# ═══════════════════════════════════════════════════════════════


def merton_char_func(
    rate: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    T: float,
) -> Callable[[complex], complex]:
    """Merton jump-diffusion characteristic function.

    φ(u) for log(S_T/S_0) under risk-neutral measure.

    Args:
        rate: risk-free rate.
        sigma: diffusion volatility.
        lam: jump intensity (Poisson rate).
        mu_j: mean of log-jump size.
        sigma_j: std of log-jump size.
        T: time to maturity.
    """
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1  # compensator

    def phi(u: complex) -> complex:
        diff = 1j * u * (rate - lam * k - 0.5 * sigma**2) * T \
               - 0.5 * u**2 * sigma**2 * T
        jump_cf = cmath.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2) - 1
        return cmath.exp(diff + lam * T * jump_cf)

    return phi


def vg_char_func(
    sigma: float,
    nu: float,
    theta: float,
    rate: float,
    T: float,
) -> Callable[[complex], complex]:
    """Variance Gamma characteristic function.

    φ(u) for log(S_T/S_0) under risk-neutral measure.

    Args:
        sigma: volatility of BM component.
        nu: variance rate of Gamma subordinator.
        theta: drift of BM component.
        rate: risk-free rate.
        T: time to maturity.
    """
    omega = (1.0 / nu) * math.log(1 - theta * nu - 0.5 * sigma**2 * nu)

    def phi(u: complex) -> complex:
        drift = 1j * u * (rate + omega) * T
        inner = 1 - 1j * u * theta * nu + 0.5 * u**2 * sigma**2 * nu
        return cmath.exp(drift) * inner ** (-T / nu)

    return phi


def kou_char_func(
    rate: float,
    sigma: float,
    T: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    div_yield: float = 0.0,
) -> Callable[[complex], complex]:
    """Kou double-exponential jump-diffusion characteristic function.

    φ(u) for log(S_T/S_0) under risk-neutral measure.

    Args:
        rate: risk-free rate.
        sigma: diffusion volatility.
        T: time to maturity.
        lam: jump intensity.
        p: probability of up-jump.
        eta1: up-jump rate (mean up-jump = 1/eta1). Requires eta1 > 1.
        eta2: down-jump rate (mean down-jump = 1/eta2).
        div_yield: continuous dividend yield.

    The jump-size density: f(x) = p·η₁·exp(-η₁x)·1_{x>0} + (1-p)·η₂·exp(η₂x)·1_{x<0}
    """
    # Compensator: ζ = E[e^ξ - 1] = p·η₁/(η₁-1) + (1-p)·η₂/(η₂+1) - 1
    zeta = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1

    def phi(u: complex) -> complex:
        # Diffusion
        drift = 1j * u * (rate - div_yield - lam * zeta - 0.5 * sigma**2) * T
        diff = -0.5 * u**2 * sigma**2 * T
        # Double-exponential jump CF: E[e^{iuξ}] = p·η₁/(η₁-iu) + (1-p)·η₂/(η₂+iu)
        jump_cf = p * eta1 / (eta1 - 1j * u) + (1 - p) * eta2 / (eta2 + 1j * u)
        jump_part = lam * T * (jump_cf - 1)
        return cmath.exp(drift + diff + jump_part)

    return phi


def bates_char_func(
    rate: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    T: float,
    div_yield: float = 0.0,
) -> Callable[[complex], complex]:
    """Bates (Heston + Merton jumps) characteristic function.

    Combines Heston stochastic volatility CF with Merton jump CF.

    φ_Bates(u) = φ_Heston(u) × φ_jumps(u)

    Args:
        rate: risk-free rate.
        v0: initial variance.
        kappa: mean-reversion speed.
        theta: long-run variance.
        xi: vol-of-vol.
        rho: correlation (spot, vol).
        lam: jump intensity.
        mu_j: mean log-jump size.
        sigma_j: std log-jump size.
        T: time to maturity.
        div_yield: continuous dividend yield.
    """
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1  # jump compensator

    def phi(u: complex) -> complex:
        # Heston CF (Schoutens form)
        d = cmath.sqrt(
            (rho * xi * 1j * u - kappa) ** 2
            + xi**2 * (1j * u + u**2)
        )
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        exp_dT = cmath.exp(-d * T)
        C = (rate - div_yield) * 1j * u * T + (kappa * theta / xi**2) * (
            (kappa - rho * xi * 1j * u - d) * T
            - 2 * cmath.log((1 - g * exp_dT) / (1 - g))
        )
        D = ((kappa - rho * xi * 1j * u - d) / xi**2) * (
            (1 - exp_dT) / (1 - g * exp_dT)
        )

        heston_part = cmath.exp(C + D * v0)

        # Merton jump part
        jump_drift = -lam * k * 1j * u * T
        jump_cf = lam * T * (cmath.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2) - 1)
        jump_part = cmath.exp(jump_drift + jump_cf)

        return heston_part * jump_part

    return phi


# Alias: SVJ is Bates under a different name
svj_char_func = bates_char_func
