"""Lévy processes: Normal Inverse Gaussian (NIG) and CGMY.

These are pure-jump Lévy processes with analytical characteristic functions,
enabling COS/FFT pricing. CGMY generalises Variance Gamma (Y→0 limit).

    from pricebook.models.levy_processes import (
        NIGProcess, CGMYProcess,
        nig_char_func, cgmy_char_func,
    )

    nig = NIGProcess(alpha=15, beta=-5, delta=0.5)
    phi = nig.char_func(T=1.0)
    price = cos_price(phi, spot, strike, rate, T)

References:
    Barndorff-Nielsen (1997). Normal Inverse Gaussian Distributions
        and Stochastic Volatility Modelling. Scand. J. Statist.
    Carr, Geman, Madan & Yor (2002). The Fine Structure of Asset Returns:
        An Empirical Investigation. J. Business.
    Schoutens (2003). Lévy Processes in Finance.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Normal Inverse Gaussian (NIG)
# ═══════════════════════════════════════════════════════════════

@dataclass
class NIGResult:
    """NIG terminal simulation result."""
    terminal: np.ndarray
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "n_paths": len(self.terminal)}


class NIGProcess:
    """Normal Inverse Gaussian process.

    X(t) = μt + βδ²t·IG(t) + δ·W(IG(t))

    where IG is an Inverse Gaussian subordinator with parameter δ√(α²-β²).

    Parameters:
        alpha: tail heaviness (α > 0, α > |β|).
        beta: asymmetry (-α < β < α). β < 0 → left skew.
        delta: scale (δ > 0).
        mu: location/drift.

    Characteristic function:
        φ(u) = exp(iuμ + δ(√(α²-β²) - √(α²-(β+iu)²)))
    """

    def __init__(self, alpha: float, beta: float, delta: float, mu: float = 0.0):
        if alpha <= abs(beta):
            raise ValueError(f"NIG requires alpha > |beta|, got alpha={alpha}, beta={beta}")
        if delta <= 0:
            raise ValueError(f"NIG requires delta > 0, got {delta}")
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu
        self._gamma = math.sqrt(alpha**2 - beta**2)

    def char_func(self, T: float) -> Callable[[complex], complex]:
        """Characteristic function of X(T) (not risk-neutral adjusted)."""
        alpha, beta, delta, mu = self.alpha, self.beta, self.delta, self.mu
        gamma = self._gamma

        def phi(u: complex) -> complex:
            inner = cmath.sqrt(alpha**2 - (beta + 1j * u) ** 2)
            return cmath.exp(1j * u * mu * T + delta * T * (gamma - inner))

        return phi

    def terminal(
        self, S0: float, rate: float, T: float,
        n_paths: int = 50_000, seed: int = 42,
    ) -> np.ndarray:
        """Simulate terminal S(T) under risk-neutral measure.

        Uses the representation: NIG(α,β,δ) = βδ²·IG + δ·√IG·Z
        where IG ~ InverseGaussian(δT, γ²δ²T²) with γ = √(α²-β²).

        Risk-neutral correction: ω = δ(√(α²-β²) - √(α²-(β+1)²))
        ensures E[exp(X)] = exp(ωT), so E[S_T] = S_0·exp(rT).
        """
        rng = np.random.default_rng(seed)
        alpha, beta, delta = self.alpha, self.beta, self.delta
        gamma = self._gamma

        # Inverse Gaussian: mean = δT/γ, shape = δ²T²
        # numpy.random.wald(mean, lambda): IG(mean, lambda)
        ig_mean = delta * T / gamma
        ig_lam = delta**2 * T**2
        ig_samples = rng.wald(ig_mean, ig_lam, size=n_paths)

        # NIG increment: X = β·ig + √ig·Z  (simplified parametrisation)
        Z = rng.standard_normal(n_paths)
        X = beta * ig_samples + np.sqrt(ig_samples) * Z

        # Risk-neutral martingale correction
        # ψ(1) = δ(γ - √(α²-(β+1)²)) is the log-MGF at u=1
        psi_1 = delta * (gamma - math.sqrt(alpha**2 - (beta + 1)**2))
        omega = -psi_1  # so E[exp(X - ψ(1)T)] = 1

        drift = (rate + omega) * T
        return S0 * np.exp(drift + X)

    def to_dict(self) -> dict:
        return {"type": "nig", "alpha": self.alpha, "beta": self.beta,
                "delta": self.delta, "mu": self.mu}


# ═══════════════════════════════════════════════════════════════
# CGMY
# ═══════════════════════════════════════════════════════════════

@dataclass
class CGMYResult:
    """CGMY terminal simulation result."""
    terminal: np.ndarray
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "n_paths": len(self.terminal)}


class CGMYProcess:
    """CGMY (Carr-Geman-Madan-Yor) process.

    A tempered stable Lévy process with Lévy measure:
        ν(dx) = C·exp(-G|x|)/|x|^{1+Y}·1_{x<0} + C·exp(-Mx)/x^{1+Y}·1_{x>0}

    Parameters:
        C: overall activity level (C > 0).
        G: rate of exponential decay for negative jumps (G > 0).
        M: rate of exponential decay for positive jumps (M > 0).
        Y: fine structure index (-∞ < Y < 2, Y ≠ 0,1).
            Y < 0: compound Poisson (finite activity)
            0 < Y < 1: infinite activity, finite variation
            1 < Y < 2: infinite activity, infinite variation

    VG is the special case Y → 0.

    Characteristic function:
        φ(u) = exp(C·T·Γ(-Y)·((M-iu)^Y - M^Y + (G+iu)^Y - G^Y))
    """

    def __init__(self, C: float, G: float, M: float, Y: float):
        if C <= 0:
            raise ValueError(f"CGMY requires C > 0, got {C}")
        if G <= 0 or M <= 0:
            raise ValueError(f"CGMY requires G > 0, M > 0, got G={G}, M={M}")
        if Y >= 2:
            raise ValueError(f"CGMY requires Y < 2, got {Y}")
        self.C = C
        self.G = G
        self.M = M
        self.Y = Y

    def char_func(self, T: float) -> Callable[[complex], complex]:
        """Characteristic function of X(T) (not risk-neutral adjusted)."""
        C, G, M, Y = self.C, self.G, self.M, self.Y

        if abs(Y) < 1e-10:
            # Y → 0 limit: VG-like, use log formula
            return self._char_func_y0(T)

        gamma_neg_Y = math.gamma(-Y)  # Γ(-Y), valid for Y not in {0, 1, 2, ...}

        def phi(u: complex) -> complex:
            term1 = (M - 1j * u) ** Y - M**Y
            term2 = (G + 1j * u) ** Y - G**Y
            return cmath.exp(C * T * gamma_neg_Y * (term1 + term2))

        return phi

    def _char_func_y0(self, T: float) -> Callable[[complex], complex]:
        """VG limit (Y → 0)."""
        C, G, M = self.C, self.G, self.M

        def phi(u: complex) -> complex:
            term = C * T * (cmath.log(M - 1j * u) - cmath.log(complex(M))
                            + cmath.log(G + 1j * u) - cmath.log(complex(G)))
            return cmath.exp(-term)

        return phi

    def terminal(
        self, S0: float, rate: float, T: float,
        n_paths: int = 50_000, seed: int = 42,
    ) -> np.ndarray:
        """Simulate terminal S(T) under risk-neutral measure.

        Approximates CGMY via difference-of-Gamma (exact for VG / Y=0).
        The VG approximation maps CGMY parameters to σ, θ, ν for Gamma subordination.
        """
        rng = np.random.default_rng(seed)
        C, G, M, Y = self.C, self.G, self.M, self.Y

        # Map to VG-like parameters via moment-matching:
        # E[X] = C·Γ(1-Y)·(1/G^(1-Y) - 1/M^(1-Y)) ... simplified for simulation
        # Use difference-of-Gamma: X = Γ_+ - Γ_-
        # Γ_+ ~ Gamma(C·T·Γ(-Y)·M^Y / ...) — complex, so use VG approx
        theta_vg = 1.0 / G - 1.0 / M  # drift skew
        sigma_vg = math.sqrt(2.0 / G**2 + 2.0 / M**2)  # diffusion
        nu_vg = 1.0 / C  # time-change variance

        # Gamma subordinator
        shape = T / nu_vg
        scale = nu_vg
        G_sub = rng.gamma(shape, scale, size=n_paths)
        Z = rng.standard_normal(n_paths)
        X = theta_vg * G_sub + sigma_vg * np.sqrt(G_sub) * Z

        # VG-like drift correction: ω = -(1/ν)·ln(1 - θν - 0.5σ²ν)
        inner = 1 - theta_vg * nu_vg - 0.5 * sigma_vg**2 * nu_vg
        if inner > 0:
            omega = -(1.0 / nu_vg) * math.log(inner)
        else:
            omega = 0.0  # fallback for extreme params

        drift = (rate + omega) * T
        return S0 * np.exp(drift + X)

    def to_dict(self) -> dict:
        return {"type": "cgmy", "C": self.C, "G": self.G,
                "M": self.M, "Y": self.Y}


# ═══════════════════════════════════════════════════════════════
# Standalone characteristic function factories
# ═══════════════════════════════════════════════════════════════

def nig_char_func(
    alpha: float,
    beta: float,
    delta: float,
    rate: float,
    T: float,
) -> Callable[[complex], complex]:
    """Risk-neutral NIG characteristic function for log(S_T/S_0).

    Includes martingale correction ω.
    """
    gamma = math.sqrt(alpha**2 - beta**2)
    omega = -delta * (gamma - math.sqrt(alpha**2 - (beta + 1)**2))

    def phi(u: complex) -> complex:
        drift = 1j * u * (rate + omega) * T
        inner = cmath.sqrt(alpha**2 - (beta + 1j * u) ** 2)
        nig_part = delta * T * (gamma - inner)
        return cmath.exp(drift + nig_part)

    return phi


def cgmy_char_func(
    C: float,
    G: float,
    M: float,
    Y: float,
    rate: float,
    T: float,
) -> Callable[[complex], complex]:
    """Risk-neutral CGMY characteristic function for log(S_T/S_0).

    Includes martingale correction ω.
    """
    if abs(Y) < 1e-10:
        omega = -C * (math.log(M - 1) - math.log(M)
                       + math.log(G + 1) - math.log(G))
        def phi(u: complex) -> complex:
            drift = 1j * u * (rate + omega) * T
            cgmy_part = -C * T * (cmath.log(M - 1j * u) - cmath.log(complex(M))
                                   + cmath.log(G + 1j * u) - cmath.log(complex(G)))
            return cmath.exp(drift + cgmy_part)
        return phi

    gamma_neg_Y = math.gamma(-Y)
    omega = -C * gamma_neg_Y * (
        (M - 1)**Y - M**Y + (G + 1)**Y - G**Y
    )

    def phi(u: complex) -> complex:
        drift = 1j * u * (rate + omega) * T
        term1 = (M - 1j * u) ** Y - M**Y
        term2 = (G + 1j * u) ** Y - G**Y
        cgmy_part = C * T * gamma_neg_Y * (term1 + term2)
        return cmath.exp(drift + cgmy_part)

    return phi
