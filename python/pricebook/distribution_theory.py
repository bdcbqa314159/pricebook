"""Distribution theory: Schwartz distributions, Sobolev spaces, Green's functions, Feynman-Kac.

Phase M10 slices 202-204 consolidated.

The mathematical bedrock for PDE pricing: distributions give rigorous
meaning to delta functions (digital payoffs), Heaviside (barriers),
and Green's functions (the BS kernel). Sobolev spaces quantify FEM
convergence. Feynman-Kac links PDEs to expectations.

* :class:`Distribution` — distributions as functionals on test functions.
* :func:`dirac_delta` / :func:`heaviside_dist` — standard distributions.
* :func:`sobolev_norm` — H^s norm for regularity analysis.
* :func:`greens_function_heat` — fundamental solution of the heat equation.
* :func:`greens_function_bs` — Black-Scholes kernel (= lognormal density).
* :func:`feynman_kac_pde` — given SDE coefficients, return PDE coefficients.
* :func:`feynman_kac_verify` — verify PDE solution matches MC expectation.

References:
    Schwartz, *Théorie des Distributions*, Hermann, 1966.
    Evans, *Partial Differential Equations*, AMS, 2010.
    Shreve, *Stochastic Calculus for Finance II*, Springer, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---- Distributions as functionals ----

class Distribution:
    """A distribution (generalised function) acting on test functions.

    ⟨T, φ⟩ = ∫ T(x) φ(x) dx  (for regular distributions)
    ⟨δ, φ⟩ = φ(0)              (for Dirac delta)

    This class represents T via its action on test functions φ.
    """

    def __init__(self, action: Callable[[Callable], float], name: str = ""):
        self._action = action
        self.name = name

    def __call__(self, phi: Callable[[float], float]) -> float:
        """Evaluate ⟨T, φ⟩."""
        return self._action(phi)

    def derivative(self) -> "Distribution":
        """Distributional derivative: ⟨T', φ⟩ = −⟨T, φ'⟩.

        Requires φ to be differentiable. Uses central differences.
        """
        def action(phi):
            eps = 1e-6
            dphi = lambda x: (phi(x + eps) - phi(x - eps)) / (2 * eps)
            return -self._action(dphi)
        return Distribution(action, f"D({self.name})")


def dirac_delta(x0: float = 0.0) -> Distribution:
    """Dirac delta distribution at x₀: ⟨δ_{x₀}, φ⟩ = φ(x₀)."""
    return Distribution(lambda phi: phi(x0), f"δ({x0})")


def heaviside_dist(x0: float = 0.0) -> Distribution:
    """Heaviside distribution: ⟨H_{x₀}, φ⟩ = ∫_{x₀}^∞ φ(x) dx.

    Approximated numerically via quadrature on [x₀, x₀ + L].
    """
    def action(phi):
        L = 20.0
        x = np.linspace(x0, x0 + L, 1000)
        dx = x[1] - x[0]
        return float(np.sum([phi(xi) for xi in x]) * dx)
    return Distribution(action, f"H({x0})")


def regular_distribution(f: Callable[[float], float]) -> Distribution:
    """Regular distribution from an L¹_loc function: ⟨T_f, φ⟩ = ∫ f(x) φ(x) dx."""
    def action(phi):
        x = np.linspace(-20, 20, 2000)
        dx = x[1] - x[0]
        return float(np.sum([f(xi) * phi(xi) for xi in x]) * dx)
    return Distribution(action, "T_f")


# ---- Sobolev norms ----

@dataclass
class SobolevNormResult:
    """Sobolev H^s norm of a function on a grid."""
    h0_norm: float  # L² norm
    h1_norm: float  # H¹ norm = sqrt(||f||² + ||f'||²)
    hs_norm: float  # H^s norm (for given s)
    s: float


def sobolev_norm(
    f_values: np.ndarray | list[float],
    dx: float,
    s: float = 1.0,
) -> SobolevNormResult:
    """Compute Sobolev H^s norm of a function given on a uniform grid.

    H⁰ = L²: ||f||₀ = sqrt(∫ f² dx)
    H¹:      ||f||₁ = sqrt(∫ f² + f'² dx)
    H^s:     ||f||_s via Fourier (approximated by discrete FFT).

    Args:
        f_values: function values on a uniform grid.
        dx: grid spacing.
        s: Sobolev exponent.
    """
    f = np.asarray(f_values, dtype=float)
    n = len(f)

    # L² norm
    h0 = math.sqrt(float(np.sum(f**2) * dx))

    # H¹ norm: add ||f'||²
    fp = np.gradient(f, dx)
    h1 = math.sqrt(float(np.sum(f**2 + fp**2) * dx))

    # H^s norm via Fourier: ||f||_s² = Σ (1 + |k|²)^s |f̂_k|²
    fhat = np.fft.fft(f)
    freqs = np.fft.fftfreq(n, d=dx)
    hs_sq = float(np.sum((1 + (2 * math.pi * freqs)**2)**s * np.abs(fhat)**2) / n * dx)
    hs = math.sqrt(max(hs_sq, 0.0))

    return SobolevNormResult(h0, h1, hs, s)


# ---- Green's functions ----

def greens_function_heat(x: float, t: float, D: float = 1.0) -> float:
    """Fundamental solution of the heat equation: ∂u/∂t = D ∂²u/∂x².

    G(x, t) = (4πDt)^{-1/2} exp(−x²/(4Dt))

    This IS the Gaussian density — the heat kernel.
    """
    if t <= 0:
        return 0.0
    return math.exp(-x**2 / (4 * D * t)) / math.sqrt(4 * math.pi * D * t)


def greens_function_bs(
    S: float,
    K: float,
    rate: float,
    vol: float,
    T: float,
) -> float:
    """Green's function of the Black-Scholes operator.

    This is the risk-neutral transition density: the probability
    density of S_T given S_0 = S, which is a lognormal.

    p(S_T | S) = (1/(S_T σ√T)) φ((log(S_T/S) − (r−σ²/2)T)/(σ√T))

    Evaluated at S_T = K (the strike acts as the terminal point).
    """
    if T <= 0 or vol <= 0 or K <= 0 or S <= 0:
        return 0.0
    d = (math.log(S / K) + (rate - 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    phi = math.exp(-0.5 * d**2) / math.sqrt(2 * math.pi)
    return phi / (K * vol * math.sqrt(T))


# ---- Feynman-Kac ----

@dataclass
class FeynmanKacPDE:
    """PDE coefficients derived from an SDE via Feynman-Kac.

    The SDE: dX = μ(X) dt + σ(X) dW
    produces the PDE: ∂u/∂t + μ(x) ∂u/∂x + 0.5 σ²(x) ∂²u/∂x² − r u = 0

    Attributes:
        drift_coeff: μ(x) — first-order coefficient.
        diffusion_coeff: 0.5 σ²(x) — second-order coefficient.
        killing_rate: r — discount/killing rate.
    """
    drift_coeff: Callable[[float], float]
    diffusion_coeff: Callable[[float], float]
    killing_rate: float
    description: str


def feynman_kac_pde(
    drift: Callable[[float], float],
    diffusion: Callable[[float], float],
    rate: float = 0.0,
) -> FeynmanKacPDE:
    """Given SDE coefficients, derive the Feynman-Kac PDE.

    SDE: dX = μ(X) dt + σ(X) dW
    PDE: ∂u/∂t + μ(x) ∂u/∂x + 0.5 σ²(x) ∂²u/∂x² − r u = 0

    The PDE solution u(x, t) = E[e^{-r(T-t)} g(X_T) | X_t = x]
    where g is the terminal condition (payoff).
    """
    def diff_coeff(x):
        s = diffusion(x)
        return 0.5 * s * s

    desc = "∂u/∂t + μ(x)∂u/∂x + ½σ²(x)∂²u/∂x² − ru = 0"

    return FeynmanKacPDE(drift, diff_coeff, rate, desc)


@dataclass
class FeynmanKacVerification:
    """Result of Feynman-Kac consistency check."""
    pde_value: float
    mc_value: float
    relative_error: float
    passed: bool


def feynman_kac_verify(
    pde_value: float,
    mc_value: float,
    tol: float = 0.05,
) -> FeynmanKacVerification:
    """Verify that a PDE price matches the MC expectation (Feynman-Kac).

    This is the fundamental consistency check: the PDE and MC
    approaches must agree because they compute the same quantity.

    Args:
        pde_value: price from PDE solver.
        mc_value: price from MC simulation.
        tol: relative tolerance.
    """
    if abs(pde_value) < 1e-10:
        rel_err = abs(mc_value)
    else:
        rel_err = abs(pde_value - mc_value) / abs(pde_value)

    return FeynmanKacVerification(
        pde_value, mc_value, rel_err, rel_err < tol,
    )
