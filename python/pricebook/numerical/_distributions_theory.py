"""Extended Schwartz distribution theory: tempered, Fourier, convolution.

    from pricebook.numerical import TemperedDistribution, schwartz_test_function

Extends distribution_theory.py with richer functional analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class SchwartzTestFunction:
    """Test function in Schwartz space S(R): smooth and rapidly decreasing.

    phi(x) = exp(-x^2 / (2*sigma^2)) * polynomial(x)
    """

    def __init__(self, sigma: float = 1.0, polynomial_coeffs: list[float] | None = None):
        self.sigma = sigma
        self.coeffs = polynomial_coeffs or [1.0]

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(x)
        poly = sum(c * x ** k for k, c in enumerate(self.coeffs))
        return poly * np.exp(-x ** 2 / (2 * self.sigma ** 2))

    def derivative(self, x: float | np.ndarray) -> float | np.ndarray:
        """Numerical derivative via central differences."""
        h = 1e-6
        return (self(x + h) - self(x - h)) / (2 * h)

    def fourier(self, xi: float | np.ndarray) -> complex | np.ndarray:
        """Fourier transform of the test function (numerical)."""
        x = np.linspace(-20 * self.sigma, 20 * self.sigma, 4000)
        dx = x[1] - x[0]
        xi = np.asarray(xi)
        if xi.ndim == 0:
            integrand = self(x) * np.exp(-1j * xi * x)
            return complex(np.trapz(integrand, dx=dx))
        result = np.zeros(len(xi), dtype=complex)
        for k, xik in enumerate(xi):
            integrand = self(x) * np.exp(-1j * xik * x)
            result[k] = np.trapz(integrand, dx=dx)
        return result


class TemperedDistribution:
    """Tempered distribution in S'(R): continuous linear functional on S(R).

    A tempered distribution T acts on test functions φ ∈ S:
        ⟨T, φ⟩ = T(φ) ∈ R

    Supports Fourier transform, convolution, derivative.
    """

    def __init__(self, action):
        """
        Args:
            action: callable(phi) → float, where phi is a SchwartzTestFunction or callable.
        """
        self._action = action

    def __call__(self, phi) -> float:
        """Evaluate ⟨T, φ⟩."""
        return self._action(phi)

    def derivative(self) -> TemperedDistribution:
        """Distributional derivative: ⟨T', φ⟩ = -⟨T, φ'⟩."""
        parent = self

        def deriv_action(phi):
            # Approximate phi' via central differences
            def phi_prime(x):
                h = 1e-6
                return (phi(x + h) - phi(x - h)) / (2 * h)
            return -parent._action(phi_prime)

        return TemperedDistribution(deriv_action)

    def fourier_transform(self) -> TemperedDistribution:
        """Fourier transform: ⟨F[T], φ⟩ = ⟨T, F[φ]⟩.

        Where F[φ](x) = ∫ φ(ξ) e^{-ixξ} dξ.
        """
        parent = self

        def ft_action(phi):
            # ⟨F[T], φ⟩ = ⟨T, F[φ]⟩
            # F[φ](x) = numerical Fourier of phi
            def fourier_phi(x):
                xi = np.linspace(-20, 20, 2000)
                dxi = xi[1] - xi[0]
                vals = np.array([phi(xik) for xik in xi])
                integrand = vals * np.exp(-1j * x * xi)
                return float(np.real(np.trapz(integrand, dx=dxi)))

            return parent._action(fourier_phi)

        return TemperedDistribution(ft_action)

    def convolve(self, other: TemperedDistribution) -> TemperedDistribution:
        """Convolution: ⟨T * S, φ⟩ = ⟨T, S̃ * φ⟩ where S̃(x) = S(-x).

        For regular distributions, (T*S)(x) = ∫ T(y) S(x-y) dy.
        """
        parent_t = self
        parent_s = other

        def conv_action(phi):
            # ⟨T*S, φ⟩ = ⟨T_x, ⟨S_y, φ(x+y)⟩⟩
            def inner(x):
                return parent_s._action(lambda y: phi(x + y))
            return parent_t._action(inner)

        return TemperedDistribution(conv_action)


# ═══════════════════════════════════════════════════════════════
# Standard distributions in S'
# ═══════════════════════════════════════════════════════════════

def dirac_delta(x0: float = 0.0) -> TemperedDistribution:
    """Dirac delta at x0: ⟨δ_{x0}, φ⟩ = φ(x0)."""
    return TemperedDistribution(lambda phi: float(phi(x0)))


def heaviside(x0: float = 0.0) -> TemperedDistribution:
    """Heaviside step function at x0: ⟨H_{x0}, φ⟩ = ∫_{x0}^∞ φ(x) dx."""
    def action(phi):
        x = np.linspace(x0, x0 + 30, 3000)
        dx = x[1] - x[0]
        return float(np.sum(np.array([phi(xi) for xi in x])) * dx)
    return TemperedDistribution(action)


def regular(f) -> TemperedDistribution:
    """Regular distribution from L^1_loc function: ⟨T_f, φ⟩ = ∫ f(x)φ(x) dx."""
    def action(phi):
        x = np.linspace(-20, 20, 4000)
        dx = x[1] - x[0]
        return float(np.sum(np.array([f(xi) * phi(xi) for xi in x])) * dx)
    return TemperedDistribution(action)


# ═══════════════════════════════════════════════════════════════
# Sobolev spaces
# ═══════════════════════════════════════════════════════════════

@dataclass
class SobolevNorm:
    """Sobolev norm results."""
    h0: float    # L2 norm
    h1: float    # H1 norm (includes first derivative)
    hs: float    # H^s norm
    s: float     # Sobolev index

    def to_dict(self) -> dict:
        return vars(self)


def sobolev_norm(
    f_values: np.ndarray,
    dx: float,
    s: float = 1.0,
) -> SobolevNorm:
    """Compute Sobolev H^s norm via FFT.

    ||f||_{H^s} = (∫ (1 + |ξ|²)^s |F[f](ξ)|² dξ)^{1/2}

    Args:
        f_values: function sampled on uniform grid.
        dx: grid spacing.
        s: Sobolev index (0 = L2, 1 = H1, etc.).
    """
    f = np.asarray(f_values)
    n = len(f)

    # L2 norm
    h0 = float(math.sqrt(np.sum(f ** 2) * dx))

    # H1 norm (includes gradient)
    grad = np.gradient(f, dx)
    h1 = float(math.sqrt(np.sum(f ** 2 + grad ** 2) * dx))

    # H^s norm via FFT
    F = np.fft.fft(f) * dx
    freqs = np.fft.fftfreq(n, d=dx) * 2 * math.pi
    weight = (1 + freqs ** 2) ** s
    hs = float(math.sqrt(np.sum(weight * np.abs(F) ** 2) / n))

    return SobolevNorm(h0, h1, hs, s)
