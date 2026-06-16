"""Rough Heston characteristic function via fractional Riccati ODE.

The rough Heston model replaces the standard Heston variance process
with a fractional kernel, giving power-law decay of the ATM skew.

* :func:`rough_heston_char_func` — CF via Adams scheme on fractional Riccati.
* :func:`rough_heston_price` — European option via COS + rough Heston CF.

References:
    El Euch & Rosenbaum, *The Characteristic Function of Rough Heston
    Models*, Mathematical Finance, 2019.
    Gatheral, Jaisson & Rosenbaum, *Volatility is Rough*, QF, 2018.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as gamma_fn


@dataclass
class RoughHestonParams:
    """Rough Heston model parameters."""
    v0: float       # initial variance
    kappa: float    # mean reversion (of the kernel, not rate)
    theta: float    # long-run variance
    xi: float       # vol-of-vol (ν)
    rho: float      # spot-variance correlation
    H: float        # Hurst parameter (0 < H < 0.5 for rough)

    def to_dict(self) -> dict:
        return dict(vars(self))


def rough_heston_char_func(
    params: RoughHestonParams,
    rate: float,
    T: float,
    div_yield: float = 0.0,
    n_steps: int = 200,
):
    """Characteristic function of the rough Heston model.

    The log-price CF satisfies:
    φ(u) = exp(iu(log S₀ + (r-q)T) + θ ∫₀ᵀ h(T-s) ds + v₀ I^α h(0⁺))

    where h solves the fractional Riccati ODE:
    D^α h(t) = F(h(t))
    F(h) = ½(u² + iu) + (iuρξ − κ)h + ½ξ²h²

    D^α is the Caputo fractional derivative of order α = H + 0.5.
    Solved via Adams scheme (predictor-corrector).

    Args:
        params: rough Heston parameters.
        rate: risk-free rate.
        T: time to expiry.
        n_steps: Adams scheme time steps.

    Returns:
        Callable φ(u) → complex.
    """
    H = params.H
    alpha = H + 0.5  # fractional order
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho
    v0 = params.v0

    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)

    # Precompute kernel weights for Adams scheme
    # K(t) = t^{α-1} / Γ(α)
    gamma_alpha = gamma_fn(alpha)

    def _solve_fractional_riccati(u: complex) -> complex:
        """Solve fractional Riccati for a given u."""
        # F(h) = 0.5*(u² + iu) + (iu*rho*xi - kappa)*h + 0.5*xi²*h²
        a_coeff = 0.5 * xi**2
        b_coeff = 1j * u * rho * xi - kappa
        c_coeff = 0.5 * (-(u**2) + 1j * u)

        def F(h):
            return c_coeff + b_coeff * h + a_coeff * h**2

        # Adams-Bashforth-Moulton for fractional ODE
        # h(t_n) = (1/Γ(α)) Σ_{j=0}^{n-1} ∫_{t_j}^{t_{j+1}} (t_n - s)^{α-1} F(h(s)) ds
        h = np.zeros(n_steps + 1, dtype=complex)
        h[0] = 0.0  # initial condition

        for n in range(1, n_steps + 1):
            # Predictor (Adams-Bashforth): explicit
            h_pred = 0.0 + 0j
            for j in range(n):
                # Weight: ∫_{t_j}^{t_{j+1}} (t_n - s)^{α-1} ds / Γ(α)
                w = ((n - j) * dt)**alpha - ((n - j - 1) * dt)**alpha
                w /= (alpha * gamma_alpha)
                h_pred += w * F(h[j])

            # Corrector (Adams-Moulton): implicit (one step)
            w_0 = dt**alpha / (alpha * gamma_alpha)
            # h[n] = h_pred_without_n + w_0 * F(h[n])
            # Solve: h = h_pred + w_0 * (c + b*h + a*h²)
            # Quadratic: a*w_0*h² + (b*w_0 - 1)*h + (h_pred + c*w_0) = 0
            A = a_coeff * w_0
            B = b_coeff * w_0 - 1.0
            C = h_pred + c_coeff * w_0

            discriminant = B**2 - 4 * A * C
            sqrt_disc = discriminant**0.5

            # Choose root with smaller magnitude (stability)
            h1 = (-B + sqrt_disc) / (2 * A) if abs(A) > 1e-15 else -C / B
            h2 = (-B - sqrt_disc) / (2 * A) if abs(A) > 1e-15 else h1

            h[n] = h1 if abs(h1) < abs(h2) else h2

        # Compute integrals for the CF
        # ∫₀ᵀ h(T-s) ds via trapezoidal
        integral_h = float(np.sum(0.5 * (h[:-1] + h[1:]) * np.diff(t_grid)))

        # I^α h(0⁺): fractional integral at origin
        # Approximation: h[1] * dt^α / Γ(α+1)
        I_alpha_h0 = h[-1]  # value at T (simplified)

        # CF: exp(iu*(log-price drift) + θ * integral_h + v0 * I_alpha_h0)
        log_cf = 1j * u * (rate - div_yield) * T + theta * integral_h + v0 * I_alpha_h0
        return log_cf

    def char_func(u):
        """Characteristic function φ(u)."""
        if np.isscalar(u):
            return np.exp(_solve_fractional_riccati(complex(u)))
        return np.array([np.exp(_solve_fractional_riccati(complex(ui))) for ui in u])

    return char_func


def rough_heston_price(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    params: RoughHestonParams,
    is_call: bool = True,
    div_yield: float = 0.0,
    N: int = 128,
    n_ode_steps: int = 200,
) -> float:
    """Price European option under rough Heston via COS method.

    Args:
        params: rough Heston parameters (H < 0.5 for rough regime).
        N: COS method terms.
        n_ode_steps: Adams scheme steps for fractional Riccati.
    """
    from pricebook.models.cos_method import cos_price
    from pricebook.models.black76 import OptionType

    cf = rough_heston_char_func(params, rate, T, div_yield, n_ode_steps)
    otype = OptionType.CALL if is_call else OptionType.PUT
    return cos_price(cf, spot, strike, rate, T, otype, div_yield, N)
