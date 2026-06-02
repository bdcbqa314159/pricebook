"""Jump-diffusion PIDE solver: Merton, Kou, general Lévy.

Integro-PDE: diffusion (FD) + jump integral (quadrature).
Operator splitting: diffusion step + jump step.

* :func:`merton_pide` — Merton jump-diffusion via operator splitting.
* :func:`kou_pide` — Kou double-exponential jump-diffusion.

References:
    Cont & Voltchkova, *A Finite Difference Scheme for Option Pricing
    in Jump Diffusion and Exponential Lévy Models*, SIAM JNA, 2005.
    d'Halluin, Forsyth & Vetzal, *Robust Numerical Methods for Contingent
    Claims Under Jump Diffusion Processes*, JCAM, 2005.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.pde_protocol import PDEPricingResult, PDEConvergenceInfo
from pricebook.numerical._pde import build_grid, GridType, extract_greeks


@dataclass
class PIDEResult:
    """PIDE pricing result."""
    price: float
    delta: float
    gamma: float
    theta: float
    jump_contribution: float    # fraction of price from jumps

    def to_dict(self) -> dict:
        return vars(self)


def merton_pide(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    jump_intensity: float = 0.5,
    jump_mean: float = -0.05,
    jump_vol: float = 0.10,
    is_call: bool = True,
    n_space: int = 200,
    n_time: int = 200,
) -> PIDEResult:
    """Merton jump-diffusion via operator splitting.

    PDE part: standard BS with adjusted drift.
    Jump integral: J[V](x) = λ∫ [V(x+y) − V(x)] f(y) dy
    where f(y) = lognormal density of jump sizes.

    Operator splitting each time step:
    1. Diffusion half-step (Crank-Nicolson on BS operator)
    2. Jump full-step (explicit quadrature of integral)

    Args:
        jump_intensity: λ, expected jumps per year.
        jump_mean: mean of log-jump size (μ_J).
        jump_vol: std of log-jump size (σ_J).
    """
    # Compensated drift: μ* = r − q − λ(e^{μ_J + σ_J²/2} − 1)
    kappa = math.exp(jump_mean + 0.5 * jump_vol**2) - 1
    drift_adj = rate - jump_intensity * kappa

    # Grid in log-spot
    x0 = math.log(spot)
    width = max(4 * vol * math.sqrt(T), 2.0)
    x_min = x0 - width
    x_max = x0 + width
    x = np.linspace(x_min, x_max, n_space)
    dx = x[1] - x[0]
    S = np.exp(x)
    dt = T / n_time

    # Terminal condition
    if is_call:
        V = np.maximum(S - strike, 0)
    else:
        V = np.maximum(strike - S, 0)

    # Diffusion coefficients (in log-space)
    alpha = 0.5 * vol**2
    beta = drift_adj - 0.5 * vol**2
    a_coeff = alpha / dx**2 - beta / (2 * dx)
    b_coeff = -2 * alpha / dx**2 - rate
    c_coeff = alpha / dx**2 + beta / (2 * dx)

    # Jump integral kernel: precompute weights
    # f(y) = (1/σ_J√2π) exp(−(y−μ_J)²/(2σ_J²))
    n_jump = min(n_space, 100)  # quadrature points for jump
    y_max = 4 * jump_vol + abs(jump_mean)
    y = np.linspace(-y_max, y_max, n_jump)
    dy = y[1] - y[0]
    jump_density = np.exp(-0.5 * ((y - jump_mean) / jump_vol)**2) / (jump_vol * math.sqrt(2 * math.pi))
    jump_weights = jump_density * dy * jump_intensity

    V_prev = V.copy()

    for step in range(n_time):
        # Step 1: Diffusion half-step (Crank-Nicolson)
        rhs = V.copy()
        for i in range(1, n_space - 1):
            rhs[i] = V[i] + 0.5 * dt * (a_coeff * V[i - 1] + b_coeff * V[i] + c_coeff * V[i + 1])

        # Solve implicit part: (I − 0.5 dt L) V_new = rhs
        lower = np.full(n_space - 2, -0.5 * dt * a_coeff)
        diag = np.full(n_space - 2, 1 - 0.5 * dt * b_coeff)
        upper = np.full(n_space - 2, -0.5 * dt * c_coeff)
        V_diff = _thomas(lower, diag, upper, rhs[1:-1])

        V_new = V.copy()
        V_new[1:-1] = V_diff

        # Step 2: Jump integral (explicit)
        V_jump = V_new.copy()
        for i in range(1, n_space - 1):
            integral = 0.0
            for k in range(n_jump):
                # V(x + y_k) via interpolation
                x_shifted = x[i] + y[k]
                v_shifted = float(np.interp(x_shifted, x, V_new))
                integral += jump_weights[k] * (v_shifted - V_new[i])
            V_jump[i] = V_new[i] + dt * integral

        # Boundary conditions
        if is_call:
            V_jump[0] = 0
            V_jump[-1] = S[-1] - strike * math.exp(-rate * (step + 1) * dt)
        else:
            V_jump[0] = strike * math.exp(-rate * (step + 1) * dt) - S[0]
            V_jump[-1] = 0

        V_prev = V.copy()
        V = V_jump

    greeks = extract_greeks(S, V, spot, V_prev, dt)

    # Jump contribution estimate
    # Price without jumps
    from pricebook.models.pde_protocol import pde_price
    no_jump = pde_price(spot, strike, vol, rate, T, is_call, n_space=n_space, n_time=n_time)
    jump_frac = (greeks["price"] - no_jump.price) / greeks["price"] if abs(greeks["price"]) > 1e-10 else 0

    return PIDEResult(
        price=greeks["price"],
        delta=greeks["delta"],
        gamma=greeks["gamma"],
        theta=greeks["theta"],
        jump_contribution=jump_frac,
    )


def kou_pide(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    jump_intensity: float = 0.5,
    p_up: float = 0.4,
    eta_up: float = 10.0,
    eta_down: float = 5.0,
    is_call: bool = True,
    n_space: int = 200,
    n_time: int = 200,
) -> PIDEResult:
    """Kou double-exponential jump-diffusion.

    Jump sizes: f(y) = p × η₊ exp(−η₊y) 1_{y>0} + (1−p) × η₋ exp(η₋y) 1_{y<0}.

    The double-exponential has semi-analytical Fourier transform,
    but here we use the same operator-splitting PDE approach.

    Args:
        p_up: probability of upward jump.
        eta_up: rate of upward exponential (1/mean_up).
        eta_down: rate of downward exponential (1/mean_down).
    """
    # Compensated drift
    kappa = p_up * eta_up / (eta_up - 1) + (1 - p_up) * eta_down / (eta_down + 1) - 1
    drift_adj = rate - jump_intensity * kappa

    x0 = math.log(spot)
    width = max(4 * vol * math.sqrt(T), 2.0)
    x = np.linspace(x0 - width, x0 + width, n_space)
    dx = x[1] - x[0]
    S = np.exp(x)
    dt = T / n_time

    if is_call:
        V = np.maximum(S - strike, 0)
    else:
        V = np.maximum(strike - S, 0)

    alpha = 0.5 * vol**2
    beta = drift_adj - 0.5 * vol**2
    a_c = alpha / dx**2 - beta / (2 * dx)
    b_c = -2 * alpha / dx**2 - rate
    c_c = alpha / dx**2 + beta / (2 * dx)

    # Kou jump kernel
    n_jump = 80
    y_max = 3.0
    y = np.linspace(-y_max, y_max, n_jump)
    dy_j = y[1] - y[0]
    kou_density = np.where(
        y > 0,
        p_up * eta_up * np.exp(-eta_up * y),
        (1 - p_up) * eta_down * np.exp(eta_down * y),
    )
    jump_weights = kou_density * dy_j * jump_intensity

    V_prev = V.copy()

    for step in range(n_time):
        # Diffusion CN step
        rhs = V.copy()
        for i in range(1, n_space - 1):
            rhs[i] = V[i] + 0.5 * dt * (a_c * V[i-1] + b_c * V[i] + c_c * V[i+1])

        lower = np.full(n_space - 2, -0.5 * dt * a_c)
        diag = np.full(n_space - 2, 1 - 0.5 * dt * b_c)
        upper = np.full(n_space - 2, -0.5 * dt * c_c)
        V_new = V.copy()
        V_new[1:-1] = _thomas(lower, diag, upper, rhs[1:-1])

        # Jump step
        for i in range(1, n_space - 1):
            integral = 0.0
            for k in range(n_jump):
                x_shifted = x[i] + y[k]
                v_shifted = float(np.interp(x_shifted, x, V_new))
                integral += jump_weights[k] * (v_shifted - V_new[i])
            V_new[i] += dt * integral

        if is_call:
            V_new[0] = 0
            V_new[-1] = S[-1] - strike * math.exp(-rate * (step + 1) * dt)
        else:
            V_new[0] = strike * math.exp(-rate * (step + 1) * dt) - S[0]
            V_new[-1] = 0

        V_prev = V.copy()
        V = V_new

    greeks = extract_greeks(S, V, spot, V_prev, dt)
    return PIDEResult(greeks["price"], greeks["delta"], greeks["gamma"], greeks["theta"], 0.0)


def _thomas(lower, diag, upper, rhs):
    """Thomas algorithm."""
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)
    c[0] = upper[0] / diag[0] if abs(diag[0]) > 1e-15 else 0
    d[0] = rhs[0] / diag[0] if abs(diag[0]) > 1e-15 else 0
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        if abs(denom) < 1e-15:
            c[i] = 0
            d[i] = 0
        else:
            c[i] = upper[i] / denom if i < n - 1 else 0
            d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x
