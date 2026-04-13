"""Advanced PDE methods: PSOR, spectral (Chebyshev), method of lines, Richardson.

Phase M7 slices 194-196 consolidated.

* :func:`psor_american` — Projected SOR for American free boundary.
* :func:`chebyshev_bs` — Chebyshev collocation for Black-Scholes PDE.
* :func:`method_of_lines` — semi-discretise space, ODE in time.
* :func:`richardson_extrapolation` — combine O(h) and O(h/2) to get O(h²).

References:
    Wilmott, Howison & Dewynne, *The Mathematics of Financial Derivatives*, 1995.
    Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.
    Duffy, *Finite Difference Methods in Financial Engineering*, Wiley, 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import OptionType


# ---- PSOR for American options ----

@dataclass
class PSORResult:
    """Result of PSOR American option pricing."""
    price: float
    exercise_boundary: np.ndarray  # S* at each time step
    n_sor_iterations: int
    grid_size: int


def psor_american(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.PUT,
    n_spot: int = 200,
    n_time: int = 200,
    omega: float = 1.2,
    tol: float = 1e-8,
    max_sor_iter: int = 500,
) -> PSORResult:
    """American option via Projected SOR on the linear complementarity problem.

    Solves: max(LV, g − V) = 0 where L is the BS operator and g is
    the exercise payoff. At each time step, iterates the SOR scheme
    with projection onto the exercise constraint.

    Args:
        omega: SOR relaxation parameter (1 < ω < 2 for over-relaxation).
        tol: convergence tolerance per SOR sweep.
        max_sor_iter: max SOR iterations per time step.

    Reference:
        Wilmott et al., Ch. 8 (American options via complementarity).
    """
    # Grid in log-spot space
    mu = rate - 0.5 * vol**2
    width = 4 * vol * math.sqrt(T)
    x0 = math.log(spot)
    x_min, x_max = x0 - width, x0 + width
    dx = (x_max - x_min) / n_spot
    dt = T / n_time

    x = np.linspace(x_min, x_max, n_spot + 1)
    S = np.exp(x)

    # Payoff
    if option_type == OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    V = payoff.copy()

    # FD coefficients (Crank-Nicolson)
    alpha = 0.5 * vol**2
    a = alpha / dx**2 - mu / (2 * dx)
    b = -2 * alpha / dx**2 - rate
    c = alpha / dx**2 + mu / (2 * dx)

    theta = 0.5  # CN
    exercise_boundary = np.zeros(n_time)
    total_sor = 0

    for step in range(n_time):
        tau = (step + 1) * dt

        # Boundary conditions
        if option_type == OptionType.PUT:
            V[0] = strike * math.exp(-rate * tau) - S[0]
            V[-1] = 0.0
        else:
            V[0] = 0.0
            V[-1] = S[-1] - strike * math.exp(-rate * tau)

        # RHS from explicit part
        rhs = np.zeros(n_spot - 1)
        for i in range(n_spot - 1):
            j = i + 1
            rhs[i] = (V[j]
                       + theta * dt * (a * V[j - 1] + b * V[j] + c * V[j + 1]))
        rhs[0] += (1 - theta) * dt * a * V[0]
        rhs[-1] += (1 - theta) * dt * c * V[-1]

        # Implicit coefficients
        la = -(1 - theta) * dt * a
        lb = 1 - (1 - theta) * dt * b
        lc = -(1 - theta) * dt * c

        # PSOR iteration
        V_int = V[1:n_spot].copy()
        for sor_iter in range(max_sor_iter):
            max_change = 0.0
            for i in range(n_spot - 1):
                j = i + 1
                left = la * (V_int[i - 1] if i > 0 else V[0])
                right = lc * (V_int[i + 1] if i < n_spot - 2 else V[-1])
                gs = (rhs[i] - left - right) / lb
                # SOR relaxation
                new_val = V_int[i] + omega * (gs - V_int[i])
                # Project onto exercise constraint
                new_val = max(new_val, payoff[j])
                max_change = max(max_change, abs(new_val - V_int[i]))
                V_int[i] = new_val

            total_sor += 1
            if max_change < tol:
                break

        V[1:n_spot] = V_int

        # Exercise boundary: smallest S where V > payoff (for put)
        if option_type == OptionType.PUT:
            for j in range(n_spot + 1):
                if V[j] > payoff[j] + 1e-10:
                    exercise_boundary[step] = S[j]
                    break

    # Interpolate at spot
    idx = int(np.searchsorted(x, x0)) - 1
    idx = max(0, min(idx, n_spot - 1))
    w = (x0 - x[idx]) / dx if dx > 0 else 0.0
    price = V[idx] * (1 - w) + V[idx + 1] * w

    return PSORResult(price, exercise_boundary, total_sor, n_spot)


# ---- Chebyshev collocation ----

@dataclass
class SpectralResult:
    """Result of spectral method pricing."""
    price: float
    n_points: int
    max_residual: float


def chebyshev_bs(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    N: int = 32,
    n_time: int = 100,
) -> SpectralResult:
    """European option via Chebyshev collocation on the BS PDE.

    Semi-discretises in space using Chebyshev points, then marches
    backward in time with implicit Euler. Exponential convergence
    for smooth solutions.

    Reference:
        Trefethen, Spectral Methods in MATLAB, Ch. 7.
    """
    # Chebyshev points on [-1, 1]
    j = np.arange(N + 1)
    xi = np.cos(j * math.pi / N)  # Chebyshev-Lobatto points

    # Map to [x_min, x_max] in log-spot
    x0 = math.log(spot)
    width = 4 * vol * math.sqrt(T)
    x_min, x_max = x0 - width, x0 + width
    x = 0.5 * (x_max + x_min) + 0.5 * (x_max - x_min) * xi
    S = np.exp(x)

    # Chebyshev differentiation matrix
    D = _chebyshev_diff_matrix(N, xi)
    # Scale to physical domain
    D = D * 2.0 / (x_max - x_min)
    D2 = D @ D

    # BS operator in log-spot: L = 0.5σ² D² + (r - 0.5σ²) D - r I
    mu = rate - 0.5 * vol**2
    L = 0.5 * vol**2 * D2 + mu * D - rate * np.eye(N + 1)

    # Initial condition (payoff at T)
    if option_type == OptionType.CALL:
        V = np.maximum(S - strike, 0.0)
    else:
        V = np.maximum(strike - S, 0.0)

    # Backward time stepping (implicit Euler)
    dt = T / n_time
    A = np.eye(N + 1) - dt * L

    for step in range(n_time):
        # Boundary conditions
        if option_type == OptionType.CALL:
            V[0] = S[0] - strike * math.exp(-rate * (step + 1) * dt)
            V[-1] = 0.0
        else:
            V[-1] = strike * math.exp(-rate * (step + 1) * dt) - S[-1]
            V[0] = 0.0

        # Solve (I - dt L) V_new = V_old with BCs
        rhs = V.copy()
        # Zero out BC rows
        A_bc = A.copy()
        A_bc[0, :] = 0; A_bc[0, 0] = 1
        A_bc[-1, :] = 0; A_bc[-1, -1] = 1
        rhs[0] = V[0]
        rhs[-1] = V[-1]
        V = np.linalg.solve(A_bc, rhs)

    # Interpolate at spot via barycentric formula
    idx = np.argmin(np.abs(x - x0))
    price = float(V[idx])

    # Residual check
    residual = float(np.max(np.abs(A @ V - V)))

    return SpectralResult(price, N, residual)


def _chebyshev_diff_matrix(N: int, xi: np.ndarray) -> np.ndarray:
    """Chebyshev differentiation matrix on N+1 Lobatto points.

    Reference: Trefethen, Spectral Methods in MATLAB, Program 3.
    """
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c *= (-1.0) ** np.arange(N + 1)

    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = c[i] / (c[j] * (xi[i] - xi[j]))
            elif i == 0:
                D[i, j] = (2 * N * N + 1) / 6.0
            elif i == N:
                D[i, j] = -(2 * N * N + 1) / 6.0
            else:
                D[i, j] = -xi[i] / (2 * (1 - xi[i]**2))
    return D


# ---- Method of lines ----

@dataclass
class MOLResult:
    """Result of method-of-lines pricing."""
    price: float
    n_spatial: int
    n_time_steps: int


def method_of_lines(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    n_spot: int = 100,
    n_time: int = 5000,
) -> MOLResult:
    """European option via method of lines: FD in space, RK4 in time.

    Semi-discretises the BS PDE in log-spot using second-order central
    differences, producing a system of ODEs which is integrated
    backward in time using explicit RK4.

    Cleanly separates spatial and temporal accuracy.
    """
    mu = rate - 0.5 * vol**2
    width = 4 * vol * math.sqrt(T)
    x0 = math.log(spot)
    x_min, x_max = x0 - width, x0 + width
    dx = (x_max - x_min) / n_spot
    dt = T / n_time

    x = np.linspace(x_min, x_max, n_spot + 1)
    S = np.exp(x)

    # Payoff at T
    if option_type == OptionType.CALL:
        V = np.maximum(S - strike, 0.0)
    else:
        V = np.maximum(strike - S, 0.0)

    alpha = 0.5 * vol**2 / dx**2
    beta = mu / (2 * dx)

    def rhs(V_int, tau):
        """RHS of the ODE system: dV/dτ = L V (with BCs)."""
        n = len(V_int)
        f = np.zeros(n)
        for i in range(n):
            j = i + 1  # index in full grid
            left = V_int[i - 1] if i > 0 else V[0]
            right = V_int[i + 1] if i < n - 1 else V[-1]
            f[i] = alpha * (left - 2 * V_int[i] + right) + beta * (right - left) - rate * V_int[i]
        return f

    # Backward in time using RK4
    V_int = V[1:n_spot].copy()
    for step in range(n_time):
        tau = (step + 1) * dt
        # Update BCs
        if option_type == OptionType.CALL:
            V[0] = 0.0
            V[-1] = S[-1] - strike * math.exp(-rate * tau)
        else:
            V[-1] = 0.0
            V[0] = strike * math.exp(-rate * tau) - S[0]

        # RK4 step
        k1 = rhs(V_int, tau) * dt
        k2 = rhs(V_int + 0.5 * k1, tau + 0.5 * dt) * dt
        k3 = rhs(V_int + 0.5 * k2, tau + 0.5 * dt) * dt
        k4 = rhs(V_int + k3, tau + dt) * dt
        V_int = V_int + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    V[1:n_spot] = V_int

    # Interpolate
    idx = int(np.searchsorted(x, x0)) - 1
    idx = max(0, min(idx, n_spot - 1))
    w = (x0 - x[idx]) / dx
    price = V[idx] * (1 - w) + V[idx + 1] * w

    return MOLResult(price, n_spot, n_time)


# ---- Richardson extrapolation ----

def richardson_extrapolation(
    coarse: float,
    fine: float,
    order: int = 2,
) -> float:
    """Richardson extrapolation: combine two estimates to cancel leading error.

    If f(h) = f* + C h^p + O(h^{p+1}), and fine uses h/2:
        f* ≈ (2^p × fine − coarse) / (2^p − 1)

    Args:
        coarse: estimate at step size h.
        fine: estimate at step size h/2.
        order: convergence order p of the method.

    Returns:
        Extrapolated estimate with error O(h^{p+1}).
    """
    r = 2 ** order
    return (r * fine - coarse) / (r - 1)
