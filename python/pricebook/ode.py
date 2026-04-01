"""
ODE solvers for model dynamics.

    from pricebook.ode import rk4, rk45, ODEResult

    result = rk4(f=lambda t, y: -y, t_span=(0, 5), y0=[1.0], dt=0.01)
    print(result.t, result.y)  # t array, y array

    result = rk45(f=lambda t, y: -y, t_span=(0, 5), y0=[1.0], tol=1e-8)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ODEResult:
    """Result of an ODE solver."""

    t: np.ndarray
    y: np.ndarray
    n_evaluations: int
    method: str


def rk4(
    f,
    t_span: tuple[float, float],
    y0: list[float] | np.ndarray,
    dt: float = 0.01,
) -> ODEResult:
    """Classical 4th-order Runge-Kutta (fixed step).

    Args:
        f: dy/dt = f(t, y). y can be scalar or vector.
        t_span: (t0, t_end).
        y0: initial condition.
        dt: step size.
    """
    t0, t_end = t_span
    y = np.asarray(y0, dtype=float)
    t = t0
    n_eval = 0

    ts = [t]
    ys = [y.copy()]

    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        n_eval += 4

        ts.append(t)
        ys.append(y.copy())

    return ODEResult(
        t=np.array(ts),
        y=np.array(ys),
        n_evaluations=n_eval,
        method="rk4",
    )


def rk45(
    f,
    t_span: tuple[float, float],
    y0: list[float] | np.ndarray,
    tol: float = 1e-6,
    dt_init: float = 0.01,
    dt_min: float = 1e-12,
    dt_max: float = 1.0,
    max_steps: int = 100_000,
) -> ODEResult:
    """Adaptive Runge-Kutta-Fehlberg (4th/5th order pair).

    Dormand-Prince coefficients for embedded error estimation.
    Automatically adjusts step size to meet tolerance.
    """
    # Dormand-Prince coefficients
    a2, a3, a4, a5, a6 = 1/5, 3/10, 4/5, 8/9, 1.0
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 44/45, -56/15, 32/9
    b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

    # 5th order weights
    c1, c3, c4, c5, c6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

    # 4th order weights (for error)
    d1, d3, d4, d5, d6 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100
    d7 = 1/40

    t0, t_end = t_span
    y = np.asarray(y0, dtype=float)
    t = t0
    dt = dt_init
    n_eval = 0

    ts = [t]
    ys = [y.copy()]

    for _ in range(max_steps):
        if t >= t_end - 1e-12:
            break

        dt = min(dt, t_end - t)

        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + a2*dt, y + dt*b21*k1), dtype=float)
        k3 = np.asarray(f(t + a3*dt, y + dt*(b31*k1 + b32*k2)), dtype=float)
        k4 = np.asarray(f(t + a4*dt, y + dt*(b41*k1 + b42*k2 + b43*k3)), dtype=float)
        k5 = np.asarray(f(t + a5*dt, y + dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4)), dtype=float)
        k6 = np.asarray(f(t + a6*dt, y + dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5)), dtype=float)
        n_eval += 6

        # 5th order solution
        y_new = y + dt * (c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6)

        # 4th order solution (for error)
        k7 = np.asarray(f(t + dt, y_new), dtype=float)
        n_eval += 1
        y_4th = y + dt * (d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6 + d7*k7)

        # Error estimate
        err = np.max(np.abs(y_new - y_4th))
        if err < 1e-20:
            err = 1e-20

        # Accept or reject
        if err <= tol:
            t += dt
            y = y_new
            ts.append(t)
            ys.append(y.copy())

        # Adjust step size
        safety = 0.9
        factor = safety * (tol / err) ** 0.2
        factor = max(0.2, min(5.0, factor))
        dt = max(dt_min, min(dt_max, dt * factor))

    return ODEResult(
        t=np.array(ts),
        y=np.array(ys),
        n_evaluations=n_eval,
        method="rk45",
    )


def bdf(
    f,
    t_span: tuple[float, float],
    y0: list[float] | np.ndarray,
    tol: float = 1e-6,
    max_steps: int = 100_000,
) -> ODEResult:
    """BDF solver for stiff systems (wraps scipy)."""
    from scipy.integrate import solve_ivp

    y0 = np.asarray(y0, dtype=float)

    res = solve_ivp(
        f, t_span, y0, method="BDF",
        rtol=tol, atol=tol, max_step=max_steps,
    )

    return ODEResult(
        t=res.t,
        y=res.y.T,  # scipy returns (n_vars, n_points), we want (n_points, n_vars)
        n_evaluations=res.nfev,
        method="bdf",
    )
