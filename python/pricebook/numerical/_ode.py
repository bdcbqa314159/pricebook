"""ODE integrators: RK45 adaptive, BDF stiff, Adams multi-step.

    from pricebook.numerical import rk45, bdf, euler

Wraps scipy.integrate behind a clean interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ODEResult:
    """ODE solution result."""
    t: np.ndarray           # time points
    y: np.ndarray           # solution at each time point (n_times, n_states)
    success: bool
    n_evaluations: int
    method: str

    def to_dict(self) -> dict:
        return {"method": self.method, "n_points": len(self.t),
                "success": self.success, "n_evaluations": self.n_evaluations}


def euler(
    f,
    t_span: tuple[float, float],
    y0: np.ndarray | float,
    n_steps: int = 100,
) -> ODEResult:
    """Forward Euler: y_{n+1} = y_n + h f(t_n, y_n).

    First-order, explicit. Simple but requires small step for stability.
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        y[i + 1] = y[i] + h * np.asarray(f(t[i], y[i]))

    return ODEResult(t, y, True, n_steps, "euler")


def rk4(
    f,
    t_span: tuple[float, float],
    y0: np.ndarray | float,
    n_steps: int = 100,
) -> ODEResult:
    """Classical Runge-Kutta 4th order (fixed step).

    Fourth-order, explicit. Good accuracy for smooth problems.
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        k1 = h * np.asarray(f(t[i], y[i]))
        k2 = h * np.asarray(f(t[i] + 0.5 * h, y[i] + 0.5 * k1))
        k3 = h * np.asarray(f(t[i] + 0.5 * h, y[i] + 0.5 * k2))
        k4 = h * np.asarray(f(t[i] + h, y[i] + k3))
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return ODEResult(t, y, True, 4 * n_steps, "rk4")


def rk45(
    f,
    t_span: tuple[float, float],
    y0: np.ndarray | float,
    tol: float = 1e-6,
    max_step: float | None = None,
) -> ODEResult:
    """Runge-Kutta-Fehlberg 4(5) with adaptive step size.

    Embedded pair: 4th-order solution + 5th-order error estimate.
    Automatically adjusts step size to meet tolerance.
    """
    from scipy.integrate import solve_ivp

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    result = solve_ivp(f, t_span, y0, method="RK45", rtol=tol, atol=tol,
                       max_step=max_step or np.inf, dense_output=False)

    return ODEResult(
        t=result.t,
        y=result.y.T,  # scipy returns (n_states, n_times) → transpose
        success=result.success,
        n_evaluations=result.nfev,
        method="rk45",
    )


def bdf(
    f,
    t_span: tuple[float, float],
    y0: np.ndarray | float,
    jac=None,
    tol: float = 1e-6,
) -> ODEResult:
    """Backward Differentiation Formula for stiff systems.

    Implicit multi-step method. Good for stiff ODEs (e.g., chemical kinetics,
    circuit simulation, some rate models).

    Args:
        jac: Jacobian function (optional). If None, computed numerically.
    """
    from scipy.integrate import solve_ivp

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    result = solve_ivp(f, t_span, y0, method="BDF", jac=jac,
                       rtol=tol, atol=tol, dense_output=False)

    return ODEResult(
        t=result.t,
        y=result.y.T,
        success=result.success,
        n_evaluations=result.nfev,
        method="bdf",
    )


def adams(
    f,
    t_span: tuple[float, float],
    y0: np.ndarray | float,
    tol: float = 1e-6,
) -> ODEResult:
    """Adams-Bashforth-Moulton predictor-corrector (LSODA).

    Non-stiff multi-step method. Automatically switches to BDF if stiffness detected.
    """
    from scipy.integrate import solve_ivp

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    result = solve_ivp(f, t_span, y0, method="LSODA",
                       rtol=tol, atol=tol, dense_output=False)

    return ODEResult(
        t=result.t,
        y=result.y.T,
        success=result.success,
        n_evaluations=result.nfev,
        method="adams",
    )
