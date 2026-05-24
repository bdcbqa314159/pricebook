"""ODE solvers — flexible, class-based, with runtime method selection.

Supports explicit (Euler, RK4, RK45), implicit (BDF, Radau), and adaptive
(LSODA) methods. Jacobian callbacks, event detection, dense output,
backward integration, and financial-specific helpers.

    from pricebook.numerical._ode import (
        ODESolver, ODEMethod, ODEResult,
        solve_ode, solve_riccati, solve_backward,
    )

    # Simple: solve with auto method selection
    result = solve_ode(f, (0, 10), y0)

    # Explicit method choice
    result = solve_ode(f, (0, 10), y0, method=ODEMethod.BDF, jac=jac_fn)

    # Class-based for reuse
    solver = ODESolver(method=ODEMethod.RK45, tol=1e-8)
    result = solver.solve(f, (0, 10), y0)

References:
    Hairer, Norsett & Wanner (1993). Solving Ordinary Differential Equations I.
    Hairer & Wanner (1996). Solving Ordinary Differential Equations II (Stiff).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ODEMethod(Enum):
    """Available ODE integration methods."""
    EULER = "euler"              # Forward Euler, O(h), explicit
    RK4 = "rk4"                 # Classical RK4, O(h⁴), explicit, fixed step
    RK45 = "rk45"               # Dormand-Prince 4(5), adaptive, explicit
    RK23 = "rk23"               # Bogacki-Shampine 2(3), adaptive, explicit
    BDF = "bdf"                 # Backward differentiation, implicit, stiff
    RADAU = "radau"             # Radau IIA, implicit, stiff, L-stable
    LSODA = "lsoda"             # Auto-switching explicit/implicit (Adams/BDF)
    DOP853 = "dop853"           # Dormand-Prince 8(5,3), high-order adaptive
    IMPLICIT_EULER = "implicit_euler"  # Backward Euler, O(h), implicit


@dataclass
class ODEResult:
    """ODE solution result."""
    t: np.ndarray               # (N,) time points
    y: np.ndarray               # (N, D) solution values
    success: bool
    n_evaluations: int
    method: str
    dense_output: object | None = None  # scipy OdeSolution for interpolation
    events: list | None = None   # detected events
    message: str = ""

    def __call__(self, t_query: float | np.ndarray) -> np.ndarray:
        """Evaluate solution at arbitrary time(s) via dense output."""
        if self.dense_output is not None:
            return self.dense_output(t_query).T
        # Fallback: linear interpolation
        t_q = np.atleast_1d(t_query)
        result = np.zeros((len(t_q), self.y.shape[1]))
        for j in range(self.y.shape[1]):
            result[:, j] = np.interp(t_q, self.t, self.y[:, j])
        return result.squeeze()

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_points": len(self.t),
            "success": self.success,
            "n_evaluations": self.n_evaluations,
            "message": self.message,
        }


class ODESolver:
    """Configurable ODE solver with method selection and options.

    Args:
        method: integration method (from ODEMethod enum).
        tol: tolerance (used as both rtol and atol unless overridden).
        rtol: relative tolerance (overrides tol).
        atol: absolute tolerance (overrides tol).
        max_step: maximum step size.
        n_steps: number of steps for fixed-step methods (Euler, RK4).
        dense_output: if True, return interpolant for arbitrary evaluation.
    """

    def __init__(
        self,
        method: ODEMethod = ODEMethod.RK45,
        tol: float = 1e-6,
        rtol: float | None = None,
        atol: float | None = None,
        max_step: float | None = None,
        n_steps: int = 100,
        dense_output: bool = False,
    ):
        self.method = method
        self.rtol = rtol or tol
        self.atol = atol or tol
        self.max_step = max_step
        self.n_steps = n_steps
        self.dense_output = dense_output

    def solve(
        self,
        f: callable,
        t_span: tuple[float, float],
        y0: np.ndarray | float | list,
        jac: callable | None = None,
        t_eval: np.ndarray | None = None,
        events: callable | list[callable] | None = None,
    ) -> ODEResult:
        """Solve the ODE dy/dt = f(t, y).

        Args:
            f: right-hand side function f(t, y) → dy/dt.
            t_span: (t_start, t_end). t_end < t_start for backward integration.
            y0: initial condition (scalar, array, or list).
            jac: Jacobian df/dy (for implicit methods). Optional.
            t_eval: specific times to evaluate at.
            events: event functions for detection (event(t, y) = 0 triggers).

        Returns:
            ODEResult with solution arrays and metadata.
        """
        y0 = np.atleast_1d(np.asarray(y0, dtype=float))

        if self.method in (ODEMethod.EULER, ODEMethod.RK4, ODEMethod.IMPLICIT_EULER):
            return self._solve_fixed(f, t_span, y0, jac)
        else:
            return self._solve_scipy(f, t_span, y0, jac, t_eval, events)

    def _solve_fixed(self, f, t_span, y0, jac):
        """Fixed-step methods (Euler, RK4, Implicit Euler)."""
        t = np.linspace(t_span[0], t_span[1], self.n_steps + 1)
        h = t[1] - t[0]
        y = np.zeros((self.n_steps + 1, len(y0)))
        y[0] = y0
        n_evals = 0

        for i in range(self.n_steps):
            if self.method == ODEMethod.EULER:
                y[i + 1] = y[i] + h * np.asarray(f(t[i], y[i]))
                n_evals += 1
            elif self.method == ODEMethod.RK4:
                k1 = h * np.asarray(f(t[i], y[i]))
                k2 = h * np.asarray(f(t[i] + 0.5 * h, y[i] + 0.5 * k1))
                k3 = h * np.asarray(f(t[i] + 0.5 * h, y[i] + 0.5 * k2))
                k4 = h * np.asarray(f(t[i] + h, y[i] + k3))
                y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                n_evals += 4
            elif self.method == ODEMethod.IMPLICIT_EULER:
                y[i + 1] = self._implicit_euler_step(f, t[i], y[i], h, jac)
                n_evals += 3  # approximate (Newton iterations)

        return ODEResult(t, y, True, n_evals, self.method.value)

    def _solve_scipy(self, f, t_span, y0, jac, t_eval, events):
        """Adaptive methods via scipy.integrate.solve_ivp."""
        from scipy.integrate import solve_ivp

        method_map = {
            ODEMethod.RK45: "RK45",
            ODEMethod.RK23: "RK23",
            ODEMethod.BDF: "BDF",
            ODEMethod.RADAU: "Radau",
            ODEMethod.LSODA: "LSODA",
            ODEMethod.DOP853: "DOP853",
        }

        scipy_method = method_map[self.method]
        kwargs = {
            "method": scipy_method,
            "rtol": self.rtol,
            "atol": self.atol,
            "dense_output": self.dense_output,
        }
        if self.max_step is not None:
            kwargs["max_step"] = self.max_step
        if jac is not None and self.method in (ODEMethod.BDF, ODEMethod.RADAU):
            kwargs["jac"] = jac
        if t_eval is not None:
            kwargs["t_eval"] = t_eval
        if events is not None:
            if callable(events):
                events = [events]
            kwargs["events"] = events

        result = solve_ivp(f, t_span, y0, **kwargs)

        dense = result.sol if self.dense_output and hasattr(result, "sol") else None
        evt = None
        if events is not None and hasattr(result, "t_events"):
            evt = result.t_events

        return ODEResult(
            t=result.t,
            y=result.y.T,
            success=result.success,
            n_evaluations=result.nfev,
            method=self.method.value,
            dense_output=dense,
            events=evt,
            message=result.message if hasattr(result, "message") else "",
        )

    @staticmethod
    def _implicit_euler_step(f, t, y, h, jac, max_iter=10, tol=1e-10):
        """Single implicit Euler step via Newton iteration.

        Solve: y_{n+1} = y_n + h × f(t_{n+1}, y_{n+1})
        Newton: g(z) = z - y_n - h × f(t+h, z) = 0
        """
        z = y + h * np.asarray(f(t, y))  # initial guess (explicit Euler)
        for _ in range(max_iter):
            g = z - y - h * np.asarray(f(t + h, z))
            if np.linalg.norm(g) < tol:
                break
            if jac is not None:
                J = np.eye(len(y)) - h * np.asarray(jac(t + h, z))
                try:
                    dz = np.linalg.solve(J, -g)
                except np.linalg.LinAlgError:
                    dz = -g
            else:
                dz = -g  # simplified: no Jacobian → fixed-point iteration
            z = z + dz
        return z


# ═══════════════════════════════════════════════════════════════
# Convenience functions (backward compatible + new)
# ═══════════════════════════════════════════════════════════════


def solve_ode(
    f: callable,
    t_span: tuple[float, float],
    y0: np.ndarray | float,
    method: ODEMethod = ODEMethod.RK45,
    jac: callable | None = None,
    tol: float = 1e-6,
    t_eval: np.ndarray | None = None,
    events: callable | list[callable] | None = None,
    dense_output: bool = False,
    n_steps: int = 100,
    max_step: float | None = None,
) -> ODEResult:
    """Solve an ODE with flexible method selection.

    This is the main entry point. Picks the right solver based on method.
    """
    solver = ODESolver(method, tol=tol, n_steps=n_steps,
                        dense_output=dense_output, max_step=max_step)
    return solver.solve(f, t_span, y0, jac, t_eval, events)


def solve_backward(
    f: callable,
    t_span: tuple[float, float],
    y_terminal: np.ndarray | float,
    method: ODEMethod = ODEMethod.BDF,
    jac: callable | None = None,
    tol: float = 1e-6,
) -> ODEResult:
    """Solve an ODE backward in time (for PDE time-stepping).

    dy/dt = f(t, y) integrated from t_span[1] back to t_span[0].
    The terminal condition y(T) = y_terminal.
    """
    # Reverse time: integrate from T to 0
    t_start, t_end = t_span
    result = solve_ode(f, (t_end, t_start), y_terminal, method, jac, tol)
    # Reverse the output to be in forward time order
    result.t = result.t[::-1]
    result.y = result.y[::-1]
    return result


def solve_riccati(
    a: callable | float,
    b: callable | float,
    c: callable | float,
    t_span: tuple[float, float],
    y0: float | complex = 0.0,
    method: ODEMethod = ODEMethod.BDF,
    tol: float = 1e-8,
) -> ODEResult:
    """Solve a Riccati ODE: dy/dt = a(t) + b(t)×y + c(t)×y².

    Common in affine models (Heston, CIR++, Hull-White):
    - Heston characteristic function: dB/dτ = -½u² - iuρσ + (κ-iuρσ)B - ½σ²B²
    - Affine bond pricing: dA/dτ = a + bA + cA²

    Args:
        a, b, c: coefficients (callable(t) → float, or constant float/complex).
        t_span: integration interval.
        y0: initial condition.
        method: ODE method (BDF recommended for stiff Riccati).
        tol: tolerance.
    """
    # Wrap constants as callables
    a_fn = a if callable(a) else lambda t: a
    b_fn = b if callable(b) else lambda t: b
    c_fn = c if callable(c) else lambda t: c

    def f(t, y):
        return np.array([a_fn(t) + b_fn(t) * y[0] + c_fn(t) * y[0]**2])

    def jac(t, y):
        return np.array([[b_fn(t) + 2 * c_fn(t) * y[0]]])

    y0_arr = np.array([complex(y0) if isinstance(y0, complex) else float(y0)])

    # For complex Riccati, split into real and imaginary parts
    if isinstance(y0, complex) or isinstance(a, complex):
        return _solve_complex_riccati(a_fn, b_fn, c_fn, t_span, y0, method, tol)

    return solve_ode(f, t_span, y0_arr, method, jac, tol)


def _solve_complex_riccati(a_fn, b_fn, c_fn, t_span, y0, method, tol):
    """Solve complex Riccati by splitting into real + imaginary system."""
    y0c = complex(y0)

    def f(t, y):
        z = y[0] + 1j * y[1]
        a = complex(a_fn(t))
        b = complex(b_fn(t))
        c = complex(c_fn(t))
        dz = a + b * z + c * z**2
        return np.array([dz.real, dz.imag])

    result = solve_ode(f, t_span, np.array([y0c.real, y0c.imag]), method, tol=tol)
    return result


def solve_system(
    f: callable,
    t_span: tuple[float, float],
    y0: np.ndarray,
    method: ODEMethod = ODEMethod.LSODA,
    jac: callable | None = None,
    tol: float = 1e-6,
    stiff_detection: bool = True,
) -> ODEResult:
    """Solve a system of ODEs with automatic stiffness detection.

    If stiff_detection=True and method is LSODA, scipy automatically
    switches between Adams (non-stiff) and BDF (stiff).
    """
    if stiff_detection:
        method = ODEMethod.LSODA
    return solve_ode(f, t_span, y0, method, jac, tol)


# ═══════════════════════════════════════════════════════════════
# Backward compatibility: old function names still work
# ═══════════════════════════════════════════════════════════════

def euler(f, t_span, y0, n_steps=100):
    return solve_ode(f, t_span, y0, ODEMethod.EULER, n_steps=n_steps)

def rk4(f, t_span, y0, n_steps=100):
    return solve_ode(f, t_span, y0, ODEMethod.RK4, n_steps=n_steps)

def rk45(f, t_span, y0, tol=1e-6, max_step=None):
    return solve_ode(f, t_span, y0, ODEMethod.RK45, tol=tol, max_step=max_step)

def bdf(f, t_span, y0, jac=None, tol=1e-6):
    return solve_ode(f, t_span, y0, ODEMethod.BDF, jac=jac, tol=tol)

def adams(f, t_span, y0, tol=1e-6):
    return solve_ode(f, t_span, y0, ODEMethod.LSODA, tol=tol)
