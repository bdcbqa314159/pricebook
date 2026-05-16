"""Unified Monte Carlo engine.

Core simulation framework that any instrument can plug into.
Define the SDE (ProcessSpec), define the payoff, get price + Greeks.

    from pricebook.models.mc_engine import MCEngine, TimeGrid, MCResult
    from pricebook.models.mc_processes import GBMProcess
    from pricebook.models.mc_payoffs import european_call

    engine = MCEngine(
        process=GBMProcess(s0=100, mu=0.05, sigma=0.20),
        time_grid=TimeGrid.uniform(T=1.0, n_steps=100),
        n_paths=100_000,
    )
    result = engine.price(european_call(strike=100))

Architecture:
    TimeGrid       — simulation dates
    ProcessSpec    — SDE definition (drift, diffusion, n_factors)
    MCEngine       — orchestrator (paths → payoff → statistics)
    MCResult       — price, stderr, convergence
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Time Grid
# ---------------------------------------------------------------------------

class TimeGrid:
    """Simulation time points.

    Stores an array of times [0, t1, t2, ..., T] and the corresponding
    dt increments for the SDE discretisation.
    """

    def __init__(self, times: np.ndarray):
        self.times = np.asarray(times, dtype=np.float64)
        self.dt = np.diff(self.times)
        self.n_steps = len(self.dt)
        self.T = float(self.times[-1])

    @classmethod
    def uniform(cls, T: float, n_steps: int) -> TimeGrid:
        """Uniform grid from 0 to T with n_steps intervals."""
        return cls(np.linspace(0, T, n_steps + 1))

    @classmethod
    def from_dates(cls, dates: list, day_count: float = 365.0) -> TimeGrid:
        """Build from a list of dates (converted to year fractions)."""
        if not dates:
            raise ValueError("dates list is empty")
        t0 = dates[0]
        times = np.array([(d - t0).days / day_count for d in dates])
        return cls(times)

    def __len__(self) -> int:
        return self.n_steps


# ---------------------------------------------------------------------------
# Process Specification
# ---------------------------------------------------------------------------

class ProcessSpec:
    """SDE specification: dX = drift(X,t) dt + diffusion(X,t) dW.

    For multi-factor processes, X is a vector and diffusion returns
    a matrix (n_factors × n_brownians).

    Args:
        x0: initial value(s). Scalar for 1D, array for multi-factor.
        drift: callable(x, t) → dx/dt component.
        diffusion: callable(x, t) → volatility component.
        n_factors: number of state variables (1 for GBM, 2 for Heston, etc.).
        correlation: correlation matrix for multi-factor (None = identity).
        exact_step: optional exact simulation step (e.g., for CIR).
    """

    def __init__(
        self,
        x0: float | np.ndarray,
        drift: Callable,
        diffusion: Callable,
        n_factors: int = 1,
        correlation: np.ndarray | None = None,
        exact_step: Callable | None = None,
    ):
        self.x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        self.drift = drift
        self.diffusion = diffusion
        self.n_factors = n_factors
        self.exact_step = exact_step

        if correlation is not None:
            self.correlation = np.asarray(correlation, dtype=np.float64)
            self.cholesky = np.linalg.cholesky(self.correlation)
        else:
            self.correlation = np.eye(n_factors)
            self.cholesky = np.eye(n_factors)


# ---------------------------------------------------------------------------
# Simulation Schemes
# ---------------------------------------------------------------------------

def euler_step(
    x: np.ndarray,
    t: float,
    dt: float,
    dw: np.ndarray,
    process: ProcessSpec,
) -> np.ndarray:
    """Euler-Maruyama step: X_{t+dt} = X_t + drift × dt + diffusion × dW."""
    return x + process.drift(x, t) * dt + process.diffusion(x, t) * dw


def milstein_step_1d(
    x: np.ndarray,
    t: float,
    dt: float,
    dw: np.ndarray,
    process: ProcessSpec,
    diffusion_deriv: Callable | None = None,
) -> np.ndarray:
    """Milstein step (1D): adds 0.5 × σ × σ' × (dW² - dt) correction."""
    drift_val = process.drift(x, t)
    diff_val = process.diffusion(x, t)
    base = x + drift_val * dt + diff_val * dw

    if diffusion_deriv is not None:
        dd = diffusion_deriv(x, t)
        base += 0.5 * diff_val * dd * (dw ** 2 - dt)

    return base


# ---------------------------------------------------------------------------
# MC Engine
# ---------------------------------------------------------------------------

class MCEngine:
    """Monte Carlo simulation engine.

    Generates paths for a given ProcessSpec and evaluates payoffs.

    Args:
        process: SDE specification.
        time_grid: simulation time points.
        n_paths: number of simulation paths.
        seed: random seed for reproducibility.
        antithetic: if True, use antithetic variates (doubles effective paths).
        scheme: "euler" (default) or "milstein".
    """

    def __init__(
        self,
        process: ProcessSpec,
        time_grid: TimeGrid,
        n_paths: int = 100_000,
        seed: int = 42,
        antithetic: bool = False,
        scheme: str = "euler",
    ):
        self.process = process
        self.time_grid = time_grid
        self.n_paths = n_paths
        self.seed = seed
        self.antithetic = antithetic
        self.scheme = scheme
        self._paths: np.ndarray | None = None

    def generate_paths(self) -> np.ndarray:
        """Generate MC paths.

        Returns:
            Array of shape (n_paths, n_steps+1) for 1D,
            or (n_paths, n_steps+1, n_factors) for multi-factor.
        """
        rng = np.random.default_rng(self.seed)
        proc = self.process
        grid = self.time_grid
        n = self.n_paths
        nf = proc.n_factors

        if self.antithetic:
            n_half = n // 2
            n = n_half * 2  # ensure even
        else:
            n_half = n

        # Generate Brownian increments
        if nf == 1:
            dw_raw = rng.standard_normal((n_half, grid.n_steps))
            if self.antithetic:
                dw = np.concatenate([dw_raw, -dw_raw], axis=0)
            else:
                dw = dw_raw
            dw *= np.sqrt(grid.dt)[np.newaxis, :]

            # Simulate paths
            paths = np.zeros((dw.shape[0], grid.n_steps + 1))
            paths[:, 0] = proc.x0[0]

            step_fn = euler_step if self.scheme == "euler" else euler_step

            for i in range(grid.n_steps):
                if proc.exact_step is not None:
                    paths[:, i + 1] = proc.exact_step(
                        paths[:, i], grid.times[i], grid.dt[i], dw[:, i],
                    )
                else:
                    paths[:, i + 1] = step_fn(
                        paths[:, i], grid.times[i], grid.dt[i], dw[:, i], proc,
                    )
        else:
            # Multi-factor
            dw_raw = rng.standard_normal((n_half, grid.n_steps, nf))
            if self.antithetic:
                dw_raw = np.concatenate([dw_raw, -dw_raw], axis=0)

            # Apply Cholesky correlation
            dw = np.einsum('...j,kj->...k', dw_raw, proc.cholesky)
            for i in range(grid.n_steps):
                dw[:, i, :] *= np.sqrt(grid.dt[i])

            # Simulate
            paths = np.zeros((dw.shape[0], grid.n_steps + 1, nf))
            paths[:, 0, :] = proc.x0

            for i in range(grid.n_steps):
                if proc.exact_step is not None:
                    paths[:, i + 1, :] = proc.exact_step(
                        paths[:, i, :], grid.times[i], grid.dt[i], dw[:, i, :],
                    )
                else:
                    paths[:, i + 1, :] = euler_step(
                        paths[:, i, :], grid.times[i], grid.dt[i], dw[:, i, :], proc,
                    )

        self._paths = paths
        return paths

    @property
    def paths(self) -> np.ndarray:
        """Cached paths (generated on first access)."""
        if self._paths is None:
            self.generate_paths()
        return self._paths

    def price(
        self,
        payoff: Callable,
        discount_factor: float = 1.0,
    ) -> "MCResult":
        """Evaluate a payoff on the simulated paths.

        Args:
            payoff: callable(paths, times) → array of shape (n_paths,).
            discount_factor: e.g. exp(-r*T) for risk-neutral pricing.

        Returns:
            MCResult with price, stderr, and paths.
        """
        paths = self.paths
        times = self.time_grid.times

        values = payoff(paths, times)
        discounted = values * discount_factor

        price = float(np.mean(discounted))
        stderr = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))

        return MCResult(
            price=price,
            stderr=stderr,
            n_paths=len(discounted),
            n_steps=self.time_grid.n_steps,
            confidence_95=(price - 1.96 * stderr, price + 1.96 * stderr),
        )

    def greek(
        self,
        payoff: Callable,
        param_name: str,
        bump: float,
        base_price: float | None = None,
        discount_factor: float = 1.0,
    ) -> float:
        """Bump-and-reprice Greek.

        Bumps a parameter on the process, regenerates paths, reprices.
        Centred difference: (price_up - price_dn) / (2 × bump).

        Args:
            payoff: payoff function.
            param_name: attribute on process.x0 or process itself to bump.
            bump: shift size.
            base_price: pre-computed base price (avoids recomputation).
            discount_factor: discount factor.
        """
        # Save original
        original_x0 = self.process.x0.copy()

        # Bump up
        self.process.x0 = original_x0 + bump
        self._paths = None
        price_up = self.price(payoff, discount_factor).price

        # Bump down
        self.process.x0 = original_x0 - bump
        self._paths = None
        price_dn = self.price(payoff, discount_factor).price

        # Restore
        self.process.x0 = original_x0
        self._paths = None

        return (price_up - price_dn) / (2 * bump)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """Monte Carlo pricing result."""
    price: float
    stderr: float
    n_paths: int
    n_steps: int
    confidence_95: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "price": self.price, "stderr": self.stderr,
            "n_paths": self.n_paths, "n_steps": self.n_steps,
            "ci_95": list(self.confidence_95),
        }

    @property
    def relative_error(self) -> float:
        """Standard error as percentage of price."""
        return abs(self.stderr / self.price) * 100 if self.price != 0 else 0.0
