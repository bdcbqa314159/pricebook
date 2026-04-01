"""
Protocols for the numerics layer.

Defines the contracts that numerical methods must satisfy. Implementations
can be swapped without changing client code.

These are structural (duck-typing) protocols — no inheritance required.
Any object with the right methods satisfies the protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

@runtime_checkable
class RootFinder(Protocol):
    """Protocol for scalar root-finding methods."""

    def solve(self, f, *args, **kwargs) -> "SolverResult":
        """Find x such that f(x) ≈ 0."""
        ...


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

@runtime_checkable
class Integrator(Protocol):
    """Protocol for numerical integration methods."""

    def integrate(self, f, a: float, b: float, **kwargs) -> "QuadratureResult":
        """Compute ∫f(x)dx from a to b."""
        ...


# ---------------------------------------------------------------------------
# Option pricing engines
# ---------------------------------------------------------------------------

@runtime_checkable
class OptionPricer(Protocol):
    """Protocol for option pricing engines (tree, PDE, MC)."""

    def price(self, **kwargs) -> float:
        """Compute the option price."""
        ...


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

@runtime_checkable
class MCEngine(Protocol):
    """Protocol for Monte Carlo pricing engines."""

    def price(self, **kwargs) -> "MCResult":
        """Compute option price via Monte Carlo."""
        ...


# ---------------------------------------------------------------------------
# Vol model
# ---------------------------------------------------------------------------

@runtime_checkable
class VolModel(Protocol):
    """Protocol for volatility models that produce implied vol."""

    def implied_vol(self, forward: float, strike: float, T: float) -> float:
        """Implied Black vol at (forward, strike, T)."""
        ...


# ---------------------------------------------------------------------------
# Vol surface
# ---------------------------------------------------------------------------

@runtime_checkable
class VolSurface(Protocol):
    """Protocol for any object that provides vol(expiry, strike)."""

    def vol(self, expiry, strike: float | None = None) -> float:
        """Volatility at (expiry, strike)."""
        ...


# ---------------------------------------------------------------------------
# Result types (canonical definitions)
# ---------------------------------------------------------------------------

# These are imported from their home modules to avoid duplication.
# The protocol module re-exports them for convenience.

from pricebook.solvers import SolverResult  # noqa: E402
from pricebook.quadrature import QuadratureResult  # noqa: E402
from pricebook.mc_pricer import MCResult  # noqa: E402


__all__ = [
    "RootFinder",
    "Integrator",
    "OptionPricer",
    "MCEngine",
    "VolModel",
    "VolSurface",
    "SolverResult",
    "QuadratureResult",
    "MCResult",
]
