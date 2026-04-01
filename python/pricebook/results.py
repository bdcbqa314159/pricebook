"""
Unified result types for numerical methods.

Canonical result dataclasses used across the numerics layer.
Some are defined in their home module (SolverResult, QuadratureResult, MCResult)
and re-exported here. Others (TreeResult, PDEResult) are defined here.

    from pricebook.results import TreeResult, PDEResult, MCResult
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-export from home modules
from pricebook.solvers import SolverResult
from pricebook.quadrature import QuadratureResult
from pricebook.mc_pricer import MCResult
from pricebook.optimization import OptimizerResult


@dataclass
class TreeResult:
    """Result of a tree-based option pricing computation."""

    price: float
    delta: float | None = None
    gamma: float | None = None
    n_steps: int = 0
    method: str = ""


@dataclass
class PDEResult:
    """Result of a PDE-based option pricing computation."""

    price: float
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    n_spot: int = 0
    n_time: int = 0
    scheme: str = ""


__all__ = [
    "SolverResult",
    "QuadratureResult",
    "MCResult",
    "TreeResult",
    "PDEResult",
    "OptimizerResult",
]
