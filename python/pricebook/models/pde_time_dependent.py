"""Time-dependent PDE coefficients: σ(t), r(t), q(t).

Extends the PDE framework with term-structure-aware coefficients.

* :func:`time_dependent_pde` — solve BS PDE with r(t), σ(t), q(t).
* :class:`TermStructureCoefficients` — interpolated term structures.

References:
    Duffy, *Finite Difference Methods in Financial Engineering*, Ch. 12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.pde_protocol import (
    PDECoefficients, PDESpec, PDEEngine, PDEPricingResult,
)


@dataclass
class TermStructureCoefficients:
    """Piecewise-linear term structures for r(t), σ(t), q(t).

    Interpolates between pillars for continuous evaluation.
    """
    rate_pillars: list[tuple[float, float]]   # [(t, r), ...]
    vol_pillars: list[tuple[float, float]]    # [(t, σ), ...]
    div_pillars: list[tuple[float, float]] | None = None

    def rate(self, t: float) -> float:
        return _interpolate(self.rate_pillars, t)

    def vol(self, t: float) -> float:
        return _interpolate(self.vol_pillars, t)

    def div_yield(self, t: float) -> float:
        if self.div_pillars:
            return _interpolate(self.div_pillars, t)
        return 0.0

    def to_pde_coefficients(self) -> PDECoefficients:
        """Convert to PDE coefficients with time-dependent callables."""
        return PDECoefficients(
            diffusion=lambda S, t: 0.5 * self.vol(t)**2 * S**2,
            convection=lambda S, t: (self.rate(t) - self.div_yield(t)) * S,
            reaction=lambda S, t: -self.rate(t),
        )


def _interpolate(pillars: list[tuple[float, float]], t: float) -> float:
    """Piecewise-linear interpolation on (time, value) pillars."""
    if not pillars:
        return 0.0
    if t <= pillars[0][0]:
        return pillars[0][1]
    if t >= pillars[-1][0]:
        return pillars[-1][1]
    for i in range(len(pillars) - 1):
        t0, v0 = pillars[i]
        t1, v1 = pillars[i + 1]
        if t0 <= t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return v0 + w * (v1 - v0)
    return pillars[-1][1]


def time_dependent_pde(
    spot: float,
    strike: float,
    T: float,
    term_structure: TermStructureCoefficients,
    is_call: bool = True,
    is_american: bool = False,
    n_space: int = 200,
    n_time: int = 200,
) -> PDEPricingResult:
    """Solve BS PDE with time-dependent r(t), σ(t), q(t).

    At each time step, coefficients are evaluated at the current
    calendar time, giving non-constant drift and diffusion.

    Args:
        term_structure: term structures for rate, vol, dividend yield.
    """
    coeffs = term_structure.to_pde_coefficients()

    # Use average rate for BCs (approximation)
    avg_rate = sum(r for _, r in term_structure.rate_pillars) / len(term_structure.rate_pillars)

    s_min = max(spot * 0.01, 1e-4)
    s_max = spot * 5.0
    payoff = (lambda S: np.maximum(S - strike, 0)) if is_call else (lambda S: np.maximum(strike - S, 0))

    if is_call:
        bc_lo = lambda S, t: 0.0
        bc_hi = lambda S, t: S - strike * math.exp(-avg_rate * t)
    else:
        bc_lo = lambda S, t: strike * math.exp(-avg_rate * t) - S
        bc_hi = lambda S, t: 0.0

    spec = PDESpec(
        coefficients=coeffs,
        s_min=s_min, s_max=s_max, T=T,
        payoff=payoff,
        bc_lower=bc_lo, bc_upper=bc_hi,
        is_american=is_american,
        exercise_payoff=payoff if is_american else None,
    )

    engine = PDEEngine("crank_nicolson", "log", n_space, n_time)
    return engine.solve(spec, spot)
