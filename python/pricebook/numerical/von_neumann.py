"""Von Neumann stability analysis for finite difference schemes.

Computes amplification factor for θ-scheme on uniform grid.

* :func:`amplification_factor` — |g(θ, ν)| for a given scheme.
* :func:`stability_region` — stable region in (ν, θ) space.
* :func:`scheme_analysis` — full stability report for a scheme.

References:
    Strikwerda, *Finite Difference Schemes and PDEs*, 2nd ed., Ch. 3.
    Duffy, *Finite Difference Methods in Financial Engineering*, Ch. 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StabilityResult:
    """Von Neumann stability analysis result."""
    max_amplification: float    # max |g| over all frequencies
    stable: bool                # |g| ≤ 1 + O(dt)
    cfl_limit: float            # maximum stable ν (for explicit)
    theta: float
    nu: float                   # ν = σ²dt/dx²

    def to_dict(self) -> dict:
        return vars(self)


def amplification_factor(
    theta: float,
    nu: float,
    xi: float,
) -> complex:
    """Von Neumann amplification factor g(ξ) for θ-scheme.

    For the heat equation u_t = a u_xx discretised as:
    (1 + 2θν(1−cos ξ)) u^{n+1} = (1 − 2(1−θ)ν(1−cos ξ)) u^n

    g(ξ) = (1 − 2(1−θ)ν(1−cos ξ)) / (1 + 2θν(1−cos ξ))

    Args:
        theta: scheme parameter (0=explicit, 0.5=CN, 1=implicit).
        nu: CFL number ν = a×dt/dx².
        xi: Fourier mode (frequency).
    """
    c = 1 - math.cos(xi)
    numerator = 1 - 2 * (1 - theta) * nu * c
    denominator = 1 + 2 * theta * nu * c
    if abs(denominator) < 1e-15:
        return complex(1e10, 0)
    return complex(numerator / denominator, 0)


def max_amplification(
    theta: float,
    nu: float,
    n_modes: int = 100,
) -> float:
    """Maximum |g| over all Fourier modes.

    Args:
        theta: scheme parameter.
        nu: CFL number.
        n_modes: number of modes to test.
    """
    xi_grid = np.linspace(0, math.pi, n_modes)
    max_g = 0.0
    for xi in xi_grid:
        g = amplification_factor(theta, nu, xi)
        max_g = max(max_g, abs(g))
    return max_g


def stability_region(
    theta: float,
    nu_range: tuple[float, float] = (0, 2),
    n_points: int = 200,
) -> list[dict]:
    """Compute stability as function of ν.

    Args:
        theta: scheme parameter.
        nu_range: range of CFL numbers to test.

    Returns:
        List of {"nu", "max_g", "stable"} dicts.
    """
    nus = np.linspace(nu_range[0], nu_range[1], n_points)
    results = []
    for nu in nus:
        mg = max_amplification(theta, nu)
        results.append({
            "nu": float(nu),
            "max_g": mg,
            "stable": mg <= 1.0 + 1e-10,
        })
    return results


def cfl_limit(theta: float) -> float:
    """Compute CFL limit for a given theta.

    For θ = 0 (explicit): ν ≤ 0.5.
    For θ ≥ 0.5 (CN, implicit): unconditionally stable.
    For 0 < θ < 0.5: ν ≤ 1/(2(1−2θ)).
    """
    if theta >= 0.5:
        return float('inf')
    if theta == 0:
        return 0.5
    return 1.0 / (2 * (1 - 2 * theta))


def scheme_analysis(
    theta: float,
    nu: float,
) -> StabilityResult:
    """Full stability analysis for a (θ, ν) pair.

    Args:
        theta: 0=explicit, 0.5=CN, 1=implicit.
        nu: CFL number σ²dt/dx².
    """
    mg = max_amplification(theta, nu)
    limit = cfl_limit(theta)
    stable = nu <= limit + 1e-10

    return StabilityResult(
        max_amplification=mg,
        stable=stable,
        cfl_limit=limit,
        theta=theta,
        nu=nu,
    )
