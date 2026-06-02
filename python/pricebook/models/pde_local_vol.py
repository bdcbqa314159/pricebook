"""Local volatility PDE solver.

Takes a Dupire local vol surface and solves the BS PDE with σ(S,t).
Supports European, American, and barrier options.

* :func:`local_vol_pde` — price under local vol via PDE.
* :func:`local_vol_barrier_pde` — barrier option under local vol.

References:
    Dupire, *Pricing with a Smile*, Risk, 1994.
    Andersen & Brotherton-Ratcliffe, *The Equity Option Volatility Smile*,
    JD, 1998.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.pde_protocol import (
    PDECoefficients, PDESpec, PDEEngine, PDEPricingResult,
)


def local_vol_pde(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    vol_surface,
    is_call: bool = True,
    is_american: bool = False,
    div_yield: float = 0.0,
    n_space: int = 200,
    n_time: int = 200,
    method: str = "crank_nicolson",
) -> PDEPricingResult:
    """Price under local volatility via PDE.

    The local vol surface σ(S,t) is interpolated at each grid
    node and time step, giving a fully non-constant diffusion.

    Args:
        vol_surface: object with .vol(T, K) → σ_loc, or callable(S, t) → σ.
    """
    coeffs = PDECoefficients.local_vol(vol_surface, rate, div_yield)

    s_min = max(spot * 0.01, 1e-4)
    s_max = spot * 5.0
    payoff = (lambda S: np.maximum(S - strike, 0)) if is_call else (lambda S: np.maximum(strike - S, 0))

    if is_call:
        bc_lo = lambda S, t: 0.0
        bc_hi = lambda S, t: S - strike * math.exp(-rate * t)
    else:
        bc_lo = lambda S, t: strike * math.exp(-rate * t) - S
        bc_hi = lambda S, t: 0.0

    spec = PDESpec(
        coefficients=coeffs,
        s_min=s_min, s_max=s_max, T=T,
        payoff=payoff,
        bc_lower=bc_lo, bc_upper=bc_hi,
        is_american=is_american,
        exercise_payoff=payoff if is_american else None,
    )

    engine = PDEEngine(method, "log", n_space, n_time)
    return engine.solve(spec, spot)


def local_vol_barrier_pde(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    T: float,
    vol_surface,
    is_call: bool = True,
    is_up: bool = True,
    is_knock_out: bool = True,
    div_yield: float = 0.0,
    n_space: int = 200,
    n_time: int = 200,
) -> PDEPricingResult:
    """Barrier option under local vol via PDE.

    Applies absorbing boundary at the barrier level.

    Args:
        barrier: barrier level.
        is_up: True for up barrier, False for down.
        is_knock_out: True for knock-out, False for knock-in (via parity).
    """
    coeffs = PDECoefficients.local_vol(vol_surface, rate, div_yield)

    # Adjust domain to include barrier
    if is_up:
        s_min = max(spot * 0.01, 1e-4)
        s_max = barrier
    else:
        s_min = barrier
        s_max = spot * 5.0

    payoff = (lambda S: np.maximum(S - strike, 0)) if is_call else (lambda S: np.maximum(strike - S, 0))

    # Barrier BC: value = 0 at barrier
    if is_up:
        bc_lo = lambda S, t: 0.0 if not is_call else 0.0
        bc_hi = lambda S, t: 0.0  # knocked out at barrier
    else:
        bc_lo = lambda S, t: 0.0  # knocked out at barrier
        bc_hi = lambda S, t: S - strike * math.exp(-rate * t) if is_call else 0.0

    spec = PDESpec(
        coefficients=coeffs,
        s_min=s_min, s_max=s_max, T=T,
        payoff=payoff,
        bc_lower=bc_lo, bc_upper=bc_hi,
    )

    engine = PDEEngine("crank_nicolson", "sinh", n_space, n_time, compute_vega=False)
    ko_result = engine.solve(spec, spot)

    if is_knock_out:
        return ko_result

    # Knock-in via parity: KI = vanilla - KO
    from pricebook.models.pde_protocol import pde_price
    vanilla = pde_price(spot, strike, 0.20, rate, T, is_call, vol_surface=vol_surface,
                         n_space=n_space, n_time=n_time)
    ki_price = vanilla.price - ko_result.price

    return PDEPricingResult(
        price=ki_price,
        delta=vanilla.delta - ko_result.delta,
        gamma=vanilla.gamma - ko_result.gamma,
        theta=vanilla.theta - ko_result.theta,
        vega=0.0,
        convergence=ko_result.convergence,
    )
