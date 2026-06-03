"""Feynman-Kac bridge: formal connection between SDE and PDE.

Given SDE coefficients, auto-derive PDE coefficients and vice versa.
Cross-validate MC vs PDE pricing.

* :func:`sde_to_pde` — derive PDE coefficients from SDE dynamics.
* :func:`verify_feynman_kac` — price via MC and PDE, compare.
* :func:`pde_to_sde` — extract SDE dynamics from PDE coefficients.

References:
    Kac, *On Distributions of Certain Wiener Functionals*, TAMS, 1949.
    Shreve, *Stochastic Calculus for Finance II*, Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FeynmanKacResult:
    """Cross-validation result: MC vs PDE."""
    mc_price: float
    pde_price: float
    difference: float
    relative_diff_pct: float
    mc_stderr: float
    consistent: bool            # within 3 standard errors

    def to_dict(self) -> dict:
        return vars(self)


def sde_to_pde(
    mu_fn,
    sigma_fn,
    rate_fn=None,
) -> dict:
    """Derive PDE coefficients from SDE dynamics.

    SDE: dS = μ(S,t) dt + σ(S,t) dW
    PDE: ∂V/∂t + ½σ²(S,t) ∂²V/∂S² + μ(S,t) ∂V/∂S − r(t) V = 0

    Under risk-neutral measure:
    μ(S,t) = (r − q) × S  (for equity)
    σ(S,t) = σ × S         (for GBM)

    Args:
        mu_fn: drift callable(S, t) → float.
        sigma_fn: diffusion callable(S, t) → float.
        rate_fn: discount rate callable(t) → float, or constant.

    Returns:
        Dict with PDE coefficient callables: diffusion, convection, reaction.
    """
    if rate_fn is None:
        rate_fn = lambda t: 0.04

    if not callable(rate_fn):
        r_const = float(rate_fn)
        rate_fn = lambda t: r_const

    return {
        "diffusion": lambda S, t: 0.5 * sigma_fn(S, t) ** 2,
        "convection": lambda S, t: mu_fn(S, t),
        "reaction": lambda S, t: -rate_fn(t),
    }


def pde_to_sde(
    diffusion_fn,
    convection_fn,
) -> dict:
    """Extract SDE dynamics from PDE coefficients.

    PDE: a(S,t) ∂²V/∂S² + b(S,t) ∂V/∂S + c(S,t) V = 0
    SDE: dS = b(S,t) dt + √(2a(S,t)) dW

    Args:
        diffusion_fn: a(S, t) = ½σ²(S,t).
        convection_fn: b(S, t) = μ(S,t).

    Returns:
        Dict with SDE coefficient callables: drift, volatility.
    """
    return {
        "drift": convection_fn,
        "volatility": lambda S, t: math.sqrt(max(2 * diffusion_fn(S, t), 0)),
    }


def verify_feynman_kac(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    is_call: bool = True,
    div_yield: float = 0.0,
    n_paths: int = 200_000,
    n_space: int = 200,
    n_time: int = 200,
    seed: int = 42,
) -> FeynmanKacResult:
    """Cross-validate MC and PDE pricing for the same instrument.

    Both use the same model (GBM with given parameters).
    If Feynman-Kac holds, prices should agree within MC standard error.

    Args:
        n_paths: MC paths.
        n_space: PDE spatial grid points.
        n_time: PDE time steps.
    """
    # MC price
    from pricebook.models.mc_engine import MCEngine, TimeGrid
    from pricebook.models.mc_processes import GBMProcess
    from pricebook.models.mc_payoffs import european_call, european_put

    proc = GBMProcess(s0=spot, mu=rate - div_yield, sigma=vol)
    grid = TimeGrid.uniform(T, 100)
    engine = MCEngine(proc, grid, n_paths, seed, antithetic=True)
    payoff = european_call(strike) if is_call else european_put(strike)
    mc_result = engine.price(payoff, math.exp(-rate * T))

    # PDE price
    from pricebook.models.pde_protocol import pde_price
    pde_result = pde_price(spot, strike, vol, rate, T, is_call, div_yield=div_yield,
                            n_space=n_space, n_time=n_time)

    diff = abs(mc_result.price - pde_result.price)
    rel = diff / abs(mc_result.price) * 100 if mc_result.price != 0 else 0
    consistent = diff < 3 * mc_result.stderr

    return FeynmanKacResult(
        mc_price=mc_result.price,
        pde_price=pde_result.price,
        difference=diff,
        relative_diff_pct=rel,
        mc_stderr=mc_result.stderr,
        consistent=consistent,
    )
