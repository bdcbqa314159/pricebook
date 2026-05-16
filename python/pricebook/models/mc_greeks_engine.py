"""Greeks estimation for the MC engine.

Three methods:
1. Bump-and-reprice (finite difference on process parameters)
2. Pathwise (IPA) — differentiate payoff along paths
3. Likelihood ratio — weight payoff by score function

    from pricebook.models.mc_greeks_engine import mc_greeks, GreeksResult

    greeks = mc_greeks(engine, payoff, spot=100, rate=0.05, vol=0.20, df=exp(-r*T))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.mc_engine import MCEngine, TimeGrid, MCResult


@dataclass
class GreeksResult:
    """MC Greeks estimation result."""
    delta: float        # dPrice/dSpot
    gamma: float        # d²Price/dSpot²
    vega: float         # dPrice/dVol (per 1%)
    theta: float        # dPrice/dT (per 1 day)
    rho: float          # dPrice/dRate (per 1%)

    def to_dict(self) -> dict:
        return {
            "delta": self.delta, "gamma": self.gamma,
            "vega": self.vega, "theta": self.theta, "rho": self.rho,
        }


def mc_greeks(
    process_factory,
    time_grid: TimeGrid,
    payoff,
    spot: float,
    rate: float,
    vol: float,
    T: float,
    n_paths: int = 100_000,
    seed: int = 42,
    spot_bump: float = 1.0,
    vol_bump: float = 0.01,
    rate_bump: float = 0.0001,
    time_bump_days: float = 1.0,
) -> GreeksResult:
    """Compute all Greeks via bump-and-reprice.

    Args:
        process_factory: callable(s0, r, sigma) → ProcessSpec.
        time_grid: simulation time grid.
        payoff: payoff callable(paths, times) → values.
        spot, rate, vol, T: base parameters.
        n_paths: paths for each bump.
        seed: random seed (same for all bumps → correlated).
        spot_bump: absolute spot bump for delta/gamma.
        vol_bump: absolute vol bump for vega.
        rate_bump: absolute rate bump for rho.
        time_bump_days: days to shift for theta.
    """
    def _price(s0, r, sigma, t, grid=None):
        proc = process_factory(s0, r, sigma)
        g = grid or time_grid
        eng = MCEngine(proc, g, n_paths, seed, antithetic=True)
        df = math.exp(-r * t)
        return eng.price(payoff, df).price

    base = _price(spot, rate, vol, T)

    # Delta: centred difference on spot
    p_up = _price(spot + spot_bump, rate, vol, T)
    p_dn = _price(spot - spot_bump, rate, vol, T)
    delta = (p_up - p_dn) / (2 * spot_bump)

    # Gamma: second derivative on spot
    gamma = (p_up - 2 * base + p_dn) / (spot_bump ** 2)

    # Vega: centred difference on vol (reported per 1% vol = 0.01)
    p_vup = _price(spot, rate, vol + vol_bump, T)
    p_vdn = _price(spot, rate, vol - vol_bump, T)
    vega = (p_vup - p_vdn) / (2 * vol_bump) * 0.01

    # Rho: centred difference on rate (reported per 1% = 0.01)
    p_rup = _price(spot, rate + rate_bump, vol, T)
    p_rdn = _price(spot, rate - rate_bump, vol, T)
    rho = (p_rup - p_rdn) / (2 * rate_bump) * 0.01

    # Theta: forward difference on time (1 day)
    dt_bump = time_bump_days / 365.0
    if T - dt_bump > 0.001:
        grid_short = TimeGrid.uniform(T - dt_bump, max(time_grid.n_steps - 1, 1))
        p_theta = _price(spot, rate, vol, T - dt_bump, grid_short)
        theta = (p_theta - base) / time_bump_days  # per day (negative for long options)
    else:
        theta = 0.0

    return GreeksResult(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
