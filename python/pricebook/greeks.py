"""Unified Greeks dataclass and bump-and-reprice framework.

Every option pricer should return a Greeks object with standard sensitivities.
For analytical pricers, populate directly. For MC/PDE, use bump_greeks() helper.

    from pricebook.greeks import Greeks, bump_greeks

    # Analytical
    g = Greeks(price=5.23, delta=0.55, gamma=0.03, vega=15.2, theta=-0.05, rho=0.12)

    # Bump-and-reprice
    g = bump_greeks(price_func, spot=100, vol=0.20, rate=0.04, T=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Greeks:
    """Standard option sensitivities."""
    price: float
    delta: float = 0.0       # ∂V/∂S
    gamma: float = 0.0       # ∂²V/∂S²
    vega: float = 0.0        # ∂V/∂σ (per 1% vol shift, i.e. × 0.01)
    theta: float = 0.0       # ∂V/∂t (per 1 day, i.e. × 1/365)
    rho: float = 0.0         # ∂V/∂r (per 1% rate shift, i.e. × 0.01)
    vanna: float = 0.0       # ∂²V/(∂S∂σ)
    volga: float = 0.0       # ∂²V/∂σ²

    @property
    def dollar_delta(self) -> float:
        """Delta in dollar terms: delta × price_of_underlying."""
        return self.delta * self.price  # approximate

    @property
    def dollar_gamma(self) -> float:
        """Gamma P&L for a 1% spot move: 0.5 × gamma × S² × 0.01²."""
        return 0.5 * self.gamma  # caller should multiply by S² × move²


def bump_greeks(
    price_func,
    spot: float,
    vol: float,
    rate: float,
    T: float,
    spot_bump: float = 0.01,
    vol_bump: float = 0.001,
    rate_bump: float = 0.0001,
    time_bump: float = 1.0 / 365.0,
) -> Greeks:
    """Compute Greeks via bump-and-reprice.

    Args:
        price_func: callable(spot, vol, rate, T) → price.
        spot: current spot.
        vol: current vol.
        rate: current rate.
        T: time to expiry.
        spot_bump: absolute spot bump for delta/gamma.
        vol_bump: absolute vol bump for vega.
        rate_bump: absolute rate bump for rho.
        time_bump: time bump for theta (in years).
    """
    base = price_func(spot, vol, rate, T)

    # Delta and gamma
    up = price_func(spot + spot_bump, vol, rate, T)
    down = price_func(spot - spot_bump, vol, rate, T)
    delta = (up - down) / (2 * spot_bump)
    gamma = (up - 2 * base + down) / (spot_bump ** 2)

    # Vega
    vega_up = price_func(spot, vol + vol_bump, rate, T)
    vega = (vega_up - base) / vol_bump * 0.01  # per 1% vol shift

    # Theta
    if T > time_bump:
        theta_val = price_func(spot, vol, rate, T - time_bump)
        theta = (theta_val - base)  # already per time_bump
    else:
        theta = 0.0

    # Rho
    rho_up = price_func(spot, vol, rate + rate_bump, T)
    rho = (rho_up - base) / rate_bump * 0.01  # per 1% rate shift

    # Vanna (∂²V/∂S∂σ)
    up_vol_up = price_func(spot + spot_bump, vol + vol_bump, rate, T)
    vanna = (up_vol_up - up - vega_up + base) / (spot_bump * vol_bump)

    # Volga (∂²V/∂σ²)
    vega_down = price_func(spot, vol - vol_bump, rate, T)
    volga = (vega_up - 2 * base + vega_down) / (vol_bump ** 2)

    return Greeks(
        price=base,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        volga=volga,
    )
