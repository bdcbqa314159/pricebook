"""Unified Greeks dataclass and bump-and-reprice framework.

Every option pricer should return a Greeks object with standard sensitivities.
For analytical pricers, populate directly. For MC/PDE, use bump_greeks() helper.

    from pricebook.risk.greeks import Greeks, bump_greeks
"""

from __future__ import annotations

# Greeks dataclass lives in core (L0) so instruments (L2) can import
# it without depending on risk (L3). Re-exported here for backward compat.
from pricebook.core.greeks import Greeks  # noqa: F401


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

    Conventions:
      - vega is per +1 vol-point (e.g. σ → σ+0.01) — scaled by ``0.01``.
      - rho is per +1 rate-point — scaled by ``0.01``.
      - theta is per ``time_bump`` years (default = 1 day), negative for
        long options (price decays with calendar time).

    Fix T4-RISK2: pre-fix called ``price_func(spot, vol - vol_bump, ...)``
    twice — once stored as ``vega_down_v`` (used in vega), once as
    ``vega_down`` (used in volga).  The two were identical recomputes,
    wasting a pricer call.  Real cost for any expensive pricer (MC, PDE).
    Now computed once and reused.

    Also: pre-fix rho used a forward difference ``(rho_up − base)/h``
    while delta, gamma, vega all used central differences.  Switched
    to central difference for consistency and O(h²) accuracy.
    """
    if spot_bump <= 0 or vol_bump <= 0 or rate_bump <= 0 or time_bump <= 0:
        raise ValueError("all bump sizes must be strictly positive")
    if vol_bump >= vol:
        raise ValueError(
            f"vol_bump ({vol_bump}) must be smaller than vol ({vol}) — "
            f"vol - vol_bump would be non-positive"
        )

    base = price_func(spot, vol, rate, T)
    up = price_func(spot + spot_bump, vol, rate, T)
    down = price_func(spot - spot_bump, vol, rate, T)
    delta = (up - down) / (2 * spot_bump)
    gamma = (up - 2 * base + down) / (spot_bump ** 2)

    vega_up = price_func(spot, vol + vol_bump, rate, T)
    vega_down = price_func(spot, vol - vol_bump, rate, T)
    vega = (vega_up - vega_down) / (2 * vol_bump) * 0.01

    if T > time_bump:
        theta_val = price_func(spot, vol, rate, T - time_bump)
        theta = theta_val - base
    else:
        theta = 0.0

    rho_up = price_func(spot, vol, rate + rate_bump, T)
    rho_down = price_func(spot, vol, rate - rate_bump, T)
    rho = (rho_up - rho_down) / (2 * rate_bump) * 0.01

    up_vol_up = price_func(spot + spot_bump, vol + vol_bump, rate, T)
    vanna = (up_vol_up - up - vega_up + base) / (spot_bump * vol_bump)

    volga = (vega_up - 2 * base + vega_down) / (vol_bump ** 2)

    return Greeks(price=base, delta=delta, gamma=gamma, vega=vega,
                  theta=theta, rho=rho, vanna=vanna, volga=volga)
