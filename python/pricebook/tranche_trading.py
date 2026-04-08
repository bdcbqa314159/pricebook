"""Tranche correlation trading: delta, correlation skew, and sensitivities.

Tranche delta, spread sensitivity, correlation skew analysis, and
base correlation bumps for CDO tranche trading.

    from pricebook.tranche_trading import (
        tranche_delta, tranche_cs01, correlation_skew, skew_bump,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.cdo import (
    portfolio_loss_distribution, tranche_expected_loss,
    tranche_spread, base_correlation,
)


# ---- Tranche delta ----

def tranche_delta(
    pd: float,
    rho: float,
    lgd: float,
    attach: float,
    detach: float,
    T: float = 5.0,
    risk_free_rate: float = 0.05,
    spread_bump_bps: float = 1.0,
) -> float:
    """Tranche delta: dPV/d(index spread) per bp.

    Approximated by bumping the portfolio PD (which drives the index spread)
    and measuring the tranche spread change.

    Equity tranche: positive delta (spread widening hurts equity).
    Senior tranche: negative delta (spread widening may help senior via
    increased subordination value, but tranche loss also increases).
    """
    loss_grid, density = portfolio_loss_distribution(pd, rho, lgd)
    base_spread = tranche_spread(loss_grid, density, attach, detach, T, risk_free_rate)

    # Bump PD by 1bp equivalent
    pd_bump = spread_bump_bps / 10000.0
    pd_up = min(pd + pd_bump, 0.9999)
    loss_grid_up, density_up = portfolio_loss_distribution(pd_up, rho, lgd)
    bumped_spread = tranche_spread(loss_grid_up, density_up, attach, detach, T, risk_free_rate)

    return (bumped_spread - base_spread) / spread_bump_bps


def tranche_cs01(
    pd: float,
    rho: float,
    lgd: float,
    attach: float,
    detach: float,
    notional: float = 10_000_000.0,
    T: float = 5.0,
    risk_free_rate: float = 0.05,
    bump_bps: float = 1.0,
) -> float:
    """Tranche CS01: PV change per 1bp move in index spread.

    CS01 ≈ tranche_delta × tranche_notional × risky_annuity
    """
    import math
    delta = tranche_delta(pd, rho, lgd, attach, detach, T, risk_free_rate, bump_bps)
    thickness = detach - attach
    tranche_not = notional * thickness
    annuity = sum(math.exp(-risk_free_rate * t) for t in range(1, int(T) + 1))
    return delta * tranche_not * annuity


# ---- Correlation sensitivity ----

def correlation_sensitivity(
    pd: float,
    rho: float,
    lgd: float,
    attach: float,
    detach: float,
    T: float = 5.0,
    risk_free_rate: float = 0.05,
    rho_bump: float = 0.01,
) -> float:
    """Tranche spread sensitivity to correlation (rho bump).

    d(tranche_spread) / d(rho) per unit rho change.
    """
    loss_grid, density = portfolio_loss_distribution(pd, rho, lgd)
    base_spread = tranche_spread(loss_grid, density, attach, detach, T, risk_free_rate)

    rho_up = min(rho + rho_bump, 0.999)
    loss_grid_up, density_up = portfolio_loss_distribution(pd, rho_up, lgd)
    bumped_spread = tranche_spread(loss_grid_up, density_up, attach, detach, T, risk_free_rate)

    return (bumped_spread - base_spread) / rho_bump


# ---- Correlation skew ----

@dataclass
class SkewPoint:
    """One point on the base correlation curve."""
    detach: float
    market_spread: float
    base_corr: float


@dataclass
class CorrelationSkew:
    """Base correlation curve with skew measures."""
    points: list[SkewPoint]
    skew: float  # difference between senior and equity base corr


def correlation_skew(
    market_spreads: dict[float, float],
    pd: float,
    lgd: float,
    T: float = 5.0,
    risk_free_rate: float = 0.05,
) -> CorrelationSkew:
    """Build base correlation curve from market tranche spreads.

    Args:
        market_spreads: detachment -> market tranche spread.
            Each tranche is [0, detach] (equity-up).
        pd: portfolio default probability.
        lgd: loss given default.

    Returns:
        CorrelationSkew with base correlation at each detachment point.
    """
    points = []
    for detach in sorted(market_spreads.keys()):
        spread = market_spreads[detach]
        try:
            bc = base_correlation(spread, detach, pd, lgd, T, risk_free_rate)
        except Exception:
            bc = float("nan")
        points.append(SkewPoint(detach=detach, market_spread=spread, base_corr=bc))

    skew = 0.0
    if len(points) >= 2:
        skew = points[-1].base_corr - points[0].base_corr

    return CorrelationSkew(points=points, skew=skew)


def skew_bump(
    pd: float,
    rho: float,
    lgd: float,
    detachment_points: list[float],
    bump_type: str = "parallel",
    bump_size: float = 0.01,
    T: float = 5.0,
    risk_free_rate: float = 0.05,
) -> dict[float, tuple[float, float]]:
    """Bump the correlation and measure tranche spread impact.

    Args:
        detachment_points: list of detachment levels.
        bump_type: "parallel" (all same), "tilt" (more at senior end).
        bump_size: correlation bump size.

    Returns:
        dict of detach -> (base_spread, bumped_spread).
    """
    results = {}
    for i, detach in enumerate(sorted(detachment_points)):
        loss_grid, density = portfolio_loss_distribution(pd, rho, lgd)
        base = tranche_spread(loss_grid, density, 0.0, detach, T, risk_free_rate)

        if bump_type == "parallel":
            rho_bumped = min(rho + bump_size, 0.999)
        elif bump_type == "tilt":
            # Tilt: more bump at higher detachment
            frac = i / max(len(detachment_points) - 1, 1)
            rho_bumped = min(rho + bump_size * frac, 0.999)
        else:
            rho_bumped = min(rho + bump_size, 0.999)

        loss_grid_b, density_b = portfolio_loss_distribution(pd, rho_bumped, lgd)
        bumped = tranche_spread(loss_grid_b, density_b, 0.0, detach, T, risk_free_rate)

        results[detach] = (base, bumped)

    return results
