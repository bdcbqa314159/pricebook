"""Bond futures basis trading: CTD switches, delivery option, switch trades.

Extends :mod:`pricebook.govt_bond_trading` (``basis_decomposition``,
``ctd_switch_monitor``) with scenario-based CTD switch analysis,
delivery option valuation, and switch trade construction.

* :func:`ctd_switch_scenarios` — which bond becomes CTD under parallel
  yield shifts.
* :func:`delivery_option_value` — wild-card, quality, and timing
  option components.
* :class:`SwitchTrade` — sell old CTD, buy new CTD, with P&L.
* :func:`basis_at_delivery` — basis convergence as delivery approaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.govt_bond_trading import basis_decomposition, BasisDecomposition


# ---- CTD switch under scenarios ----

@dataclass
class CTDScenario:
    """CTD outcome under a specific yield shift."""
    yield_shift_bps: float
    ctd_bond: str
    ctd_implied_repo: float
    all_bonds: list[BasisDecomposition]


def ctd_switch_scenarios(
    bonds: list[dict],
    futures_price: float,
    repo_rate: float,
    days_to_delivery: int,
    shifts_bps: list[float] | None = None,
) -> list[CTDScenario]:
    """Identify the CTD bond under a range of parallel yield shifts.

    Each bond in *bonds* is a dict with keys:
    ``name``, ``price``, ``cf``, ``coupon_rate``, ``duration``.

    The shifted bond price is approximated as:
    ``price_shifted ≈ price × (1 − duration × shift_bps / 10000)``.

    Args:
        bonds: list of deliverable bond dicts.
        futures_price: current futures price.
        repo_rate: repo financing rate.
        days_to_delivery: calendar days to delivery.
        shifts_bps: list of parallel shifts (default: −50 to +50 in 10bp steps).

    Returns:
        list of :class:`CTDScenario`, one per shift.
    """
    if shifts_bps is None:
        shifts_bps = list(range(-50, 55, 10))

    results = []
    for shift in shifts_bps:
        decomps = []
        for b in bonds:
            shifted_price = b["price"] * (1 - b["duration"] * shift / 10_000)
            # Futures price also shifts (approximately by CTD duration × shift)
            # but for ranking CTD the relative ordering is what matters,
            # so we keep futures flat as a simplification.
            bd = basis_decomposition(
                b["name"], shifted_price, futures_price, b["cf"],
                b["coupon_rate"], repo_rate, days_to_delivery,
            )
            decomps.append(bd)

        best = max(decomps, key=lambda d: d.implied_repo)
        results.append(CTDScenario(
            yield_shift_bps=shift,
            ctd_bond=best.bond_name,
            ctd_implied_repo=best.implied_repo,
            all_bonds=decomps,
        ))

    return results


def ctd_switch_probability(
    scenarios: list[CTDScenario],
    current_ctd: str,
) -> float:
    """Fraction of scenarios where the CTD switches away from *current_ctd*.

    A simple estimate of switch risk: higher = more likely to switch.
    """
    if not scenarios:
        return 0.0
    switches = sum(1 for s in scenarios if s.ctd_bond != current_ctd)
    return switches / len(scenarios)


# ---- Delivery option value ----

@dataclass
class DeliveryOptionValue:
    """Components of the delivery option embedded in a bond future.

    * **quality option** — freedom to choose *which* bond to deliver
      (different from the CTD at inception).
    * **timing option** — flexibility on *when* within the delivery month
      to deliver (wild-card for intraday moves after futures close).
    * **total** — quality + timing (approximation; these are not additive
      in theory but are in practice for rough estimates).
    """
    quality_option: float
    timing_option: float
    total: float


def delivery_option_value(
    net_basis_ctd: float,
    avg_net_basis_others: float,
    daily_vol_bps: float = 5.0,
    delivery_window_days: int = 5,
) -> DeliveryOptionValue:
    """Estimate the delivery option value.

    * Quality option ≈ net basis of CTD − average net basis of other bonds.
      This is the value from being able to switch to a cheaper deliverable.
    * Timing option ≈ daily_vol × √(delivery_window_days) × scaling_factor.
      This is a rough proxy for the wild-card option.

    Args:
        net_basis_ctd: net basis of the current CTD (should be the lowest).
        avg_net_basis_others: average net basis of other deliverable bonds.
        daily_vol_bps: daily yield volatility in bps.
        delivery_window_days: number of days in the delivery window.
    """
    import math
    quality = max(avg_net_basis_others - net_basis_ctd, 0.0)
    timing = daily_vol_bps / 10_000.0 * math.sqrt(delivery_window_days) * 100.0
    return DeliveryOptionValue(
        quality_option=quality,
        timing_option=timing,
        total=quality + timing,
    )


# ---- Switch trade ----

@dataclass
class SwitchTrade:
    """Sell old CTD, buy new CTD — a basis switch trade.

    P&L from the switch = net_basis_old − net_basis_new.
    Positive if the new CTD is cheaper to deliver.
    """
    old_ctd: str
    new_ctd: str
    old_net_basis: float
    new_net_basis: float
    switch_pnl: float
    face_amount: float


def construct_switch_trade(
    old_ctd: BasisDecomposition,
    new_ctd: BasisDecomposition,
    face_amount: float = 1_000_000,
) -> SwitchTrade:
    """Build a basis switch: sell old CTD, buy new CTD."""
    switch_pnl = (old_ctd.net_basis - new_ctd.net_basis) * face_amount / 100.0
    return SwitchTrade(
        old_ctd=old_ctd.bond_name,
        new_ctd=new_ctd.bond_name,
        old_net_basis=old_ctd.net_basis,
        new_net_basis=new_ctd.net_basis,
        switch_pnl=switch_pnl,
        face_amount=face_amount,
    )


# ---- Basis convergence ----

def basis_at_delivery(
    gross_basis: float,
    days_to_delivery: int,
    current_day: int,
) -> float:
    """Estimate the gross basis as delivery approaches.

    The basis converges linearly to zero at delivery:
        basis(t) = gross_basis × (days_remaining / total_days).

    Args:
        gross_basis: initial gross basis.
        days_to_delivery: total days from now to delivery.
        current_day: day number (0 = now, days_to_delivery = delivery).

    Returns:
        Estimated gross basis at *current_day*.
    """
    if days_to_delivery <= 0:
        return 0.0
    remaining = max(days_to_delivery - current_day, 0)
    return gross_basis * remaining / days_to_delivery
