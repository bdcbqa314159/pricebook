"""Bond futures delivery options: EOM, quality, timing, net basis.

* :func:`end_of_month_option` — wild card option value.
* :func:`quality_option` — CTD switch probability and value.
* :func:`timing_option` — early/late delivery value.
* :func:`net_basis_decomposition` — gross − carry = delivery options.
* :func:`joint_delivery_option_value` — combined delivery options.

References:
    Burghardt & Belton, *The Treasury Bond Basis*, McGraw-Hill, 2005.
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012, Ch. 14.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EOMOptionResult:
    """End-of-month (wild card) option value."""
    value: float
    n_business_days_remaining: int
    daily_option_value: float


def end_of_month_option(
    futures_dv01: float,
    daily_vol_bps: float,
    n_business_days: int = 5,
    coupon_accrual_per_day: float = 0.0,
) -> EOMOptionResult:
    """End-of-month wild card option.

    After the last trading day, the short can still deliver for several
    business days at the final settlement price. If rates move, a different
    bond may become cheaper to deliver.

    Approximate value ≈ Σ daily_straddle_value.
    Daily straddle ≈ 0.8 × DV01 × daily_vol (normal approx).

    Args:
        futures_dv01: DV01 of the futures contract.
        daily_vol_bps: daily yield vol in basis points.
        n_business_days: days between last trade and last delivery.
        coupon_accrual_per_day: daily carry benefit of holding bond.
    """
    daily_option = 0.8 * futures_dv01 * daily_vol_bps / 100
    total = n_business_days * daily_option + n_business_days * coupon_accrual_per_day
    return EOMOptionResult(
        value=float(max(total, 0.0)),
        n_business_days_remaining=n_business_days,
        daily_option_value=float(daily_option),
    )


@dataclass
class QualityOptionResult:
    """Quality option (CTD switch) result."""
    value: float
    current_ctd: str
    ctd_switch_probability: float
    n_deliverables: int


def quality_option(
    deliverable_bases: dict[str, float],
    yield_vol_bps: float,
    futures_dv01: float,
    T: float = 0.25,
) -> QualityOptionResult:
    """Quality option: option to deliver cheapest bond.

    When yields move, a different bond may become CTD. The option to
    switch CTD has value proportional to:
    - Number of near-CTD bonds (narrow basis spread).
    - Yield volatility.
    - Time to delivery.

    Simplified: value ≈ max(0, second_cheapest_basis − cheapest_basis)
    × probability of crossing × DV01.

    Args:
        deliverable_bases: {bond_id: net_basis} — lower = cheaper to deliver.
        yield_vol_bps: annual yield vol in bps.
        futures_dv01: contract DV01.
        T: time to delivery.
    """
    sorted_bonds = sorted(deliverable_bases.items(), key=lambda x: x[1])
    ctd = sorted_bonds[0][0]
    ctd_basis = sorted_bonds[0][1]

    if len(sorted_bonds) < 2:
        return QualityOptionResult(0.0, ctd, 0.0, 1)

    second_basis = sorted_bonds[1][1]
    gap = second_basis - ctd_basis

    # Probability of switch: approximate with normal CDF
    # gap in same units as vol × DV01 × sqrt(T)
    vol_effect = yield_vol_bps / 100 * futures_dv01 * math.sqrt(T)
    if vol_effect > 1e-10:
        from scipy.stats import norm
        switch_prob = norm.cdf(-gap / vol_effect)
    else:
        switch_prob = 0.0

    value = switch_prob * gap * futures_dv01

    return QualityOptionResult(
        value=float(max(value, 0.0)),
        current_ctd=ctd,
        ctd_switch_probability=float(switch_prob),
        n_deliverables=len(deliverable_bases),
    )


@dataclass
class TimingOptionResult:
    """Timing option value."""
    value: float
    accrued_carry: float
    optimal_delivery_day: str


def timing_option(
    futures_price: float,
    coupon_rate: float,
    repo_rate: float,
    conversion_factor: float,
    days_first: int = 1,
    days_last: int = 30,
) -> TimingOptionResult:
    """Timing option: short chooses when in the delivery month to deliver.

    If positive carry (coupon > repo), deliver late (earn carry).
    If negative carry, deliver early (avoid cost).

    Value ≈ |carry_per_day| × days_flexibility.

    Args:
        futures_price: current futures price.
        coupon_rate: annual coupon of CTD.
        repo_rate: repo financing rate.
        conversion_factor: CF for CTD.
    """
    daily_coupon = coupon_rate * 100 / 360  # per $100 face
    daily_repo_cost = repo_rate * futures_price * conversion_factor / 360
    daily_carry = daily_coupon - daily_repo_cost

    n_days = days_last - days_first
    if daily_carry > 0:
        # Deliver late to earn carry
        optimal = "last_day"
        value = daily_carry * n_days
    else:
        # Deliver early to avoid negative carry
        optimal = "first_day"
        value = -daily_carry * n_days

    return TimingOptionResult(
        value=float(max(value, 0.0)),
        accrued_carry=float(daily_carry * n_days),
        optimal_delivery_day=optimal,
    )


@dataclass
class NetBasisResult:
    """Net basis decomposition."""
    gross_basis: float
    carry: float
    net_basis: float                # = gross - carry ≈ delivery option value
    delivery_option_value: float


def net_basis_decomposition(
    bond_price: float,
    conversion_factor: float,
    futures_price: float,
    coupon_accrued: float,
    repo_cost: float,
) -> NetBasisResult:
    """Net basis = gross basis − carry.

    Gross basis = (bond_dirty × CF) − futures × CF  (simplified)
                = bond_price − futures × CF − accrued_interest_adjustment
    Carry = coupon_income − financing_cost over delivery period.
    Net basis = gross − carry ≈ sum of delivery options.

    Args:
        bond_price: clean price of deliverable bond.
        conversion_factor: CF for the bond.
        futures_price: futures settlement price.
        coupon_accrued: coupon income earned during delivery period.
        repo_cost: repo financing cost during delivery period.
    """
    gross_basis = bond_price - futures_price * conversion_factor
    carry = coupon_accrued - repo_cost
    net_basis = gross_basis - carry

    return NetBasisResult(
        gross_basis=float(gross_basis),
        carry=float(carry),
        net_basis=float(net_basis),
        delivery_option_value=float(max(net_basis, 0.0)),
    )


@dataclass
class JointDeliveryResult:
    """Joint delivery option valuation."""
    total_value: float
    eom_value: float
    quality_value: float
    timing_value: float


def joint_delivery_option_value(
    eom: EOMOptionResult,
    quality: QualityOptionResult,
    timing: TimingOptionResult,
    correlation_discount: float = 0.7,
) -> JointDeliveryResult:
    """Combined delivery option value (with correlation discount).

    Options are not fully additive because they're exercised jointly.
    Approximate: total ≈ (eom + quality + timing) × discount.
    """
    raw_sum = eom.value + quality.value + timing.value
    total = raw_sum * correlation_discount

    return JointDeliveryResult(
        total_value=float(total),
        eom_value=float(eom.value),
        quality_value=float(quality.value),
        timing_value=float(timing.value),
    )
