"""Basis trading: CDS-bond basis, trade construction, and monitoring.

The CDS-bond basis measures the difference between credit protection
cost (CDS spread) and bond credit risk (Z-spread or ASW spread).

    from pricebook.basis_trade import (
        cds_bond_basis, negative_basis_trade, basis_monitor,
    )

    basis = cds_bond_basis(cds, bond, market_price, discount_curve, survival_curve)
    trade = negative_basis_trade(cds, bond, market_price, discount_curve, survival_curve)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.cds import CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.risky_bond import RiskyBond, z_spread, asset_swap_spread
from pricebook.survival_curve import SurvivalCurve


# ---- CDS-bond basis ----

@dataclass
class BasisResult:
    """CDS-bond basis calculation."""
    cds_spread: float
    z_spread_val: float
    asw_spread: float
    basis_z: float       # CDS spread - Z-spread
    basis_asw: float     # CDS spread - ASW spread


def cds_bond_basis(
    cds: CDS,
    bond: RiskyBond,
    bond_market_price: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> BasisResult:
    """Compute CDS-bond basis.

    Basis = CDS spread - bond spread measure.
    Negative basis = CDS cheaper than bond (protection is cheap).
    Positive basis = CDS more expensive than bond.

    Args:
        cds: the CDS contract.
        bond: the risky bond.
        bond_market_price: observed bond price.
        discount_curve: risk-free curve.
        survival_curve: credit curve for CDS par spread.
    """
    cds_par = cds.par_spread(discount_curve, survival_curve)
    z_sp = z_spread(bond, bond_market_price, discount_curve)
    asw_sp = asset_swap_spread(bond, bond_market_price, discount_curve)

    return BasisResult(
        cds_spread=cds_par,
        z_spread_val=z_sp,
        asw_spread=asw_sp,
        basis_z=cds_par - z_sp,
        basis_asw=cds_par - asw_sp,
    )


# ---- Basis trade construction ----

@dataclass
class BasisTradeResult:
    """P&L and carry of a basis trade."""
    trade_type: str  # "negative_basis" or "positive_basis"
    bond_pv: float
    cds_pv: float
    net_pv: float
    carry: float     # net coupon income per period
    basis: float


def negative_basis_trade(
    cds: CDS,
    bond: RiskyBond,
    bond_market_price: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> BasisTradeResult:
    """Negative basis trade: buy bond + buy CDS protection.

    Profits when: basis is negative (CDS protection is cheap) and
    converges toward zero over time.

    Carry = bond coupon - CDS premium - funding cost.
    """
    bond_pv = bond.dirty_price(discount_curve, survival_curve)
    cds_pv = cds.pv(discount_curve, survival_curve)
    cds_par = cds.par_spread(discount_curve, survival_curve)
    z_sp = z_spread(bond, bond_market_price, discount_curve)
    basis = cds_par - z_sp

    # Carry: bond coupon income minus CDS premium cost
    carry = bond.coupon_rate * bond.notional - cds.spread * cds.notional

    return BasisTradeResult(
        trade_type="negative_basis",
        bond_pv=bond_pv,
        cds_pv=cds_pv,
        net_pv=bond_pv + cds_pv,
        carry=carry,
        basis=basis,
    )


def positive_basis_trade(
    cds: CDS,
    bond: RiskyBond,
    bond_market_price: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> BasisTradeResult:
    """Positive basis trade: sell bond + sell CDS protection.

    Profits when: basis is positive (CDS is expensive) and
    converges toward zero.
    """
    bond_pv = bond.dirty_price(discount_curve, survival_curve)
    cds_pv = cds.pv(discount_curve, survival_curve)
    cds_par = cds.par_spread(discount_curve, survival_curve)
    z_sp = z_spread(bond, bond_market_price, discount_curve)
    basis = cds_par - z_sp

    # Carry: receive CDS spread, pay bond coupon (short)
    carry = cds.spread * cds.notional - bond.coupon_rate * bond.notional

    return BasisTradeResult(
        trade_type="positive_basis",
        bond_pv=-bond_pv,
        cds_pv=-cds_pv,
        net_pv=-(bond_pv + cds_pv),
        carry=carry,
        basis=basis,
    )


# ---- Basis monitor ----

@dataclass
class BasisSignal:
    """Basis level with historical z-score and signal."""
    name: str
    basis: float
    z_score: float | None
    percentile: float | None
    signal: str  # "negative", "positive", "fair"


def basis_monitor(
    name: str,
    current_basis: float,
    history: list[float] | None = None,
    threshold: float = 2.0,
) -> BasisSignal:
    """Monitor basis level with z-score signal.

    Args:
        name: reference entity name.
        current_basis: current CDS-bond basis.
        history: historical basis values.
        threshold: z-score threshold for signal.
    """
    z_score = None
    percentile = None
    signal = "fair"

    if history and len(history) >= 2:
        mean = sum(history) / len(history)
        var = sum((h - mean) ** 2 for h in history) / len(history)
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 1e-12:
            z_score = (current_basis - mean) / std
            if z_score < -threshold:
                signal = "negative"  # basis unusually negative → buy basis
            elif z_score > threshold:
                signal = "positive"  # basis unusually positive → sell basis
        sorted_hist = sorted(history)
        rank = sum(1 for h in sorted_hist if h <= current_basis)
        percentile = rank / len(sorted_hist) * 100.0

    return BasisSignal(
        name=name,
        basis=current_basis,
        z_score=z_score,
        percentile=percentile,
        signal=signal,
    )
