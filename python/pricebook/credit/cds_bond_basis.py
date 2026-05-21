"""Sovereign CDS-bond basis analysis.

Decomposes the CDS-bond basis into its components and tracks basis
dynamics for relative value trading.

    from pricebook.credit.cds_bond_basis import (
        compute_basis, BasisResult, basis_drivers,
    )

The CDS-bond basis = CDS spread - bond Z-spread.

Positive basis (CDS > bond): common, driven by funding, delivery option, restructuring.
Negative basis (CDS < bond): rare, arbitrage signal (buy protection + buy bond).

References:
    Bai & Collin-Dufresne (2019). The CDS-Bond Basis.
    De Wit (2006). Exploring the CDS-Bond Basis. ECB Working Paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BasisResult:
    """CDS-bond basis analysis result."""
    cds_spread_bp: float
    bond_spread_bp: float        # Z-spread or ASW spread
    basis_bp: float              # CDS - bond spread
    funding_component_bp: float
    delivery_option_bp: float
    restructuring_bp: float
    residual_bp: float
    signal: str                  # "NEGATIVE_BASIS" (arb), "POSITIVE_BASIS", "NEUTRAL"

    def to_dict(self) -> dict:
        return vars(self)


def compute_basis(
    cds_spread_bp: float,
    bond_spread_bp: float,
    repo_spread_bp: float = 0.0,
    has_restructuring: bool = True,
    is_deliverable_basket: bool = True,
    funding_spread_bp: float = 0.0,
) -> BasisResult:
    """Compute and decompose the CDS-bond basis.

    Args:
        cds_spread_bp: CDS par spread (bp).
        bond_spread_bp: bond Z-spread or ASW spread (bp).
        repo_spread_bp: special repo spread (bp, negative = on special).
        has_restructuring: CDS includes restructuring event (widens CDS).
        is_deliverable_basket: CDS has delivery option (cheapest-to-deliver).
        funding_spread_bp: investor's funding spread above risk-free.
    """
    basis = cds_spread_bp - bond_spread_bp

    # Component decomposition (Bai & Collin-Dufresne framework)

    # Funding: buying a bond requires funding; CDS is unfunded
    # Positive funding → bond is more expensive → widens CDS-bond basis
    funding = funding_spread_bp

    # Delivery option: CDS buyer can deliver cheapest bond → CDS wider
    delivery = 5.0 if is_deliverable_basket else 0.0

    # Restructuring: if CDS triggers on restructuring but bond doesn't default
    restructuring = 10.0 if has_restructuring else 0.0

    # Repo: bond on special (negative repo) → depresses bond spread → widens basis
    # Positive repo spread → narrows basis
    repo_effect = -repo_spread_bp

    residual = basis - funding - delivery - restructuring - repo_effect

    # Signal
    if basis < -20:
        signal = "NEGATIVE_BASIS"
    elif basis > 20:
        signal = "POSITIVE_BASIS"
    else:
        signal = "NEUTRAL"

    return BasisResult(
        cds_spread_bp=cds_spread_bp,
        bond_spread_bp=bond_spread_bp,
        basis_bp=basis,
        funding_component_bp=funding,
        delivery_option_bp=delivery,
        restructuring_bp=restructuring,
        residual_bp=residual,
        signal=signal,
    )


def basis_z_score(
    current_basis_bp: float,
    historical_mean_bp: float,
    historical_std_bp: float,
) -> float:
    """Z-score of current basis vs historical distribution.

    Z > 2: basis unusually wide (potential negative basis trade entry)
    Z < -2: basis unusually tight (unwind signal)
    """
    if historical_std_bp <= 0:
        return 0.0
    return (current_basis_bp - historical_mean_bp) / historical_std_bp


def negative_basis_pnl(
    entry_basis_bp: float,
    exit_basis_bp: float,
    notional: float,
    hold_years: float,
    carry_bp: float = 0.0,
) -> dict:
    """P&L of a negative basis trade (buy bond + buy CDS protection).

    Entry: pay bond spread, receive CDS spread → collect negative basis
    Exit: unwind at new basis level

    Args:
        entry_basis_bp: basis at entry (negative for the trade to work).
        exit_basis_bp: basis at exit.
        notional: trade notional.
        hold_years: holding period.
        carry_bp: net carry (coupon - funding - CDS premium) in bp.
    """
    # Mark-to-market P&L from basis convergence
    # Negative basis trade is SHORT basis (you profit when basis becomes more negative)
    basis_change = exit_basis_bp - entry_basis_bp
    # Profit when basis tightens (entry more negative than exit → basis_change > 0 → loss)
    # So PnL = -(basis_change) × duration × notional (short basis position)
    duration = 5.0
    mtm_pnl = -(basis_change) / 10_000 * duration * notional

    # Carry P&L
    carry_pnl = carry_bp / 10_000 * hold_years * notional

    return {
        "entry_basis_bp": entry_basis_bp,
        "exit_basis_bp": exit_basis_bp,
        "mtm_pnl": mtm_pnl,
        "carry_pnl": carry_pnl,
        "total_pnl": mtm_pnl + carry_pnl,
        "hold_years": hold_years,
    }
