"""CVA desk tools: sensitivities, hedging, and incremental CVA.

Compute CVA CS01 and IR01, determine CDS hedge notionals,
and measure incremental CVA from adding trades.

    from pricebook.cva_desk import cva_cs01, cva_ir01, cva_hedge, incremental_cva
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.credit_risk import _bump_survival_curve
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.survival_curve import SurvivalCurve
from pricebook.xva import (
    simulate_exposures, expected_positive_exposure, cva,
)


# ---- CVA sensitivities ----

def cva_cs01(
    epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
    shift_bps: float = 1.0,
) -> float:
    """CVA CS01: sensitivity of CVA to a 1bp parallel shift in credit spreads.

    Bumps the survival curve (via hazard rate shift) and recomputes CVA.
    """
    base_cva = cva(epe, time_grid, discount_curve, survival_curve, recovery)
    shift = shift_bps / 10000.0
    bumped_sc = _bump_survival_curve(survival_curve, shift)
    bumped_cva = cva(epe, time_grid, discount_curve, bumped_sc, recovery)
    return (bumped_cva - base_cva) / shift_bps


def cva_ir01(
    epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
    shift_bps: float = 1.0,
) -> float:
    """CVA IR01: sensitivity of CVA to a 1bp parallel shift in interest rates.

    Bumps the discount curve and recomputes CVA.
    """
    base_cva = cva(epe, time_grid, discount_curve, survival_curve, recovery)
    shift = shift_bps / 10000.0
    bumped_dc = discount_curve.bumped(shift)
    bumped_cva = cva(epe, time_grid, bumped_dc, survival_curve, recovery)
    return (bumped_cva - base_cva) / shift_bps


# ---- CVA by trade ----

def cva_by_trade(
    trade_epes: dict[str, np.ndarray],
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
) -> dict[str, float]:
    """CVA contribution per trade (stand-alone, not marginal).

    Args:
        trade_epes: trade_id -> EPE array (one per time point).
    """
    return {
        trade_id: cva(epe, time_grid, discount_curve, survival_curve, recovery)
        for trade_id, epe in trade_epes.items()
    }


# ---- CVA hedging ----

@dataclass
class CVAHedgeResult:
    """Result of CVA hedge computation."""
    portfolio_cva: float
    portfolio_cs01: float
    hedge_notional: float
    hedge_cs01: float
    residual_cs01: float


def cva_hedge(
    epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    cds_cs01_per_notional: float,
    recovery: float = 0.4,
    shift_bps: float = 1.0,
) -> CVAHedgeResult:
    """Compute CDS hedge notional to offset CVA CS01.

    Args:
        epe: portfolio expected positive exposure.
        time_grid: time points.
        discount_curve: risk-free curve.
        survival_curve: counterparty survival curve.
        cds_cs01_per_notional: CS01 per unit notional of the hedge CDS.
        recovery: recovery rate.

    Returns:
        CVAHedgeResult with hedge notional and residual CS01.
    """
    portfolio_cva = cva(epe, time_grid, discount_curve, survival_curve, recovery)
    portfolio_cs01 = cva_cs01(epe, time_grid, discount_curve, survival_curve, recovery, shift_bps)

    if abs(cds_cs01_per_notional) < 1e-15:
        return CVAHedgeResult(portfolio_cva, portfolio_cs01, 0.0, 0.0, portfolio_cs01)

    # Hedge notional to offset CS01
    hedge_notional = -portfolio_cs01 / cds_cs01_per_notional
    hedge_cs01 = hedge_notional * cds_cs01_per_notional
    residual = portfolio_cs01 + hedge_cs01

    return CVAHedgeResult(
        portfolio_cva=portfolio_cva,
        portfolio_cs01=portfolio_cs01,
        hedge_notional=hedge_notional,
        hedge_cs01=hedge_cs01,
        residual_cs01=residual,
    )


# ---- Incremental CVA ----

def incremental_cva(
    portfolio_epe: np.ndarray,
    new_trade_epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
) -> dict[str, float]:
    """Incremental CVA: change in portfolio CVA from adding a new trade.

    Incremental CVA = CVA(portfolio + new) - CVA(portfolio).
    This captures netting effects.

    Args:
        portfolio_epe: existing portfolio EPE.
        new_trade_epe: new trade EPE (stand-alone).
    """
    base_cva = cva(portfolio_epe, time_grid, discount_curve, survival_curve, recovery)

    # Combined EPE: EPE of the netted portfolio
    # Approximate: EPE(A+B) ≤ EPE(A) + EPE(B), but netting can reduce
    combined_epe = portfolio_epe + new_trade_epe
    combined_cva = cva(combined_epe, time_grid, discount_curve, survival_curve, recovery)

    standalone_cva = cva(new_trade_epe, time_grid, discount_curve, survival_curve, recovery)

    return {
        "portfolio_cva": base_cva,
        "combined_cva": combined_cva,
        "incremental_cva": combined_cva - base_cva,
        "standalone_cva": standalone_cva,
        "netting_benefit": standalone_cva - (combined_cva - base_cva),
    }
