"""Regulatory capital: SA-CCR and FRTB for IR trades.

SA-CCR: Standardised Approach for Counterparty Credit Risk.
FRTB SBA: Sensitivities-Based Approach for market risk.

    from pricebook.regulatory import sa_ccr_addon, frtb_delta_charge

    addon = sa_ccr_addon(notional=10_000_000, maturity=5.0, direction=1)
    charge = frtb_delta_charge(sensitivities, scenario="medium")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pricebook.discount_curve import DiscountCurve
from pricebook.swap import InterestRateSwap, SwapDirection


# ---- SA-CCR for IR ----

# Supervisory factor for IR asset class (Basel III SA-CCR)
IR_SUPERVISORY_FACTOR = 0.005

# Maturity buckets for hedging set netting
SA_CCR_BUCKETS = [(0, 1), (1, 5), (5, float("inf"))]
SA_CCR_BUCKET_LABELS = ["<1Y", "1Y-5Y", ">5Y"]


def _maturity_factor(maturity_years: float) -> float:
    """SA-CCR maturity factor: min(M, 1) for margined, sqrt(min(M,1)/1) approx."""
    return math.sqrt(min(maturity_years, 1.0))


def _supervisory_delta(direction: int, is_option: bool = False) -> float:
    """SA-CCR supervisory delta adjustment.

    For swaps: +1 (receive fixed / long) or -1 (pay fixed / short).
    Options would use Black-Scholes delta (not implemented here).
    """
    return float(direction)


def _maturity_bucket(maturity_years: float) -> int:
    """Assign maturity to SA-CCR bucket index."""
    if maturity_years <= 1.0:
        return 0
    elif maturity_years <= 5.0:
        return 1
    else:
        return 2


@dataclass
class SaCcrTrade:
    """Trade-level inputs for SA-CCR."""
    notional: float
    maturity_years: float
    direction: int  # +1 for receiver, -1 for payer
    currency: str = "USD"


@dataclass
class SaCcrResult:
    """SA-CCR computation result."""
    trade_level: list[dict[str, float]]
    bucket_effective_notionals: list[float]
    effective_notional: float
    addon: float
    details: dict[str, Any] = field(default_factory=dict)


def sa_ccr_addon(
    trades: list[SaCcrTrade],
) -> SaCcrResult:
    """Compute SA-CCR add-on for a set of IR trades in one currency.

    Add-on = SF × EffectiveNotional
    EffectiveNotional = sqrt(D1² + D2² + D3² + 1.4·D1·D2 + 1.4·D2·D3 + 0.6·D1·D3)

    where Di = sum of adjusted notionals in bucket i.
    """
    # Trade-level adjusted notionals
    trade_details = []
    bucket_sums = [0.0, 0.0, 0.0]

    for t in trades:
        delta = _supervisory_delta(t.direction)
        mf = _maturity_factor(t.maturity_years)
        adjusted = delta * t.notional * mf
        bucket = _maturity_bucket(t.maturity_years)
        bucket_sums[bucket] += adjusted
        trade_details.append({
            "notional": t.notional,
            "maturity": t.maturity_years,
            "direction": t.direction,
            "delta": delta,
            "maturity_factor": mf,
            "adjusted_notional": adjusted,
            "bucket": SA_CCR_BUCKET_LABELS[bucket],
        })

    D1, D2, D3 = bucket_sums

    # Effective notional with cross-bucket correlations
    eff_notional_sq = (
        D1 ** 2 + D2 ** 2 + D3 ** 2
        + 1.4 * D1 * D2
        + 1.4 * D2 * D3
        + 0.6 * D1 * D3
    )
    effective_notional = math.sqrt(max(eff_notional_sq, 0.0))
    addon = IR_SUPERVISORY_FACTOR * effective_notional

    return SaCcrResult(
        trade_level=trade_details,
        bucket_effective_notionals=bucket_sums,
        effective_notional=effective_notional,
        addon=addon,
        details={"supervisory_factor": IR_SUPERVISORY_FACTOR},
    )


def sa_ccr_single(
    notional: float,
    maturity_years: float,
    direction: int,
) -> float:
    """Quick SA-CCR add-on for a single trade."""
    result = sa_ccr_addon([SaCcrTrade(notional, maturity_years, direction)])
    return result.addon


# ---- FRTB Sensitivities-Based Approach (IR Delta) ----

# FRTB tenor buckets
FRTB_TENORS = [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30]
FRTB_TENOR_LABELS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y", "15Y", "20Y", "30Y"]

# Risk weights per tenor bucket (in %)
FRTB_IR_RISK_WEIGHTS = [
    0.017, 0.017, 0.016, 0.013, 0.012,
    0.011, 0.011, 0.011, 0.011, 0.011,
]

# Correlation between tenor buckets i,j: rho = max(e^{-theta*|Ti-Tj|/min(Ti,Tj)}, 0.4)
FRTB_THETA = 0.03


def _frtb_correlation(i: int, j: int, scenario: str = "medium") -> float:
    """FRTB intra-bucket correlation between tenor points."""
    if i == j:
        return 1.0
    Ti = FRTB_TENORS[i]
    Tj = FRTB_TENORS[j]
    rho = max(math.exp(-FRTB_THETA * abs(Ti - Tj) / min(Ti, Tj)), 0.40)
    if scenario == "low":
        rho = max(2 * rho - 1, 0.75 * rho)
    elif scenario == "high":
        rho = min(1.0, 1.25 * rho)
    return rho


@dataclass
class FrtbDeltaResult:
    """FRTB delta risk charge result."""
    weighted_sensitivities: list[float]
    risk_charge: float
    scenario: str
    details: dict[str, Any] = field(default_factory=dict)


def frtb_delta_charge(
    sensitivities: dict[str, float],
    scenario: str = "medium",
) -> FrtbDeltaResult:
    """Compute FRTB IR delta risk charge.

    Args:
        sensitivities: mapping of tenor label (e.g. "5Y") to DV01 in currency.
        scenario: correlation scenario ("low", "medium", "high").

    Returns:
        FrtbDeltaResult with risk charge.
    """
    n = len(FRTB_TENORS)

    # Weighted sensitivities: WS_k = s_k * RW_k
    ws = []
    for k in range(n):
        label = FRTB_TENOR_LABELS[k]
        s = sensitivities.get(label, 0.0)
        ws.append(s * FRTB_IR_RISK_WEIGHTS[k])

    # Risk charge: sqrt(sum_i sum_j rho_ij * WS_i * WS_j)
    total = 0.0
    for i in range(n):
        for j in range(n):
            rho = _frtb_correlation(i, j, scenario)
            total += rho * ws[i] * ws[j]

    charge = math.sqrt(max(total, 0.0))

    return FrtbDeltaResult(
        weighted_sensitivities=ws,
        risk_charge=charge,
        scenario=scenario,
        details={
            "tenor_labels": FRTB_TENOR_LABELS,
            "risk_weights": FRTB_IR_RISK_WEIGHTS,
        },
    )


def frtb_delta_charge_all_scenarios(
    sensitivities: dict[str, float],
) -> dict[str, FrtbDeltaResult]:
    """Compute FRTB delta charge under all three correlation scenarios.

    The binding charge is the maximum across scenarios.
    """
    return {
        scenario: frtb_delta_charge(sensitivities, scenario)
        for scenario in ("low", "medium", "high")
    }
