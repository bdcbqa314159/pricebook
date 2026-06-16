"""Operational risk SMA (Standardised Measurement Approach, OPE25).

Basel III replacement for BIA/TSA/AMA. Capital = BIC × ILM.

    from pricebook.regulatory.operational_risk import (
        calculate_sma_full, SMAInputs, SMAResult,
    )

References:
    Basel Committee (2017). Basel III: Finalising post-crisis reforms, OPE25.
    EBA (2022). Guidelines on SMA for operational risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# SMA Constants (OPE25)
# ═══════════════════════════════════════════════════════════════

# BIC marginal coefficients by bucket (OPE25.7)
BIC_BUCKETS = [
    (1_000_000_000, 0.12),    # Bucket 1: BI ≤ 1bn → 12%
    (30_000_000_000, 0.15),   # Bucket 2: 1bn < BI ≤ 30bn → 15% marginal
    (float('inf'), 0.18),     # Bucket 3: BI > 30bn → 18% marginal
]

# Bucket 1 threshold for ILM applicability (OPE25.11)
BUCKET_1_THRESHOLD = 1_000_000_000


@dataclass
class SMAInputs:
    """3-year P&L data for Business Indicator calculation.

    Each list must have exactly 3 elements (years 1-3).
    """
    interest_income: list[float]
    interest_expense: list[float]
    lease_income: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    lease_expense: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    dividend_income: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fee_income: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fee_expense: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    other_operating_income: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    other_operating_expense: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    net_trading_income: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    net_banking_book_pnl: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Loss data (10 years for ILM)
    annual_op_losses: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "interest_income": self.interest_income,
            "interest_expense": self.interest_expense,
            "fee_income": self.fee_income,
            "net_trading_income": self.net_trading_income,
            "annual_op_losses": self.annual_op_losses,
        }


@dataclass
class SMAResult:
    """SMA capital calculation result."""
    bi_yearly: list[float]
    bi_average: float
    bucket: int
    bic: float
    loss_component: float
    ilm: float
    use_ilm: bool
    capital_requirement: float
    rwa: float
    legacy_bia: float

    def to_dict(self) -> dict:
        return dict(vars(self))


# ═══════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════

def _business_indicator_year(inputs: SMAInputs, year: int) -> float:
    """Business Indicator for a single year (OPE25.4-6).

    BI = ILDC + SC + FC
    ILDC = min(|II - IE|, 2.25% × IEA) + |LeaseI - LeaseE| + DivI
    SC = max(FeeI, FeeE) + max(OtherI, OtherE)
    FC = |NetTrading| + |NetBanking|

    Simplified: uses absolute values of income components.
    """
    i = year  # index into lists

    # Interest, Lease & Dividend Component (ILDC)
    net_interest = abs(inputs.interest_income[i] - inputs.interest_expense[i])
    net_lease = abs(inputs.lease_income[i] - inputs.lease_expense[i])
    div = abs(inputs.dividend_income[i])
    ildc = net_interest + net_lease + div

    # Services Component (SC)
    sc = max(inputs.fee_income[i], inputs.fee_expense[i]) + \
         max(inputs.other_operating_income[i], inputs.other_operating_expense[i])

    # Financial Component (FC)
    fc = abs(inputs.net_trading_income[i]) + abs(inputs.net_banking_book_pnl[i])

    return ildc + sc + fc


def calculate_business_indicator(inputs: SMAInputs) -> list[float]:
    """Calculate BI for each of the 3 years."""
    return [_business_indicator_year(inputs, i) for i in range(3)]


def sma_bucket(bi: float) -> int:
    """Determine SMA bucket from BI."""
    if bi <= BIC_BUCKETS[0][0]:
        return 1
    elif bi <= BIC_BUCKETS[1][0]:
        return 2
    return 3


def calculate_bic(bi: float) -> float:
    """Business Indicator Component (OPE25.7).

    BIC = 12% × min(BI, 1bn) + 15% × max(min(BI, 30bn) - 1bn, 0) + 18% × max(BI - 30bn, 0)
    """
    bic = 0.0
    prev_threshold = 0.0
    for threshold, coeff in BIC_BUCKETS:
        slice_amount = min(bi, threshold) - prev_threshold
        if slice_amount > 0:
            bic += coeff * slice_amount
        prev_threshold = threshold
    return bic


def calculate_ilm(bic: float, loss_component: float) -> float:
    """Internal Loss Multiplier (OPE25.11).

    ILM = ln(exp(1) - 1 + (LC/BIC)^0.8)

    Where LC = 15 × average annual op loss (10-year).
    ILM floored at 1.0 for bucket 1 banks (national discretion allows
    setting ILM=1 for all buckets).
    """
    if bic <= 0:
        return 1.0
    ratio = loss_component / bic
    return math.log(math.exp(1) - 1 + ratio ** 0.8)


def calculate_sma_full(inputs: SMAInputs) -> SMAResult:
    """Full SMA calculation with 3-year averaging and ILM.

    Args:
        inputs: SMAInputs with 3-year P&L and optional 10-year loss data.
    """
    # BI per year and average
    bi_yearly = calculate_business_indicator(inputs)
    bi_avg = sum(bi_yearly) / 3.0

    bucket = sma_bucket(bi_avg)
    bic = calculate_bic(bi_avg)

    # Loss component: 15 × average annual op loss
    if inputs.annual_op_losses:
        avg_loss = sum(inputs.annual_op_losses) / len(inputs.annual_op_losses)
        loss_component = 15.0 * avg_loss
    else:
        loss_component = bic  # default: ILM = 1

    # ILM (only for bucket 2/3 by default)
    use_ilm = bucket >= 2 and len(inputs.annual_op_losses) >= 5
    ilm = calculate_ilm(bic, loss_component) if use_ilm else 1.0

    capital = bic * ilm
    rwa = capital / 0.08  # RWA = capital / 8%

    # Legacy comparison: BIA
    gross_income = [
        (inputs.interest_income[i] - inputs.interest_expense[i] +
         inputs.fee_income[i] - inputs.fee_expense[i] +
         inputs.net_trading_income[i] + inputs.other_operating_income[i])
        for i in range(3)
    ]
    positive_gi = [max(gi, 0) for gi in gross_income]
    legacy_bia = sum(positive_gi) / 3.0 * 0.15 if any(gi > 0 for gi in positive_gi) else 0.0

    return SMAResult(
        bi_yearly=bi_yearly, bi_average=bi_avg,
        bucket=bucket, bic=bic,
        loss_component=loss_component, ilm=ilm,
        use_ilm=use_ilm, capital_requirement=capital,
        rwa=rwa, legacy_bia=legacy_bia,
    )


def sma_sensitivity(
    bi: float,
    loss_range: tuple[float, float] = (0.0, 1.0),
    n_points: int = 20,
) -> list[dict]:
    """SMA capital sensitivity to loss component / BIC ratio.

    Args:
        bi: business indicator.
        loss_range: range of LC/BIC ratios to test.
        n_points: number of grid points.
    """
    bic = calculate_bic(bi)
    results = []
    for i in range(n_points):
        ratio = loss_range[0] + (loss_range[1] - loss_range[0]) * i / max(n_points - 1, 1)
        lc = bic * ratio
        ilm = calculate_ilm(bic, lc)
        cap = bic * ilm
        results.append({"lc_bic_ratio": ratio, "ilm": ilm, "capital": cap})
    return results
