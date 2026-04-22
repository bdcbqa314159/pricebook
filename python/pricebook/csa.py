"""CSA and funding framework — generic collateral and funding adjustments for any trade.

A CSA (Credit Support Annex) defines the collateral terms between two counterparties.
The funding model computes the cost or benefit of collateral posting and
adjusts any trade's PV accordingly.

Usage:
    csa = CSA(threshold=1_000_000, mta=50_000, rounding=10_000)
    funding = FundingModel(secured_rate=0.05, unsecured_rate=0.052)

    # Works with any trade that has a pv(ctx) method
    adj = collateral_adjusted_pv(trade, ctx, csa, funding)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext


# ---------------------------------------------------------------------------
# CSA terms
# ---------------------------------------------------------------------------


class CollateralType(Enum):
    CASH = "cash"
    GOVERNMENT_BOND = "government_bond"
    CORPORATE_BOND = "corporate_bond"


class MarginFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class CSA:
    """Credit Support Annex — collateral agreement terms.

    Args:
        threshold: exposure below which no collateral is required.
        mta: minimum transfer amount (smallest collateral move).
        rounding: collateral amounts rounded to this.
        margin_frequency: how often collateral is exchanged.
        eligible_collateral: types of collateral accepted.
        haircut: percentage haircut on non-cash collateral.
        rehypothecation: whether posted collateral can be reused.
        initial_margin: independent amount (posted regardless of exposure).
        currency: collateral currency.
    """

    threshold: float = 0.0
    mta: float = 0.0
    rounding: float = 1.0
    margin_frequency: MarginFrequency = MarginFrequency.DAILY
    eligible_collateral: list[CollateralType] = field(
        default_factory=lambda: [CollateralType.CASH]
    )
    haircut: float = 0.0
    rehypothecation: bool = True
    initial_margin: float = 0.0
    currency: str = "USD"

    @property
    def is_fully_collateralised(self) -> bool:
        return self.threshold == 0.0 and self.mta == 0.0

    @property
    def margin_period_days(self) -> int:
        if self.margin_frequency == MarginFrequency.DAILY:
            return 1
        elif self.margin_frequency == MarginFrequency.WEEKLY:
            return 7
        return 30


def required_collateral(exposure: float, csa: CSA) -> float:
    """Compute collateral required given current exposure and CSA terms.

    Args:
        exposure: current mark-to-market (positive = we are owed).
        csa: the CSA terms.

    Returns:
        Collateral amount the counterparty must post (>= 0).
    """
    excess = max(exposure - csa.threshold, 0.0)
    if excess < csa.mta:
        return 0.0

    # Apply rounding
    if csa.rounding > 0:
        excess = math.floor(excess / csa.rounding) * csa.rounding

    # Add initial margin
    return excess + csa.initial_margin


def uncollateralised_exposure(exposure: float, csa: CSA) -> float:
    """The portion of exposure not covered by collateral."""
    collateral = required_collateral(exposure, csa)
    return max(exposure - collateral, 0.0)


# ---------------------------------------------------------------------------
# Funding model
# ---------------------------------------------------------------------------


@dataclass
class FundingModel:
    """Funding costs for secured and unsecured positions.

    Args:
        secured_rate: funding rate for collateralised exposure (e.g. OIS).
        unsecured_rate: funding rate for uncollateralised exposure.
        collateral_rate: rate earned/paid on posted collateral.
    """

    secured_rate: float = 0.05
    unsecured_rate: float = 0.055
    collateral_rate: float = 0.05  # typically OIS

    @property
    def funding_spread(self) -> float:
        """Spread between unsecured and secured funding."""
        return self.unsecured_rate - self.secured_rate


# ---------------------------------------------------------------------------
# Generic PV adjustments — works with any trade
# ---------------------------------------------------------------------------


def _get_trade_pv(trade, ctx: PricingContext) -> float:
    """Extract PV from any trade-like object."""
    if hasattr(trade, "pv"):
        # Trade wrapper or standalone instrument with pv(ctx)
        try:
            return trade.pv(ctx)
        except TypeError:
            pass
    if hasattr(trade, "pv_ctx"):
        return trade.pv_ctx(ctx)
    raise ValueError(f"Object {type(trade).__name__} has no pv or pv_ctx method")


def collateral_adjusted_pv(
    trade,
    ctx: PricingContext,
    csa: CSA,
    funding: FundingModel,
    horizon: float = 1.0,
) -> dict:
    """Compute collateral-adjusted PV for any trade.

    Returns a dict with:
        - base_pv: unadjusted PV
        - collateral: required collateral amount
        - uncollateralised: exposure not covered
        - funding_cost: cost of funding uncollateralised exposure
        - collateral_cost: net cost/benefit of posting collateral
        - adjusted_pv: base_pv - funding_cost - collateral_cost
    """
    base_pv = _get_trade_pv(trade, ctx)

    collateral = required_collateral(base_pv, csa)
    uncoll = max(base_pv - collateral, 0.0)

    # Funding cost: pay funding spread on uncollateralised portion
    funding_cost = uncoll * funding.funding_spread * horizon

    # Collateral cost: difference between what we earn on collateral vs our funding
    # If we receive collateral: we earn collateral_rate, save unsecured_rate
    # Net benefit = collateral * (unsecured_rate - collateral_rate) * horizon
    # If we post collateral (negative exposure): cost = |collateral| * (funding - collateral_rate)
    if base_pv >= 0:
        # We receive collateral → funding benefit
        collateral_cost = -collateral * (funding.unsecured_rate - funding.collateral_rate) * horizon
    else:
        # We post collateral → funding cost
        collateral_cost = abs(collateral) * (funding.unsecured_rate - funding.collateral_rate) * horizon

    adjusted_pv = base_pv - funding_cost - collateral_cost

    return {
        "base_pv": base_pv,
        "collateral": collateral,
        "uncollateralised": uncoll,
        "funding_cost": funding_cost,
        "collateral_cost": collateral_cost,
        "adjusted_pv": adjusted_pv,
    }


def funding_benefit_analysis(
    trade,
    ctx: PricingContext,
    csa_options: list[tuple[str, CSA]],
    funding: FundingModel,
    horizon: float = 1.0,
) -> list[dict]:
    """Compare funding impact across different CSA terms.

    Useful for analysing the benefit of tighter collateral agreements.

    Args:
        trade: any priceable object.
        ctx: pricing context.
        csa_options: list of (name, CSA) pairs to compare.
        funding: funding model.
        horizon: time horizon for funding cost calculation.

    Returns:
        List of dicts with name + all collateral_adjusted_pv fields.
    """
    results = []
    for name, csa in csa_options:
        adj = collateral_adjusted_pv(trade, ctx, csa, funding, horizon)
        adj["name"] = name
        results.append(adj)
    return results


# ---------------------------------------------------------------------------
# CSA-aware discounting (COL1)
# ---------------------------------------------------------------------------


@dataclass
class CSADiscountResult:
    """Result of CSA-aware discount curve selection."""
    csa: CSA
    collateral_currency: str
    discount_curve_name: str
    is_cleared: bool


def csa_discount_curve(
    csa: CSA,
    trade_currency: str,
    discount_curves: dict[str, DiscountCurve],
    xccy_basis_curves: dict[str, DiscountCurve] | None = None,
) -> DiscountCurve:
    """Select the correct discount curve based on CSA terms.

    Under CSA collateralisation, the discount curve depends on the
    collateral currency, not the trade currency:

    - If collateral currency == trade currency: use OIS curve
    - If collateral currency != trade currency: use OIS in collateral
      currency + cross-currency basis adjustment

    Args:
        csa: CSA terms (contains collateral currency).
        trade_currency: currency of the trade cashflows.
        discount_curves: {currency → OIS DiscountCurve}.
        xccy_basis_curves: {currency_pair → basis-adjusted curve} (optional).

    Returns:
        The appropriate DiscountCurve for PV calculation.
    """
    coll_ccy = csa.currency

    if coll_ccy == trade_currency:
        # Same currency: use OIS curve
        if coll_ccy in discount_curves:
            return discount_curves[coll_ccy]
        raise ValueError(f"No discount curve for {coll_ccy}")

    # Cross-currency: look for a basis-adjusted curve
    if xccy_basis_curves is not None:
        pair_key = f"{trade_currency}_{coll_ccy}"
        alt_key = f"{coll_ccy}_{trade_currency}"
        if pair_key in xccy_basis_curves:
            return xccy_basis_curves[pair_key]
        if alt_key in xccy_basis_curves:
            return xccy_basis_curves[alt_key]

    # Fallback: use collateral currency OIS curve
    if coll_ccy in discount_curves:
        return discount_curves[coll_ccy]

    # Last resort: use trade currency
    if trade_currency in discount_curves:
        return discount_curves[trade_currency]

    raise ValueError(f"No suitable discount curve for CSA({coll_ccy}) on {trade_currency} trade")


def colva(
    exposure_profile: list[float],
    collateral_profile: list[float],
    collateral_rate: float,
    discount_rate: float,
    dt: float,
) -> float:
    """Collateral Value Adjustment (ColVA).

    ColVA = cost of posting collateral at a rate different from the
    discount rate. If we earn collateral_rate on posted collateral
    but discount at discount_rate, the difference is a cost (or benefit).

    ColVA = Σ (discount_rate - collateral_rate) × collateral_t × dt × df_t

    Args:
        exposure_profile: expected exposure at each time step.
        collateral_profile: posted collateral at each time step.
        collateral_rate: rate earned/paid on collateral.
        discount_rate: rate used for discounting.
        dt: time step.
    """
    import math
    total = 0.0
    spread = discount_rate - collateral_rate
    for i, (exp, coll) in enumerate(zip(exposure_profile, collateral_profile)):
        t = (i + 1) * dt
        df = math.exp(-discount_rate * t)
        total += spread * coll * dt * df
    return total
