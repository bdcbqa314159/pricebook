"""Portfolio-wide LCR/NSFR with product-type classification.

    from pricebook.regulatory.liquidity import (
        calculate_portfolio_lcr, LiquidityPosition, PortfolioLiquidityResult,
    )

References:
    Basel Committee (2013). LIQ30: Liquidity Coverage Ratio.
    Basel Committee (2014). LIQ40: Net Stable Funding Ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.regulatory.liquidity_op import (
    calculate_hqla, calculate_cash_outflows, calculate_cash_inflows,
    calculate_lcr,
)


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class LiquidityPosition:
    """Single position for liquidity classification."""
    position_id: str
    product_type: str      # "bond", "repo", "loan", "derivative", "deposit", "equity", "cash"
    notional: float
    market_value: float = 0.0
    maturity_days: int = 365
    rating: str = "BBB"
    is_asset: bool = True
    counterparty_type: str = "corporate"  # "sovereign", "bank", "corporate", "retail"
    hqla_level: str = ""   # "L1", "L2A", "L2B", ""
    is_secured: bool = False

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class PortfolioLiquidityResult:
    """Portfolio-level LCR and NSFR result."""
    # LCR
    hqla_total: float
    total_outflows: float
    total_inflows: float
    net_outflows: float
    lcr_pct: float
    lcr_compliant: bool
    # NSFR
    asf_total: float
    rsf_total: float
    nsfr_pct: float
    nsfr_compliant: bool
    # Breakdown
    product_breakdown: dict[str, dict]

    def to_dict(self) -> dict:
        return vars(self)


# ═══════════════════════════════════════════════════════════════
# Product-Type LCR Rules
# ═══════════════════════════════════════════════════════════════

# Map product_type + attributes → LCR HQLA / outflow / inflow categories
def _classify_asset_for_lcr(pos: LiquidityPosition) -> dict | None:
    """Classify an asset position for LCR HQLA."""
    if pos.hqla_level == "L1":
        return {"asset_type": "L1_sovereign_0pct", "amount": pos.market_value or pos.notional}
    elif pos.hqla_level == "L2A":
        return {"asset_type": "L2A_corporate_bonds_AA", "amount": pos.market_value or pos.notional}
    elif pos.hqla_level == "L2B":
        return {"asset_type": "L2B_corporate_bonds_A_BBB", "amount": pos.market_value or pos.notional}

    if pos.product_type == "cash":
        return {"asset_type": "L1_cash", "amount": pos.notional}
    if pos.product_type == "bond" and pos.counterparty_type == "sovereign" and pos.rating in ("AAA", "AA"):
        return {"asset_type": "L1_sovereign_0pct", "amount": pos.market_value or pos.notional}
    if pos.product_type == "bond" and pos.rating in ("AAA", "AA"):
        return {"asset_type": "L2A_corporate_bonds_AA", "amount": pos.market_value or pos.notional}

    return None  # not HQLA


def _classify_liability_for_lcr(pos: LiquidityPosition) -> dict | None:
    """Classify a liability/outflow position for LCR."""
    if pos.is_asset:
        return None

    if pos.product_type == "deposit":
        if pos.counterparty_type == "retail":
            return {"liability_type": "retail_stable", "amount": pos.notional}
        elif pos.counterparty_type == "corporate":
            return {"liability_type": "wholesale_non_operational_uninsured", "amount": pos.notional}
        else:
            return {"liability_type": "wholesale_financial_institution", "amount": pos.notional}

    if pos.product_type == "repo":
        if pos.is_secured:
            return {"liability_type": "secured_L1", "amount": pos.notional}
        return {"liability_type": "secured_other", "amount": pos.notional}

    if pos.product_type == "derivative":
        return {"liability_type": "derivative_outflows", "amount": abs(pos.market_value)}

    return {"liability_type": "other_contingent", "amount": pos.notional}


def _classify_inflow_for_lcr(pos: LiquidityPosition) -> dict | None:
    """Classify an asset position as LCR inflow (maturing within 30 days)."""
    if not pos.is_asset or pos.maturity_days > 30:
        return None

    if pos.product_type == "loan":
        if pos.counterparty_type == "retail":
            return {"receivable_type": "retail_loans", "amount": pos.notional}
        elif pos.counterparty_type in ("bank", "financial"):
            return {"receivable_type": "wholesale_financial", "amount": pos.notional}
        return {"receivable_type": "wholesale_non_financial", "amount": pos.notional}

    if pos.product_type == "repo" and pos.is_secured:
        return {"receivable_type": "secured_L1", "amount": pos.notional}

    return {"receivable_type": "wholesale_non_financial", "amount": pos.notional}


# ═══════════════════════════════════════════════════════════════
# NSFR Factors
# ═══════════════════════════════════════════════════════════════

# Required Stable Funding factors by product type and maturity
def _rsf_factor(pos: LiquidityPosition) -> float:
    """RSF factor for an asset position."""
    if pos.product_type == "cash":
        return 0.0
    if pos.hqla_level == "L1":
        return 0.05
    if pos.hqla_level == "L2A":
        return 0.15
    if pos.hqla_level == "L2B":
        return 0.50

    if pos.maturity_days <= 180:
        if pos.counterparty_type in ("bank", "financial"):
            return 0.10
        return 0.50
    if pos.maturity_days <= 365:
        return 0.50
    # > 1 year
    if pos.product_type == "loan" and pos.counterparty_type == "retail":
        return 0.65  # retail mortgage ≤ 35% RW
    return 0.85  # corporate loans > 1Y


def _asf_factor(pos: LiquidityPosition) -> float:
    """Available Stable Funding factor for a liability position."""
    if pos.product_type == "equity":
        return 1.0  # regulatory capital
    if pos.maturity_days > 365:
        return 1.0  # > 1Y funding
    if pos.maturity_days > 180:
        return 0.50
    if pos.product_type == "deposit" and pos.counterparty_type == "retail":
        return 0.90  # stable retail deposits
    if pos.product_type == "deposit":
        return 0.50  # wholesale deposits < 6M
    return 0.0  # short-term wholesale


# ═══════════════════════════════════════════════════════════════
# Portfolio LCR/NSFR
# ═══════════════════════════════════════════════════════════════

def calculate_portfolio_lcr(
    positions: list[LiquidityPosition],
    inflow_cap_rate: float = 0.75,
) -> PortfolioLiquidityResult:
    """Portfolio-wide LCR and NSFR from typed positions.

    Args:
        positions: list of LiquidityPosition objects.
        inflow_cap_rate: inflow cap as fraction of outflows (75% standard).
    """
    hqla_assets = []
    liabilities = []
    receivables = []
    product_breakdown: dict[str, dict] = {}

    # Classify each position
    for pos in positions:
        pt = pos.product_type
        if pt not in product_breakdown:
            product_breakdown[pt] = {"count": 0, "notional": 0.0, "hqla": 0.0, "outflow": 0.0, "inflow": 0.0}
        product_breakdown[pt]["count"] += 1
        product_breakdown[pt]["notional"] += pos.notional

        if pos.is_asset:
            # HQLA classification
            hqla = _classify_asset_for_lcr(pos)
            if hqla:
                hqla_assets.append(hqla)
                product_breakdown[pt]["hqla"] += hqla["amount"]

            # Inflow (maturing within 30 days)
            inflow = _classify_inflow_for_lcr(pos)
            if inflow:
                receivables.append(inflow)
                product_breakdown[pt]["inflow"] += inflow["amount"]
        else:
            # Outflow
            outflow = _classify_liability_for_lcr(pos)
            if outflow:
                liabilities.append(outflow)
                product_breakdown[pt]["outflow"] += outflow["amount"]

    # LCR via liquidity_op
    lcr_result = calculate_lcr(hqla_assets, liabilities, receivables, inflow_cap_rate)
    hqla_total = lcr_result["hqla"]
    total_out = lcr_result["gross_outflows"]
    total_in = lcr_result["gross_inflows"]
    net_out = lcr_result["net_outflows"]
    lcr_pct = lcr_result["lcr_pct"]

    # NSFR
    asf = 0.0
    rsf = 0.0
    for pos in positions:
        if pos.is_asset:
            rsf += pos.notional * _rsf_factor(pos)
        else:
            asf += pos.notional * _asf_factor(pos)

    nsfr_pct = asf / rsf * 100 if rsf > 0 else float('inf')

    return PortfolioLiquidityResult(
        hqla_total=hqla_total,
        total_outflows=total_out,
        total_inflows=total_in,
        net_outflows=net_out,
        lcr_pct=lcr_pct,
        lcr_compliant=lcr_pct >= 100,
        asf_total=asf,
        rsf_total=rsf,
        nsfr_pct=nsfr_pct,
        nsfr_compliant=nsfr_pct >= 100,
        product_breakdown=product_breakdown,
    )


def liquidity_stress(
    positions: list[LiquidityPosition],
    outflow_multiplier: float = 1.5,
    hqla_haircut: float = 0.10,
) -> PortfolioLiquidityResult:
    """Stressed LCR under adverse scenario.

    Args:
        outflow_multiplier: multiply outflows (e.g. 1.5 = 50% higher run-off).
        hqla_haircut: additional haircut on HQLA values.
    """
    stressed = []
    for pos in positions:
        notional = pos.notional * outflow_multiplier if not pos.is_asset else pos.notional
        mv = (pos.market_value or pos.notional) * (1 - hqla_haircut) if pos.is_asset else pos.market_value
        stressed.append(LiquidityPosition(
            position_id=pos.position_id, product_type=pos.product_type,
            notional=notional, market_value=mv,
            maturity_days=pos.maturity_days, rating=pos.rating,
            is_asset=pos.is_asset, counterparty_type=pos.counterparty_type,
            hqla_level=pos.hqla_level, is_secured=pos.is_secured,
        ))
    return calculate_portfolio_lcr(stressed)
