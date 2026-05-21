"""Bond spread decomposition.

Decomposes a corporate/sovereign bond spread into its components:
credit risk, liquidity premium, tax effect, and optionality.

    from pricebook.credit.spread_decomposition import (
        decompose_spread, SpreadComponents,
    )

    result = decompose_spread(
        bond_spread_bp=250,
        cds_spread_bp=180,
        bid_ask_bp=10,
        tax_rate=0.0,
        oas_bp=None,
    )

The total spread can be written as:
    s_bond = s_credit + s_liquidity + s_tax + s_optionality + s_residual

References:
    Longstaff, Mithal & Neis (2005). Corporate Yield Spreads: Default Risk
    or Liquidity? Journal of Finance.
    Collin-Dufresne, Goldstein & Martin (2001). The Determinants of Credit
    Spread Changes. Journal of Finance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpreadComponents:
    """Decomposed bond spread."""
    total_spread_bp: float
    credit_bp: float             # CDS-implied credit component
    liquidity_bp: float          # Residual non-credit, non-tax spread
    tax_bp: float                # Tax effect (muni bonds, some sovereigns)
    optionality_bp: float        # Embedded option value (callable, putable)
    residual_bp: float           # Unexplained (model error, technicals)

    @property
    def non_credit_bp(self) -> float:
        """Total non-credit spread."""
        return self.liquidity_bp + self.tax_bp + self.optionality_bp + self.residual_bp

    @property
    def credit_share(self) -> float:
        """Fraction of total spread attributable to credit."""
        if self.total_spread_bp == 0:
            return 0.0
        return self.credit_bp / self.total_spread_bp

    def to_dict(self) -> dict:
        return {
            **vars(self),
            "non_credit_bp": self.non_credit_bp,
            "credit_share": self.credit_share,
        }


def decompose_spread(
    bond_spread_bp: float,
    cds_spread_bp: float | None = None,
    bid_ask_bp: float = 0.0,
    tax_rate: float = 0.0,
    oas_bp: float | None = None,
    risk_free_bp: float = 0.0,
) -> SpreadComponents:
    """Decompose a bond spread into components.

    The methodology:
    1. Credit = CDS spread (if available) — market-implied default risk
    2. Tax = tax_rate × (bond_yield - risk_free) approximation
    3. Optionality = bond_spread - OAS (if OAS provided)
    4. Liquidity = bid_ask heuristic + residual allocation
    5. Residual = total - credit - liquidity - tax - optionality

    Args:
        bond_spread_bp: total bond spread over risk-free (bp).
        cds_spread_bp: CDS spread if available (bp). Best credit proxy.
        bid_ask_bp: bid-ask spread (bp). Liquidity proxy.
        tax_rate: marginal tax rate (0 for tax-exempt investors).
        oas_bp: option-adjusted spread (bp). If the bond has embedded options.
        risk_free_bp: risk-free yield in bp (for tax calculation).
    """
    total = bond_spread_bp

    # 1. Credit component
    if cds_spread_bp is not None:
        credit = min(cds_spread_bp, total)
    else:
        # No CDS: estimate credit as ~70% of spread (Longstaff et al. 2005)
        credit = total * 0.70

    # 2. Tax component
    tax = tax_rate * (total + risk_free_bp) / 100 * 10_000 if tax_rate > 0 else 0.0
    tax = min(tax, max(total - credit, 0))

    # 3. Optionality component
    if oas_bp is not None:
        optionality = max(total - oas_bp, 0.0)
    else:
        optionality = 0.0

    # 4. Liquidity component
    # Heuristic: liquidity ≈ 2 × bid-ask + base (Longstaff et al. find ~50bp avg)
    liquidity_est = bid_ask_bp * 2.0 + 5.0  # base liquidity premium
    remaining = total - credit - tax - optionality
    liquidity = min(liquidity_est, max(remaining, 0))

    # 5. Residual
    residual = total - credit - liquidity - tax - optionality
    if residual < 0:
        # Over-attributed: reduce liquidity
        liquidity = max(liquidity + residual, 0)
        residual = total - credit - liquidity - tax - optionality

    return SpreadComponents(
        total_spread_bp=total,
        credit_bp=credit,
        liquidity_bp=liquidity,
        tax_bp=tax,
        optionality_bp=optionality,
        residual_bp=residual,
    )


def cds_bond_basis(
    bond_spread_bp: float,
    cds_spread_bp: float,
) -> float:
    """CDS-bond basis: bond spread minus CDS spread (bp).

    Positive basis = bond spread > CDS → bond is cheap vs CDS (or illiquid).
    Negative basis = CDS > bond → CDS is cheap (rare, arbitrage opportunity).

    The basis captures liquidity, funding, and delivery option effects.
    """
    return bond_spread_bp - cds_spread_bp


def decompose_portfolio(
    bonds: list[dict],
) -> dict:
    """Decompose spreads for a portfolio of bonds.

    Args:
        bonds: list of dicts with keys: name, bond_spread_bp, cds_spread_bp (optional),
               bid_ask_bp (optional), weight (optional, default 1.0).

    Returns:
        Portfolio-level decomposition with weighted averages.
    """
    if not bonds:
        raise ValueError("At least one bond required")

    total_weight = sum(b.get("weight", 1.0) for b in bonds)
    decomps = []
    for b in bonds:
        d = decompose_spread(
            b["bond_spread_bp"],
            b.get("cds_spread_bp"),
            b.get("bid_ask_bp", 5.0),
        )
        decomps.append((d, b.get("weight", 1.0)))

    # Weighted averages
    def wavg(attr):
        return sum(getattr(d, attr) * w for d, w in decomps) / total_weight

    return {
        "n_bonds": len(bonds),
        "total_spread_bp": wavg("total_spread_bp"),
        "credit_bp": wavg("credit_bp"),
        "liquidity_bp": wavg("liquidity_bp"),
        "tax_bp": wavg("tax_bp"),
        "optionality_bp": wavg("optionality_bp"),
        "residual_bp": wavg("residual_bp"),
        "avg_credit_share": wavg("credit_share"),
    }
