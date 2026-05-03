"""Repo regulatory capital: SFT framework (Basel CRE50), LCR, NSFR.

Secured Financing Transaction (SFT) capital treatment under Basel III/IV.
Reuses existing regulatory infrastructure — no duplication.

    from pricebook.regulatory.repo_capital import (
        sft_ead, repo_rwa, repo_lcr_outflow, repo_nsfr_rsf,
        repo_capital_summary,
    )

References:
    Basel Committee (2017). Basel III: Finalising post-crisis reforms (CRE50).
    Basel Committee (2013). Basel III: Liquidity Coverage Ratio (LIQ30).
    Basel Committee (2014). Basel III: Net Stable Funding Ratio (LIQ40).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# SFT EAD (Basel CRE50)
# ---------------------------------------------------------------------------

# SA risk weights by counterparty type
SA_RISK_WEIGHTS = {
    "sovereign": 0.0,
    "central_bank": 0.0,
    "bank": 0.20,
    "broker_dealer": 0.20,
    "corporate": 1.00,
    "hedge_fund": 1.50,
    "ccp": 0.02,
    "other": 1.00,
}


def sft_ead(
    cash_lent: float,
    collateral_value: float,
    supervisory_haircut: float = 0.0,
) -> float:
    """Exposure at Default for a Secured Financing Transaction.

    EAD = max(0, E - C × (1 - Hs))

    where E = cash lent, C = collateral market value,
    Hs = supervisory haircut on the collateral.

    For over-collateralised repos: EAD = 0 (no credit exposure).

    Args:
        cash_lent: amount of cash lent (exposure).
        collateral_value: market value of collateral received.
        supervisory_haircut: Basel supervisory haircut on collateral.
    """
    adjusted_collateral = collateral_value * (1 - supervisory_haircut)
    return max(0.0, cash_lent - adjusted_collateral)


def repo_rwa(
    ead: float,
    counterparty_type: str = "bank",
) -> float:
    """Risk-Weighted Assets for a repo.

    RWA = EAD × risk_weight.

    Args:
        ead: from sft_ead().
        counterparty_type: "sovereign", "bank", "corporate", etc.
    """
    rw = SA_RISK_WEIGHTS.get(counterparty_type, 1.0)
    return ead * rw


def repo_capital_requirement(
    rwa: float,
    capital_ratio: float = 0.08,
) -> float:
    """Minimum capital = RWA × capital ratio (8% Basel minimum)."""
    return rwa * capital_ratio


# ---------------------------------------------------------------------------
# LCR treatment (LIQ30)
# ---------------------------------------------------------------------------

# Cash outflow rates for secured funding by collateral level
LCR_OUTFLOW_RATES = {
    "L1": 0.0,      # Level 1 HQLA (govt): 0% outflow (assumed to roll)
    "L2A": 0.15,    # Level 2A (agency, covered bonds): 15%
    "L2B": 0.25,    # Level 2B (IG corp, equity): 25%
    "other": 1.00,  # Non-HQLA: 100% outflow (assumed not to roll)
}

# Collateral type → HQLA level mapping
COLLATERAL_HQLA = {
    "sovereign": "L1",
    "govt": "L1",
    "GC": "L1",
    "agency": "L2A",
    "ig_corp": "L2B",
    "equity": "L2B",
    "hy_corp": "other",
    "special": "L1",  # specials are typically govt bonds
}


def repo_lcr_outflow(
    cash_amount: float,
    collateral_type: str = "GC",
    remaining_days: int = 30,
) -> float:
    """LCR cash outflow for a repo maturing within 30 days.

    Only repos maturing within the LCR horizon (30d) count.
    Outflow rate depends on collateral quality.

    Returns: expected cash outflow amount.
    """
    if remaining_days > 30:
        return 0.0  # beyond LCR horizon
    hqla_level = COLLATERAL_HQLA.get(collateral_type, "other")
    outflow_rate = LCR_OUTFLOW_RATES.get(hqla_level, 1.0)
    return cash_amount * outflow_rate


# ---------------------------------------------------------------------------
# NSFR treatment (LIQ40)
# ---------------------------------------------------------------------------

# RSF factors for SFTs by collateral and maturity
NSFR_RSF = {
    ("L1", "short"):  0.10,  # < 6M, L1 collateral: 10%
    ("L1", "long"):   0.15,  # ≥ 6M, L1 collateral: 15%
    ("L2A", "short"): 0.15,
    ("L2A", "long"):  0.50,
    ("L2B", "short"): 0.50,
    ("L2B", "long"):  0.50,
    ("other", "short"): 0.50,
    ("other", "long"):  1.00,
}


def repo_nsfr_rsf(
    cash_amount: float,
    collateral_type: str = "GC",
    remaining_days: int = 30,
) -> float:
    """NSFR Required Stable Funding for a repo.

    RSF depends on collateral quality and remaining maturity.

    Returns: RSF amount (required stable funding).
    """
    hqla_level = COLLATERAL_HQLA.get(collateral_type, "other")
    bucket = "short" if remaining_days < 180 else "long"
    rsf_factor = NSFR_RSF.get((hqla_level, bucket), 1.0)
    return cash_amount * rsf_factor


# ---------------------------------------------------------------------------
# Capital summary for repo book
# ---------------------------------------------------------------------------

@dataclass
class RepoCapitalSummary:
    """Regulatory capital summary for a repo book."""
    n_trades: int
    total_ead: float
    total_rwa: float
    total_capital: float
    total_lcr_outflow: float
    total_nsfr_rsf: float

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "total_ead": self.total_ead,
            "total_rwa": self.total_rwa,
            "total_capital": self.total_capital,
            "lcr_outflow": self.total_lcr_outflow,
            "nsfr_rsf": self.total_nsfr_rsf,
        }


def repo_capital_summary(
    trades: list[dict],
    capital_ratio: float = 0.08,
) -> RepoCapitalSummary:
    """Aggregate capital summary across a repo book.

    Args:
        trades: list of {cash_lent, collateral_value, supervisory_haircut,
                         counterparty_type, collateral_type, remaining_days}.
    """
    total_ead = 0.0
    total_rwa = 0.0
    total_lcr = 0.0
    total_nsfr = 0.0

    for t in trades:
        ead = sft_ead(t["cash_lent"], t["collateral_value"],
                       t.get("supervisory_haircut", 0.0))
        rwa = repo_rwa(ead, t.get("counterparty_type", "bank"))
        lcr = repo_lcr_outflow(t["cash_lent"], t.get("collateral_type", "GC"),
                                t.get("remaining_days", 30))
        nsfr = repo_nsfr_rsf(t["cash_lent"], t.get("collateral_type", "GC"),
                              t.get("remaining_days", 30))

        total_ead += ead
        total_rwa += rwa
        total_lcr += lcr
        total_nsfr += nsfr

    return RepoCapitalSummary(
        n_trades=len(trades),
        total_ead=total_ead,
        total_rwa=total_rwa,
        total_capital=total_rwa * capital_ratio,
        total_lcr_outflow=total_lcr,
        total_nsfr_rsf=total_nsfr,
    )
