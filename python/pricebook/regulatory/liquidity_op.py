"""Liquidity (LCR/NSFR) + Operational risk (SMA).

LCR (LIQ30): HQLA / Net Cash Outflows ≥ 100%
NSFR (LIQ40): ASF / RSF ≥ 100%
SMA (OPE25): K = BIC × ILM (Business Indicator Component × Internal Loss Multiplier)

    from pricebook.regulatory.liquidity_op import (
        calculate_hqla, calculate_lcr,
        calculate_asf, calculate_rsf, calculate_nsfr,
        calculate_business_indicator, calculate_bic, calculate_ilm,
        calculate_sma_capital,
    )
"""

from __future__ import annotations

import math


# =============================================================================
# LCR (LIQ30)
# =============================================================================

# HQLA haircuts (LIQ30.35-48)
HQLA_HAIRCUTS: dict[str, float] = {
    "L1_cash": 0.0,
    "L1_central_bank_reserves": 0.0,
    "L1_sovereign_0pct": 0.0,
    "L2A_sovereign_20pct": 0.15,
    "L2A_covered_bonds_AA": 0.15,
    "L2A_corporate_bonds_AA": 0.15,
    "L2B_rmbs_AA": 0.25,
    "L2B_corporate_bonds_A_BBB": 0.50,
    "L2B_equity_major_index": 0.50,
}

# Cash outflow rates (LIQ30.52-76)
CASH_OUTFLOW_RATES: dict[str, float] = {
    "retail_stable": 0.03, "retail_less_stable": 0.10, "retail_high_runoff": 0.20,
    "wholesale_operational": 0.25,
    "wholesale_non_operational_insured": 0.40,
    "wholesale_non_operational_uninsured": 1.00,
    "wholesale_financial_institution": 1.00,
    "secured_L1": 0.0, "secured_L2A": 0.15, "secured_L2B": 0.25, "secured_other": 1.00,
    "committed_credit_retail": 0.05,
    "committed_credit_corporate": 0.10,
    "committed_credit_financial": 1.00,
    "committed_liquidity_facility": 1.00,
    "derivative_outflows": 1.00,
    "other_contingent": 1.00,
}

# Cash inflow rates (LIQ30.77-90)
CASH_INFLOW_RATES: dict[str, float] = {
    "retail_loans": 0.50,
    "wholesale_non_financial": 0.50,
    "wholesale_financial": 1.00,
    "secured_L1": 0.0, "secured_L2A": 0.15, "secured_L2B": 0.25, "secured_other": 1.00,
}


def calculate_hqla(assets: list[dict]) -> dict:
    """Calculate HQLA stock with L1/L2A/L2B caps.

    L2A capped at 40/60 of L1; L2B capped at 15/85 of L1.
    """
    level1 = level2a = level2b = 0.0
    for asset in assets:
        amt = asset.get("amount", 0)
        atype = asset.get("asset_type", "")
        haircut = HQLA_HAIRCUTS.get(atype, 0.50)
        adjusted = amt * (1 - haircut)
        if atype.startswith("L1"):
            level1 += adjusted
        elif atype.startswith("L2A"):
            level2a += adjusted
        else:
            level2b += adjusted

    # Apply caps
    max_level2 = level1 * (40 / 60)
    max_level2b = level1 * (15 / 85)
    level2b_adj = min(level2b, max_level2b)
    level2a_adj = min(level2a, max_level2 - level2b_adj)
    total_hqla = level1 + level2a_adj + level2b_adj

    return {
        "level1": level1,
        "level2a_gross": level2a, "level2a_adjusted": level2a_adj,
        "level2b_gross": level2b, "level2b_adjusted": level2b_adj,
        "total_hqla": total_hqla,
    }


def calculate_cash_outflows(liabilities: list[dict]) -> dict:
    """Total expected cash outflows over 30 days."""
    total = 0.0
    details = []
    for lia in liabilities:
        amt = lia.get("amount", 0)
        ltype = lia.get("liability_type", "")
        rate = CASH_OUTFLOW_RATES.get(ltype, 1.0)
        outflow = amt * rate
        total += outflow
        details.append({"liability_type": ltype, "amount": amt, "outflow_rate": rate, "outflow": outflow})
    return {"total_outflows": total, "details": details}


def calculate_cash_inflows(receivables: list[dict]) -> dict:
    """Total expected cash inflows over 30 days (uncapped)."""
    total = 0.0
    details = []
    for r in receivables:
        amt = r.get("amount", 0)
        rtype = r.get("receivable_type", "")
        rate = CASH_INFLOW_RATES.get(rtype, 0.50)
        inflow = amt * rate
        total += inflow
        details.append({"receivable_type": rtype, "amount": amt, "inflow_rate": rate, "inflow": inflow})
    return {"total_inflows_gross": total, "details": details}


def calculate_lcr(
    hqla_assets: list[dict],
    liabilities: list[dict],
    receivables: list[dict],
    inflow_cap_rate: float = 0.75,
) -> dict:
    """LCR = HQLA / max(Outflows - min(Inflows, 75%×Outflows), 25%×Outflows)."""
    hqla = calculate_hqla(hqla_assets)
    out = calculate_cash_outflows(liabilities)
    inf = calculate_cash_inflows(receivables)

    inflow_cap = out["total_outflows"] * inflow_cap_rate
    capped_inflows = min(inf["total_inflows_gross"], inflow_cap)
    net_outflows = out["total_outflows"] - capped_inflows

    lcr = hqla["total_hqla"] / net_outflows if net_outflows > 0 else float("inf")

    return {
        "hqla": hqla["total_hqla"], "hqla_breakdown": hqla,
        "gross_outflows": out["total_outflows"],
        "gross_inflows": inf["total_inflows_gross"],
        "capped_inflows": capped_inflows,
        "net_outflows": net_outflows,
        "lcr": lcr, "lcr_pct": lcr * 100,
        "minimum_requirement": 100,
        "is_compliant": lcr >= 1.0,
        "surplus_deficit": hqla["total_hqla"] - net_outflows,
    }


# =============================================================================
# NSFR (LIQ40)
# =============================================================================

ASF_FACTORS: dict[str, float] = {
    "tier1_capital": 1.00, "tier2_capital": 1.00, "other_capital": 1.00,
    "long_term_debt_1y": 1.00,
    "retail_stable_deposits": 0.95,
    "retail_less_stable_deposits": 0.90,
    "wholesale_operational": 0.50,
    "wholesale_non_operational_1y": 1.00,
    "wholesale_non_operational_6m_1y": 0.50,
    "wholesale_non_operational_lt_6m": 0.0,
    "all_other_liabilities_1y": 1.00,
    "all_other_liabilities_lt_1y": 0.0,
}

RSF_FACTORS: dict[str, float] = {
    "L1_assets": 0.0, "L2A_assets": 0.15, "L2B_assets": 0.50,
    "loans_to_financials_lt_6m": 0.10,
    "loans_to_financials_6m_1y": 0.50,
    "loans_to_financials_gt_1y": 1.00,
    "loans_to_non_financials_lt_1y": 0.50,
    "loans_to_non_financials_gt_1y_lt_35rw": 0.65,
    "loans_to_non_financials_gt_1y_ge_35rw": 0.85,
    "mortgages_gt_1y_lt_35rw": 0.65,
    "mortgages_gt_1y_ge_35rw": 0.85,
    "unencumbered_equity": 0.85,
    "physical_commodities": 0.85,
    "other_assets": 1.00,
    "off_balance_sheet": 0.05,
}


def calculate_asf(funding_sources: list[dict]) -> dict:
    """Available Stable Funding."""
    total = 0.0
    details = []
    for src in funding_sources:
        amt = src.get("amount", 0)
        ftype = src.get("funding_type", "")
        factor = ASF_FACTORS.get(ftype, 0.0)
        weighted = amt * factor
        total += weighted
        details.append({"funding_type": ftype, "amount": amt, "asf_factor": factor, "weighted_amount": weighted})
    return {"total_asf": total, "details": details}


def calculate_rsf(assets: list[dict], off_balance_sheet: float = 0) -> dict:
    """Required Stable Funding."""
    total = 0.0
    details = []
    for asset in assets:
        amt = asset.get("amount", 0)
        atype = asset.get("asset_type", "")
        factor = RSF_FACTORS.get(atype, 1.0)
        weighted = amt * factor
        total += weighted
        details.append({"asset_type": atype, "amount": amt, "rsf_factor": factor, "weighted_amount": weighted})

    obs_rsf = off_balance_sheet * RSF_FACTORS["off_balance_sheet"]
    total += obs_rsf

    return {"total_rsf": total, "details": details, "off_balance_sheet_rsf": obs_rsf}


def calculate_nsfr(
    funding_sources: list[dict],
    assets: list[dict],
    off_balance_sheet: float = 0,
) -> dict:
    """NSFR = ASF / RSF (≥ 100%)."""
    asf = calculate_asf(funding_sources)
    rsf = calculate_rsf(assets, off_balance_sheet)
    nsfr = asf["total_asf"] / rsf["total_rsf"] if rsf["total_rsf"] > 0 else float("inf")

    return {
        "asf": asf["total_asf"], "asf_breakdown": asf,
        "rsf": rsf["total_rsf"], "rsf_breakdown": rsf,
        "nsfr": nsfr, "nsfr_pct": nsfr * 100,
        "minimum_requirement": 100,
        "is_compliant": nsfr >= 1.0,
        "surplus_deficit": asf["total_asf"] - rsf["total_rsf"],
    }


# =============================================================================
# Operational Risk SMA (OPE25)
# =============================================================================

# Business Indicator Component marginal coefficients
BIC_BUCKETS = [
    (0, 1_000_000_000, 0.12),
    (1_000_000_000, 30_000_000_000, 0.15),
    (30_000_000_000, float("inf"), 0.18),
]

ILM_COEFFICIENT = 0.8


def calculate_business_indicator(
    interest_income: float,
    interest_expense: float,
    interest_earning_assets: float,
    fee_income: float,
    fee_expense: float,
    other_operating_income: float = 0,
    other_operating_expense: float = 0,
    dividend_income: float = 0,
    leasing_income: float = 0,
    leasing_expense: float = 0,
    trading_book_pnl: float = 0,
    banking_book_pnl: float = 0,
) -> dict:
    """Business Indicator (BI) = ILDC + SC + FC.

    ILDC: Interest, Leasing, Dividend Component
    SC: Services Component
    FC: Financial Component
    """
    # ILDC
    net_interest = abs(interest_income - interest_expense)
    interest_cap = 0.0225 * interest_earning_assets
    interest_component = min(net_interest, interest_cap)
    net_leasing = abs(leasing_income - leasing_expense)
    ildc = interest_component + net_leasing + dividend_income

    # SC
    sc = max(fee_income, fee_expense) + max(other_operating_income, other_operating_expense)

    # FC
    fc = abs(trading_book_pnl) + abs(banking_book_pnl)

    bi = ildc + sc + fc
    return {"bi": bi, "ildc": ildc, "sc": sc, "fc": fc}


def calculate_bic(bi: float) -> float:
    """Business Indicator Component via piecewise linear marginal coefficients."""
    if bi <= 0:
        return 0.0
    bic = 0.0
    for lo, hi, coef in BIC_BUCKETS:
        if bi <= lo:
            break
        portion = min(bi, hi) - lo
        bic += portion * coef
    return bic


def calculate_loss_component(average_annual_loss: float) -> float:
    """LC = 15 × average annual operational loss (10-year average)."""
    return 15 * average_annual_loss


def calculate_ilm(bic: float, lc: float, use_ilm: bool = True) -> float:
    """Internal Loss Multiplier: ILM = ln(e - 1 + (LC/BIC)^0.8).

    If bank doesn't use ILM (small banks, bucket 1), ILM = 1.
    """
    if not use_ilm or bic <= 0:
        return 1.0
    ratio = lc / bic
    ilm = math.log(math.exp(1) - 1 + ratio ** ILM_COEFFICIENT)
    return max(ilm, 0)


def calculate_sma_capital(
    bi: float,
    average_annual_loss: float = 0,
    use_ilm: bool = True,
) -> dict:
    """SMA capital = BIC × ILM."""
    bic = calculate_bic(bi)
    lc = calculate_loss_component(average_annual_loss)
    ilm = calculate_ilm(bic, lc, use_ilm)
    capital = bic * ilm
    return {
        "approach": "SMA",
        "bi": bi, "bic": bic, "lc": lc, "ilm": ilm,
        "capital_requirement": capital,
        "rwa": capital * 12.5,
    }
