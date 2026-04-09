"""Capital framework: output floor, leverage ratio, large exposures, G-SIB, TLAC.

Basel III/IV (CAP30, LEV30, LEX30) plus G-SIB framework and TLAC/MREL.

    from pricebook.regulatory.capital_framework import (
        calculate_output_floor, calculate_leverage_ratio,
        calculate_large_exposure, calculate_gsib_score,
        calculate_tlac_requirement, calculate_mrel_requirement,
    )
"""

from __future__ import annotations

import math


# ---- Output Floor (CAP30) ----

OUTPUT_FLOOR_PERCENTAGE = 0.725

OUTPUT_FLOOR_TRANSITION: dict[int, float] = {
    2023: 0.50, 2024: 0.55, 2025: 0.60,
    2026: 0.65, 2027: 0.70, 2028: 0.725,
}


def calculate_output_floor(
    rwa_irb: float,
    rwa_standardised: float,
    year: int = 2028,
) -> dict:
    """Output Floor: Floored RWA = max(RWA_IRB, floor% × RWA_SA)."""
    floor_pct = OUTPUT_FLOOR_TRANSITION.get(year, OUTPUT_FLOOR_PERCENTAGE)
    floor_rwa = floor_pct * rwa_standardised
    floored_rwa = max(rwa_irb, floor_rwa)
    floor_addon = max(floor_rwa - rwa_irb, 0)

    return {
        "rwa_irb": rwa_irb, "rwa_standardised": rwa_standardised,
        "floor_percentage": floor_pct, "floor_rwa": floor_rwa,
        "floored_rwa": floored_rwa, "floor_addon": floor_addon,
        "floor_is_binding": floor_rwa > rwa_irb,
    }


def calculate_output_floor_by_risk_type(
    credit_risk_irb: float,
    credit_risk_sa: float,
    market_risk_ima: float = 0,
    market_risk_sa: float = 0,
    operational_risk: float = 0,
    cva_risk: float = 0,
    year: int = 2028,
) -> dict:
    """Output floor with risk-type breakdown."""
    total_irb = credit_risk_irb + market_risk_ima + operational_risk + cva_risk
    total_sa = credit_risk_sa + market_risk_sa + operational_risk + cva_risk
    result = calculate_output_floor(total_irb, total_sa, year)
    result["breakdown"] = {
        "credit_risk_irb": credit_risk_irb, "credit_risk_sa": credit_risk_sa,
        "market_risk_ima": market_risk_ima, "market_risk_sa": market_risk_sa,
        "operational_risk": operational_risk, "cva_risk": cva_risk,
    }
    return result


# ---- Leverage Ratio (LEV30) ----

LEVERAGE_RATIO_MINIMUM = 0.03


def calculate_leverage_ratio(
    tier1_capital: float,
    on_balance_sheet: float,
    derivatives_exposure: float = 0,
    sft_exposure: float = 0,
    off_balance_sheet: float = 0,
    is_gsib: bool = False,
    gsib_buffer_pct: float = 0,
) -> dict:
    """Leverage Ratio = Tier1 / Total Exposure Measure."""
    total_exposure = on_balance_sheet + derivatives_exposure + sft_exposure + off_balance_sheet
    leverage_ratio = tier1_capital / total_exposure if total_exposure > 0 else 0

    min_req = LEVERAGE_RATIO_MINIMUM
    if is_gsib:
        min_req += gsib_buffer_pct * 0.5  # 50% of G-SIB buffer

    is_compliant = leverage_ratio >= min_req

    return {
        "tier1_capital": tier1_capital, "total_exposure": total_exposure,
        "leverage_ratio": leverage_ratio, "leverage_ratio_pct": leverage_ratio * 100,
        "minimum_requirement": min_req, "minimum_requirement_pct": min_req * 100,
        "is_compliant": is_compliant,
        "buffer": leverage_ratio - min_req,
        "required_tier1": min_req * total_exposure,
        "excess_capital": tier1_capital - min_req * total_exposure,
    }


# ---- Large Exposures (LEX30) ----

LARGE_EXPOSURE_THRESHOLD = 0.10  # 10% reporting
LARGE_EXPOSURE_LIMIT = 0.25      # 25% maximum
GSIB_INTERBANK_LIMIT = 0.15      # 15% G-SIB to G-SIB


def calculate_large_exposure(
    exposure_value: float,
    tier1_capital: float,
    counterparty_type: str = "corporate",
    is_gsib_to_gsib: bool = False,
) -> dict:
    """Large exposure check vs Tier 1 capital."""
    if tier1_capital <= 0:
        return {"exposure_value": exposure_value, "tier1_capital": 0, "is_breach": True}

    pct = exposure_value / tier1_capital
    limit = GSIB_INTERBANK_LIMIT if is_gsib_to_gsib else LARGE_EXPOSURE_LIMIT
    is_reportable = pct >= LARGE_EXPOSURE_THRESHOLD
    is_breach = pct > limit
    excess = max(exposure_value - limit * tier1_capital, 0)

    return {
        "exposure_value": exposure_value, "tier1_capital": tier1_capital,
        "exposure_pct": pct * 100, "limit_pct": limit * 100,
        "is_reportable": is_reportable, "is_breach": is_breach,
        "headroom": limit * tier1_capital - exposure_value,
        "excess_above_limit": excess,
    }


# ---- Credit Risk Mitigation: collateral haircuts ----

# Standard supervisory haircuts (Basel III)
SUPERVISORY_HAIRCUTS: dict[str, float] = {
    "cash": 0.0,
    "sovereign_AAA_AA": 0.005,
    "sovereign_A": 0.02,
    "sovereign_BBB": 0.04,
    "corporate_AAA_AA": 0.04,
    "corporate_A": 0.06,
    "equity_main_index": 0.15,
    "equity_other": 0.25,
    "gold": 0.15,
}


def calculate_collateral_haircut(collateral_type: str) -> float:
    """Supervisory haircut for collateral type."""
    return SUPERVISORY_HAIRCUTS.get(collateral_type, 0.25)


def calculate_exposure_with_collateral(
    exposure: float,
    collateral_value: float,
    collateral_type: str = "cash",
    fx_haircut: float = 0.0,
) -> dict:
    """Adjusted exposure after collateral haircut.

    Adjusted exposure = max(0, E × (1 + He) - C × (1 - Hc - Hfx))
    """
    Hc = calculate_collateral_haircut(collateral_type)
    He = 0.0  # Exposure haircut (0 for non-securities)
    adj_collateral = collateral_value * (1 - Hc - fx_haircut)
    adj_exposure = max(exposure * (1 + He) - adj_collateral, 0)

    return {
        "exposure": exposure, "collateral_value": collateral_value,
        "collateral_type": collateral_type, "haircut": Hc,
        "adjusted_collateral": adj_collateral,
        "adjusted_exposure": adj_exposure,
        "exposure_reduction_pct": (1 - adj_exposure / exposure) * 100 if exposure > 0 else 0,
    }


# ---- Off-balance sheet CCF ----

CREDIT_CONVERSION_FACTORS: dict[str, float] = {
    "unconditionally_cancellable": 0.10,  # 10% under Basel IV
    "uncommitted": 0.10,
    "1y_or_less": 0.20,
    "over_1y": 0.50,
    "letters_of_credit_self_liquidating": 0.20,
    "letters_of_credit_non_self_liquidating": 0.50,
    "guarantees_substitute": 1.00,
    "asset_sale_recourse": 1.00,
    "forward_purchase_commitment": 1.00,
}


def calculate_ead_off_balance_sheet(
    notional: float,
    facility_type: str = "1y_or_less",
) -> dict:
    """EAD for off-balance-sheet exposures using CCF."""
    ccf = CREDIT_CONVERSION_FACTORS.get(facility_type, 0.50)
    ead = notional * ccf
    return {
        "notional": notional, "facility_type": facility_type,
        "ccf": ccf, "ead": ead,
    }


# ---- G-SIB Framework ----

GSIB_CATEGORIES: dict[str, dict] = {
    "size": {
        "weight": 0.20,
        "indicators": {"total_exposures": 1.0},
    },
    "interconnectedness": {
        "weight": 0.20,
        "indicators": {
            "intra_financial_assets": 1/3,
            "intra_financial_liabilities": 1/3,
            "securities_outstanding": 1/3,
        },
    },
    "substitutability": {
        "weight": 0.20,
        "indicators": {
            "payments_activity": 1/3,
            "assets_under_custody": 1/3,
            "underwriting_activity": 1/3,
        },
    },
    "complexity": {
        "weight": 0.20,
        "indicators": {
            "otc_derivatives_notional": 1/3,
            "level_3_assets": 1/3,
            "trading_securities": 1/3,
        },
    },
    "cross_jurisdictional": {
        "weight": 0.20,
        "indicators": {
            "cross_jurisdictional_claims": 0.5,
            "cross_jurisdictional_liabilities": 0.5,
        },
    },
}

GSIB_BUCKETS: dict[int, dict] = {
    1: {"score_range": (130, 229), "buffer": 0.010},
    2: {"score_range": (230, 329), "buffer": 0.015},
    3: {"score_range": (330, 429), "buffer": 0.020},
    4: {"score_range": (430, 529), "buffer": 0.025},
    5: {"score_range": (530, float("inf")), "buffer": 0.035},
}

# Illustrative global denominators
GSIB_DENOMINATORS: dict[str, float] = {
    "total_exposures": 100_000_000_000_000,
    "intra_financial_assets": 20_000_000_000_000,
    "intra_financial_liabilities": 20_000_000_000_000,
    "securities_outstanding": 15_000_000_000_000,
    "payments_activity": 500_000_000_000_000,
    "assets_under_custody": 150_000_000_000_000,
    "underwriting_activity": 10_000_000_000_000,
    "otc_derivatives_notional": 400_000_000_000_000,
    "level_3_assets": 2_000_000_000_000,
    "trading_securities": 20_000_000_000_000,
    "cross_jurisdictional_claims": 30_000_000_000_000,
    "cross_jurisdictional_liabilities": 25_000_000_000_000,
}


def calculate_gsib_score(bank_data: dict[str, float]) -> dict:
    """Total G-SIB score: 5 categories × indicators × global denominators × 10000.

    bank_data: {indicator_name: value}
    """
    category_results = {}
    total_score = 0.0

    for category, config in GSIB_CATEGORIES.items():
        weighted_sum = 0.0
        indicator_scores = {}
        for ind_name, ind_weight in config["indicators"].items():
            value = bank_data.get(ind_name, 0)
            denom = GSIB_DENOMINATORS.get(ind_name, 1)
            score = (value / denom) * 10000 if denom > 0 else 0
            weighted = score * ind_weight
            indicator_scores[ind_name] = {
                "value": value, "score": score, "weight": ind_weight,
                "weighted_score": weighted,
            }
            weighted_sum += weighted

        cat_score = weighted_sum * config["weight"]
        category_results[category] = {
            "category_weight": config["weight"],
            "indicator_scores": indicator_scores,
            "category_score": cat_score,
        }
        total_score += cat_score

    # Determine bucket
    bucket = None
    buffer = 0.0
    for bucket_num, bcfg in GSIB_BUCKETS.items():
        lo, hi = bcfg["score_range"]
        if lo <= total_score < hi:
            bucket = bucket_num
            buffer = bcfg["buffer"]
            break

    return {
        "total_score": total_score,
        "is_gsib": bucket is not None,
        "bucket": bucket,
        "buffer_requirement": buffer,
        "buffer_requirement_pct": buffer * 100,
        "category_scores": category_results,
    }


# ---- TLAC ----

TLAC_MINIMUM_RWA = 0.18
TLAC_MINIMUM_LEVERAGE = 0.0675


def calculate_tlac_requirement(
    rwa: float,
    leverage_exposure: float,
    gsib_buffer: float = 0,
) -> dict:
    """TLAC minimum = max(18%×RWA + buffers, 6.75%×leverage)."""
    rwa_req = TLAC_MINIMUM_RWA + gsib_buffer
    tlac_rwa = rwa * rwa_req
    tlac_lev = leverage_exposure * TLAC_MINIMUM_LEVERAGE

    requirement = max(tlac_rwa, tlac_lev)
    binding = "RWA" if tlac_rwa >= tlac_lev else "Leverage"

    return {
        "rwa": rwa, "leverage_exposure": leverage_exposure,
        "gsib_buffer": gsib_buffer,
        "rwa_requirement_pct": rwa_req * 100,
        "leverage_requirement_pct": TLAC_MINIMUM_LEVERAGE * 100,
        "tlac_rwa_based": tlac_rwa, "tlac_leverage_based": tlac_lev,
        "tlac_requirement": requirement, "binding_constraint": binding,
    }


# ---- MREL ----

def calculate_mrel_requirement(
    rwa: float,
    leverage_exposure: float,
    loss_absorption_amount: float = 0,
    recapitalisation_amount: float = 0,
    is_resolution_entity: bool = True,
) -> dict:
    """MREL = loss absorption + recapitalisation amount.

    Subject to floor of 8% of RWA or 3% of leverage exposure for resolution entities.
    """
    mrel_loss_recap = loss_absorption_amount + recapitalisation_amount

    floor_rwa = 0.08 * rwa if is_resolution_entity else 0.0
    floor_lev = 0.03 * leverage_exposure if is_resolution_entity else 0.0

    requirement = max(mrel_loss_recap, floor_rwa, floor_lev)
    return {
        "rwa": rwa, "leverage_exposure": leverage_exposure,
        "loss_absorption": loss_absorption_amount,
        "recapitalisation": recapitalisation_amount,
        "mrel_loss_recap": mrel_loss_recap,
        "floor_rwa": floor_rwa, "floor_leverage": floor_lev,
        "mrel_requirement": requirement,
    }
