"""Counterparty Credit Risk: Full SA-CCR, BA-CVA, SA-CVA, CCP exposures.

Basel III/IV (CRE52, MAR50, CRE54):
- SA-CCR: Standardised Approach for Counterparty Credit Risk
- BA-CVA: Basic Approach for Credit Valuation Adjustment
- SA-CVA: Standardised Approach for CVA
- CCP exposures: trade and default fund contributions

    from pricebook.regulatory.counterparty import (
        calculate_sa_ccr_ead, calculate_ba_cva, calculate_sa_cva,
        calculate_ccp_trade_exposure, calculate_ccp_default_fund,
        SA_CCR_SUPERVISORY_FACTORS, BA_CVA_RISK_WEIGHTS,
    )
"""

from __future__ import annotations

import math


# ---- SA-CCR Supervisory Factors (CRE52.72) ----

SA_CCR_SUPERVISORY_FACTORS: dict[str, dict[str, float]] = {
    # Interest Rate
    "IR": {"SF": 0.50, "correlation": 1.0},
    # Foreign Exchange
    "FX": {"SF": 4.0, "correlation": 1.0},
    # Credit (single name)
    "CR_AAA_AA": {"SF": 0.38, "correlation": 0.50},
    "CR_A":      {"SF": 0.42, "correlation": 0.50},
    "CR_BBB":    {"SF": 0.54, "correlation": 0.50},
    "CR_BB":     {"SF": 1.06, "correlation": 0.50},
    "CR_B":      {"SF": 1.6,  "correlation": 0.50},
    "CR_CCC":    {"SF": 6.0,  "correlation": 0.50},
    # Credit indices
    "CR_INDEX_IG": {"SF": 0.38, "correlation": 0.80},
    "CR_INDEX_SG": {"SF": 1.06, "correlation": 0.80},
    # Equity
    "EQ_SINGLE": {"SF": 32.0, "correlation": 0.50},
    "EQ_INDEX":  {"SF": 20.0, "correlation": 0.80},
    # Commodity
    "COM_ELECTRICITY":  {"SF": 40.0, "correlation": 0.40},
    "COM_OIL_GAS":      {"SF": 18.0, "correlation": 0.40},
    "COM_METALS":       {"SF": 18.0, "correlation": 0.40},
    "COM_AGRICULTURAL": {"SF": 18.0, "correlation": 0.40},
    "COM_OTHER":        {"SF": 18.0, "correlation": 0.40},
}

SA_CCR_OPTION_VOLATILITY = {
    "IR": 0.50, "FX": 0.15, "CR": 1.00, "EQ": 1.20, "COM": 1.50,
}


# ---- Maturity factor ----

def calculate_maturity_factor(
    maturity: float,
    is_margined: bool = False,
    mpor_days: int = 10,
) -> float:
    """SA-CCR maturity factor.

    Margined: MF = 1.5 × sqrt(MPOR / 1Y)
    Unmargined: MF = sqrt(min(M, 1) / 1Y) where M ≥ 10/250
    """
    if is_margined:
        return 1.5 * math.sqrt(mpor_days / 250)
    m = max(10 / 250, min(maturity, 1))
    return math.sqrt(m)


def calculate_supervisory_duration(maturity: float) -> float:
    """SA-CCR supervisory duration: SD = (1 - exp(-0.05 × M)) / 0.05."""
    return (1 - math.exp(-0.05 * maturity)) / 0.05


def calculate_supervisory_delta(
    is_long: bool,
    is_option: bool = False,
    option_type: str | None = None,
) -> float:
    """SA-CCR supervisory delta. ±1 for non-options, ±0.5 default for options."""
    if not is_option:
        return 1.0 if is_long else -1.0
    if option_type == "call":
        sign = 1 if is_long else -1
    else:
        sign = -1 if is_long else 1
    return sign * 0.5


def calculate_adjusted_notional(
    notional: float,
    asset_class: str,
    maturity: float = 1.0,
    is_margined: bool = False,
    mpor_days: int = 10,
) -> float:
    """Adjusted notional: notional × supervisory_duration (IR/CR) or × MF (others)."""
    if asset_class.startswith("IR") or asset_class.startswith("CR"):
        return notional * calculate_supervisory_duration(maturity)
    mf = calculate_maturity_factor(maturity, is_margined, mpor_days)
    return notional * mf


# ---- Trade-level add-on ----

def calculate_addon_single_trade(
    notional: float,
    asset_class: str,
    maturity: float = 1.0,
    delta: float = 1.0,
    is_margined: bool = False,
    mpor_days: int = 10,
) -> float:
    """Single-trade add-on: AddOn = SF × |δ| × adjusted_notional."""
    sf_data = SA_CCR_SUPERVISORY_FACTORS.get(asset_class, {"SF": 0.15, "correlation": 0.50})
    sf = sf_data["SF"] / 100  # convert from %
    adj_notional = calculate_adjusted_notional(notional, asset_class, maturity, is_margined, mpor_days)
    return sf * abs(delta) * adj_notional


# ---- Replacement Cost ----

def calculate_replacement_cost(
    mtm: float,
    collateral_held: float = 0,
    collateral_posted: float = 0,
    is_margined: bool = False,
    threshold: float = 0,
    mta: float = 0,
    nica: float = 0,
) -> float:
    """SA-CCR Replacement Cost.

    Unmargined: RC = max(V - C, 0)
    Margined: RC = max(V - C, TH + MTA - NICA, 0)
    """
    c = collateral_held - collateral_posted
    v = mtm
    if is_margined:
        return max(v - c, threshold + mta - nica, 0)
    return max(v - c, 0)


# ---- PFE multiplier ----

def calculate_pfe_multiplier(
    mtm: float,
    collateral: float,
    addon_aggregate: float,
    floor: float = 0.05,
) -> float:
    """PFE multiplier: min(1, floor + (1-floor) × exp((V-C) / (2(1-floor) × AddOn)))."""
    v_minus_c = mtm - collateral
    if addon_aggregate <= 0:
        return 1.0
    if v_minus_c >= 0:
        return 1.0
    exponent = v_minus_c / (2 * (1 - floor) * addon_aggregate)
    return min(1.0, floor + (1 - floor) * math.exp(exponent))


# ---- Total SA-CCR EAD ----

def calculate_sa_ccr_ead(
    trades: list[dict],
    collateral_held: float = 0,
    collateral_posted: float = 0,
    is_margined: bool = False,
    threshold: float = 0,
    mta: float = 0,
    nica: float = 0,
    mpor_days: int = 10,
    alpha: float = 1.4,
) -> dict:
    """Total SA-CCR EAD: EAD = α × (RC + PFE).

    trades: list of {notional, asset_class, maturity, mtm, delta}.
    """
    total_mtm = sum(t.get("mtm", 0) for t in trades)
    rc = calculate_replacement_cost(
        total_mtm, collateral_held, collateral_posted, is_margined, threshold, mta, nica,
    )

    addons_by_class: dict[str, float] = {}
    for t in trades:
        ac = t["asset_class"]
        addon = calculate_addon_single_trade(
            t["notional"], ac, t.get("maturity", 1.0),
            t.get("delta", 1.0), is_margined, mpor_days,
        )
        addons_by_class[ac] = addons_by_class.get(ac, 0) + addon

    addon_aggregate = sum(addons_by_class.values())
    net_collateral = collateral_held - collateral_posted
    multiplier = calculate_pfe_multiplier(total_mtm, net_collateral, addon_aggregate)
    pfe = multiplier * addon_aggregate
    ead = alpha * (rc + pfe)

    return {
        "approach": "SA-CCR",
        "replacement_cost": rc, "pfe": pfe,
        "addon_aggregate": addon_aggregate, "addons_by_class": addons_by_class,
        "multiplier": multiplier, "alpha": alpha,
        "ead": ead, "total_mtm": total_mtm, "net_collateral": net_collateral,
    }


# ---- BA-CVA (MAR50) ----

BA_CVA_RISK_WEIGHTS: dict[str, float] = {
    "AAA": 0.007, "AA": 0.007, "A": 0.008, "BBB": 0.010,
    "BB": 0.020, "B": 0.030, "CCC": 0.100,
    "unrated": 0.015, "unrated_financial": 0.010,
}

BA_CVA_CORRELATION = 0.5
CVA_DISCOUNT_RATE = 0.05


def calculate_supervisory_discount(maturity: float) -> float:
    """Supervisory discount: DF = (1 - exp(-0.05 × M)) / (0.05 × M)."""
    if maturity <= 0:
        return 1.0
    return (1 - math.exp(-CVA_DISCOUNT_RATE * maturity)) / (CVA_DISCOUNT_RATE * maturity)


def calculate_ba_cva(counterparties: list[dict]) -> dict:
    """BA-CVA capital (MAR50.5).

    K_CVA = α × √[ρ² × (Σ_c S_c × CVA_c)² + (1 - ρ²) × Σ_c (S_c × CVA_c)²]
    where α = 2.33 (99% confidence) and ρ = 0.50 (supervisory correlation).
    """
    results = []
    sum_sq_individual = 0.0  # Σ (S_c × CVA_c)² — idiosyncratic
    sum_systematic = 0.0     # Σ S_c × CVA_c — systematic

    for cp in counterparties:
        ead = cp.get("ead", 0)
        rating = cp.get("rating", "unrated")
        maturity = cp.get("maturity", 1.0)

        rw = BA_CVA_RISK_WEIGHTS.get(rating, BA_CVA_RISK_WEIGHTS["unrated"])
        df = calculate_supervisory_discount(maturity)
        cva_estimate = ead * rw * maturity * df

        s_c_cva = rw * cva_estimate
        sum_sq_individual += s_c_cva ** 2
        sum_systematic += s_c_cva

        results.append({
            "ead": ead, "rating": rating, "maturity": maturity,
            "risk_weight": rw, "discount_factor": df,
            "cva_estimate": cva_estimate, "weighted_cva": s_c_cva,
        })

    rho = BA_CVA_CORRELATION
    k_cva = 2.33 * math.sqrt(
        rho ** 2 * sum_systematic ** 2 + (1 - rho ** 2) * sum_sq_individual
    )
    return {
        "approach": "BA-CVA",
        "k_cva": k_cva, "rwa": k_cva * 12.5,
        "counterparties": results,
        "sum_individual_sq": sum_sq_individual,
        "sum_systematic": sum_systematic,
        "correlation": rho,
    }


# ---- SA-CVA (MAR50) ----

# SA-CVA risk weights per risk class (simplified)
SA_CVA_DELTA_WEIGHTS = {
    "IR": 0.017, "FX": 0.11, "CR_counterparty": 0.05,
    "CR_reference": 0.05, "EQ": 0.55, "COM": 0.16,
}


def calculate_sa_cva(
    delta_sensitivities: dict[str, float],
    vega_sensitivities: dict[str, float] | None = None,
) -> dict:
    """SA-CVA capital charge (simplified).

    K_SA_CVA = m_CVA × sqrt(K_delta² + K_vega²)
    where m_CVA = 1.25 (multiplier).
    """
    m_cva = 1.25
    vega_sensitivities = vega_sensitivities or {}

    # Delta charge
    k_delta_sq = 0.0
    for risk_class, sens in delta_sensitivities.items():
        rw = SA_CVA_DELTA_WEIGHTS.get(risk_class, 0.10)
        k_delta_sq += (sens * rw) ** 2

    # Vega charge (use 100% RW as simplification)
    k_vega_sq = sum(s ** 2 for s in vega_sensitivities.values())

    k_sa_cva = m_cva * math.sqrt(k_delta_sq + k_vega_sq)
    return {
        "approach": "SA-CVA",
        "k_sa_cva": k_sa_cva, "rwa": k_sa_cva * 12.5,
        "k_delta": math.sqrt(k_delta_sq), "k_vega": math.sqrt(k_vega_sq),
        "multiplier": m_cva,
    }


# ---- CCP exposures (CRE54) ----

CCP_RISK_WEIGHTS = {
    "qualifying_ccp_trade": 2.0,
    "non_qualifying_ccp_trade": 100.0,
}


def calculate_ccp_trade_exposure(ead: float, is_qualifying_ccp: bool = True) -> dict:
    """CCP trade exposure: 2% RW for QCCP, 100% for non-QCCP."""
    rw = CCP_RISK_WEIGHTS["qualifying_ccp_trade"] if is_qualifying_ccp else CCP_RISK_WEIGHTS["non_qualifying_ccp_trade"]
    return {
        "exposure_type": "CCP Trade",
        "ead": ead, "is_qualifying_ccp": is_qualifying_ccp,
        "risk_weight_pct": rw, "rwa": ead * rw / 100,
    }


def calculate_ccp_default_fund(
    df_contribution: float,
    k_ccp: float = 0,
    total_df: float = 0,
    ccp_capital: float = 0,
    is_qualifying_ccp: bool = True,
) -> dict:
    """CCP default fund contribution RWA.

    QCCP: K_i = max(K_CCP × DF_i/DF_total - c × DF_i/DF_total × CCP_capital, 0)
    Non-QCCP: 100% RW.
    """
    if not is_qualifying_ccp:
        return {
            "exposure_type": "CCP Default Fund",
            "df_contribution": df_contribution,
            "is_qualifying_ccp": False,
            "risk_weight_pct": 100, "rwa": df_contribution,
        }

    df_share = df_contribution / total_df if total_df > 0 else 0
    c = 2
    k_cm = max(k_ccp * df_share - c * df_share * ccp_capital, 0)
    rwa = k_cm * 12.5
    rw = (rwa / df_contribution * 100) if df_contribution > 0 else 0

    return {
        "exposure_type": "CCP Default Fund",
        "df_contribution": df_contribution,
        "k_ccp": k_ccp, "total_df": total_df, "ccp_capital": ccp_capital,
        "df_share": df_share,
        "is_qualifying_ccp": True,
        "capital_requirement": k_cm,
        "risk_weight_pct": rw, "rwa": rwa,
    }
