"""Basel II / II.5 legacy framework.

Original Basel II (2004) standardised approach, IRB, and operational risk
plus Basel II.5 stressed VaR and IRC charges. These coexist with B3/4
in pricebook.regulatory and can be used for legacy reporting.

    from pricebook.regulatory.basel2 import (
        b2_get_corporate_rw, b2_get_bank_rw, b2_calculate_sa_rwa,
        b2_calculate_irb_rwa,
        b2_calculate_bia_capital, b2_calculate_tsa_capital, b2_calculate_ama_capital,
        b2_calculate_cem_ead, b2_calculate_cem_netting,
        b25_stressed_var, b25_market_risk_capital,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from scipy.stats import norm


# =============================================================================
# Basel II Standardised Approach (Credit)
# =============================================================================

# B2 risk weights (2004 framework, before B3 reforms)
B2_SA_SOVEREIGN_RW: dict[str, float] = {
    "AAA": 0, "AA+": 0, "AA": 0, "AA-": 0,
    "A+": 20, "A": 20, "A-": 20,
    "BBB+": 50, "BBB": 50, "BBB-": 50,
    "BB+": 100, "BB": 100, "BB-": 100,
    "B+": 100, "B": 100, "B-": 100,
    "below_B-": 150, "unrated": 100,
}

# B2 bank Option 2 (based on bank's own rating)
B2_SA_BANK_RW: dict[str, float] = {
    "AAA": 20, "AA+": 20, "AA": 20, "AA-": 20,
    "A+": 50, "A": 50, "A-": 50,
    "BBB+": 50, "BBB": 50, "BBB-": 50,
    "BB+": 100, "BB": 100, "BB-": 100,
    "B+": 100, "B": 100, "B-": 100,
    "below_B-": 150, "unrated": 50,
}

B2_SA_CORPORATE_RW: dict[str, float] = {
    "AAA": 20, "AA+": 20, "AA": 20, "AA-": 20,
    "A+": 50, "A": 50, "A-": 50,
    "BBB+": 100, "BBB": 100, "BBB-": 100,  # B2: BBB → 100% (vs B3 75%)
    "BB+": 100, "BB": 100, "BB-": 100,
    "below_BB-": 150, "unrated": 100,
}

B2_SA_RETAIL_RW = 75
B2_SA_RESIDENTIAL_MORTGAGE_RW = 35
B2_SA_COMMERCIAL_MORTGAGE_RW = 100


def b2_get_sovereign_rw(rating: str = "unrated") -> float:
    """Basel II sovereign risk weight."""
    return B2_SA_SOVEREIGN_RW.get(rating, 100)


def b2_get_bank_rw(rating: str = "unrated", option: int = 2) -> float:
    """Basel II bank risk weight (Option 2 by default)."""
    if option == 1:
        # Option 1: one notch worse than sovereign
        sov_rw = b2_get_sovereign_rw(rating)
        mapping = {0: 20, 20: 50, 50: 100, 100: 100, 150: 150}
        return mapping.get(sov_rw, 100)
    return B2_SA_BANK_RW.get(rating, 50)


def b2_get_corporate_rw(rating: str = "unrated") -> float:
    """Basel II corporate risk weight."""
    return B2_SA_CORPORATE_RW.get(rating, 100)


def b2_calculate_sa_rwa(
    ead: float,
    asset_class: str = "corporate",
    rating: str = "unrated",
) -> dict:
    """Basel II SA RWA for one exposure."""
    if asset_class == "sovereign":
        rw = b2_get_sovereign_rw(rating)
    elif asset_class == "bank":
        rw = b2_get_bank_rw(rating)
    elif asset_class == "corporate":
        rw = b2_get_corporate_rw(rating)
    elif asset_class == "retail":
        rw = B2_SA_RETAIL_RW
    elif asset_class == "residential_mortgage":
        rw = B2_SA_RESIDENTIAL_MORTGAGE_RW
    elif asset_class == "commercial_mortgage":
        rw = B2_SA_COMMERCIAL_MORTGAGE_RW
    else:
        rw = 100

    rwa = ead * rw / 100
    return {
        "approach": "B2-SA",
        "ead": ead, "asset_class": asset_class, "rating": rating,
        "risk_weight_pct": rw, "rwa": rwa,
        "capital_requirement": rwa * 0.08,
    }


# =============================================================================
# Basel II IRB
# =============================================================================

def b2_calculate_correlation(pd: float, asset_class: str = "corporate") -> float:
    """Basel II IRB correlation (same Vasicek formula as B3)."""
    if asset_class == "retail_mortgage":
        return 0.15
    if asset_class == "retail_revolving":
        return 0.04
    if asset_class == "retail_other":
        r_min, r_max, k = 0.03, 0.16, 35
    else:
        r_min, r_max, k = 0.12, 0.24, 50
    exp_factor = (1 - math.exp(-k * pd)) / (1 - math.exp(-k))
    return r_min * exp_factor + r_max * (1 - exp_factor)


def b2_calculate_irb_rwa(
    ead: float,
    pd: float,
    lgd: float = 0.45,
    maturity: float = 2.5,
    asset_class: str = "corporate",
) -> dict:
    """Basel II IRB RWA calculation (no Basel IV LGD floors)."""
    pd = max(pd, 0.0003)
    pd = min(pd, 1.0)

    r = b2_calculate_correlation(pd, asset_class)
    g_pd = norm.ppf(pd)
    g_999 = norm.ppf(0.999)
    cond_pd = norm.cdf((1 - r) ** -0.5 * g_pd + (r / (1 - r)) ** 0.5 * g_999)

    expected_loss = pd * lgd
    unexpected_loss = lgd * cond_pd - expected_loss

    if asset_class.startswith("retail"):
        k = max(unexpected_loss, 0)
    else:
        b = (0.11852 - 0.05478 * math.log(pd)) ** 2
        maturity_factor = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
        k = max(unexpected_loss * maturity_factor, 0)

    rwa = k * 12.5 * ead
    return {
        "approach": "B2-IRB",
        "ead": ead, "pd": pd, "lgd": lgd, "maturity": maturity,
        "correlation": r,
        "capital_requirement_k": k,
        "risk_weight_pct": k * 12.5 * 100,
        "rwa": rwa,
        "expected_loss": pd * lgd * ead,
    }


# =============================================================================
# Basel II Operational Risk: BIA / TSA / AMA
# =============================================================================

class BusinessLine(Enum):
    CORPORATE_FINANCE = "corporate_finance"
    TRADING_SALES = "trading_sales"
    RETAIL_BANKING = "retail_banking"
    COMMERCIAL_BANKING = "commercial_banking"
    PAYMENT_SETTLEMENT = "payment_settlement"
    AGENCY_SERVICES = "agency_services"
    ASSET_MANAGEMENT = "asset_management"
    RETAIL_BROKERAGE = "retail_brokerage"


TSA_BETA_FACTORS: dict[BusinessLine, float] = {
    BusinessLine.CORPORATE_FINANCE: 0.18,
    BusinessLine.TRADING_SALES: 0.18,
    BusinessLine.RETAIL_BANKING: 0.12,
    BusinessLine.COMMERCIAL_BANKING: 0.15,
    BusinessLine.PAYMENT_SETTLEMENT: 0.18,
    BusinessLine.AGENCY_SERVICES: 0.15,
    BusinessLine.ASSET_MANAGEMENT: 0.12,
    BusinessLine.RETAIL_BROKERAGE: 0.12,
}


def b2_calculate_bia_capital(
    gross_income_year1: float,
    gross_income_year2: float,
    gross_income_year3: float,
    alpha: float = 0.15,
) -> dict:
    """Basic Indicator Approach: K = α × average positive GI."""
    positive = [gi for gi in [gross_income_year1, gross_income_year2, gross_income_year3] if gi > 0]
    avg = sum(positive) / len(positive) if positive else 0
    capital = alpha * avg
    return {
        "approach": "BIA", "alpha": alpha,
        "average_gross_income": avg,
        "positive_years_count": len(positive),
        "capital_requirement": capital,
        "rwa": capital * 12.5,
    }


@dataclass
class BusinessLineIncome:
    business_line: BusinessLine
    gross_income_year1: float
    gross_income_year2: float
    gross_income_year3: float


def b2_calculate_tsa_capital(business_line_incomes: list[BusinessLineIncome]) -> dict:
    """Standardised Approach: K = avg over 3y of max(Σ β_j × GI_j, 0)."""
    yearly = []
    for year_idx in range(3):
        year_total = 0.0
        for bli in business_line_incomes:
            beta = TSA_BETA_FACTORS[bli.business_line]
            gi = [bli.gross_income_year1, bli.gross_income_year2, bli.gross_income_year3][year_idx]
            year_total += beta * gi
        yearly.append(max(year_total, 0))
    capital = sum(yearly) / 3
    return {
        "approach": "TSA",
        "yearly_capitals": yearly,
        "capital_requirement": capital,
        "rwa": capital * 12.5,
    }


@dataclass
class AMAParameters:
    expected_loss: float
    unexpected_loss_999: float
    correlation_adjustment: float = 1.0
    insurance_mitigation: float = 0.0
    diversification_benefit: float = 0.0


def b2_calculate_ama_capital(
    params: AMAParameters,
    business_environment_factor: float = 1.0,
    internal_control_factor: float = 1.0,
) -> dict:
    """Advanced Measurement Approach: internal models with adjustments.

    Insurance mitigation capped at 20%.
    """
    base = params.unexpected_loss_999 * params.correlation_adjustment
    base *= business_environment_factor * internal_control_factor

    max_insurance = 0.20 * base
    insurance = min(params.insurance_mitigation, max_insurance)
    final = (base - insurance) * (1 - params.diversification_benefit)

    return {
        "approach": "AMA",
        "base_capital": base,
        "insurance_mitigation": insurance,
        "diversification_benefit": params.diversification_benefit,
        "capital_requirement": final,
        "rwa": final * 12.5,
    }


# =============================================================================
# Basel II Counterparty: Current Exposure Method (CEM)
# =============================================================================

class DerivativeType(Enum):
    INTEREST_RATE = "interest_rate"
    FX_GOLD = "fx_gold"
    EQUITY = "equity"
    PRECIOUS_METALS = "precious_metals"
    OTHER_COMMODITIES = "other_commodities"
    CREDIT_DERIVATIVE = "credit_derivative"


# CEM add-on factors (% of notional) by maturity bucket
CEM_ADDON_FACTORS: dict[DerivativeType, dict[str, float]] = {
    DerivativeType.INTEREST_RATE: {"up_to_1y": 0.0, "1y_to_5y": 0.005, "over_5y": 0.015},
    DerivativeType.FX_GOLD: {"up_to_1y": 0.01, "1y_to_5y": 0.05, "over_5y": 0.075},
    DerivativeType.EQUITY: {"up_to_1y": 0.06, "1y_to_5y": 0.08, "over_5y": 0.10},
    DerivativeType.PRECIOUS_METALS: {"up_to_1y": 0.07, "1y_to_5y": 0.07, "over_5y": 0.08},
    DerivativeType.OTHER_COMMODITIES: {"up_to_1y": 0.10, "1y_to_5y": 0.12, "over_5y": 0.15},
    DerivativeType.CREDIT_DERIVATIVE: {"up_to_1y": 0.05, "1y_to_5y": 0.05, "over_5y": 0.05},
}


def _cem_bucket(maturity: float) -> str:
    if maturity <= 1:
        return "up_to_1y"
    if maturity <= 5:
        return "1y_to_5y"
    return "over_5y"


def b2_get_cem_addon_factor(
    derivative_type: DerivativeType,
    maturity: float,
) -> float:
    """CEM add-on factor for a derivative type and maturity."""
    return CEM_ADDON_FACTORS[derivative_type][_cem_bucket(maturity)]


def b2_calculate_cem_ead(
    notional: float,
    mark_to_market: float,
    derivative_type: DerivativeType,
    maturity: float,
) -> dict:
    """CEM EAD = max(0, MTM) + add-on."""
    current_exposure = max(0, mark_to_market)
    addon_factor = b2_get_cem_addon_factor(derivative_type, maturity)
    addon = notional * addon_factor
    ead = current_exposure + addon
    return {
        "approach": "CEM",
        "notional": notional, "mark_to_market": mark_to_market,
        "current_exposure": current_exposure,
        "addon_factor": addon_factor, "addon": addon,
        "ead": ead,
    }


def b2_calculate_cem_netting(trades: list[dict]) -> dict:
    """CEM with netting: EAD = Net CE + (0.4 × A_gross + 0.6 × NGR × A_gross)."""
    if not trades:
        return {"ead": 0, "trades": 0}

    sum_mtm = sum(t.get("mark_to_market", 0) for t in trades)
    net_ce = max(0, sum_mtm)
    gross_ce = sum(max(0, t.get("mark_to_market", 0)) for t in trades)

    addons = []
    for t in trades:
        f = b2_get_cem_addon_factor(t["derivative_type"], t["maturity"])
        addons.append(t["notional"] * f)
    a_gross = sum(addons)

    ngr = (net_ce / gross_ce) if gross_ce > 0 else 1.0
    a_net = 0.4 * a_gross + 0.6 * ngr * a_gross
    ead = net_ce + a_net

    return {
        "approach": "CEM-netted",
        "net_ce": net_ce, "gross_ce": gross_ce, "ngr": ngr,
        "a_gross": a_gross, "a_net": a_net,
        "ead": ead, "trades": len(trades),
    }


# =============================================================================
# Basel II.5 Stressed VaR + Market Risk
# =============================================================================

def b25_stressed_var(
    var_1day: float,
    multiplier: float = 3.0,
    plus_factor: float = 0.0,
    holding_days: int = 10,
) -> dict:
    """Basel II.5 Stressed VaR capital.

    K_sVaR = max(sVaR_t-1, m_c × sVaR_avg) × sqrt(holding_days)
    """
    var_scaled = var_1day * math.sqrt(holding_days)
    m_c = multiplier + plus_factor
    capital = m_c * var_scaled
    return {
        "approach": "B2.5 sVaR",
        "var_1day": var_1day,
        "var_holding_period": var_scaled,
        "multiplier": m_c,
        "capital_requirement": capital,
        "rwa": capital * 12.5,
    }


def b25_market_risk_capital(
    var_capital: float,
    stressed_var_capital: float,
    irc_capital: float = 0,
    crm_capital: float = 0,
    specific_risk_capital: float = 0,
) -> dict:
    """Basel II.5 total market risk capital.

    K_mr = K_VaR + K_sVaR + K_IRC + K_CRM + K_SpecificRisk
    """
    total = var_capital + stressed_var_capital + irc_capital + crm_capital + specific_risk_capital
    return {
        "approach": "B2.5 Market Risk",
        "var": var_capital,
        "stressed_var": stressed_var_capital,
        "irc": irc_capital,
        "crm": crm_capital,
        "specific_risk": specific_risk_capital,
        "total_capital": total,
        "total_rwa": total * 12.5,
    }
