"""Credit RWA: SA-CR, F-IRB, A-IRB, and specialised lending.

Standardised Approach (SA-CR) and Internal Ratings-Based (F-IRB / A-IRB)
risk-weighted assets calculation under Basel III/IV.

The IRB capital requirement uses the Vasicek single-factor formula:
    K = LGD × [Φ((1-R)^(-0.5) × Φ⁻¹(PD) + (R/(1-R))^0.5 × Φ⁻¹(0.999)) - PD]
        × [(1 + (M-2.5)×b) / (1 - 1.5×b)]

    from pricebook.regulatory.credit_rwa import (
        get_sa_corporate_rw, get_sa_sovereign_rw, get_sa_bank_rw,
        calculate_correlation, calculate_capital_requirement,
        calculate_sa_rwa, calculate_irb_rwa, calculate_airb_rwa,
        slotting_risk_weight, sme_correlation_adjustment,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


# ---- SA-CR risk weights (Basel III/IV) ----

SA_SOVEREIGN_RW: dict[str, float] = {
    "AAA": 0, "AA+": 0, "AA": 0, "AA-": 0,
    "A+": 20, "A": 20, "A-": 20,
    "BBB+": 50, "BBB": 50, "BBB-": 50,
    "BB+": 100, "BB": 100, "BB-": 100,
    "B+": 100, "B": 100, "B-": 100,
    "below_B-": 150, "unrated": 100,
}

SA_BANK_RW: dict[str, float] = {
    "AAA": 20, "AA+": 20, "AA": 20, "AA-": 20,
    "A+": 30, "A": 30, "A-": 30,
    "BBB+": 50, "BBB": 50, "BBB-": 50,
    "BB+": 100, "BB": 100, "BB-": 100,
    "B+": 100, "B": 100, "B-": 100,
    "below_B-": 150, "unrated": 50,
}

SA_BANK_SHORT_TERM_RW: dict[str, float] = {
    "AAA": 20, "AA+": 20, "AA": 20, "AA-": 20,
    "A+": 20, "A": 20, "A-": 20,
    "BBB+": 20, "BBB": 20, "BBB-": 20,
    "BB+": 50, "BB": 50, "BB-": 50,
    "B+": 50, "B": 50, "B-": 50,
    "below_B-": 150, "unrated": 20,
}

SA_BANK_SCRA_RW: dict[str, float] = {
    "A": 40,   # Grade A
    "B": 75,   # Grade B
    "C": 150,  # Grade C
}

SA_CORPORATE_RW: dict[str, float] = {
    "AAA": 20, "AA+": 20, "AA": 20, "AA-": 20,
    "A+": 50, "A": 50, "A-": 50,
    "BBB+": 75, "BBB": 75, "BBB-": 75,
    "BB+": 100, "BB": 100, "BB-": 100,
    "below_BB-": 150, "unrated": 100,
}

SA_RETAIL_RW: dict[str, float] = {
    "regulatory_retail": 75,
    "transactor": 45,
}

# Residential RE risk weights by LTV bucket (general, income-producing)
SA_RESIDENTIAL_RE_RW: dict[str, tuple[float, float]] = {
    "ltv_50": (20, 30),
    "ltv_60": (25, 35),
    "ltv_80": (30, 45),
    "ltv_90": (40, 60),
    "ltv_100": (50, 75),
    "ltv_above_100": (70, 105),
}

SA_COMMERCIAL_RE_RW: dict[str, float] = {
    "ltv_60": 70,
    "ltv_80": 90,
    "above_ltv_80": 110,
}

SA_SUBORDINATED_RW = 150
SA_EQUITY_RW: dict[str, float] = {
    "speculative_unlisted": 400,
    "other": 250,
    "banking_book_listed": 100,
}

SA_DEFAULTED_RW: dict[str, float] = {
    "unsecured": 150,
    "secured_residential": 100,
}


def get_sa_sovereign_rw(rating: str = "unrated") -> float:
    """SA risk weight for sovereign exposures."""
    return SA_SOVEREIGN_RW.get(rating, 100)


def get_sa_bank_rw(rating: str = "unrated", short_term: bool = False) -> float:
    """SA risk weight for bank exposures (ECRA approach)."""
    table = SA_BANK_SHORT_TERM_RW if short_term else SA_BANK_RW
    return table.get(rating, 50)


def get_sa_bank_scra_rw(grade: str = "B", short_term: bool = False) -> float:
    """SA risk weight for bank exposures (SCRA approach)."""
    if short_term:
        return {"A": 20, "B": 50, "C": 150}.get(grade, 50)
    return SA_BANK_SCRA_RW.get(grade, 75)


def get_sa_corporate_rw(rating: str = "unrated", is_sme: bool = False) -> float:
    """SA risk weight for corporate exposures."""
    rw = SA_CORPORATE_RW.get(rating, 100)
    # SME supporting factor: 85% of base RW for qualifying SMEs (CRE20.32)
    if is_sme and rating == "unrated":
        rw = 85
    return rw


def get_sa_retail_rw(category: str = "regulatory_retail") -> float:
    """SA risk weight for retail exposures."""
    return SA_RETAIL_RW.get(category, 75)


def get_sa_residential_re_rw(ltv: float, income_producing: bool = False) -> float:
    """SA risk weight for residential real estate by LTV."""
    if ltv <= 0.50:
        bucket = "ltv_50"
    elif ltv <= 0.60:
        bucket = "ltv_60"
    elif ltv <= 0.80:
        bucket = "ltv_80"
    elif ltv <= 0.90:
        bucket = "ltv_90"
    elif ltv <= 1.00:
        bucket = "ltv_100"
    else:
        bucket = "ltv_above_100"
    rw = SA_RESIDENTIAL_RE_RW[bucket]
    return rw[1] if income_producing else rw[0]


def get_sa_commercial_re_rw(ltv: float) -> float:
    """SA risk weight for commercial real estate by LTV."""
    if ltv <= 0.60:
        return SA_COMMERCIAL_RE_RW["ltv_60"]
    elif ltv <= 0.80:
        return SA_COMMERCIAL_RE_RW["ltv_80"]
    return SA_COMMERCIAL_RE_RW["above_ltv_80"]


# ---- SA-CR RWA calculation ----

@dataclass
class SAExposure:
    """Standardised Approach exposure."""
    ead: float
    asset_class: str  # "sovereign", "bank", "corporate", "retail", "residential_re", "commercial_re"
    rating: str = "unrated"
    ltv: float | None = None
    is_sme: bool = False
    short_term: bool = False
    income_producing: bool = False


def calculate_sa_rwa(exposure: SAExposure) -> dict:
    """Compute SA-CR RWA for a single exposure."""
    cls = exposure.asset_class

    if cls == "sovereign":
        rw = get_sa_sovereign_rw(exposure.rating)
    elif cls == "bank":
        rw = get_sa_bank_rw(exposure.rating, exposure.short_term)
    elif cls == "corporate":
        rw = get_sa_corporate_rw(exposure.rating, exposure.is_sme)
    elif cls == "retail":
        rw = get_sa_retail_rw("regulatory_retail")
    elif cls == "residential_re":
        if exposure.ltv is None:
            raise ValueError("LTV required for residential RE")
        rw = get_sa_residential_re_rw(exposure.ltv, exposure.income_producing)
    elif cls == "commercial_re":
        if exposure.ltv is None:
            raise ValueError("LTV required for commercial RE")
        rw = get_sa_commercial_re_rw(exposure.ltv)
    elif cls == "subordinated":
        rw = SA_SUBORDINATED_RW
    elif cls == "equity":
        rw = SA_EQUITY_RW.get("other", 250)
    else:
        rw = 100  # default

    rwa = exposure.ead * rw / 100.0
    return {
        "approach": "SA-CR",
        "asset_class": cls, "rating": exposure.rating,
        "ead": exposure.ead, "risk_weight_pct": rw, "rwa": rwa,
        "capital_requirement": rwa * 0.08,
    }


# ---- IRB correlation and capital ----

# PD floors per asset class
PD_FLOORS: dict[str, float] = {
    "corporate": 0.0005,
    "sme_corporate": 0.0005,
    "bank": 0.0005,
    "sovereign": 0.0003,
    "retail_mortgage": 0.0005,
    "retail_revolving": 0.0005,
    "retail_other": 0.0005,
    "hvcre": 0.0005,
}

# A-IRB LGD floors (Basel IV)
LGD_FLOORS: dict[str, float] = {
    "senior_unsecured": 0.25,
    "senior_secured": 0.15,
    "subordinated": 0.50,
    "retail_secured_rre": 0.05,
    "retail_secured_other": 0.10,
    "retail_unsecured": 0.30,
    "retail_revolving": 0.50,
}


def sme_correlation_adjustment(sales_eur_millions: float | None) -> float:
    """SME firm-size correlation adjustment (CRE31.8).

    Reduces correlation by up to 4 pp for SMEs with sales 5-50 EUR M.
    """
    if sales_eur_millions is None or sales_eur_millions >= 50:
        return 0.0
    s = max(min(sales_eur_millions, 50), 5)
    return 0.04 * (1 - (s - 5) / 45)


def calculate_correlation(
    pd: float,
    asset_class: str = "corporate",
    sales_turnover: float | None = None,
) -> float:
    """Asset correlation R for the IRB Vasicek formula.

    Corporate: R = 0.12 × (1 - e^{-50×PD})/(1 - e^{-50}) + 0.24 × [1 - (...)]
    """
    if asset_class == "retail_mortgage":
        return 0.15
    if asset_class == "retail_revolving":
        return 0.04
    if asset_class == "retail_other":
        r_min, r_max, k = 0.03, 0.16, 35
    else:
        # corporate, bank, sovereign, sme_corporate, hvcre
        r_min, r_max, k = 0.12, 0.24, 50

    exp_factor = (1 - math.exp(-k * pd)) / (1 - math.exp(-k))
    r = r_min * exp_factor + r_max * (1 - exp_factor)

    if asset_class in ("corporate", "sme_corporate") and sales_turnover is not None:
        adj = sme_correlation_adjustment(sales_turnover)
        r = max(r - adj, r_min)

    return r


def calculate_maturity_adjustment(pd: float) -> float:
    """Maturity adjustment factor: b(PD) = (0.11852 - 0.05478 × ln(PD))²."""
    pd = max(pd, 0.0001)
    return (0.11852 - 0.05478 * math.log(pd)) ** 2


def calculate_capital_requirement(
    pd: float,
    lgd: float,
    maturity: float = 2.5,
    asset_class: str = "corporate",
    sales_turnover: float | None = None,
) -> float:
    """IRB capital requirement K via Vasicek formula.

    K = [LGD × N(...) - PD × LGD] × maturity_factor
    """
    pd_floor = PD_FLOORS.get(asset_class, 0.0003)
    pd = max(pd, pd_floor)
    pd = min(pd, 1.0)

    r = calculate_correlation(pd, asset_class, sales_turnover)

    g_pd = norm.ppf(pd)
    g_999 = norm.ppf(0.999)
    conditional_pd = norm.cdf(
        (1 - r) ** (-0.5) * g_pd + (r / (1 - r)) ** 0.5 * g_999
    )

    expected_loss = pd * lgd
    unexpected_loss = lgd * conditional_pd - expected_loss

    # Maturity adjustment (not applied to retail)
    if asset_class.startswith("retail"):
        return max(unexpected_loss, 0)

    b = calculate_maturity_adjustment(pd)
    maturity_factor = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    k = unexpected_loss * maturity_factor
    return max(k, 0)


# ---- F-IRB / A-IRB RWA ----

def calculate_irb_rwa(
    ead: float,
    pd: float,
    lgd: float = 0.45,  # F-IRB default for senior unsecured
    maturity: float = 2.5,
    asset_class: str = "corporate",
    sales_turnover: float | None = None,
) -> dict:
    """F-IRB RWA calculation. Default LGD = 0.45 (senior unsecured)."""
    k = calculate_capital_requirement(pd, lgd, maturity, asset_class, sales_turnover)
    rwa = k * 12.5 * ead
    return {
        "approach": "F-IRB",
        "ead": ead, "pd": pd, "lgd": lgd, "maturity": maturity,
        "asset_class": asset_class,
        "correlation": calculate_correlation(pd, asset_class, sales_turnover),
        "capital_requirement_k": k,
        "risk_weight_pct": k * 12.5 * 100,
        "rwa": rwa,
        "expected_loss": pd * lgd * ead,
    }


def calculate_airb_rwa(
    ead: float,
    pd: float,
    lgd: float,
    maturity: float = 2.5,
    asset_class: str = "corporate",
    collateral_type: str = "senior_unsecured",
    sales_turnover: float | None = None,
) -> dict:
    """A-IRB RWA with LGD floor (Basel IV)."""
    # Apply LGD floor
    if asset_class.startswith("retail"):
        if collateral_type == "secured_rre":
            floor = LGD_FLOORS.get("retail_secured_rre", 0.05)
        elif collateral_type == "secured_other":
            floor = LGD_FLOORS.get("retail_secured_other", 0.10)
        elif asset_class == "retail_revolving":
            floor = LGD_FLOORS.get("retail_revolving", 0.50)
        else:
            floor = LGD_FLOORS.get("retail_unsecured", 0.30)
    else:
        floor = LGD_FLOORS.get(collateral_type, 0.25)

    lgd_floored = max(lgd, floor)

    k = calculate_capital_requirement(pd, lgd_floored, maturity, asset_class, sales_turnover)
    rwa = k * 12.5 * ead

    return {
        "approach": "A-IRB",
        "ead": ead, "pd": pd, "lgd": lgd, "lgd_floored": lgd_floored, "lgd_floor": floor,
        "maturity": maturity, "asset_class": asset_class,
        "capital_requirement_k": k,
        "risk_weight_pct": k * 12.5 * 100,
        "rwa": rwa,
        "expected_loss": pd * lgd_floored * ead,
    }


# ---- Specialised lending slotting (CRE34) ----

# Slotting risk weights (CRE34.5) for project finance, object finance,
# commodities finance, IPRE, HVCRE
SLOTTING_RW: dict[str, dict[str, float]] = {
    "strong":      {"PF": 70,  "OF": 70,  "CF": 70,  "IPRE": 70,  "HVCRE": 95},
    "good":        {"PF": 90,  "OF": 90,  "CF": 90,  "IPRE": 90,  "HVCRE": 120},
    "satisfactory": {"PF": 115, "OF": 115, "CF": 115, "IPRE": 115, "HVCRE": 140},
    "weak":        {"PF": 250, "OF": 250, "CF": 250, "IPRE": 250, "HVCRE": 250},
    "default":     {"PF": 0,   "OF": 0,   "CF": 0,   "IPRE": 0,   "HVCRE": 0},
}


def slotting_risk_weight(category: str, sl_class: str = "PF") -> float:
    """Specialised lending slotting risk weight (CRE34.5).

    Args:
        category: "strong", "good", "satisfactory", "weak", "default".
        sl_class: PF (project), OF (object), CF (commodities), IPRE, HVCRE.
    """
    cat = category.lower()
    if cat not in SLOTTING_RW:
        raise ValueError(f"Unknown slotting category: {category}")
    return SLOTTING_RW[cat].get(sl_class.upper(), 115)


def calculate_slotting_rwa(
    ead: float,
    category: str,
    sl_class: str = "PF",
) -> dict:
    """Slotting RWA for specialised lending."""
    rw = slotting_risk_weight(category, sl_class)
    rwa = ead * rw / 100.0
    return {
        "approach": "Slotting",
        "ead": ead, "category": category, "sl_class": sl_class,
        "risk_weight_pct": rw, "rwa": rwa,
        "capital_requirement": rwa * 0.08,
    }


# ---- Comparison ----

def compare_sa_vs_irb(
    ead: float,
    pd: float,
    lgd: float,
    maturity: float,
    asset_class: str,
    rating: str,
) -> dict:
    """Compare SA-CR vs F-IRB RWA for the same exposure."""
    sa_class = "corporate" if asset_class == "sme_corporate" else asset_class
    sa = calculate_sa_rwa(SAExposure(ead=ead, asset_class=sa_class, rating=rating))
    irb = calculate_irb_rwa(ead, pd, lgd, maturity, asset_class)
    return {
        "sa_rwa": sa["rwa"], "irb_rwa": irb["rwa"],
        "sa_rw_pct": sa["risk_weight_pct"], "irb_rw_pct": irb["risk_weight_pct"],
        "irb_savings_pct": (1 - irb["rwa"] / sa["rwa"]) * 100 if sa["rwa"] > 0 else 0,
        "sa_detail": sa, "irb_detail": irb,
    }
