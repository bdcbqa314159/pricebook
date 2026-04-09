"""Specialty regulatory modules: crypto, step-in risk, Pillar 3, FX, simplified SA.

- Crypto-assets (BCBS d545): Group 1a/1b/2a/2b classification and RWA
- Step-in risk (BCBS 398): assessment and capital impact
- Pillar 3 disclosure helpers
- FX rate conversion with market conventions
- Simplified SA for market risk

    from pricebook.regulatory.specialty import (
        classify_crypto_asset, calculate_group1a_rwa, calculate_group2_rwa,
        check_group2_exposure_limit,
        UnconsolidatedEntity, assess_step_in_indicators,
        FXRates, get_default_fx_rates,
        simplified_sa_market_risk,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Crypto-assets (BCBS d545)
# =============================================================================

CRYPTO_GROUPS = {
    "1a": {"description": "Tokenised traditional assets", "base_rw": None},
    "1b": {"description": "Stablecoins with effective stabilisation",
           "base_rw": None, "redemption_risk_addon": 0.025},
    "2a": {"description": "Crypto with hedging recognition", "base_rw": 100},
    "2b": {"description": "Other crypto", "base_rw": 1250},
}

GROUP2_EXPOSURE_LIMIT = 0.02   # 2% of Tier 1
GROUP2B_EXPOSURE_LIMIT = 0.01  # 1% of Tier 1


def classify_crypto_asset(
    is_tokenised_traditional: bool = False,
    has_effective_stabilisation: bool = False,
    passes_redemption_test: bool = False,
    passes_reserve_test: bool = False,
    infrastructure_risk_acceptable: bool = True,
) -> str:
    """Classify crypto into Group 1a, 1b, 2a, or 2b."""
    if not infrastructure_risk_acceptable:
        return "2b"
    if is_tokenised_traditional:
        return "1a"
    if has_effective_stabilisation and passes_redemption_test and passes_reserve_test:
        return "1b"
    return "2b"


def calculate_group1a_rwa(
    exposure: float,
    underlying_rw: float,
    infrastructure_addon_pct: float = 0.025,
) -> dict:
    """Group 1a RWA: same as underlying + infrastructure add-on."""
    infrastructure_addon = exposure * infrastructure_addon_pct
    rwa = exposure * underlying_rw / 100 + infrastructure_addon
    return {
        "group": "1a", "exposure": exposure,
        "underlying_rw": underlying_rw,
        "infrastructure_addon": infrastructure_addon,
        "rwa": rwa,
    }


def calculate_group1b_rwa(
    exposure: float,
    weighted_reserve_rw: float,
    redemption_risk_addon: float = 0.025,
    infrastructure_addon_pct: float = 0.025,
) -> dict:
    """Group 1b stablecoin RWA: reserve RW + redemption + infrastructure."""
    base_rwa = exposure * weighted_reserve_rw / 100
    redemption_addon = exposure * redemption_risk_addon
    infrastructure_addon = exposure * infrastructure_addon_pct
    total_rwa = base_rwa + redemption_addon + infrastructure_addon
    return {
        "group": "1b", "exposure": exposure,
        "weighted_reserve_rw": weighted_reserve_rw,
        "base_rwa": base_rwa,
        "redemption_addon": redemption_addon,
        "infrastructure_addon": infrastructure_addon,
        "total_rwa": total_rwa,
    }


def calculate_group2_rwa(
    long_exposure: float,
    short_exposure: float = 0,
    is_group_2a: bool = False,
) -> dict:
    """Group 2 crypto RWA. 2a: net × 100%; 2b: gross × 1250%."""
    if is_group_2a:
        net = abs(long_exposure - short_exposure)
        gross = long_exposure + short_exposure
        return {
            "group": "2a", "long_exposure": long_exposure, "short_exposure": short_exposure,
            "net_exposure": net, "gross_exposure": gross,
            "risk_weight_pct": 100, "rwa": net,
            "hedging_benefit": (gross - net),
        }
    gross = long_exposure + abs(short_exposure)
    return {
        "group": "2b", "long_exposure": long_exposure, "short_exposure": short_exposure,
        "gross_exposure": gross, "risk_weight_pct": 1250,
        "rwa": gross * 12.5, "hedging_benefit": 0,
    }


def check_group2_exposure_limit(
    group2a_exposure: float,
    group2b_exposure: float,
    tier1_capital: float,
) -> dict:
    """Check Group 2 exposure limits (2% total, 1% for 2b alone)."""
    total = group2a_exposure + group2b_exposure
    g2_ratio = total / tier1_capital if tier1_capital > 0 else 0
    g2b_ratio = group2b_exposure / tier1_capital if tier1_capital > 0 else 0

    g2_excess = max(0, total - tier1_capital * GROUP2_EXPOSURE_LIMIT)
    g2b_excess = max(0, group2b_exposure - tier1_capital * GROUP2B_EXPOSURE_LIMIT)

    return {
        "group2a_exposure": group2a_exposure,
        "group2b_exposure": group2b_exposure,
        "total_group2": total,
        "tier1_capital": tier1_capital,
        "group2_ratio_pct": g2_ratio * 100,
        "group2b_ratio_pct": g2b_ratio * 100,
        "group2_limit_breached": g2_ratio > GROUP2_EXPOSURE_LIMIT,
        "group2b_limit_breached": g2b_ratio > GROUP2B_EXPOSURE_LIMIT,
        "group2_excess": g2_excess,
        "group2b_excess": g2b_excess,
        "excess_rwa": (g2_excess + g2b_excess) * 12.5,
    }


# =============================================================================
# Step-in Risk (BCBS 398)
# =============================================================================

class EntityType(Enum):
    SECURITIZATION_VEHICLE = "securitization_vehicle"
    INVESTMENT_FUND = "investment_fund"
    MONEY_MARKET_FUND = "mmf"
    PENSION_FUND = "pension_fund"
    STRUCTURED_ENTITY = "structured_entity"
    CONDUIT = "conduit"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    OTHER_SPV = "other_spv"


class StepInIndicator(Enum):
    SPONSORSHIP = "sponsorship"
    IMPLICIT_SUPPORT = "implicit_support"
    INVOLVEMENT = "involvement"
    REPUTATION = "reputation"
    INVESTOR_EXPECTATIONS = "investor_expectations"
    NAME_ASSOCIATION = "name_association"
    CREDIT_ENHANCEMENT = "credit_enhancement"
    LIQUIDITY_SUPPORT = "liquidity_support"
    REVENUE_DEPENDENCE = "revenue_dependence"


INDICATOR_WEIGHTS: dict[StepInIndicator, float] = {
    StepInIndicator.SPONSORSHIP: 0.20,
    StepInIndicator.IMPLICIT_SUPPORT: 0.25,
    StepInIndicator.INVOLVEMENT: 0.10,
    StepInIndicator.REPUTATION: 0.15,
    StepInIndicator.INVESTOR_EXPECTATIONS: 0.15,
    StepInIndicator.NAME_ASSOCIATION: 0.05,
    StepInIndicator.CREDIT_ENHANCEMENT: 0.05,
    StepInIndicator.LIQUIDITY_SUPPORT: 0.03,
    StepInIndicator.REVENUE_DEPENDENCE: 0.02,
}

CAPITAL_TREATMENT: dict[str, dict] = {
    "high":   {"approach": "full_consolidation", "capital_charge_factor": 1.0},
    "medium": {"approach": "proportional",       "capital_charge_factor": 0.5},
    "low":    {"approach": "monitoring",         "capital_charge_factor": 0.0},
}


@dataclass
class UnconsolidatedEntity:
    """An entity subject to step-in risk assessment."""
    entity_id: str
    entity_name: str
    entity_type: EntityType
    total_assets: float
    is_sponsored: bool = False
    uses_bank_name: bool = False
    contractual_exposure: float = 0
    ownership_percentage: float = 0
    past_support_provided: bool = False


@dataclass
class StepInAssessment:
    entity: UnconsolidatedEntity
    indicator_scores: dict = field(default_factory=dict)
    overall_score: float = 0.0
    risk_level: str = "low"


def assess_step_in_indicators(
    entity: UnconsolidatedEntity,
    has_implicit_support_expectation: bool = False,
    involvement_level: float = 0.0,
    reputational_impact: float = 0.0,
    investor_expectation_level: float = 0.0,
    provides_credit_enhancement: bool = False,
    provides_liquidity_support: bool = False,
    revenue_from_entity_significant: bool = False,
) -> StepInAssessment:
    """Score step-in indicators and return assessment."""
    scores = {
        StepInIndicator.SPONSORSHIP: 1.0 if entity.is_sponsored else 0.0,
        StepInIndicator.IMPLICIT_SUPPORT: 1.0 if (has_implicit_support_expectation or entity.past_support_provided) else 0.0,
        StepInIndicator.INVOLVEMENT: involvement_level,
        StepInIndicator.REPUTATION: reputational_impact,
        StepInIndicator.INVESTOR_EXPECTATIONS: investor_expectation_level,
        StepInIndicator.NAME_ASSOCIATION: 1.0 if entity.uses_bank_name else 0.0,
        StepInIndicator.CREDIT_ENHANCEMENT: 1.0 if provides_credit_enhancement else 0.0,
        StepInIndicator.LIQUIDITY_SUPPORT: 1.0 if provides_liquidity_support else 0.0,
        StepInIndicator.REVENUE_DEPENDENCE: 1.0 if revenue_from_entity_significant else 0.0,
    }

    overall = sum(scores[ind] * INDICATOR_WEIGHTS[ind] for ind in scores)

    if overall >= 0.6:
        level = "high"
    elif overall >= 0.3:
        level = "medium"
    else:
        level = "low"

    return StepInAssessment(entity=entity, indicator_scores=scores,
                             overall_score=overall, risk_level=level)


def calculate_step_in_capital_impact(
    assessment: StepInAssessment,
    entity_rwa_if_consolidated: float,
) -> dict:
    """Capital impact based on step-in risk level."""
    treatment = CAPITAL_TREATMENT[assessment.risk_level]
    factor = treatment["capital_charge_factor"]
    implied_rwa = entity_rwa_if_consolidated * factor
    return {
        "entity_id": assessment.entity.entity_id,
        "step_in_risk_level": assessment.risk_level,
        "overall_score": assessment.overall_score,
        "treatment_approach": treatment["approach"],
        "capital_charge_factor": factor,
        "entity_rwa_if_consolidated": entity_rwa_if_consolidated,
        "implied_rwa": implied_rwa,
        "capital_requirement": implied_rwa * 0.08,
    }


# =============================================================================
# FX rates with market conventions
# =============================================================================

# (base, quote) for each market-convention pair
MARKET_CONVENTION: dict[str, tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"), "EURGBP": ("EUR", "GBP"), "EURJPY": ("EUR", "JPY"),
    "EURCHF": ("EUR", "CHF"), "GBPUSD": ("GBP", "USD"), "GBPJPY": ("GBP", "JPY"),
    "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"), "USDCAD": ("USD", "CAD"),
    "USDCNY": ("USD", "CNY"), "USDHKD": ("USD", "HKD"), "USDSGD": ("USD", "SGD"),
    "USDKRW": ("USD", "KRW"), "USDINR": ("USD", "INR"), "USDBRL": ("USD", "BRL"),
    "USDMXN": ("USD", "MXN"), "USDZAR": ("USD", "ZAR"),
}


@dataclass
class FXRates:
    """FX rate store with automatic convention handling."""
    _rates: dict = field(default_factory=dict)

    def set_spot(self, pair: str, rate: float) -> None:
        self._rates[self._normalize(pair)] = rate

    def set_rates(self, rates: dict) -> None:
        for p, r in rates.items():
            self.set_spot(p, r)

    def get_spot(self, pair: str) -> float | None:
        return self._rates.get(self._normalize(pair))

    def convert(self, amount: float, from_ccy: str, to_ccy: str) -> float:
        from_ccy = from_ccy.upper()
        to_ccy = to_ccy.upper()
        if from_ccy == to_ccy:
            return amount

        rate = self._get_rate(from_ccy, to_ccy)
        if rate is not None:
            return amount * rate

        # Triangulate via USD
        if from_ccy != "USD" and to_ccy != "USD":
            r1 = self._get_rate(from_ccy, "USD")
            r2 = self._get_rate("USD", to_ccy)
            if r1 is not None and r2 is not None:
                return amount * r1 * r2

        # Triangulate via EUR
        if from_ccy != "EUR" and to_ccy != "EUR":
            r1 = self._get_rate(from_ccy, "EUR")
            r2 = self._get_rate("EUR", to_ccy)
            if r1 is not None and r2 is not None:
                return amount * r1 * r2

        raise ValueError(f"No FX rate for {from_ccy} → {to_ccy}")

    def _get_rate(self, from_ccy: str, to_ccy: str) -> float | None:
        pair1 = from_ccy + to_ccy
        pair2 = to_ccy + from_ccy

        if pair1 in self._rates:
            base, _quote = MARKET_CONVENTION.get(pair1, (pair1[:3], pair1[3:]))
            return self._rates[pair1] if base == from_ccy else 1.0 / self._rates[pair1]

        if pair2 in self._rates:
            base, _quote = MARKET_CONVENTION.get(pair2, (pair2[:3], pair2[3:]))
            return self._rates[pair2] if base == from_ccy else 1.0 / self._rates[pair2]

        return None

    def _normalize(self, pair: str) -> str:
        return pair.replace("/", "").replace("-", "").upper()


def get_default_fx_rates() -> FXRates:
    """Approximate FX rates for testing/demos."""
    fx = FXRates()
    fx.set_rates({
        "EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 150.0, "USDCHF": 0.88,
        "USDCAD": 1.36, "AUDUSD": 0.66, "NZDUSD": 0.61, "USDCNY": 7.20,
        "USDHKD": 7.83, "USDSGD": 1.34, "USDKRW": 1320.0, "USDINR": 83.0,
        "USDBRL": 5.00, "USDMXN": 17.0, "USDZAR": 18.5,
    })
    return fx


# =============================================================================
# Simplified SA for market risk
# =============================================================================

# Simplified SA risk weights for small banks (Basel IV reduced framework)
SIMPLIFIED_SA_RW: dict[str, float] = {
    "GIRR": 0.02,        # 2% on net IR sensitivity
    "FX": 0.10,          # 10% on net FX position
    "EQ": 0.20,          # 20% on net equity position
    "COM": 0.20,         # 20% on net commodity position
    "CR": 0.10,          # 10% on net credit position
}


def simplified_sa_market_risk(
    girr_sensitivity: float = 0,
    fx_net_position: float = 0,
    equity_net_position: float = 0,
    commodity_net_position: float = 0,
    credit_net_position: float = 0,
) -> dict:
    """Simplified SA for market risk (small bank framework).

    Sum of |sensitivity| × supervisory weight.
    """
    components = {
        "GIRR": abs(girr_sensitivity) * SIMPLIFIED_SA_RW["GIRR"],
        "FX": abs(fx_net_position) * SIMPLIFIED_SA_RW["FX"],
        "EQ": abs(equity_net_position) * SIMPLIFIED_SA_RW["EQ"],
        "COM": abs(commodity_net_position) * SIMPLIFIED_SA_RW["COM"],
        "CR": abs(credit_net_position) * SIMPLIFIED_SA_RW["CR"],
    }
    total = sum(components.values())
    return {
        "approach": "Simplified SA",
        "components": components,
        "total_capital": total,
        "total_rwa": total * 12.5,
    }


# =============================================================================
# Pillar 3 Disclosure
# =============================================================================

class DisclosureFrequency(Enum):
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


def generate_km1_template(
    cet1: float,
    tier1: float,
    total_capital: float,
    rwa: float,
    leverage_exposure: float,
    lcr_hqla: float,
    lcr_net_outflows: float,
    nsfr_asf: float,
    nsfr_rsf: float,
) -> dict:
    """KM1: Key prudential metrics template (Pillar 3)."""
    return {
        "template": "KM1",
        "frequency": DisclosureFrequency.QUARTERLY.value,
        "available_capital": {
            "cet1": cet1, "tier1": tier1, "total_capital": total_capital,
        },
        "rwa": rwa,
        "capital_ratios": {
            "cet1_ratio_pct": (cet1 / rwa * 100) if rwa > 0 else 0,
            "tier1_ratio_pct": (tier1 / rwa * 100) if rwa > 0 else 0,
            "total_capital_ratio_pct": (total_capital / rwa * 100) if rwa > 0 else 0,
        },
        "leverage_ratio_pct": (tier1 / leverage_exposure * 100) if leverage_exposure > 0 else 0,
        "lcr_pct": (lcr_hqla / lcr_net_outflows * 100) if lcr_net_outflows > 0 else 0,
        "nsfr_pct": (nsfr_asf / nsfr_rsf * 100) if nsfr_rsf > 0 else 0,
    }


def generate_ov1_template(
    credit_rwa: float,
    market_rwa: float,
    operational_rwa: float,
    cva_rwa: float = 0,
    settlement_rwa: float = 0,
) -> dict:
    """OV1: Overview of RWA template (Pillar 3)."""
    total = credit_rwa + market_rwa + operational_rwa + cva_rwa + settlement_rwa
    return {
        "template": "OV1",
        "credit_rwa": credit_rwa,
        "market_rwa": market_rwa,
        "operational_rwa": operational_rwa,
        "cva_rwa": cva_rwa,
        "settlement_rwa": settlement_rwa,
        "total_rwa": total,
        "capital_requirement": total * 0.08,
    }
