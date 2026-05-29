"""ESG bond labelling framework (D8).

Green, social, sustainability, and sustainability-linked bond classification,
greenium calculation, and ESG-adjusted spread analytics.

    from pricebook.fixed_income.esg_bonds import (
        ESGLabel, ESGBondSpec, greenium, esg_adjusted_spread,
        create_green_bond, list_esg_labels,
    )

Framework:
- ICMA Green Bond Principles (2021)
- ICMA Social Bond Principles (2021)
- ICMA Sustainability Bond Guidelines (2021)
- ICMA Sustainability-Linked Bond Principles (2020)

Greenium (green premium):
    greenium = yield_conventional - yield_green
    Typically 2-10bp for investment grade sovereigns/agencies.

References:
    Ehlers & Packer (2017). Green Bond Finance and Certification. BIS QR.
    Zerbib (2019). The Effect of Pro-Environmental Preferences on Bond Prices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from pricebook.core.serialisable import serialisable_convention


class ESGLabel(Enum):
    """ESG bond label types (ICMA classification)."""
    GREEN = "green"                           # Use of proceeds: environmental
    SOCIAL = "social"                         # Use of proceeds: social
    SUSTAINABILITY = "sustainability"         # Use of proceeds: environmental + social
    SUSTAINABILITY_LINKED = "sustainability_linked"  # KPI-linked coupon step-up/down
    TRANSITION = "transition"                 # Financing transition from brown to green
    BLUE = "blue"                             # Ocean/water-related
    UNLABELLED = "unlabelled"                 # Conventional bond (no ESG label)


class UseOfProceeds(Enum):
    """Eligible use-of-proceeds categories (ICMA taxonomy)."""
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    CLEAN_TRANSPORT = "clean_transport"
    GREEN_BUILDINGS = "green_buildings"
    BIODIVERSITY = "biodiversity"
    WATER_MANAGEMENT = "water_management"
    POLLUTION_PREVENTION = "pollution_prevention"
    AFFORDABLE_HOUSING = "affordable_housing"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    FOOD_SECURITY = "food_security"
    SOCIOECONOMIC_ADVANCEMENT = "socioeconomic_advancement"
    SME_FINANCING = "sme_financing"
    GENERAL = "general"


@serialisable_convention("esg_bond_spec")
@dataclass(frozen=True)
class ESGBondSpec:
    """ESG bond specification — labelling and use-of-proceeds metadata."""
    label: ESGLabel
    issuer: str
    currency: str
    use_of_proceeds: list[str] = field(default_factory=list)
    external_reviewer: str = ""            # e.g. "Sustainalytics", "Vigeo Eiris"
    framework: str = ""                    # e.g. "ICMA GBP 2021"
    kpi_target: str = ""                   # for sustainability-linked: KPI description
    coupon_step_up_bps: float = 0.0        # SLB: step-up if KPI missed
    coupon_step_down_bps: float = 0.0      # SLB: step-down if KPI achieved
    taxonomy_alignment: str = ""           # "EU Taxonomy", "CBI Standard"
    second_party_opinion: bool = False
    post_issuance_reporting: bool = False
    notes: str = ""


@dataclass
class GreeniumResult:
    """Result of greenium calculation."""
    greenium_bps: float         # yield difference (positive = green trades tighter)
    green_yield: float
    conventional_yield: float
    green_z_spread: float
    conventional_z_spread: float
    confidence: str             # "high" if matched pair, "medium" if interpolated

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ESGAdjustedSpreadResult:
    """ESG-adjusted spread decomposition."""
    total_spread_bps: float
    credit_spread_bps: float
    greenium_bps: float
    liquidity_premium_bps: float
    esg_label: str
    issuer: str

    def to_dict(self) -> dict:
        return vars(self)


def greenium(
    green_yield: float,
    conventional_yield: float,
    green_z_spread: float | None = None,
    conventional_z_spread: float | None = None,
) -> GreeniumResult:
    """Compute the greenium (green bond premium).

    greenium = yield_conventional - yield_green

    Positive greenium means the green bond trades at a lower yield
    (tighter spread) than its conventional equivalent — investors
    accept a lower return for the ESG label.

    Args:
        green_yield: YTM of the green bond.
        conventional_yield: YTM of the comparable conventional bond.
        green_z_spread: Z-spread of the green bond (optional).
        conventional_z_spread: Z-spread of the conventional bond (optional).
    """
    greenium_bps = (conventional_yield - green_yield) * 10_000

    return GreeniumResult(
        greenium_bps=greenium_bps,
        green_yield=green_yield,
        conventional_yield=conventional_yield,
        green_z_spread=green_z_spread or 0.0,
        conventional_z_spread=conventional_z_spread or 0.0,
        confidence="high" if green_z_spread is not None else "medium",
    )


def esg_adjusted_spread(
    total_spread_bps: float,
    estimated_greenium_bps: float = 5.0,
    liquidity_premium_bps: float = 3.0,
    esg_label: ESGLabel = ESGLabel.GREEN,
    issuer: str = "",
) -> ESGAdjustedSpreadResult:
    """Decompose a bond spread into credit + greenium + liquidity.

    total_spread = credit_spread + greenium + liquidity_premium

    The credit spread is the "true" credit risk, after removing
    the ESG-driven premium and any liquidity discount.
    """
    credit_spread_bps = total_spread_bps - estimated_greenium_bps - liquidity_premium_bps

    return ESGAdjustedSpreadResult(
        total_spread_bps=total_spread_bps,
        credit_spread_bps=max(credit_spread_bps, 0.0),
        greenium_bps=estimated_greenium_bps,
        liquidity_premium_bps=liquidity_premium_bps,
        esg_label=esg_label.value,
        issuer=issuer,
    )


def slb_coupon_adjustment(
    base_coupon: float,
    spec: ESGBondSpec,
    kpi_achieved: bool,
) -> float:
    """Compute adjusted coupon for a sustainability-linked bond.

    If KPI is missed: coupon steps up by coupon_step_up_bps.
    If KPI is achieved: coupon steps down by coupon_step_down_bps.

    Returns the adjusted annual coupon rate.
    """
    if spec.label != ESGLabel.SUSTAINABILITY_LINKED:
        return base_coupon

    if kpi_achieved:
        return base_coupon - spec.coupon_step_down_bps / 10_000
    else:
        return base_coupon + spec.coupon_step_up_bps / 10_000


def create_green_bond(
    issuer: str,
    currency: str,
    issue_date: date,
    maturity: date,
    coupon_rate: float,
    use_of_proceeds: list[str] | None = None,
    market_code: str = "UST",
    face_value: float = 100.0,
    external_reviewer: str = "",
    framework: str = "ICMA GBP 2021",
) -> tuple:
    """Create a green bond (FixedRateBond + ESGBondSpec).

    Returns (bond, esg_spec) tuple. The bond is a standard FixedRateBond
    with the sovereign convention for the given market code. The ESG spec
    carries the labelling metadata.
    """
    from pricebook.fixed_income.sovereign_bonds import get_conventions
    from pricebook.fixed_income.bond import FixedRateBond

    conv = get_conventions(market_code)
    bond = FixedRateBond.from_convention(conv, issue_date, maturity, coupon_rate, face_value)

    spec = ESGBondSpec(
        label=ESGLabel.GREEN,
        issuer=issuer,
        currency=currency,
        use_of_proceeds=use_of_proceeds or [UseOfProceeds.RENEWABLE_ENERGY.value],
        external_reviewer=external_reviewer,
        framework=framework,
        second_party_opinion=bool(external_reviewer),
    )

    return bond, spec


def list_esg_labels() -> list[str]:
    """Return available ESG label types."""
    return [e.value for e in ESGLabel]


def list_use_of_proceeds() -> list[str]:
    """Return eligible use-of-proceeds categories."""
    return [u.value for u in UseOfProceeds]
