"""ESG bond labelling and classification (D8).

Metadata module for green bonds, social bonds, sustainability-linked bonds (SLBs),
and transition bonds. Classifies use-of-proceeds and KPI triggers.

    from pricebook.fixed_income.esg_bond import (
        ESGLabel, ESGBondClassification, classify_esg_bond,
    )

References:
    ICMA (2021). Green Bond Principles.
    ICMA (2023). Sustainability-Linked Bond Principles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ESGLabel(Enum):
    """ESG bond label."""
    GREEN = "green"                          # Use-of-proceeds: environmental
    SOCIAL = "social"                        # Use-of-proceeds: social
    SUSTAINABILITY = "sustainability"        # Both green and social
    SLB = "sustainability_linked"            # KPI/target-based, general purpose
    TRANSITION = "transition"                # Financing decarbonisation
    BLUE = "blue"                            # Marine/ocean conservation
    CONVENTIONAL = "conventional"            # No ESG label


class UseOfProceeds(Enum):
    """ICMA Green Bond Principles categories."""
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity"
    CLEAN_TRANSPORT = "clean_transport"
    SUSTAINABLE_WATER = "sustainable_water"
    GREEN_BUILDINGS = "green_buildings"
    AFFORDABLE_HOUSING = "affordable_housing"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    FOOD_SECURITY = "food_security"
    SME_LENDING = "sme_lending"


@dataclass
class ESGBondClassification:
    """ESG classification for a bond."""
    label: ESGLabel
    use_of_proceeds: list[UseOfProceeds] = field(default_factory=list)
    kpi_targets: list[str] = field(default_factory=list)  # for SLBs
    step_up_bp: float = 0.0       # coupon step-up if KPI missed (SLB)
    step_down_bp: float = 0.0     # coupon step-down if KPI met
    second_party_opinion: str = "" # SPO provider (e.g. "Sustainalytics", "ISS")
    framework: str = ""            # e.g. "ICMA GBP 2021"
    greenium_bp: float = 0.0      # estimated green premium (spread tightening)

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "use_of_proceeds": [u.value for u in self.use_of_proceeds],
            "kpi_targets": self.kpi_targets,
            "step_up_bp": self.step_up_bp,
            "greenium_bp": self.greenium_bp,
            "framework": self.framework,
        }


def classify_esg_bond(
    label: str,
    use_of_proceeds: list[str] | None = None,
    kpi_targets: list[str] | None = None,
    step_up_bp: float = 0.0,
) -> ESGBondClassification:
    """Classify an ESG bond from basic inputs.

    Args:
        label: "green", "social", "sustainability", "slb", "transition".
        use_of_proceeds: list of use-of-proceeds categories.
        kpi_targets: KPI descriptions for SLBs.
        step_up_bp: coupon step-up if KPI missed.
    """
    esg_label = ESGLabel(label.lower())
    uop = [UseOfProceeds(u.lower()) for u in (use_of_proceeds or [])]

    # Estimate greenium based on label type
    greenium = {"green": 5.0, "social": 3.0, "sustainability": 4.0,
                "sustainability_linked": 2.0, "transition": 1.0,
                "blue": 3.0, "conventional": 0.0}.get(esg_label.value, 0.0)

    return ESGBondClassification(
        label=esg_label,
        use_of_proceeds=uop,
        kpi_targets=kpi_targets or [],
        step_up_bp=step_up_bp,
        greenium_bp=greenium,
    )


def list_esg_labels() -> list[str]:
    """Return available ESG labels."""
    return [l.value for l in ESGLabel]


def list_use_of_proceeds() -> list[str]:
    """Return ICMA use-of-proceeds categories."""
    return [u.value for u in UseOfProceeds]
