"""Commodity FRTB SA capital: wire commodity positions into the regulatory engine.

Bridges :mod:`pricebook.commodity_book` and
:mod:`pricebook.regulatory.market_risk_sa`. Maps commodities to FRTB SA
COM buckets (energy_solid/liquid/electricity, metals, agriculture, …),
builds the position dicts, and produces a one-call
``CommodityCapitalReport``.

Forward/Futures protocol decision
----------------------------------
After building every forward/future flavour (FXForward, EquityForward,
CommodityForwardCurve, FRA, CommodityFuture, BondFuture, IRFuture) we
assessed whether a unifying ``Forward`` or ``Future`` protocol would
reduce duplication.

**Decision: no protocol.** The pricing interfaces are too divergent —
IR futures need convexity adjustments and curve references, commodity
futures are pure price containers, bond futures need CTD/conversion
factor, FX forwards need two discount curves. A protocol would either
be so thin as to be useless (just ``price`` and ``expiry``) or force
every subclass to carry fields it doesn't need. The current approach —
each product owns its interface, with ad-hoc adapters in desk modules —
is simpler and more honest. If a need for cross-asset aggregation arises
(e.g. a unified P&L blotter), a thin ``Tradeable`` protocol with
``pv_ctx`` and ``maturity_date`` can be introduced at that point.

References:
    BCBS d457 (FRTB), MAR21 (COM risk weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pricebook.regulatory.market_risk_sa import (
    COM_RISK_WEIGHTS,
    calculate_frtb_sa,
)


# ---- Inputs ----

@dataclass
class CommodityRiskInputs:
    """Per-commodity market risk sensitivities for FRTB SA.

    Attributes:
        commodity: name identifier (WTI, Brent, gold, …).
        delta: Δ × price sensitivity (notional exposure).
        vega: vega per vol point.
        cvr_up / cvr_down: curvature scenarios.
        notional: notional for DRC / RRAO.
    """
    commodity: str
    delta: float = 0.0
    vega: float = 0.0
    cvr_up: float = 0.0
    cvr_down: float = 0.0
    notional: float = 0.0


@dataclass
class CommodityClassification:
    """Per-commodity FRTB SA classification.

    Attributes:
        commodity: name identifier.
        bucket: one of the COM_RISK_WEIGHTS keys
            (energy_solid, energy_liquid, energy_electricity, freight,
            metals_precious, metals_non_precious, agriculture_grains,
            agriculture_softs, other).
        is_exotic: exotic payoff flag (for RRAO).
    """
    commodity: str
    bucket: str
    is_exotic: bool = False


# ---- Bucket mapping ----

COMMODITY_SECTOR_MAP: dict[str, str] = {
    "crude": "energy_liquid",
    "wti": "energy_liquid",
    "brent": "energy_liquid",
    "gasoline": "energy_liquid",
    "distillate": "energy_liquid",
    "heating_oil": "energy_liquid",
    "diesel": "energy_liquid",
    "jet_fuel": "energy_liquid",
    "natgas": "energy_solid",
    "natural_gas": "energy_solid",
    "lng": "energy_solid",
    "coal": "energy_solid",
    "power": "energy_electricity",
    "electricity": "energy_electricity",
    "gold": "metals_precious",
    "silver": "metals_precious",
    "platinum": "metals_precious",
    "palladium": "metals_precious",
    "copper": "metals_non_precious",
    "aluminium": "metals_non_precious",
    "aluminum": "metals_non_precious",
    "zinc": "metals_non_precious",
    "nickel": "metals_non_precious",
    "iron_ore": "metals_non_precious",
    "wheat": "agriculture_grains",
    "corn": "agriculture_grains",
    "soybean": "agriculture_grains",
    "rice": "agriculture_grains",
    "soybean_meal": "agriculture_grains",
    "soybean_oil": "agriculture_softs",
    "sugar": "agriculture_softs",
    "coffee": "agriculture_softs",
    "cocoa": "agriculture_softs",
    "cotton": "agriculture_softs",
}


def map_to_com_bucket(commodity: str) -> str:
    """Map a commodity name to its FRTB SA COM bucket.

    Uses a built-in lookup table; falls back to ``"other"`` for
    unrecognised names.
    """
    return COMMODITY_SECTOR_MAP.get(commodity.strip().lower(), "other")


# ---- Wiring ----

def commodity_to_frtb_positions(
    inputs: list[CommodityRiskInputs],
    classifications: dict[str, CommodityClassification] | None = None,
) -> dict:
    """Convert commodity inputs into the dict format expected by
    :func:`calculate_frtb_sa`.
    """
    classifications = classifications or {}
    delta_com: list[dict] = []
    vega_com: list[dict] = []
    curv_com: list[dict] = []
    rrao_pos: list[dict] = []

    for inp in inputs:
        cls = classifications.get(inp.commodity)
        bucket = cls.bucket if cls else map_to_com_bucket(inp.commodity)
        is_exotic = cls.is_exotic if cls else False

        rw = COM_RISK_WEIGHTS.get(bucket, COM_RISK_WEIGHTS["other"])

        delta_com.append({
            "bucket": bucket,
            "sensitivity": inp.delta,
            "risk_weight": rw,
        })

        if inp.vega:
            vega_com.append({
                "bucket": bucket,
                "vega": inp.vega,
                "vega_risk_weight": 50,  # COM vega RW per MAR21
            })

        if inp.cvr_up or inp.cvr_down:
            curv_com.append({
                "bucket": bucket,
                "cvr_up": inp.cvr_up,
                "cvr_down": inp.cvr_down,
            })

        if is_exotic and inp.notional > 0:
            rrao_pos.append({
                "notional": inp.notional,
                "is_exotic": True,
            })

    return {
        "delta_positions": {"COM": delta_com},
        "vega_positions": {"COM": vega_com},
        "curvature_positions": {"COM": curv_com},
        "drc_positions": [],  # no equity-style DRC for commodities
        "rrao_positions": rrao_pos,
    }


# ---- Capital report ----

@dataclass
class CommodityCapitalReport:
    """One-call FRTB SA capital summary for a commodity desk."""
    sbm_capital: float
    delta_capital: float
    vega_capital: float
    curvature_capital: float
    rrao_capital: float
    total_capital: float
    total_rwa: float
    total_notional: float
    capital_efficiency: float
    bucket_capitals: dict[str, float]
    n_commodities: int

    @property
    def sbm_components_sum(self) -> float:
        return self.delta_capital + self.vega_capital + self.curvature_capital


def commodity_frtb_capital(
    inputs: list[CommodityRiskInputs],
    classifications: dict[str, CommodityClassification] | None = None,
) -> CommodityCapitalReport:
    """Run the full FRTB SA commodity calculation in one call."""
    positions = commodity_to_frtb_positions(inputs, classifications)
    result = calculate_frtb_sa(**positions)

    com_sbm = result["sbm_by_risk_class"].get("COM", {})
    delta_detail = com_sbm.get("delta_detail", {}) or {}
    bucket_capitals = delta_detail.get("bucket_capitals", {})

    total_notional = sum(inp.notional for inp in inputs)
    capital_eff = (
        result["total_capital"] / total_notional if total_notional > 0 else 0.0
    )

    return CommodityCapitalReport(
        sbm_capital=com_sbm.get("total_capital", 0.0),
        delta_capital=com_sbm.get("delta_capital", 0.0),
        vega_capital=com_sbm.get("vega_capital", 0.0),
        curvature_capital=com_sbm.get("curvature_capital", 0.0),
        rrao_capital=result.get("rrao_capital", 0.0),
        total_capital=result["total_capital"],
        total_rwa=result["total_rwa"],
        total_notional=total_notional,
        capital_efficiency=capital_eff,
        bucket_capitals=dict(bucket_capitals),
        n_commodities=len(inputs),
    )
