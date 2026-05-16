"""FX FRTB SA capital: wire FX positions into the regulatory engine.

Maps FX positions to the FRTB SA FX risk class. Liquid pairs get a
reduced risk weight (11.25% vs 15% for others).

References:
    BCBS d457 (FRTB), MAR21 (FX risk weights).
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.regulatory.market_risk_sa import (
    FX_RISK_WEIGHT,
    FX_LIQUID_PAIRS_RW,
    calculate_frtb_sa,
)


# ---- Inputs ----

@dataclass
class FXRiskInputs:
    """Per-pair market risk sensitivity for FRTB SA."""
    pair: str
    delta: float = 0.0
    notional: float = 0.0
    is_liquid: bool = False


# ---- Liquid pair classification ----

LIQUID_PAIRS = {
    "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD",
    "USD/CHF", "EUR/GBP", "EUR/JPY", "EUR/CHF",
}


def is_liquid_pair(pair: str) -> bool:
    """Check if a pair qualifies for the reduced FX risk weight."""
    normalized = pair.upper().replace(" ", "")
    if normalized in LIQUID_PAIRS:
        return True
    parts = normalized.split("/")
    if len(parts) == 2 and f"{parts[1]}/{parts[0]}" in LIQUID_PAIRS:
        return True
    return False


# ---- Wiring ----

def fx_to_frtb_positions(
    inputs: list[FXRiskInputs],
) -> dict:
    """Convert FX inputs into the dict format for ``calculate_frtb_sa``."""
    fx_pos: list[dict] = []
    for inp in inputs:
        rw = FX_LIQUID_PAIRS_RW if (inp.is_liquid or is_liquid_pair(inp.pair)) else FX_RISK_WEIGHT
        fx_pos.append({
            "bucket": inp.pair,
            "sensitivity": inp.delta,
            "risk_weight": rw,
        })
    return {
        "delta_positions": {"FX": fx_pos},
        "vega_positions": {},
        "curvature_positions": {},
        "drc_positions": [],
        "rrao_positions": [],
    }


# ---- Capital report ----

@dataclass
class FXCapitalReport:
    """One-call FRTB SA capital summary for an FX desk."""
    delta_capital: float
    total_capital: float
    total_rwa: float
    total_notional: float
    capital_efficiency: float
    n_pairs: int


def fx_frtb_capital(
    inputs: list[FXRiskInputs],
) -> FXCapitalReport:
    """Run the full FRTB SA FX calculation in one call."""
    positions = fx_to_frtb_positions(inputs)
    result = calculate_frtb_sa(**positions)

    fx_sbm = result["sbm_by_risk_class"].get("FX", {})
    delta_cap = fx_sbm.get("total_capital", 0.0)
    total = result["total_capital"]

    total_notional = sum(inp.notional for inp in inputs)
    eff = total / total_notional if total_notional > 0 else 0.0

    return FXCapitalReport(
        delta_capital=delta_cap,
        total_capital=total,
        total_rwa=result["total_rwa"],
        total_notional=total_notional,
        capital_efficiency=eff,
        n_pairs=len(inputs),
    )
