"""Bond FRTB SA capital: wire bond positions into GIRR + CSR + DRC.

Bridges :mod:`pricebook.bond_book` and
:mod:`pricebook.regulatory.market_risk_sa`. Bond positions carry both
interest rate risk (GIRR) and credit spread risk (CSR), plus default
risk (DRC). This module maps them to the appropriate FRTB SA buckets
and produces a one-call ``BondCapitalReport``.

References:
    BCBS d457 (FRTB), MAR21 (GIRR/CSR risk weights), MAR22 (DRC).
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.regulatory.market_risk_sa import (
    GIRR_RISK_WEIGHTS,
    CSR_RISK_WEIGHTS,
    DRC_RISK_WEIGHTS,
    DRC_LGD,
    calculate_frtb_sa,
)


# ---- Inputs ----

@dataclass
class BondRiskInputs:
    """Per-bond market risk sensitivities for FRTB SA.

    Attributes:
        issuer: bond issuer name.
        currency: currency code (used as GIRR bucket).
        sector: CSR sector (e.g. "sovereign_IG", "corporate_IG", "corporate_HY").
        rating: external rating (for DRC).
        seniority: "senior", "subordinated", "covered_bond", or "equity".
        ir_sensitivity: rate DV01 (GIRR delta sensitivity).
        cs_sensitivity: credit spread DV01 (CSR delta sensitivity).
        notional: face amount (for DRC).
        is_long: direction (True = long, False = short).
    """
    issuer: str
    currency: str = "USD"
    sector: str = "sovereign_IG"
    rating: str = "AAA"
    seniority: str = "senior"
    ir_sensitivity: float = 0.0
    cs_sensitivity: float = 0.0
    notional: float = 0.0
    is_long: bool = True


# ---- Bucket mapping ----

def _girr_rw(tenor_years: float) -> float:
    """Look up the GIRR risk weight for a given tenor."""
    best_tenor = min(GIRR_RISK_WEIGHTS.keys(), key=lambda t: abs(t - tenor_years))
    return GIRR_RISK_WEIGHTS[best_tenor]


def _csr_rw(sector: str) -> float:
    """Look up the CSR risk weight for a sector (first tenor bucket)."""
    rws = CSR_RISK_WEIGHTS.get(sector, CSR_RISK_WEIGHTS.get("corporate_IG", (1.0,)))
    return rws[0]


# ---- Wiring ----

def bond_to_frtb_positions(
    inputs: list[BondRiskInputs],
) -> dict:
    """Convert bond inputs into the dict format for ``calculate_frtb_sa``.

    Produces:
    - GIRR delta positions (one per bond, bucketed by currency).
    - CSR delta positions (one per bond, bucketed by sector).
    - DRC positions (one per issuer, with netting within issuer).
    """
    girr_pos: list[dict] = []
    csr_pos: list[dict] = []
    drc_pos: list[dict] = []

    for inp in inputs:
        # GIRR: rate sensitivity
        if inp.ir_sensitivity:
            girr_pos.append({
                "bucket": inp.currency,
                "sensitivity": inp.ir_sensitivity,
                "risk_weight": 11,  # ~1.1% mid-curve average
            })

        # CSR: credit spread sensitivity
        if inp.cs_sensitivity:
            csr_rw = _csr_rw(inp.sector)
            csr_pos.append({
                "bucket": inp.sector,
                "sensitivity": inp.cs_sensitivity,
                "risk_weight": csr_rw * 100,  # convert fraction to %
            })

        # DRC: default risk
        if inp.notional > 0:
            drc_pos.append({
                "obligor": inp.issuer,
                "notional": inp.notional,
                "rating": inp.rating,
                "seniority": inp.seniority,
                "sector": inp.sector,
                "is_long": inp.is_long,
            })

    delta_positions = {}
    if girr_pos:
        delta_positions["GIRR"] = girr_pos
    if csr_pos:
        delta_positions["CSR"] = csr_pos

    return {
        "delta_positions": delta_positions,
        "vega_positions": {},
        "curvature_positions": {},
        "drc_positions": drc_pos,
        "rrao_positions": [],
    }


# ---- Capital report ----

@dataclass
class BondCapitalReport:
    """One-call FRTB SA capital summary for a bond desk."""
    girr_capital: float
    csr_capital: float
    drc_capital: float
    total_capital: float
    total_rwa: float
    total_notional: float
    capital_efficiency: float
    n_bonds: int

    @property
    def sbm_capital(self) -> float:
        return self.girr_capital + self.csr_capital


def bond_frtb_capital(
    inputs: list[BondRiskInputs],
) -> BondCapitalReport:
    """Run the full FRTB SA bond calculation in one call.

    Returns a :class:`BondCapitalReport` with GIRR + CSR + DRC breakdown.
    """
    positions = bond_to_frtb_positions(inputs)
    result = calculate_frtb_sa(**positions)

    girr = result["sbm_by_risk_class"].get("GIRR", {}).get("total_capital", 0.0)
    csr = result["sbm_by_risk_class"].get("CSR", {}).get("total_capital", 0.0)
    drc = result.get("drc_capital", 0.0)

    total_notional = sum(inp.notional for inp in inputs)
    total = result["total_capital"]
    eff = total / total_notional if total_notional > 0 else 0.0

    return BondCapitalReport(
        girr_capital=girr,
        csr_capital=csr,
        drc_capital=drc,
        total_capital=total,
        total_rwa=result["total_rwa"],
        total_notional=total_notional,
        capital_efficiency=eff,
        n_bonds=len(inputs),
    )
