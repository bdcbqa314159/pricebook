"""FRTB-IMA desk integration: bridge desk sensitivities to IMA engine.

Maps desk-level risk metrics (delta/gamma/vega/DV01/CS01) into ESRiskFactor
and DRCPosition objects for the FRTB-IMA capital calculation.

    from pricebook.regulatory.ima_bridge import (
        aggregate_desk_ima, extract_risk_factors_from_desk,
    )

References:
    Basel Committee (2019). MAR30-33: Internal Models Approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.regulatory.market_risk_ima import (
    ESRiskFactor, DRCPosition, DeskPLA, FRTBIMAConfig,
    calculate_frtb_ima_capital, evaluate_pla,
)


# ═══════════════════════════════════════════════════════════════
# Desk-to-Risk-Class Mapping
# ═══════════════════════════════════════════════════════════════

RISK_CLASS_MAP: dict[str, tuple[str, str]] = {
    "swap": ("IR", "major"),
    "swaption": ("IR", "major"),
    "bond": ("CR", "IG_corporate"),
    "cds": ("CR", "HY"),
    "equity": ("EQ", "large_cap"),
    "fx": ("FX", "major"),
    "commodity": ("COM", "energy"),
    "repo": ("IR", "major"),
    "inflation": ("IR", "other"),
    "convertible": ("EQ", "large_cap"),
    "trs": ("CR", "IG_corporate"),
    "loan": ("CR", "IG_corporate"),
    "pe": ("EQ", "other"),
}


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class DeskRiskExtract:
    """Extracted risk data from a desk's risk_metrics."""
    desk_id: str
    desk_type: str
    risk_class: str
    sub_category: str
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    dv01: float = 0.0
    cs01: float = 0.0
    notional: float = 0.0
    obligor: str = ""
    rating: str = "BBB"
    is_long: bool = True

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class IMABridgeResult:
    """Result of IMA bridge calculation."""
    ima_capital: dict
    desk_extracts: list[DeskRiskExtract]
    n_risk_factors: int
    n_drc_positions: int
    pla_results: dict | None = None

    def to_dict(self) -> dict:
        return {
            "ima_capital": self.ima_capital,
            "n_risk_factors": self.n_risk_factors,
            "n_drc_positions": self.n_drc_positions,
            "pla_results": self.pla_results,
        }


# ═══════════════════════════════════════════════════════════════
# Extraction Functions
# ═══════════════════════════════════════════════════════════════

def extract_risk_factors_from_desk(
    extract: DeskRiskExtract,
    vol_of_returns: float = 0.01,
    stressed_vol: float | None = None,
) -> list[ESRiskFactor]:
    """Convert desk risk extract into ESRiskFactor list for IMA.

    Maps sensitivity × volatility → 10-day ES contribution.
    ES ≈ sensitivity × vol × z_97.5% × sqrt(10/252)

    Args:
        extract: desk-level risk data.
        vol_of_returns: daily return volatility for the risk factor.
        stressed_vol: stressed-period volatility (if None, uses 1.5× vol).
    """
    z_975 = 1.96
    sqrt_10_252 = math.sqrt(10.0 / 252.0)
    sv = stressed_vol or vol_of_returns * 1.5

    factors = []

    # Delta/DV01 → ES contribution
    sensitivity = abs(extract.dv01) if extract.dv01 != 0 else abs(extract.delta)
    if sensitivity > 0:
        es_10d = sensitivity * vol_of_returns * z_975 * sqrt_10_252
        ses_10d = sensitivity * sv * z_975 * sqrt_10_252
        factors.append(ESRiskFactor(
            risk_class=extract.risk_class,
            sub_category=extract.sub_category,
            es_10day=es_10d,
            is_modellable=True,
            stressed_es_10day=ses_10d,
        ))

    # Vega → separate ES factor
    if extract.vega != 0:
        vol_vol = vol_of_returns * 0.5  # vol of vol ≈ 50% of return vol
        es_vega = abs(extract.vega) * vol_vol * z_975 * sqrt_10_252
        ses_vega = abs(extract.vega) * vol_vol * 1.5 * z_975 * sqrt_10_252
        factors.append(ESRiskFactor(
            risk_class=extract.risk_class,
            sub_category=extract.sub_category + "_vega",
            es_10day=es_vega,
            is_modellable=True,
            stressed_es_10day=ses_vega,
        ))

    # CS01 → credit spread ES (if different from DV01)
    if extract.cs01 != 0 and extract.risk_class in ("CR",):
        spread_vol = vol_of_returns * 2.0  # credit spread more volatile
        es_cs = abs(extract.cs01) * spread_vol * z_975 * sqrt_10_252
        ses_cs = abs(extract.cs01) * spread_vol * 1.5 * z_975 * sqrt_10_252
        factors.append(ESRiskFactor(
            risk_class="CR",
            sub_category=extract.sub_category + "_spread",
            es_10day=es_cs,
            is_modellable=True,
            stressed_es_10day=ses_cs,
        ))

    return factors


def extract_drc_positions_from_desk(
    extract: DeskRiskExtract,
    pd: float = 0.01,
    lgd: float = 0.45,
) -> DRCPosition | None:
    """Convert desk extract into DRCPosition for IMA DRC.

    Only applicable for credit-bearing desks (bond, CDS, loan, TRS).
    """
    if extract.risk_class not in ("CR",) or extract.notional <= 0:
        return None

    return DRCPosition(
        position_id=f"{extract.desk_id}_{extract.obligor or 'pool'}",
        obligor=extract.obligor or extract.desk_id,
        notional=extract.notional,
        market_value=extract.notional,  # simplified: MV ≈ notional
        pd=pd,
        lgd=lgd,
        seniority="senior_unsecured",
        sector="corporate",
        is_long=extract.is_long,
    )


def extract_from_risk_metrics(
    desk_id: str,
    desk_type: str,
    metrics_dict: dict,
) -> DeskRiskExtract:
    """Create DeskRiskExtract from a desk's risk_metrics().to_dict() output.

    Looks for common field names across all desk RiskMetrics dataclasses.
    """
    rc, sc = RISK_CLASS_MAP.get(desk_type, ("CR", "IG_corporate"))

    return DeskRiskExtract(
        desk_id=desk_id,
        desk_type=desk_type,
        risk_class=rc,
        sub_category=sc,
        delta=metrics_dict.get("delta", 0.0),
        gamma=metrics_dict.get("gamma", 0.0),
        vega=metrics_dict.get("vega", 0.0),
        dv01=metrics_dict.get("dv01", metrics_dict.get("dv01_curve", 0.0)),
        cs01=metrics_dict.get("cs01", 0.0),
        notional=metrics_dict.get("notional", 0.0),
        rating=metrics_dict.get("rating", "BBB"),
    )


# ═══════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════

def aggregate_desk_ima(
    desk_extracts: list[DeskRiskExtract],
    desk_pla: list[DeskPLA] | None = None,
    config: FRTBIMAConfig | None = None,
    vol_of_returns: float = 0.01,
) -> IMABridgeResult:
    """Run full IMA pipeline from aggregated desk extracts.

    1. Convert each DeskRiskExtract → ESRiskFactor list
    2. Convert credit desks → DRCPosition list
    3. Run calculate_frtb_ima_capital()
    4. Run PLA evaluation if desk_pla provided
    """
    all_risk_factors = []
    all_drc = []

    for extract in desk_extracts:
        rfs = extract_risk_factors_from_desk(extract, vol_of_returns)
        all_risk_factors.extend(rfs)

        drc = extract_drc_positions_from_desk(extract)
        if drc is not None:
            all_drc.append(drc)

    # Run IMA
    cfg = config or FRTBIMAConfig()
    ima_result = calculate_frtb_ima_capital(
        risk_factors=all_risk_factors,
        drc_positions=all_drc,
        config=cfg,
    )

    # PLA
    pla_result = None
    if desk_pla:
        pla_result = evaluate_pla(desk_pla)

    return IMABridgeResult(
        ima_capital=ima_result,
        desk_extracts=desk_extracts,
        n_risk_factors=len(all_risk_factors),
        n_drc_positions=len(all_drc),
        pla_results=pla_result,
    )
