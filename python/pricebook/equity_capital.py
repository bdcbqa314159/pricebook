"""Equity FRTB SA capital: wire equity positions into the regulatory engine.

Bridges :mod:`pricebook.equity_book` and :mod:`pricebook.regulatory.market_risk_sa`.
Maps tickers to FRTB SA EQ buckets (large/small cap × developed/emerging),
builds the position dicts the engine expects, and produces a one-call
``EquityCapitalReport`` aggregating SbM (delta + vega + curvature), DRC,
and RRAO.

References:
    BCBS d457 (FRTB), MAR21 (EQ risk weights), MAR22 (DRC), MAR23 (RRAO).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pricebook.regulatory.market_risk_sa import (
    EQ_RISK_WEIGHTS,
    calculate_frtb_sa,
)


# ---- Inputs ----

@dataclass
class EquityRiskInputs:
    """Per-name market risk sensitivities for FRTB SA.

    All values are pre-computed by the desk and expressed in the
    reporting currency.

    Attributes:
        ticker: name identifier.
        delta: ∂PV/∂S × spot, expressed as a notional sensitivity (the
            FRTB engine multiplies by the bucket risk weight in %).
        vega: vega per vol point.
        cvr_up / cvr_down: curvature scenarios (already netted with the
            linear delta term — see MAR21).
        notional: equity notional, used for DRC and RRAO.
    """
    ticker: str
    delta: float = 0.0
    vega: float = 0.0
    cvr_up: float = 0.0
    cvr_down: float = 0.0
    notional: float = 0.0


@dataclass
class EquityClassification:
    """Per-name regulatory classification.

    Attributes:
        ticker: name identifier.
        market_cap: USD market capitalisation.
        region: ``"developed"`` or ``"emerging"`` (case-insensitive;
            common ISO/region tags also accepted).
        rating: external rating (used for DRC).
        is_exotic: whether the position carries an exotic payoff (RRAO).
    """
    ticker: str
    market_cap: float
    region: str
    rating: str = "BBB"
    is_exotic: bool = False


# ---- Bucket mapping ----

# BCBS large-cap threshold per MAR21.78: market cap > USD 2bn.
LARGE_CAP_THRESHOLD = 2_000_000_000.0

DEVELOPED_REGIONS = {
    "developed", "developed_market",
    "us", "usa", "north_america",
    "eu", "europe", "uk",
    "japan", "canada", "australia", "switzerland",
}


def map_to_frtb_bucket(market_cap: float, region: str) -> str:
    """Map a name's ``(market_cap, region)`` to a FRTB SA EQ bucket.

    Returns one of ``large_cap_developed``, ``large_cap_emerging``,
    ``small_cap_developed``, ``small_cap_emerging``.
    """
    is_large = market_cap >= LARGE_CAP_THRESHOLD
    is_developed = region.strip().lower() in DEVELOPED_REGIONS

    if is_large and is_developed:
        return "large_cap_developed"
    if is_large and not is_developed:
        return "large_cap_emerging"
    if not is_large and is_developed:
        return "small_cap_developed"
    return "small_cap_emerging"


# ---- Wiring ----

def equity_to_frtb_positions(
    inputs: list[EquityRiskInputs],
    classifications: dict[str, EquityClassification],
) -> dict:
    """Convert equity inputs into the dict format expected by
    :func:`calculate_frtb_sa`.

    Returns a dict with keys ``delta_positions``, ``vega_positions``,
    ``curvature_positions``, ``drc_positions``, ``rrao_positions``,
    suitable for ``calculate_frtb_sa(**result)``.
    """
    delta_eq: list[dict] = []
    vega_eq: list[dict] = []
    curv_eq: list[dict] = []
    drc_pos: list[dict] = []
    rrao_pos: list[dict] = []

    for inp in inputs:
        cls = classifications.get(inp.ticker)
        if cls is None:
            bucket = "other"
            rating = "BBB"
            is_exotic = False
        else:
            bucket = map_to_frtb_bucket(cls.market_cap, cls.region)
            rating = cls.rating
            is_exotic = cls.is_exotic

        rw_delta, rw_vega = EQ_RISK_WEIGHTS.get(bucket, EQ_RISK_WEIGHTS["other"])

        delta_eq.append({
            "bucket": bucket,
            "sensitivity": inp.delta,
            "risk_weight": rw_delta,
        })

        if inp.vega:
            vega_eq.append({
                "bucket": bucket,
                "vega": inp.vega,
                # rw_vega is stored as a fraction; FRTB engine divides by 100,
                # so convert to percent here.
                "vega_risk_weight": rw_vega * 100,
            })

        if inp.cvr_up or inp.cvr_down:
            curv_eq.append({
                "bucket": bucket,
                "cvr_up": inp.cvr_up,
                "cvr_down": inp.cvr_down,
            })

        if inp.notional > 0:
            drc_pos.append({
                "obligor": inp.ticker,
                "notional": inp.notional,
                "rating": rating,
                "seniority": "equity",
                "sector": "equity",
                "is_long": True,
            })
            if is_exotic:
                rrao_pos.append({
                    "notional": inp.notional,
                    "is_exotic": True,
                })

    return {
        "delta_positions": {"EQ": delta_eq},
        "vega_positions": {"EQ": vega_eq},
        "curvature_positions": {"EQ": curv_eq},
        "drc_positions": drc_pos,
        "rrao_positions": rrao_pos,
    }


# ---- Capital report ----

@dataclass
class EquityCapitalReport:
    """One-call FRTB SA capital summary for an equity desk."""
    sbm_capital: float
    delta_capital: float
    vega_capital: float
    curvature_capital: float
    drc_capital: float
    rrao_capital: float
    total_capital: float
    total_rwa: float
    total_notional: float
    capital_efficiency: float          # total_capital / total_notional
    bucket_capitals: dict[str, float]  # delta-leg per-bucket K_b
    n_names: int

    @property
    def sbm_components_sum(self) -> float:
        return self.delta_capital + self.vega_capital + self.curvature_capital


def equity_frtb_capital(
    inputs: list[EquityRiskInputs],
    classifications: dict[str, EquityClassification],
) -> EquityCapitalReport:
    """Run the full FRTB SA equity calculation in one call.

    Returns an :class:`EquityCapitalReport` with SbM (delta/vega/curvature),
    DRC, RRAO, total capital, RWA, total notional, and a capital
    efficiency ratio.
    """
    positions = equity_to_frtb_positions(inputs, classifications)
    result = calculate_frtb_sa(**positions)

    eq_sbm = result["sbm_by_risk_class"].get("EQ", {})
    delta_detail = eq_sbm.get("delta_detail", {}) or {}
    bucket_capitals = delta_detail.get("bucket_capitals", {})

    total_notional = sum(inp.notional for inp in inputs)
    capital_eff = (
        result["total_capital"] / total_notional if total_notional > 0 else 0.0
    )

    return EquityCapitalReport(
        sbm_capital=eq_sbm.get("total_capital", 0.0),
        delta_capital=eq_sbm.get("delta_capital", 0.0),
        vega_capital=eq_sbm.get("vega_capital", 0.0),
        curvature_capital=eq_sbm.get("curvature_capital", 0.0),
        drc_capital=result.get("drc_capital", 0.0),
        rrao_capital=result.get("rrao_capital", 0.0),
        total_capital=result["total_capital"],
        total_rwa=result["total_rwa"],
        total_notional=total_notional,
        capital_efficiency=capital_eff,
        bucket_capitals=dict(bucket_capitals),
        n_names=len(inputs),
    )
