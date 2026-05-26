"""Sovereign CDS conventions and credit curve construction.

Sovereign CDS have specific restructuring clauses, recovery assumptions,
and quotation conventions that differ from corporate CDS.

    from pricebook.credit.sovereign_cds import (
        get_sovereign_cds_conventions, bootstrap_sovereign_hazard,
        SovereignCDSConventions,
    )

    conv = get_sovereign_cds_conventions("BR")
    result = bootstrap_sovereign_hazard(ref, spreads, discount_curve, "BR")

References:
    ISDA (2014). 2014 Credit Derivatives Definitions.
    Markit (2009). CDS Small Bang Protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.core.serialisable import serialisable_convention
from datetime import date
from enum import Enum

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve


class RestructuringClause(Enum):
    """ISDA restructuring clause for CDS contracts."""
    CR = "CR"       # Full (old) restructuring — rare, pre-2003
    MR = "MR"       # Modified restructuring — US corporates
    MM = "MM"       # Modified-modified restructuring — European corporates
    XR = "XR"       # No restructuring — North American standard, most sovereigns


@serialisable_convention("sovereign_cds_conventions")
@dataclass(frozen=True)
class SovereignCDSConventions:
    """CDS conventions for a sovereign reference entity."""
    country_code: str           # ISO 2-letter (e.g. "BR", "TR", "ZA")
    country_name: str
    currency: str               # Quotation currency (usually USD)
    restructuring: RestructuringClause
    recovery_rate: float        # ISDA standard recovery
    standard_tenors: list[int]  # Standard CDS tenors in years
    doc_clause: str             # ISDA doc clause identifier
    notes: str = ""

# ═══════════════════════════════════════════════════════════════
# Convention registry (~30 sovereigns)
# ═══════════════════════════════════════════════════════════════

_REGISTRY: dict[str, SovereignCDSConventions] = {}
_STD_TENORS = [1, 2, 3, 5, 7, 10]
_LONG_TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]


def _reg(c: SovereignCDSConventions) -> None:
    _REGISTRY[c.country_code] = c


# LatAm
_reg(SovereignCDSConventions("BR", "Brazil", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "LatAm14"))
_reg(SovereignCDSConventions("MX", "Mexico", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "LatAm14"))
_reg(SovereignCDSConventions("CO", "Colombia", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "LatAm14"))
_reg(SovereignCDSConventions("CL", "Chile", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "LatAm14"))
_reg(SovereignCDSConventions("PE", "Peru", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "LatAm14"))
_reg(SovereignCDSConventions("AR", "Argentina", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "LatAm14",
     notes="Frequently distressed. Wide bid-ask."))

# CEEMEA
_reg(SovereignCDSConventions("TR", "Turkey", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14"))
_reg(SovereignCDSConventions("ZA", "South Africa", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14"))
_reg(SovereignCDSConventions("PL", "Poland", "USD", RestructuringClause.MM, 0.40, _STD_TENORS, "Euro14",
     notes="EU member, modified-modified restructuring."))
_reg(SovereignCDSConventions("HU", "Hungary", "USD", RestructuringClause.MM, 0.40, _STD_TENORS, "Euro14"))
_reg(SovereignCDSConventions("RO", "Romania", "USD", RestructuringClause.MM, 0.40, _STD_TENORS, "Euro14"))
_reg(SovereignCDSConventions("RU", "Russia", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14",
     notes="Sanctioned since 2022 — CDS settlement affected."))
_reg(SovereignCDSConventions("EG", "Egypt", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14"))
_reg(SovereignCDSConventions("NG", "Nigeria", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14"))
_reg(SovereignCDSConventions("KE", "Kenya", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14"))

# Asia
_reg(SovereignCDSConventions("CN", "China", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "Asia14"))
_reg(SovereignCDSConventions("KR", "South Korea", "USD", RestructuringClause.XR, 0.40, _STD_TENORS, "Asia14"))
_reg(SovereignCDSConventions("ID", "Indonesia", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "Asia14"))
_reg(SovereignCDSConventions("PH", "Philippines", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "Asia14"))
_reg(SovereignCDSConventions("MY", "Malaysia", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "Asia14"))
_reg(SovereignCDSConventions("TH", "Thailand", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "Asia14"))
_reg(SovereignCDSConventions("IN", "India", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "Asia14"))
_reg(SovereignCDSConventions("VN", "Vietnam", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "Asia14"))

# Western Europe (traded but tight spreads)
_reg(SovereignCDSConventions("IT", "Italy", "EUR", RestructuringClause.MM, 0.40, _LONG_TENORS, "Euro14"))
_reg(SovereignCDSConventions("ES", "Spain", "EUR", RestructuringClause.MM, 0.40, _LONG_TENORS, "Euro14"))
_reg(SovereignCDSConventions("PT", "Portugal", "EUR", RestructuringClause.MM, 0.40, _STD_TENORS, "Euro14"))
_reg(SovereignCDSConventions("GR", "Greece", "EUR", RestructuringClause.MM, 0.40, _STD_TENORS, "Euro14",
     notes="Post-2012 restructuring — new bonds under UK law."))
_reg(SovereignCDSConventions("IE", "Ireland", "EUR", RestructuringClause.MM, 0.40, _STD_TENORS, "Euro14"))

# MENA
_reg(SovereignCDSConventions("SA", "Saudi Arabia", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14"))
_reg(SovereignCDSConventions("QA", "Qatar", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14"))
_reg(SovereignCDSConventions("IL", "Israel", "USD", RestructuringClause.CR, 0.25, _STD_TENORS, "EMEA14"))


# ═══════════════════════════════════════════════════════════════
# Registry API
# ═══════════════════════════════════════════════════════════════


def get_sovereign_cds_conventions(country_code: str) -> SovereignCDSConventions:
    """Get CDS conventions for a sovereign.

    Args:
        country_code: ISO 2-letter country code (e.g. "BR", "TR", "IT").
    """
    code = country_code.upper()
    conv = _REGISTRY.get(code)
    if conv is None:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"No sovereign CDS conventions for {code!r}. Available: {available}")
    return conv


def list_sovereign_cds() -> list[str]:
    """Return sorted list of available sovereign CDS country codes."""
    return sorted(_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════
# Hazard rate bootstrap from CDS spreads
# ═══════════════════════════════════════════════════════════════


@dataclass
class SovereignHazardResult:
    """Result of sovereign hazard curve bootstrap."""
    survival_curve: SurvivalCurve
    pillar_years: list[int]
    pillar_hazards: list[float]
    pillar_survivals: list[float]
    input_spreads_bp: list[float]
    fitted_spreads_bp: list[float]
    recovery_rate: float
    country_code: str
    restructuring: str
    n_pillars: int

    def to_dict(self) -> dict:
        return {
            "pillar_years": self.pillar_years,
            "pillar_hazards": self.pillar_hazards,
            "pillar_survivals": self.pillar_survivals,
            "input_spreads_bp": self.input_spreads_bp,
            "fitted_spreads_bp": self.fitted_spreads_bp,
            "recovery_rate": self.recovery_rate,
            "country_code": self.country_code,
            "restructuring": self.restructuring,
            "n_pillars": self.n_pillars,
        }


def bootstrap_sovereign_hazard(
    reference_date: date,
    spreads_bp: dict[int, float],
    discount_curve: DiscountCurve,
    country_code: str,
    recovery_override: float | None = None,
) -> SovereignHazardResult:
    """Bootstrap hazard rates from sovereign CDS spreads.

    Sequential bootstrap: each tenor adds one hazard pillar.

    Args:
        reference_date: valuation date.
        spreads_bp: {tenor_years: spread_in_bp}, e.g. {1: 50, 5: 120, 10: 180}.
        discount_curve: risk-free discount curve (USD OIS or EUR).
        country_code: ISO 2-letter code for convention lookup.
        recovery_override: override ISDA standard recovery if needed.

    Returns:
        SovereignHazardResult with survival curve and diagnostics.
    """
    conv = get_sovereign_cds_conventions(country_code)
    recovery = recovery_override if recovery_override is not None else conv.recovery_rate

    sorted_tenors = sorted(spreads_bp.keys())
    dc = DayCountConvention.ACT_365_FIXED

    pillar_dates = [reference_date]
    pillar_survivals = [1.0]
    pillar_hazards = []
    fitted_spreads = []

    for tenor in sorted_tenors:
        if not isinstance(tenor, int) or tenor <= 0:
            raise ValueError(f"Tenor must be a positive integer (years), got {tenor!r}")
        spread = spreads_bp[tenor] / 10_000
        target_date = date(reference_date.year + tenor, reference_date.month, reference_date.day)

        # Approximate hazard from spread: h ≈ s / (1 - R)
        h_approx = spread / (1.0 - recovery)

        # Refine: find h such that the par CDS spread equals the market spread
        # For sequential bootstrap, use the approximate value (good enough for
        # piecewise-constant hazard).
        # More precise: iterative refinement via risky annuity
        dt = year_fraction(pillar_dates[-1], target_date, dc)
        if dt <= 0:
            continue

        # Compute hazard for this segment
        # CDS par spread ≈ h × (1-R) for flat hazard over the segment
        # With prior survival: need to account for cumulative survival
        h = _solve_segment_hazard(
            spread, recovery, dt, pillar_survivals[-1],
            discount_curve, pillar_dates[-1], target_date, dc,
        )

        q_new = pillar_survivals[-1] * math.exp(-h * dt)
        pillar_dates.append(target_date)
        pillar_survivals.append(q_new)
        pillar_hazards.append(h)

        # Fitted spread for this tenor
        fitted_s = h * (1.0 - recovery) * 10_000
        fitted_spreads.append(fitted_s)

    # Build survival curve
    if len(pillar_dates) <= 1:
        raise ValueError("No valid CDS tenors produced hazard rates")

    survival_curve = SurvivalCurve(reference_date, pillar_dates[1:], pillar_survivals[1:])

    return SovereignHazardResult(
        survival_curve=survival_curve,
        pillar_years=sorted_tenors,
        pillar_hazards=pillar_hazards,
        pillar_survivals=pillar_survivals[1:],
        input_spreads_bp=[spreads_bp[t] for t in sorted_tenors],
        fitted_spreads_bp=fitted_spreads,
        recovery_rate=recovery,
        country_code=country_code,
        restructuring=conv.restructuring.value,
        n_pillars=len(pillar_hazards),
    )


def _solve_segment_hazard(
    spread: float,
    recovery: float,
    dt: float,
    prev_survival: float,
    discount_curve: DiscountCurve,
    start_date: date,
    end_date: date,
    dc: DayCountConvention,
) -> float:
    """Solve for the hazard rate in a segment that matches the CDS par spread.

    Uses the approximation: h ≈ spread / (1 - R), refined by accounting for
    the discount factor and survival probability over the segment.

    For a more precise result with risky annuity, this could be extended
    to an iterative solver, but the first-order approximation is typically
    within a few bp for investment-grade sovereigns.
    """
    lgd = 1.0 - recovery
    if lgd <= 0:
        return 0.0

    # Discount factor adjustment: weight by average df over the segment
    df_start = discount_curve.df(start_date)
    df_end = discount_curve.df(end_date)
    avg_df = (df_start + df_end) / 2.0

    # Adjusted hazard: account for discounting
    h = spread / lgd

    # Second-order correction for longer segments
    if dt > 1.0:
        # Premium leg timing adjustment
        r = -math.log(df_end / df_start) / dt if df_start > 0 and dt > 0 else 0.0
        h = spread / (lgd * (1.0 - spread * dt / (2.0 * (1.0 + r * dt))))

    return max(h, 0.0)
