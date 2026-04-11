"""Credit bond tools: allocation, tracking error, concentration, sector rotation.

Mid-office analytics for an IG/HY credit bond portfolio.

* :func:`sector_allocation` — current portfolio weights by sector.
* :func:`index_tracking_error` — difference between portfolio and benchmark
  weights; zero when the book matches the index exactly.
* :func:`concentration_risk` — Herfindahl-style single-name, sector, and
  rating concentration measures.
* :func:`sector_spread_monitor` — z-score a sector spread vs history.
* :func:`cross_sector_rv` — rank sectors by richness/cheapness.
* :func:`rating_migration_impact` — estimate P&L from a one-notch
  rating transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---- Z-score core ----

def _zscore(current: float, history: list[float], threshold: float = 2.0):
    if not history or len(history) < 2:
        return None, "fair"
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-12:
        return None, "fair"
    z = (current - mean) / std
    if abs(z) >= threshold:
        signal = "wide" if z > 0 else "tight"
    else:
        signal = "fair"
    return z, signal


# ---- Sector allocation ----

@dataclass
class SectorWeight:
    """Portfolio weight of a single sector."""
    sector: str
    market_value: float
    weight: float  # fraction of total


def sector_allocation(
    positions: list[tuple[str, float]],
) -> list[SectorWeight]:
    """Compute portfolio weights by sector.

    Args:
        positions: list of ``(sector, market_value)`` tuples.

    Returns:
        list of :class:`SectorWeight` sorted by sector.
    """
    agg: dict[str, float] = {}
    for sector, mv in positions:
        agg[sector] = agg.get(sector, 0.0) + mv

    total = sum(agg.values())
    if total <= 0:
        return [
            SectorWeight(s, v, 0.0) for s, v in sorted(agg.items())
        ]

    return [
        SectorWeight(s, v, v / total)
        for s, v in sorted(agg.items())
    ]


# ---- Index tracking error ----

@dataclass
class TrackingErrorResult:
    """Tracking error vs a benchmark index."""
    active_weights: dict[str, float]
    tracking_error: float  # sum of |active_weight_i|
    max_overweight: str
    max_underweight: str


def index_tracking_error(
    portfolio_weights: dict[str, float],
    index_weights: dict[str, float],
) -> TrackingErrorResult:
    """Compute the tracking error between portfolio and index weights.

    ``active_weight_i = portfolio_weight_i - index_weight_i``.
    ``tracking_error = Σ |active_weight_i|`` (L1 norm).

    When the portfolio exactly matches the index, tracking error = 0.
    """
    all_sectors = set(portfolio_weights) | set(index_weights)
    active: dict[str, float] = {}
    for s in all_sectors:
        active[s] = portfolio_weights.get(s, 0.0) - index_weights.get(s, 0.0)

    te = sum(abs(v) for v in active.values())

    max_ow = max(active, key=lambda s: active[s]) if active else ""
    max_uw = min(active, key=lambda s: active[s]) if active else ""

    return TrackingErrorResult(
        active_weights=active,
        tracking_error=te,
        max_overweight=max_ow,
        max_underweight=max_uw,
    )


# ---- Concentration risk ----

@dataclass
class ConcentrationResult:
    """Concentration measures for a portfolio."""
    herfindahl_name: float    # Σ w_i² across names
    herfindahl_sector: float  # Σ w_s² across sectors
    herfindahl_rating: float  # Σ w_r² across rating buckets
    top_name_weight: float
    top_sector_weight: float


def concentration_risk(
    name_weights: dict[str, float],
    sector_weights: dict[str, float],
    rating_weights: dict[str, float],
) -> ConcentrationResult:
    """Compute Herfindahl-style concentration across names, sectors, ratings.

    All weights should sum to ~1.0 (fractions of total portfolio).
    HHI = Σ w_i². HHI = 1 means a single-name portfolio; HHI → 0 means
    fully diversified.
    """
    hhi_name = sum(w * w for w in name_weights.values())
    hhi_sector = sum(w * w for w in sector_weights.values())
    hhi_rating = sum(w * w for w in rating_weights.values())

    top_name = max(name_weights.values()) if name_weights else 0.0
    top_sector = max(sector_weights.values()) if sector_weights else 0.0

    return ConcentrationResult(
        herfindahl_name=hhi_name,
        herfindahl_sector=hhi_sector,
        herfindahl_rating=hhi_rating,
        top_name_weight=top_name,
        top_sector_weight=top_sector,
    )


# ---- Sector spread monitor ----

@dataclass
class SectorSpreadSignal:
    """Sector spread level with z-score signal."""
    sector: str
    current_spread_bps: float
    z_score: float | None
    signal: str  # "wide", "tight", "fair"


def sector_spread_monitor(
    sector: str,
    current_spread_bps: float,
    history_bps: list[float],
    threshold: float = 2.0,
) -> SectorSpreadSignal:
    """Z-score a sector spread vs history."""
    z, signal = _zscore(current_spread_bps, history_bps, threshold)
    return SectorSpreadSignal(sector, current_spread_bps, z, signal)


# ---- Cross-sector RV ----

@dataclass
class CrossSectorRV:
    """Ranked sectors by richness/cheapness."""
    sector: str
    spread_bps: float
    z_score: float | None
    rank: int  # 1 = cheapest (most attractive to buy)


def cross_sector_rv(
    sectors: list[tuple[str, float, list[float]]],
    threshold: float = 2.0,
) -> list[CrossSectorRV]:
    """Rank sectors by z-score (cheapest first).

    Args:
        sectors: list of ``(sector_name, current_spread_bps, history_bps)``.

    Returns:
        list of :class:`CrossSectorRV` ranked by z-score descending
        (highest z = widest vs history = cheapest).
    """
    results = []
    for name, spread, history in sectors:
        z, _ = _zscore(spread, history, threshold)
        results.append((name, spread, z))

    # Sort by z descending (widest = cheapest = rank 1)
    results.sort(key=lambda x: -(x[2] if x[2] is not None else -999))

    return [
        CrossSectorRV(name, spread, z, rank=i + 1)
        for i, (name, spread, z) in enumerate(results)
    ]


# ---- Rating migration impact ----

@dataclass
class MigrationImpact:
    """Estimated P&L from a one-notch rating transition."""
    issuer: str
    current_rating: str
    new_rating: str
    spread_change_bps: float
    market_value: float
    duration: float
    estimated_pnl: float


# Standard one-notch spread widening (bps) by current rating.
# Approximate: IG upgrade/downgrade is smaller; HY is larger.
DEFAULT_MIGRATION_SPREADS: dict[str, float] = {
    "AAA": 10, "AA": 15, "A": 25, "BBB": 50,
    "BB": 75, "B": 100, "CCC": 200,
}


def rating_migration_impact(
    issuer: str,
    current_rating: str,
    new_rating: str,
    market_value: float,
    duration: float,
    spread_change_bps: float | None = None,
    direction: str = "downgrade",
) -> MigrationImpact:
    """Estimate the P&L from a one-notch rating transition.

    For a downgrade, spreads widen → price falls → negative P&L.
    For an upgrade, spreads tighten → price rises → positive P&L.

    Args:
        spread_change_bps: explicit spread change. If None, uses the
            default table based on current_rating.
        direction: ``"downgrade"`` or ``"upgrade"``.
    """
    if spread_change_bps is None:
        spread_change_bps = DEFAULT_MIGRATION_SPREADS.get(current_rating, 50)

    sign = -1.0 if direction == "downgrade" else 1.0
    # P&L ≈ sign × duration × spread_change / 10000 × market_value
    # Downgrade: spreads widen → price falls → negative P&L
    pnl = sign * duration * spread_change_bps / 10_000.0 * market_value

    return MigrationImpact(
        issuer=issuer,
        current_rating=current_rating,
        new_rating=new_rating,
        spread_change_bps=spread_change_bps,
        market_value=market_value,
        duration=duration,
        estimated_pnl=pnl,
    )
