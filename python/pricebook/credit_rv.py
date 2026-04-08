"""Credit curve relative value: cross-name, term structure, and sector screening.

Compare CDS spreads across names, analyse curve slope, and screen
sectors for cheapest/richest names.

    from pricebook.credit_rv import (
        cross_name_rv, term_structure_rv, sector_screen,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.cds import CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


# ---- Cross-name relative value ----

@dataclass
class NameRV:
    """Relative value for a single name vs peer group."""
    name: str
    spread: float
    peer_mean: float
    peer_std: float
    z_score: float
    percentile: float
    signal: str  # "rich", "cheap", "fair"


def cross_name_rv(
    names: dict[str, tuple[CDS, SurvivalCurve]],
    discount_curve: DiscountCurve,
    threshold: float = 2.0,
) -> list[NameRV]:
    """Compare CDS spreads across names in a peer group.

    Args:
        names: mapping of name -> (CDS, SurvivalCurve).
        discount_curve: risk-free curve.
        threshold: z-score threshold for rich/cheap signal.

    Returns:
        List of NameRV, one per name, sorted by z-score.
    """
    spreads: dict[str, float] = {}
    for name, (cds, sc) in names.items():
        spreads[name] = cds.par_spread(discount_curve, sc)

    vals = list(spreads.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var) if var > 0 else 0.0

    sorted_vals = sorted(vals)
    results = []
    for name, s in spreads.items():
        z = (s - mean) / std if std > 1e-12 else 0.0
        rank = sum(1 for v in sorted_vals if v <= s)
        pct = rank / len(sorted_vals) * 100.0
        if abs(z) >= threshold:
            signal = "cheap" if z > 0 else "rich"
        else:
            signal = "fair"
        results.append(NameRV(name, s, mean, std, z, pct, signal))

    return sorted(results, key=lambda r: r.z_score)


# ---- Term structure relative value ----

@dataclass
class TermStructureRV:
    """CDS curve slope analysis for one name."""
    name: str
    short_spread: float
    long_spread: float
    slope: float
    short_tenor: int
    long_tenor: int
    z_score: float | None
    signal: str  # "steep", "flat", "fair"


def term_structure_rv(
    name: str,
    short_cds: CDS,
    long_cds: CDS,
    short_sc: SurvivalCurve,
    long_sc: SurvivalCurve,
    discount_curve: DiscountCurve,
    short_tenor: int,
    long_tenor: int,
    history: list[float] | None = None,
    threshold: float = 2.0,
) -> TermStructureRV:
    """CDS curve slope: long_spread - short_spread.

    Args:
        short_cds/long_cds: CDS at short/long tenors.
        short_sc/long_sc: survival curves (can be same curve).
        short_tenor/long_tenor: tenor labels in years.
        history: historical slope values for z-score.
    """
    short_spread = short_cds.par_spread(discount_curve, short_sc)
    long_spread = long_cds.par_spread(discount_curve, long_sc)
    slope = long_spread - short_spread

    z_score = None
    signal = "fair"
    if history and len(history) >= 2:
        mean = sum(history) / len(history)
        var = sum((h - mean) ** 2 for h in history) / len(history)
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 1e-12:
            z_score = (slope - mean) / std
            if z_score > threshold:
                signal = "steep"
            elif z_score < -threshold:
                signal = "flat"

    return TermStructureRV(
        name=name,
        short_spread=short_spread,
        long_spread=long_spread,
        slope=slope,
        short_tenor=short_tenor,
        long_tenor=long_tenor,
        z_score=z_score,
        signal=signal,
    )


# ---- Sector screening ----

@dataclass
class SectorStats:
    """Aggregate statistics for a sector."""
    sector: str
    n_names: int
    mean_spread: float
    dispersion: float
    cheapest: str
    richest: str
    cheapest_spread: float
    richest_spread: float


def sector_screen(
    names: dict[str, tuple[CDS, SurvivalCurve, str]],
    discount_curve: DiscountCurve,
) -> list[SectorStats]:
    """Screen sectors: mean spread, dispersion, cheapest/richest per sector.

    Args:
        names: name -> (CDS, SurvivalCurve, sector).
        discount_curve: risk-free curve.

    Returns:
        List of SectorStats sorted by mean spread (widest first).
    """
    sectors: dict[str, list[tuple[str, float]]] = {}
    for name, (cds, sc, sector) in names.items():
        s = cds.par_spread(discount_curve, sc)
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append((name, s))

    results = []
    for sector, entries in sectors.items():
        spreads = [s for _, s in entries]
        n = len(spreads)
        mean = sum(spreads) / n
        var = sum((s - mean) ** 2 for s in spreads) / n
        disp = math.sqrt(var) if var > 0 else 0.0

        widest = max(entries, key=lambda e: e[1])
        tightest = min(entries, key=lambda e: e[1])

        results.append(SectorStats(
            sector=sector,
            n_names=n,
            mean_spread=mean,
            dispersion=disp,
            cheapest=widest[0],
            richest=tightest[0],
            cheapest_spread=widest[1],
            richest_spread=tightest[1],
        ))

    return sorted(results, key=lambda r: -r.mean_spread)
