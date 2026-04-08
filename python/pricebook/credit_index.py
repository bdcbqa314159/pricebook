"""Credit index flow trading: index definitions, rolls, and skew analysis.

Manages CDX/iTraxx index series with on-the-run/off-the-run tracking,
roll mechanics, and index-vs-intrinsic basis (skew) analysis.

    from pricebook.credit_index import (
        IndexDefinition, IndexSeries, index_skew, index_roll_pnl,
    )

    idx = IndexDefinition("CDX.NA.IG", constituents, weights)
    skew = index_skew(idx, market_spread, discount_curve, survival_curves)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pricebook.cds import CDS, risky_annuity
from pricebook.cds_index import CDSIndex
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


# ---- Index definition ----

@dataclass
class IndexConstituent:
    """One name in a credit index."""
    name: str
    cds: CDS
    survival_curve: SurvivalCurve
    weight: float = 1.0
    sector: str = ""
    rating: str = ""


class IndexDefinition:
    """A credit index with named, weighted constituents.

    Args:
        index_name: e.g. "CDX.NA.IG", "iTraxx.EUR.Main".
        constituents: list of IndexConstituent.
        series: series number (e.g. 42).
        version: version within series (e.g. 1).
    """

    def __init__(
        self,
        index_name: str,
        constituents: list[IndexConstituent],
        series: int = 1,
        version: int = 1,
    ):
        if not constituents:
            raise ValueError("need at least 1 constituent")
        self.index_name = index_name
        self.constituents = constituents
        self.series = series
        self.version = version
        self._total_weight = sum(c.weight for c in constituents)

    @property
    def n_names(self) -> int:
        return len(self.constituents)

    def weight_normalised(self, idx: int) -> float:
        return self.constituents[idx].weight / self._total_weight

    def intrinsic_spread(self, discount_curve: DiscountCurve) -> float:
        """Notional-weighted average of constituent par spreads."""
        total = 0.0
        for c in self.constituents:
            par = c.cds.par_spread(discount_curve, c.survival_curve)
            total += (c.weight / self._total_weight) * par
        return total

    def index_pv(
        self,
        discount_curve: DiscountCurve,
        notional: float = 1_000_000.0,
    ) -> float:
        """Aggregate PV of the index."""
        total = 0.0
        for c in self.constituents:
            w = c.weight / self._total_weight
            total += w * c.cds.pv(discount_curve, c.survival_curve)
        return total * (notional / c.cds.notional)

    def constituent_spreads(self, discount_curve: DiscountCurve) -> dict[str, float]:
        """Par spread per constituent name."""
        return {
            c.name: c.cds.par_spread(discount_curve, c.survival_curve)
            for c in self.constituents
        }

    def spread_dispersion(self, discount_curve: DiscountCurve) -> float:
        """Standard deviation of constituent par spreads (weighted)."""
        spreads = []
        weights = []
        for c in self.constituents:
            spreads.append(c.cds.par_spread(discount_curve, c.survival_curve))
            weights.append(c.weight / self._total_weight)
        mean = sum(s * w for s, w in zip(spreads, weights))
        var = sum(w * (s - mean) ** 2 for s, w in zip(spreads, weights))
        return math.sqrt(var)


# ---- Index series management ----

@dataclass
class IndexSeries:
    """Tracks on-the-run and off-the-run index series."""
    index_name: str
    series: dict[int, IndexDefinition] = field(default_factory=dict)
    on_the_run: int = 0

    def add_series(self, definition: IndexDefinition, is_current: bool = False) -> None:
        self.series[definition.series] = definition
        if is_current:
            self.on_the_run = definition.series

    def current(self) -> IndexDefinition:
        if self.on_the_run not in self.series:
            raise KeyError(f"No on-the-run series set for {self.index_name}")
        return self.series[self.on_the_run]

    def off_the_run(self) -> list[IndexDefinition]:
        return [d for s, d in sorted(self.series.items()) if s != self.on_the_run]


# ---- Index skew ----

@dataclass
class SkewResult:
    """Index vs intrinsic basis analysis."""
    index_name: str
    market_spread: float
    intrinsic_spread: float
    skew: float
    dispersion: float
    n_names: int


def index_skew(
    index_def: IndexDefinition,
    market_spread: float,
    discount_curve: DiscountCurve,
) -> SkewResult:
    """Compute index vs intrinsic basis (skew).

    Skew = market_spread - intrinsic_spread.
    Positive skew = index trades wide of intrinsic (typical for IG indices).
    """
    intrinsic = index_def.intrinsic_spread(discount_curve)
    dispersion = index_def.spread_dispersion(discount_curve)

    return SkewResult(
        index_name=index_def.index_name,
        market_spread=market_spread,
        intrinsic_spread=intrinsic,
        skew=market_spread - intrinsic,
        dispersion=dispersion,
        n_names=index_def.n_names,
    )


# ---- Index roll ----

@dataclass
class RollResult:
    """Result of an index roll."""
    old_series: int
    new_series: int
    old_intrinsic: float
    new_intrinsic: float
    roll_spread_change: float
    names_added: list[str]
    names_removed: list[str]


def index_roll_pnl(
    old_def: IndexDefinition,
    new_def: IndexDefinition,
    discount_curve: DiscountCurve,
    notional: float = 1_000_000.0,
) -> RollResult:
    """Compute roll P&L and composition changes between two index series.

    Args:
        old_def: off-the-run index definition.
        new_def: on-the-run index definition.
        discount_curve: for computing par spreads.
        notional: index notional.
    """
    old_intrinsic = old_def.intrinsic_spread(discount_curve)
    new_intrinsic = new_def.intrinsic_spread(discount_curve)

    old_names = {c.name for c in old_def.constituents}
    new_names = {c.name for c in new_def.constituents}

    return RollResult(
        old_series=old_def.series,
        new_series=new_def.series,
        old_intrinsic=old_intrinsic,
        new_intrinsic=new_intrinsic,
        roll_spread_change=new_intrinsic - old_intrinsic,
        names_added=sorted(new_names - old_names),
        names_removed=sorted(old_names - new_names),
    )
