"""CDS index roll mechanics: series transition, OTR basis, roll P&L.

When a CDS index rolls from one series to the next (typically every
6 months for CDX/iTraxx), names may be added or removed. This module
computes the spread impact, roll P&L, and on-the-run basis.

* :class:`IndexRollResult` — roll transition result.
* :func:`index_roll_pnl` — P&L from rolling to new series.
* :func:`on_the_run_basis` — OTR vs off-the-run spread difference.
* :func:`series_transition` — apply name additions/removals.

References:
    Markit, *iTraxx and CDX Index Roll Mechanics*, 2007.
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 8, 2008.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve


@dataclass
class IndexRollResult:
    """Result of an index roll transition."""
    old_series: int
    new_series: int
    old_spread: float
    new_spread: float
    spread_change_bp: float
    names_added: list[str]
    names_removed: list[str]
    roll_pnl: float
    notional: float

    def to_dict(self) -> dict:
        return {
            "old_series": self.old_series,
            "new_series": self.new_series,
            "old_spread_bp": self.old_spread * 10_000,
            "new_spread_bp": self.new_spread * 10_000,
            "spread_change_bp": self.spread_change_bp,
            "n_added": len(self.names_added),
            "n_removed": len(self.names_removed),
            "roll_pnl": self.roll_pnl,
        }


@dataclass
class Constituent:
    """Index constituent with name and spread."""
    name: str
    spread: float  # par spread (decimal)
    weight: float = 1.0
    defaulted: bool = False


def series_transition(
    old_constituents: list[Constituent],
    additions: list[Constituent],
    removals: list[str],
) -> list[Constituent]:
    """Apply name additions/removals for a new series.

    Names are removed (defaulted, rating downgrade, merger) and
    new names added to maintain the target count.

    Args:
        old_constituents: current series constituents.
        additions: new names to add.
        removals: names to remove (by name string).

    Returns:
        New constituent list for the new series.
    """
    removal_set = set(removals)
    new_list = [c for c in old_constituents if c.name not in removal_set]
    new_list.extend(additions)
    return new_list


def _weighted_spread(constituents: list[Constituent]) -> float:
    """Compute weight-average spread of constituents."""
    active = [c for c in constituents if not c.defaulted]
    if not active:
        return 0.0
    total_w = sum(c.weight for c in active)
    if total_w <= 0:
        return 0.0
    return sum(c.spread * c.weight for c in active) / total_w


def index_roll_pnl(
    old_constituents: list[Constituent],
    new_constituents: list[Constituent],
    old_series: int,
    new_series: int,
    rpv01: float,
    notional: float = 10_000_000.0,
) -> IndexRollResult:
    """Compute P&L from rolling to a new index series.

    Roll P&L ≈ (old_spread − new_spread) × RPV01 × notional

    A protection buyer benefits if the new series has a lower spread
    (pays less premium going forward).

    Args:
        old_constituents: outgoing series constituents.
        new_constituents: incoming series constituents.
        old_series: outgoing series number.
        new_series: incoming series number.
        rpv01: risky annuity (PV01 of 1bp premium).
        notional: index notional.
    """
    old_spread = _weighted_spread(old_constituents)
    new_spread = _weighted_spread(new_constituents)
    change_bp = (new_spread - old_spread) * 10_000

    # Roll P&L for protection buyer: pays less if new spread is lower
    roll_pnl = (old_spread - new_spread) * rpv01 * notional

    old_names = {c.name for c in old_constituents}
    new_names = {c.name for c in new_constituents}
    added = [c.name for c in new_constituents if c.name not in old_names]
    removed = [n for n in old_names if n not in new_names]

    return IndexRollResult(
        old_series=old_series,
        new_series=new_series,
        old_spread=old_spread,
        new_spread=new_spread,
        spread_change_bp=change_bp,
        names_added=added,
        names_removed=removed,
        roll_pnl=roll_pnl,
        notional=notional,
    )


@dataclass
class OTRBasisResult:
    """On-the-run basis result."""
    otr_spread: float
    off_the_run_spread: float
    basis_bp: float
    otr_series: int
    off_run_series: int

    def to_dict(self) -> dict:
        return {
            "otr_spread_bp": self.otr_spread * 10_000,
            "off_the_run_spread_bp": self.off_the_run_spread * 10_000,
            "basis_bp": self.basis_bp,
            "otr_series": self.otr_series,
            "off_run_series": self.off_run_series,
        }


def on_the_run_basis(
    otr_constituents: list[Constituent],
    off_run_constituents: list[Constituent],
    otr_series: int,
    off_run_series: int,
) -> OTRBasisResult:
    """On-the-run vs off-the-run spread basis.

    The OTR index typically trades tighter due to liquidity premium.
    Basis = OTR − OTR-1 (usually negative).

    Args:
        otr_constituents: on-the-run (current) series.
        off_run_constituents: off-the-run (previous) series.
        otr_series: current series number.
        off_run_series: previous series number.
    """
    otr_spread = _weighted_spread(otr_constituents)
    off_spread = _weighted_spread(off_run_constituents)
    basis = (otr_spread - off_spread) * 10_000

    return OTRBasisResult(
        otr_spread=otr_spread,
        off_the_run_spread=off_spread,
        basis_bp=basis,
        otr_series=otr_series,
        off_run_series=off_run_series,
    )


def series_transition_pnl(
    old_constituents: list[Constituent],
    additions: list[Constituent],
    removals: list[str],
    old_series: int,
    rpv01: float,
    notional: float = 10_000_000.0,
) -> IndexRollResult:
    """Convenience: apply transition and compute roll P&L in one step.

    Args:
        old_constituents: outgoing series.
        additions: names being added.
        removals: names being removed.
        old_series: outgoing series number.
        rpv01: risky annuity.
        notional: index notional.
    """
    new = series_transition(old_constituents, additions, removals)
    return index_roll_pnl(
        old_constituents, new, old_series, old_series + 1,
        rpv01, notional,
    )
