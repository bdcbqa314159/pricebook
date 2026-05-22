"""Curve compression and storage.

EOD curve snapshots, sparse deltas, versioning.

    from pricebook.curves.curve_storage import (
        CurveSnapshot, CurveDelta, CurveStore,
        compress_curve, decompress_curve,
    )
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class CurveSnapshot:
    """A timestamped curve snapshot."""
    curve_id: str
    timestamp: str               # ISO format
    reference_date: date
    pillar_dates: list[date]
    zero_rates: list[float]      # continuously compounded
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "curve_id": self.curve_id,
            "timestamp": self.timestamp,
            "reference_date": self.reference_date.isoformat(),
            "pillar_dates": [d.isoformat() for d in self.pillar_dates],
            "zero_rates": self.zero_rates,
            "metadata": self.metadata,
        }

    @classmethod
    def from_curve(cls, curve: DiscountCurve, curve_id: str,
                   timestamp: str | None = None) -> CurveSnapshot:
        """Create snapshot from a DiscountCurve."""
        ref = curve.reference_date
        dc = DayCountConvention.ACT_365_FIXED
        zeros = []
        for d in curve.pillar_dates:
            t = year_fraction(ref, d, dc)
            if t > 0:
                zeros.append(-math.log(max(curve.df(d), 1e-15)) / t)
            else:
                zeros.append(0.0)
        return cls(
            curve_id=curve_id,
            timestamp=timestamp or datetime.now().isoformat(),
            reference_date=ref,
            pillar_dates=list(curve.pillar_dates),
            zero_rates=zeros,
        )

    def to_curve(self) -> DiscountCurve:
        """Reconstruct DiscountCurve from snapshot."""
        dc = DayCountConvention.ACT_365_FIXED
        dfs = []
        for d, z in zip(self.pillar_dates, self.zero_rates):
            t = year_fraction(self.reference_date, d, dc)
            dfs.append(math.exp(-z * t))
        return DiscountCurve(self.reference_date, self.pillar_dates, dfs)


@dataclass
class CurveDelta:
    """Sparse delta between two curve snapshots."""
    reference_id: str
    target_id: str
    pillar_shifts: list[float]   # zero rate shifts at each pillar (bp)

    def to_dict(self) -> dict:
        return {
            "reference_id": self.reference_id,
            "target_id": self.target_id,
            "pillar_shifts": self.pillar_shifts,
            "max_shift_bp": max(abs(s) for s in self.pillar_shifts) if self.pillar_shifts else 0,
        }


def compress_curve(
    target: CurveSnapshot,
    reference: CurveSnapshot,
) -> CurveDelta:
    """Compute the delta between two snapshots."""
    if len(target.zero_rates) != len(reference.zero_rates):
        raise ValueError("Snapshots must have same number of pillars for delta")
    shifts = [(t - r) * 10_000 for t, r in zip(target.zero_rates, reference.zero_rates)]
    return CurveDelta(reference.curve_id, target.curve_id, shifts)


def decompress_curve(
    delta: CurveDelta,
    reference: CurveSnapshot,
) -> CurveSnapshot:
    """Reconstruct a snapshot from a reference + delta."""
    zeros = [r + s / 10_000 for r, s in zip(reference.zero_rates, delta.pillar_shifts)]
    return CurveSnapshot(
        curve_id=delta.target_id,
        timestamp="",
        reference_date=reference.reference_date,
        pillar_dates=list(reference.pillar_dates),
        zero_rates=zeros,
    )


class CurveStore:
    """In-memory curve store with snapshot and delta support."""

    def __init__(self):
        self._snapshots: dict[str, CurveSnapshot] = {}
        self._history: dict[str, list[str]] = {}  # curve_id → [snapshot_ids]

    def save(self, snapshot: CurveSnapshot) -> str:
        """Save a snapshot. Returns snapshot ID."""
        snap_id = f"{snapshot.curve_id}_{snapshot.timestamp}"
        self._snapshots[snap_id] = snapshot
        self._history.setdefault(snapshot.curve_id, []).append(snap_id)
        return snap_id

    def load(self, snapshot_id: str) -> CurveSnapshot:
        """Load a snapshot by ID."""
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot {snapshot_id} not found")
        return self._snapshots[snapshot_id]

    def history(self, curve_id: str) -> list[str]:
        """Return snapshot IDs for a curve, in chronological order."""
        return self._history.get(curve_id, [])

    def diff(self, id1: str, id2: str) -> CurveDelta:
        """Compute delta between two snapshots."""
        s1 = self.load(id1)
        s2 = self.load(id2)
        return compress_curve(s2, s1)

    def n_snapshots(self) -> int:
        return len(self._snapshots)
