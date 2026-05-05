"""Cross-asset trading desk: aggregate risk across all 12 desks.

Provides a single entry point for portfolio-level risk, stress,
and capital by calling each desk's uniform protocol methods.

    from pricebook.cross_asset_desk import (
        CrossAssetDesk, CrossAssetDashboard,
    )

    desk = CrossAssetDesk()
    desk.add("trs", trs_book, trs_book.aggregate_risk)
    desk.add("cds", cds_book, cds_book.aggregate_risk)
    dashboard = desk.dashboard(date, curve)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.discount_curve import DiscountCurve


@dataclass
class DeskRiskSummary:
    """Risk summary for one desk."""
    name: str
    total_pv: float
    total_dv01: float
    n_positions: int
    total_notional: float

    def to_dict(self) -> dict:
        return {
            "name": self.name, "pv": self.total_pv, "dv01": self.total_dv01,
            "n": self.n_positions, "notional": self.total_notional,
        }


@dataclass
class CrossAssetDashboard:
    """Portfolio-level morning summary across all desks."""
    date: date
    n_desks: int
    total_pv: float
    total_dv01: float
    total_notional: float
    n_positions: int
    by_desk: list[DeskRiskSummary]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "n_desks": self.n_desks,
            "total_pv": self.total_pv,
            "total_dv01": self.total_dv01,
            "total_notional": self.total_notional,
            "n_positions": self.n_positions,
            "by_desk": [d.to_dict() for d in self.by_desk],
        }


class CrossAssetDesk:
    """Aggregates risk across multiple trading desks.

    Each desk is registered with a name and an aggregate_risk callable
    that returns a dict with at least: total_pv, total_notional, n_positions.
    Optionally total_dv01 for rate-sensitive desks.
    """

    def __init__(self):
        self._desks: dict[str, dict] = {}  # name → {book, risk_fn}

    def add(self, name: str, book, risk_fn=None) -> None:
        """Register a desk.

        Args:
            name: desk identifier (e.g. "trs", "cds", "fx").
            book: the desk's book object.
            risk_fn: callable(curve) → dict with risk metrics.
                If None, calls book.aggregate_risk.
        """
        if risk_fn is None:
            risk_fn = getattr(book, 'aggregate_risk', None)
        self._desks[name] = {"book": book, "risk_fn": risk_fn}

    @property
    def desk_names(self) -> list[str]:
        return list(self._desks.keys())

    def n_desks(self) -> int:
        return len(self._desks)

    def aggregate(self, curve: DiscountCurve) -> list[DeskRiskSummary]:
        """Compute risk summary per desk."""
        summaries = []
        for name, desk in self._desks.items():
            risk_fn = desk["risk_fn"]
            if risk_fn is None:
                continue
            try:
                risk = risk_fn(curve)
            except TypeError:
                try:
                    risk = risk_fn()
                except TypeError:
                    import warnings
                    warnings.warn(f"Desk '{name}' risk_fn failed — skipping", stacklevel=2)
                    risk = {}

            # Aggregate DV01: prefer total_dv01, fall back to total_cs01 for credit desks
            dv01 = risk.get("total_dv01", 0.0) or risk.get("total_cs01", 0.0)
            summaries.append(DeskRiskSummary(
                name=name,
                total_pv=risk.get("total_pv", 0.0),
                total_dv01=dv01,
                n_positions=risk.get("n_positions", 0),
                total_notional=risk.get("total_notional", 0.0),
            ))
        return summaries

    def dashboard(self, reference_date: date, curve: DiscountCurve) -> CrossAssetDashboard:
        """Build cross-asset morning dashboard."""
        summaries = self.aggregate(curve)
        return CrossAssetDashboard(
            date=reference_date,
            n_desks=len(summaries),
            total_pv=sum(s.total_pv for s in summaries),
            total_dv01=sum(s.total_dv01 for s in summaries),
            total_notional=sum(s.total_notional for s in summaries),
            n_positions=sum(s.n_positions for s in summaries),
            by_desk=summaries,
        )

    def total_pv(self, curve: DiscountCurve) -> float:
        return sum(s.total_pv for s in self.aggregate(curve))

    def total_dv01(self, curve: DiscountCurve) -> float:
        return sum(s.total_dv01 for s in self.aggregate(curve))
