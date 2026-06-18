"""Cross-asset volatility desk: unified vega/vanna/volga aggregation.

Aggregates vol risk across FX, equity, IR, commodity, and credit.
Provides vega ladder, cross-asset stress, and arbitrage monitoring.

    from pricebook.desks.vol_desk import (
        VolBook, VolPosition, vol_dashboard, VolDashboard,
        vol_stress_report, vol_correlation_monitor,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

@dataclass
class VolPosition:
    """A single vol exposure."""
    asset_class: str          # "fx", "equity", "ir", "commodity", "credit"
    instrument_id: str
    underlying: str           # "EURUSD", "SPX", "USD_5Y", "WTI", etc.
    expiry: date
    vega: float               # dPV/d(1% vol)
    vanna: float = 0.0        # d²PV/dS·dσ
    volga: float = 0.0        # d²PV/dσ²
    notional: float = 0.0
    implied_vol: float = 0.0
    realised_vol: float = 0.0



    def to_dict(self) -> dict:
        return dict(vars(self))
class VolBook:
    """Cross-asset vol book."""

    def __init__(self, name: str = "vol_book"):
        self.name = name
        self._positions: list[VolPosition] = []

    def add(self, position: VolPosition) -> None:
        self._positions.append(position)

    @property
    def positions(self) -> list[VolPosition]:
        return list(self._positions)

    def __len__(self) -> int:
        return len(self._positions)

    def total_vega(self) -> float:
        return sum(p.vega for p in self._positions)

    def total_vanna(self) -> float:
        return sum(p.vanna for p in self._positions)

    def total_volga(self) -> float:
        return sum(p.volga for p in self._positions)

    def by_asset_class(self) -> dict[str, list[VolPosition]]:
        result: dict[str, list[VolPosition]] = {}
        for p in self._positions:
            result.setdefault(p.asset_class, []).append(p)
        return result

    def by_underlying(self) -> dict[str, list[VolPosition]]:
        result: dict[str, list[VolPosition]] = {}
        for p in self._positions:
            result.setdefault(p.underlying, []).append(p)
        return result

    def vega_ladder(self, ref: date) -> dict[str, float]:
        """Vega aggregated by expiry bucket."""
        buckets: dict[str, float] = {}
        for p in self._positions:
            T = year_fraction(ref, p.expiry, DayCountConvention.ACT_365_FIXED)
            if T <= 0.25:
                label = "0-3M"
            elif T <= 0.5:
                label = "3-6M"
            elif T <= 1.0:
                label = "6-12M"
            elif T <= 2.0:
                label = "1-2Y"
            elif T <= 5.0:
                label = "2-5Y"
            else:
                label = "5Y+"
            buckets[label] = buckets.get(label, 0.0) + p.vega
        return buckets

    def vega_by_asset_class(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for ac, positions in self.by_asset_class().items():
            result[ac] = sum(p.vega for p in positions)
        return result

    def vol_premium(self) -> dict[str, float]:
        """Implied - realised vol premium per underlying (average if multiple)."""
        sums: dict[str, list[float]] = {}
        for p in self._positions:
            if p.implied_vol > 0 and p.realised_vol > 0:
                sums.setdefault(p.underlying, []).append(p.implied_vol - p.realised_vol)
        return {k: sum(v) / len(v) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class VolDashboard:
    """Cross-asset vol desk morning summary."""
    date: date
    n_positions: int
    total_vega: float
    total_vanna: float
    total_volga: float
    vega_by_asset_class: dict[str, float]
    vega_ladder: dict[str, float]
    vol_premium: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "n": self.n_positions,
            "total_vega": self.total_vega,
            "total_vanna": self.total_vanna,
            "total_volga": self.total_volga,
            "vega_by_asset": self.vega_by_asset_class,
            "vega_ladder": self.vega_ladder,
            "vol_premium": self.vol_premium,
        }


def vol_dashboard(book: VolBook, reference_date: date) -> VolDashboard:
    """Build cross-asset vol desk dashboard."""
    return VolDashboard(
        date=reference_date,
        n_positions=len(book),
        total_vega=book.total_vega(),
        total_vanna=book.total_vanna(),
        total_volga=book.total_volga(),
        vega_by_asset_class=book.vega_by_asset_class(),
        vega_ladder=book.vega_ladder(reference_date),
        vol_premium=book.vol_premium(),
    )


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

@dataclass
class VolStressResult:
    """Vol stress scenario result."""
    scenario: str
    description: str
    pnl: float
    pnl_by_asset: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario, "description": self.description,
            "pnl": self.pnl, "by_asset": self.pnl_by_asset,
        }


def vol_stress_report(book: VolBook, ref: date | None = None) -> list[VolStressResult]:
    """Cross-asset vol stress scenarios."""
    if ref is None:
        ref = date.today()

    vega_by_ac = book.vega_by_asset_class()
    total_vega = book.total_vega()
    total_volga = book.total_volga()

    scenarios = [
        ("vol_up_2", "All vols +2%", 2.0),
        ("vol_dn_2", "All vols -2%", -2.0),
        ("vol_up_5", "All vols +5%", 5.0),
        ("vol_dn_5", "All vols -5%", -5.0),
    ]

    results = []
    for name, desc, shock in scenarios:
        pnl = total_vega * shock + 0.5 * total_volga * shock ** 2
        pnl_by_asset = {ac: v * shock for ac, v in vega_by_ac.items()}
        results.append(VolStressResult(name, desc, pnl, pnl_by_asset))

    # Tilt: short end up, long end down
    tilt_pnl = 0.0
    for p in book.positions:
        try:
            T = year_fraction(ref, p.expiry, DayCountConvention.ACT_365_FIXED)
            tilt_pnl += p.vega * (1.0 if T < 1 else -1.0)
        except ValueError:
            pass
    results.append(VolStressResult("tilt", "Short +1%, Long -1%", tilt_pnl, {}))

    return results


# ---------------------------------------------------------------------------
# Correlation monitor
# ---------------------------------------------------------------------------

def vol_correlation_monitor(book: VolBook) -> dict[str, dict]:
    """Monitor implied vs realised vol and cross-asset correlation."""
    monitor = {}
    for p in book.positions:
        if p.implied_vol > 0 and p.realised_vol > 0:
            premium = p.implied_vol - p.realised_vol
            ratio = p.implied_vol / p.realised_vol if p.realised_vol > 0 else 0
            monitor[p.underlying] = {
                "implied": p.implied_vol,
                "realised": p.realised_vol,
                "premium": premium,
                "ratio": ratio,
                "signal": "rich" if premium > 0.02 else ("cheap" if premium < -0.02 else "fair"),
            }
    return monitor
