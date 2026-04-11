"""Cross-asset options book: unified vol/gamma/theta aggregation.

The capstone module that aggregates option risk across equity, FX, IR,
and commodity books into a single view.

* :class:`OptionEntry` — a single option position with Greeks.
* :class:`OptionsBook` — aggregate by asset class and expiry.
* :func:`vol_pnl_attribution` — daily vol P&L by asset class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# ---- Option entry ----

@dataclass
class OptionEntry:
    """A single option position with pre-computed Greeks."""
    trade_id: str
    asset_class: str      # "equity", "fx", "ir", "commodity"
    underlying: str       # ticker, pair, swap tenor, commodity
    expiry: date | None = None
    notional: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0


# ---- Aggregation dataclasses ----

@dataclass
class AssetClassExposure:
    """Aggregated Greeks for a single asset class."""
    asset_class: str
    net_vega: float
    net_gamma: float
    net_theta: float
    net_delta: float
    n_positions: int


@dataclass
class ExpiryBucket:
    """Aggregated vega by expiry bucket."""
    expiry_label: str
    net_vega: float
    n_positions: int


@dataclass
class VolPnLAttribution:
    """Daily vol P&L by asset class."""
    asset_class: str
    vega_pnl: float
    gamma_pnl: float
    theta_pnl: float
    total_pnl: float


# ---- Options book ----

class OptionsBook:
    """Cross-asset options book aggregating Greeks across all asset classes.

    Args:
        name: book name.
    """

    def __init__(self, name: str):
        self.name = name
        self._entries: list[OptionEntry] = []

    def add(self, entry: OptionEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[OptionEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def n_asset_classes(self) -> int:
        return len({e.asset_class for e in self._entries})

    # ---- Aggregations ----

    def by_asset_class(self) -> list[AssetClassExposure]:
        """Aggregate Greeks per asset class."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            ac = e.asset_class
            if ac not in agg:
                agg[ac] = {"vega": 0.0, "gamma": 0.0, "theta": 0.0,
                           "delta": 0.0, "count": 0}
            agg[ac]["vega"] += e.vega
            agg[ac]["gamma"] += e.gamma
            agg[ac]["theta"] += e.theta
            agg[ac]["delta"] += e.delta
            agg[ac]["count"] += 1

        return [
            AssetClassExposure(ac, d["vega"], d["gamma"], d["theta"],
                               d["delta"], d["count"])
            for ac, d in sorted(agg.items())
        ]

    def by_expiry(self, buckets: dict[str, tuple[date, date]] | None = None) -> list[ExpiryBucket]:
        """Aggregate vega by expiry bucket.

        If *buckets* is None, groups by exact expiry date label.
        Otherwise buckets is ``{label: (start, end)}`` date ranges.
        """
        if buckets is None:
            agg: dict[str, dict] = {}
            for e in self._entries:
                label = str(e.expiry) if e.expiry else "unknown"
                if label not in agg:
                    agg[label] = {"vega": 0.0, "count": 0}
                agg[label]["vega"] += e.vega
                agg[label]["count"] += 1
            return [
                ExpiryBucket(label, d["vega"], d["count"])
                for label, d in sorted(agg.items())
            ]

        agg = {label: {"vega": 0.0, "count": 0} for label in buckets}
        for e in self._entries:
            if e.expiry is None:
                continue
            for label, (start, end) in buckets.items():
                if start <= e.expiry <= end:
                    agg[label]["vega"] += e.vega
                    agg[label]["count"] += 1
                    break
        return [
            ExpiryBucket(label, d["vega"], d["count"])
            for label, d in sorted(agg.items())
        ]

    def total_vega(self) -> float:
        return sum(e.vega for e in self._entries)

    def total_gamma(self) -> float:
        return sum(e.gamma for e in self._entries)

    def total_theta(self) -> float:
        return sum(e.theta for e in self._entries)

    # ---- Vol P&L attribution ----

    def vol_pnl_attribution(
        self,
        vol_changes: dict[str, float],
        spot_changes: dict[str, float] | None = None,
        dt_days: float = 1.0,
    ) -> list[VolPnLAttribution]:
        """Daily vol P&L by asset class.

        Args:
            vol_changes: {underlying → Δvol} for each position.
            spot_changes: {underlying → ΔS} for gamma P&L.
            dt_days: days elapsed (for theta).

        Returns:
            list of :class:`VolPnLAttribution`, one per asset class.
        """
        spot_changes = spot_changes or {}
        agg: dict[str, dict] = {}

        for e in self._entries:
            ac = e.asset_class
            if ac not in agg:
                agg[ac] = {"vega": 0.0, "gamma": 0.0, "theta": 0.0}

            dv = vol_changes.get(e.underlying, 0.0)
            ds = spot_changes.get(e.underlying, 0.0)

            agg[ac]["vega"] += e.vega * dv
            agg[ac]["gamma"] += 0.5 * e.gamma * ds * ds
            agg[ac]["theta"] += e.theta * dt_days

        return [
            VolPnLAttribution(
                ac, d["vega"], d["gamma"], d["theta"],
                d["vega"] + d["gamma"] + d["theta"],
            )
            for ac, d in sorted(agg.items())
        ]
