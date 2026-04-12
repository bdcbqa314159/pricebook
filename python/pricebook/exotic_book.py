"""Exotic options book: position management and Greeks for non-vanilla options.

Tracks barriers, digitals, asians, and autocalls across asset classes.
Each exotic carries model-dependent Greeks that may differ from vanilla
Black-Scholes — the book supports model comparison for risk assessment.

* :class:`ExoticEntry` — a single exotic position with Greeks + model tag.
* :class:`ExoticBook` — aggregate by type and underlying.
* :func:`model_risk_comparison` — compare Greeks across models.
* :func:`hedge_exotic_book` — delta/vega hedge using vanilla instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


# ---- Exotic entry ----

@dataclass
class ExoticEntry:
    """A single exotic option position."""
    trade_id: str
    exotic_type: str      # "barrier", "digital", "asian", "autocall"
    asset_class: str      # "equity", "fx", "ir", "commodity"
    underlying: str
    expiry: date | None = None
    notional: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    model: str = "black_scholes"


@dataclass
class ExoticTypeExposure:
    """Aggregated exposure by exotic type."""
    exotic_type: str
    net_notional: float
    net_delta: float
    net_vega: float
    n_positions: int


@dataclass
class ExoticUnderlyingExposure:
    """Aggregated exposure by underlying."""
    underlying: str
    net_delta: float
    net_vega: float
    net_gamma: float
    n_positions: int


# ---- Exotic book ----

class ExoticBook:
    """Cross-asset exotic options book."""

    def __init__(self, name: str):
        self.name = name
        self._entries: list[ExoticEntry] = []

    def add(self, entry: ExoticEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[ExoticEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def by_type(self) -> list[ExoticTypeExposure]:
        """Aggregate by exotic type."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            if e.exotic_type not in agg:
                agg[e.exotic_type] = {"notional": 0.0, "delta": 0.0, "vega": 0.0, "count": 0}
            agg[e.exotic_type]["notional"] += e.notional
            agg[e.exotic_type]["delta"] += e.delta
            agg[e.exotic_type]["vega"] += e.vega
            agg[e.exotic_type]["count"] += 1
        return [
            ExoticTypeExposure(t, d["notional"], d["delta"], d["vega"], d["count"])
            for t, d in sorted(agg.items())
        ]

    def by_underlying(self) -> list[ExoticUnderlyingExposure]:
        """Aggregate by underlying."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            if e.underlying not in agg:
                agg[e.underlying] = {"delta": 0.0, "vega": 0.0, "gamma": 0.0, "count": 0}
            agg[e.underlying]["delta"] += e.delta
            agg[e.underlying]["vega"] += e.vega
            agg[e.underlying]["gamma"] += e.gamma
            agg[e.underlying]["count"] += 1
        return [
            ExoticUnderlyingExposure(u, d["delta"], d["vega"], d["gamma"], d["count"])
            for u, d in sorted(agg.items())
        ]

    def total_delta(self) -> float:
        return sum(e.delta for e in self._entries)

    def total_vega(self) -> float:
        return sum(e.vega for e in self._entries)


# ---- Model risk comparison ----

@dataclass
class ModelComparison:
    """Greeks from different models for the same position."""
    trade_id: str
    models: dict[str, dict[str, float]]  # model → {delta, gamma, vega, theta}
    max_delta_diff: float
    max_vega_diff: float


def model_risk_comparison(
    greeks_by_model: dict[str, dict[str, float]],
    trade_id: str = "",
) -> ModelComparison:
    """Compare Greeks across models for a single position.

    Args:
        greeks_by_model: {model_name → {delta, gamma, vega, theta}}.
    """
    deltas = [g.get("delta", 0.0) for g in greeks_by_model.values()]
    vegas = [g.get("vega", 0.0) for g in greeks_by_model.values()]

    max_d = max(deltas) - min(deltas) if deltas else 0.0
    max_v = max(vegas) - min(vegas) if vegas else 0.0

    return ModelComparison(trade_id, greeks_by_model, max_d, max_v)


# ---- Hedge exotic book ----

@dataclass
class ExoticHedgeResult:
    """Hedge recommendation for the exotic book."""
    hedge_delta_qty: float
    hedge_vega_qty: float
    residual_delta: float
    residual_vega: float


def hedge_exotic_book(
    book: ExoticBook,
    vanilla_delta_per_unit: float = 1.0,
    vanilla_vega_per_unit: float = 100.0,
) -> ExoticHedgeResult:
    """Compute vanilla hedge quantities to flatten exotic book delta and vega.

    Uses two instruments: underlying (delta=1, vega=0) and ATM option
    (delta≈0.5, vega>0). Simplified to independent hedges.
    """
    book_delta = book.total_delta()
    book_vega = book.total_vega()

    if abs(vanilla_vega_per_unit) > 1e-15:
        vega_qty = -book_vega / vanilla_vega_per_unit
    else:
        vega_qty = 0.0

    if abs(vanilla_delta_per_unit) > 1e-15:
        delta_qty = -book_delta / vanilla_delta_per_unit
    else:
        delta_qty = 0.0

    return ExoticHedgeResult(
        hedge_delta_qty=delta_qty,
        hedge_vega_qty=vega_qty,
        residual_delta=book_delta + delta_qty * vanilla_delta_per_unit,
        residual_vega=book_vega + vega_qty * vanilla_vega_per_unit,
    )
