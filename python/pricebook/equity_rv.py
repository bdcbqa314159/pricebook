"""Equity rich/cheap analysis and delta/vega hedging.

Mid-office tooling for an equity vol trader.

* :func:`implied_vs_historical_vol` — z-score the current implied vol
  against a window of historical realised vols.
* :func:`skew_monitor` — track 25-delta risk reversal level vs history.
* :func:`calendar_monitor` — front- vs back-month vol spread vs history.
* :func:`delta_hedge`, :func:`vega_hedge` — single-instrument hedges.
* :func:`optimal_delta_vega_hedge` — solve a 2×2 system to flatten
  both delta and vega using two distinct hedge instruments.
* :func:`hedged_exposure` — combined book + allocations exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.zscore import zscore as _zscore_impl, ZScoreSignal


def _zscore_and_signal(
    current: float,
    history: list[float],
    threshold: float = 2.0,
) -> ZScoreSignal:
    """Compute the z-score, percentile and rich/cheap signal vs history."""
    return _zscore_impl(current, history, threshold)


# ---- Rich/cheap monitors ----

def implied_vs_historical_vol(
    implied_vol: float,
    historical_vols: list[float],
    threshold: float = 2.0,
) -> ZScoreSignal:
    """Compare current implied vol to a window of historical realised vols."""
    return _zscore_and_signal(implied_vol, historical_vols, threshold)


@dataclass
class SkewLevel:
    """25-delta risk reversal monitor for a single expiry."""
    expiry: date
    rr_level: float
    mean: float
    std: float
    z_score: float | None
    percentile: float | None
    signal: str


def skew_monitor(
    expiry: date,
    rr_level: float,
    history: list[float],
    threshold: float = 2.0,
) -> SkewLevel:
    """Track current 25-delta risk reversal vs history.

    Convention: ``rr_level = vol(call_25d) - vol(put_25d)``. A positive
    z-score means the current upside skew is unusually rich (call vol
    elevated relative to put vol).
    """
    z = _zscore_and_signal(rr_level, history, threshold)
    return SkewLevel(
        expiry=expiry, rr_level=rr_level,
        mean=z.mean, std=z.std,
        z_score=z.z_score, percentile=z.percentile, signal=z.signal,
    )


@dataclass
class CalendarLevel:
    """Calendar (short - long) vol spread monitor."""
    short_expiry: date
    long_expiry: date
    short_vol: float
    long_vol: float
    spread: float
    mean: float
    std: float
    z_score: float | None
    percentile: float | None
    signal: str


def calendar_monitor(
    short_expiry: date,
    long_expiry: date,
    short_vol: float,
    long_vol: float,
    history: list[float],
    threshold: float = 2.0,
) -> CalendarLevel:
    """Track current short-vs-long vol spread vs history.

    Convention: ``spread = short_vol - long_vol``. A positive spread is
    a backwardated term structure (front month above back month).
    """
    spread = short_vol - long_vol
    z = _zscore_and_signal(spread, history, threshold)
    return CalendarLevel(
        short_expiry=short_expiry, long_expiry=long_expiry,
        short_vol=short_vol, long_vol=long_vol, spread=spread,
        mean=z.mean, std=z.std,
        z_score=z.z_score, percentile=z.percentile, signal=z.signal,
    )


# ---- Hedging ----

@dataclass
class BookExposure:
    """Aggregate book exposure to delta and vega."""
    delta: float
    vega: float


@dataclass
class HedgeInstrument:
    """A single hedge: provides per-unit delta and vega."""
    name: str
    delta: float
    vega: float


@dataclass
class HedgeAllocation:
    """Quantity of a hedge instrument."""
    instrument: HedgeInstrument
    quantity: float


def delta_hedge(
    book_delta: float,
    hedge_delta_per_unit: float = 1.0,
) -> float:
    """Quantity of underlying / futures to short to flatten book delta.

        quantity = -book_delta / hedge_delta_per_unit
    """
    if abs(hedge_delta_per_unit) < 1e-15:
        return 0.0
    return -book_delta / hedge_delta_per_unit


def vega_hedge(
    book_vega: float,
    hedge_vega_per_unit: float,
) -> float:
    """Quantity of a single vega-bearing hedge to flatten book vega."""
    if abs(hedge_vega_per_unit) < 1e-15:
        return 0.0
    return -book_vega / hedge_vega_per_unit


def optimal_delta_vega_hedge(
    book: BookExposure,
    hedge_a: HedgeInstrument,
    hedge_b: HedgeInstrument,
) -> tuple[HedgeAllocation, HedgeAllocation]:
    """Solve a 2×2 system to flatten both delta and vega.

        x_a · δ_a + x_b · δ_b = -book.delta
        x_a · ν_a + x_b · ν_b = -book.vega

    Raises:
        ValueError: if the two hedge instruments are linearly dependent
            (zero determinant) — the system has no unique solution.
    """
    a, b = hedge_a, hedge_b
    det = a.delta * b.vega - b.delta * a.vega
    if abs(det) < 1e-15:
        raise ValueError(
            f"hedge instruments {a.name} and {b.name} are linearly "
            "dependent in (delta, vega) space"
        )
    rhs1 = -book.delta
    rhs2 = -book.vega
    x_a = (rhs1 * b.vega - b.delta * rhs2) / det
    x_b = (a.delta * rhs2 - rhs1 * a.vega) / det
    return (HedgeAllocation(a, x_a), HedgeAllocation(b, x_b))


def hedged_exposure(
    book: BookExposure,
    allocations: list[HedgeAllocation],
) -> BookExposure:
    """Combined exposure of the book plus a list of hedge allocations."""
    total_delta = book.delta
    total_vega = book.vega
    for alloc in allocations:
        total_delta += alloc.quantity * alloc.instrument.delta
        total_vega += alloc.quantity * alloc.instrument.vega
    return BookExposure(total_delta, total_vega)
