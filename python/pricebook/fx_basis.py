"""FX basis strategies: cross-currency basis trading, term structure, RV.

* :func:`basis_monitor` — z-score a basis level vs history.
* :func:`basis_term_structure` — basis across tenors.
* :func:`cross_market_basis_rv` — rank basis across ccy pairs.
* :func:`basis_carry` — holding period return from a basis position.
* :class:`BasisCurveTrade` — steepener/flattener on the basis curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---- Z-score ----

def _zscore(current, history, threshold=2.0):
    if not history or len(history) < 2:
        return None, "fair"
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-12:
        return None, "fair"
    z = (current - mean) / std
    signal = "wide" if z >= threshold else ("tight" if z <= -threshold else "fair")
    return z, signal


# ---- Basis monitor ----

@dataclass
class BasisSignal:
    pair: str
    tenor: str
    basis_bps: float
    z_score: float | None
    signal: str


def basis_monitor(
    pair: str,
    tenor: str,
    basis_bps: float,
    history_bps: list[float],
    threshold: float = 2.0,
) -> BasisSignal:
    """Z-score a cross-currency basis level vs history."""
    z, signal = _zscore(basis_bps, history_bps, threshold)
    return BasisSignal(pair, tenor, basis_bps, z, signal)


# ---- Basis term structure ----

@dataclass
class BasisTermPoint:
    tenor: str
    basis_bps: float


def basis_term_structure(
    pair: str,
    tenors: list[tuple[str, float]],
) -> list[BasisTermPoint]:
    """Build a basis term structure from tenor/basis pairs.

    Args:
        tenors: list of ``(tenor_label, basis_bps)``.
    """
    return [BasisTermPoint(t, b) for t, b in tenors]


# ---- Cross-market basis RV ----

@dataclass
class BasisRVEntry:
    pair: str
    basis_bps: float
    z_score: float | None
    rank: int


def cross_market_basis_rv(
    pairs: list[tuple[str, float, list[float]]],
    threshold: float = 2.0,
) -> list[BasisRVEntry]:
    """Rank pairs by basis z-score (widest = cheapest = rank 1)."""
    results = []
    for pair, basis, history in pairs:
        z, _ = _zscore(basis, history, threshold)
        results.append((pair, basis, z))
    results.sort(key=lambda x: -(x[2] if x[2] is not None else -999))
    return [
        BasisRVEntry(p, b, z, rank=i + 1)
        for i, (p, b, z) in enumerate(results)
    ]


# ---- Basis carry ----

def basis_carry(
    notional: float,
    basis_bps: float,
    days: int,
) -> float:
    """Holding period return from a basis position.

    ``carry = notional × basis / 10000 × days / 365``.
    Positive basis = positive carry for the receiver.
    """
    return notional * basis_bps / 10_000 * days / 365.0


# ---- Basis curve trade ----

@dataclass
class BasisCurveTrade:
    """Steepener/flattener on the basis curve.

    direction = +1: steepener (receive long-tenor basis, pay short-tenor).
    direction = -1: flattener.
    """
    pair: str
    short_tenor: str
    long_tenor: str
    short_basis_bps: float
    long_basis_bps: float
    notional: float
    direction: int = 1

    @property
    def curve_spread_bps(self) -> float:
        return self.long_basis_bps - self.short_basis_bps

    def pv_change(self, new_short_bps: float, new_long_bps: float) -> float:
        """Approximate P&L from a change in the basis curve.

        For a steepener: profit when long − short widens.
        """
        old_spread = self.long_basis_bps - self.short_basis_bps
        new_spread = new_long_bps - new_short_bps
        return self.direction * self.notional * (new_spread - old_spread) / 10_000
