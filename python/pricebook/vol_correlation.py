"""Cross-asset vol correlation: correlation matrix, trading, monitoring.

* :func:`vol_correlation_matrix` — pairwise correlation of vol time series.
* :func:`correlation_monitor` — z-score a pairwise correlation vs history.
* :class:`CorrelationTrade` — long one vol, short another (zero net vega).
* :func:`correlation_sensitivity` — book P&L per unit correlation change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---- Correlation matrix ----

def vol_correlation_matrix(
    series: dict[str, list[float]],
) -> dict[tuple[str, str], float]:
    """Pairwise correlation matrix from vol time series.

    Args:
        series: {asset_class → list of vol observations} (equal length).

    Returns:
        dict keyed by ``(asset_a, asset_b)`` → correlation.
        Diagonal entries are 1.0; matrix is symmetric.
    """
    names = sorted(series.keys())
    n = len(names)
    result: dict[tuple[str, str], float] = {}

    for i in range(n):
        for j in range(n):
            a, b = names[i], names[j]
            if i == j:
                result[(a, b)] = 1.0
                continue
            if (b, a) in result:
                result[(a, b)] = result[(b, a)]
                continue

            sa, sb = series[a], series[b]
            T = min(len(sa), len(sb))
            if T < 2:
                result[(a, b)] = 0.0
                continue

            ma = sum(sa[:T]) / T
            mb = sum(sb[:T]) / T
            cov = sum((sa[t] - ma) * (sb[t] - mb) for t in range(T)) / T
            va = sum((sa[t] - ma) ** 2 for t in range(T)) / T
            vb = sum((sb[t] - mb) ** 2 for t in range(T)) / T
            denom = math.sqrt(va * vb)
            result[(a, b)] = cov / denom if denom > 0 else 0.0

    return result


def is_valid_correlation_matrix(
    matrix: dict[tuple[str, str], float],
) -> bool:
    """Check that a correlation matrix is symmetric with diagonal = 1."""
    for (a, b), v in matrix.items():
        if a == b and abs(v - 1.0) > 1e-10:
            return False
        if (b, a) in matrix and abs(matrix[(b, a)] - v) > 1e-10:
            return False
    return True


# ---- Correlation monitor ----

@dataclass
class CorrelationSignal:
    pair: tuple[str, str]
    current: float
    z_score: float | None
    signal: str


def correlation_monitor(
    pair: tuple[str, str],
    current_corr: float,
    history: list[float],
    threshold: float = 2.0,
) -> CorrelationSignal:
    """Z-score a pairwise correlation vs history."""
    if not history or len(history) < 2:
        return CorrelationSignal(pair, current_corr, None, "fair")
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-12:
        return CorrelationSignal(pair, current_corr, None, "fair")
    z = (current_corr - mean) / std
    signal = "high" if z >= threshold else ("low" if z <= -threshold else "fair")
    return CorrelationSignal(pair, current_corr, z, signal)


# ---- Correlation trade ----

@dataclass
class CorrelationTrade:
    """Long one vol, short another — zero net vega.

    If equity and FX vols are correlated, buying equity vol and selling
    FX vol profits when the correlation breaks (equity vol rises, FX
    vol doesn't follow).

    Quantities are set so that net vega = 0:
        short_qty = long_qty × long_vega_per_unit / short_vega_per_unit.
    """
    long_asset: str
    short_asset: str
    long_quantity: float
    short_quantity: float
    long_vega_per_unit: float
    short_vega_per_unit: float

    @property
    def net_vega(self) -> float:
        return (self.long_quantity * self.long_vega_per_unit
                - self.short_quantity * self.short_vega_per_unit)

    def pnl(self, long_vol_change: float, short_vol_change: float) -> float:
        """P&L from vol changes in each leg."""
        return (self.long_quantity * self.long_vega_per_unit * long_vol_change
                - self.short_quantity * self.short_vega_per_unit * short_vol_change)


def build_correlation_trade(
    long_asset: str,
    short_asset: str,
    long_quantity: float,
    long_vega_per_unit: float,
    short_vega_per_unit: float,
) -> CorrelationTrade:
    """Build a vega-neutral correlation trade."""
    if short_vega_per_unit <= 0:
        short_qty = long_quantity
    else:
        short_qty = long_quantity * long_vega_per_unit / short_vega_per_unit
    return CorrelationTrade(
        long_asset, short_asset, long_quantity, short_qty,
        long_vega_per_unit, short_vega_per_unit,
    )


# ---- Correlation sensitivity ----

def correlation_sensitivity(
    book_vega_a: float,
    book_vega_b: float,
    vol_a: float,
    vol_b: float,
) -> float:
    """Approximate P&L sensitivity to a unit change in correlation.

    For a book with vega in asset A and asset B, the cross-term in
    portfolio variance is ``2 × ρ × vega_A × vega_B × σ_A × σ_B``.
    The sensitivity ∂PV/∂ρ ≈ vega_A × vega_B × σ_A × σ_B (rough proxy).
    """
    return book_vega_a * book_vega_b * vol_a * vol_b
