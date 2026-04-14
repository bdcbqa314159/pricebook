"""Shared z-score / signal utility for rich/cheap monitors.

Used across equity, commodity, bond, FX, inflation, and credit desks.
Eliminates ~120 lines of duplicated z-score logic.

    from pricebook.zscore import zscore, ZScoreSignal

    signal = zscore(current=0.25, history=[0.18, 0.19, 0.20, 0.21],
                    threshold=2.0, labels=("rich", "cheap", "fair"))
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ZScoreSignal:
    """Generic z-score signal result."""
    current: float
    mean: float
    std: float
    z_score: float | None
    percentile: float | None
    signal: str


def zscore(
    current: float,
    history: list[float],
    threshold: float = 2.0,
    labels: tuple[str, str, str] = ("rich", "cheap", "fair"),
) -> ZScoreSignal:
    """Compute z-score, percentile, and signal from a value vs its history.

    Args:
        current: the current observation.
        history: window of past observations (need ≥ 2 for z-score).
        threshold: |z| above which the signal flips from neutral.
        labels: (high_signal, low_signal, neutral_signal).
            Default: ("rich", "cheap", "fair").
            For spreads: ("wide", "tight", "fair").
            For correlation: ("high", "low", "fair").

    Returns:
        :class:`ZScoreSignal` with z-score, percentile, and signal label.
    """
    high, low, neutral = labels

    if not history or len(history) < 2:
        return ZScoreSignal(current, current, 0.0, None, None, neutral)

    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    std = math.sqrt(var) if var > 0 else 0.0

    if std < 1e-12:
        return ZScoreSignal(current, mean, std, None, None, neutral)

    z = (current - mean) / std
    rank = sum(1 for h in history if h <= current)
    percentile = rank / len(history) * 100.0

    if z >= threshold:
        signal = high
    elif z <= -threshold:
        signal = low
    else:
        signal = neutral

    return ZScoreSignal(current, mean, std, z, percentile, signal)
