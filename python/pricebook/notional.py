"""Notional schedule normalization.

Shared utility for converting a notional input (scalar or list) into
a per-period list. Used by FixedLeg, FloatingLeg, CDS, CLN, and any
instrument that supports variable notional schedules.

    from pricebook.notional import normalize_notional

    schedule = normalize_notional(50_000_000, n_periods=10)
    # [50e6, 50e6, ..., 50e6]

    schedule = normalize_notional([50e6, 40e6, 30e6], n_periods=10)
    # [50e6, 40e6, 30e6, 30e6, 30e6, ..., 30e6]
"""

from __future__ import annotations


def normalize_notional(notional: float | list[float], n_periods: int) -> list[float]:
    """Normalize notional input to a per-period list.

    Args:
        notional: scalar (replicated) or list (extended/truncated).
        n_periods: number of periods in the schedule.

    Returns:
        List of exactly n_periods positive floats.

    Raises:
        ValueError: if notional is empty, zero, or negative.
    """
    if isinstance(notional, (int, float)):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        return [float(notional)] * n_periods
    if not notional:
        raise ValueError("notional schedule is empty")
    if any(n <= 0 for n in notional):
        raise ValueError(f"all notionals must be positive, got {notional}")
    ns = list(notional)
    if len(ns) < n_periods:
        ns += [ns[-1]] * (n_periods - len(ns))
    return ns[:n_periods]
