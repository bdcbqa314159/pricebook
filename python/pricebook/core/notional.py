"""Notional schedule normalization.

Shared utility for converting a notional input (scalar or list) into
a per-period list. Used by FixedLeg, FloatingLeg, CDS, CLN, and any
instrument that supports variable notional schedules.

    from pricebook.core.notional import normalize_notional

    schedule = normalize_notional(50_000_000, n_periods=10)
    # [50e6, 50e6, ..., 50e6]

    schedule = normalize_notional([50e6, 40e6, 30e6], n_periods=10)
    # [50e6, 40e6, 30e6, 30e6, 30e6, ..., 30e6]
"""

import math


def normalize_notional(notional: float | list[float], n_periods: int) -> list[float]:
    """Normalize notional input to a per-period list.

    Args:
        notional: scalar (replicated) or list (extended/truncated).
        n_periods: number of periods in the schedule (must be >= 0).

    Returns:
        List of exactly n_periods positive finite floats.

    Raises:
        ValueError: if n_periods is negative, or notional is empty, non-finite,
            zero, or negative.
    """
    # Negative n_periods would silently produce a wrong-length schedule via
    # slice/multiply semantics (list → ns[:-1] drops the last element; scalar
    # → []), not an error.
    if n_periods < 0:
        raise ValueError(f"n_periods must be non-negative, got {n_periods}")

    if isinstance(notional, (int, float)):
        # NaN/Inf slip a bare `<= 0` check (NaN<=0 and inf<=0 are both False),
        # yielding a schedule that poisons every PV off it.
        if not math.isfinite(notional) or notional <= 0:
            raise ValueError(f"notional must be a positive finite number, got {notional}")
        return [float(notional)] * n_periods
    if not notional:
        raise ValueError("notional schedule is empty")
    if any(not math.isfinite(n) or n <= 0 for n in notional):
        raise ValueError(f"all notionals must be positive and finite, got {notional}")
    ns = [float(n) for n in notional]
    if len(ns) < n_periods:
        ns += [ns[-1]] * (n_periods - len(ns))
    return ns[:n_periods]
