"""Vol surface arbitrage: detection and arb-free construction.

* :func:`detect_calendar_arb` — total variance must be non-decreasing in time.
* :func:`detect_butterfly_arb` — call prices must be convex in strike.
* :func:`check_surface_arbitrage` — run both checks on a surface.
* :func:`enforce_no_arb` — adjust a vol surface to remove arbitrage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


# ---- Calendar arbitrage ----

@dataclass
class CalendarArbViolation:
    """A calendar arbitrage violation."""
    short_expiry: date
    long_expiry: date
    short_total_var: float
    long_total_var: float


def detect_calendar_arb(
    expiries: list[date],
    vols: list[float],
    reference_date: date,
) -> list[CalendarArbViolation]:
    """Detect calendar arbitrage: total variance must be non-decreasing.

    Total variance = σ² × T. If σ²(T₁)×T₁ > σ²(T₂)×T₂ for T₁ < T₂,
    there is a calendar arbitrage.
    """
    violations = []
    for i in range(len(expiries) - 1):
        t1 = (expiries[i] - reference_date).days / 365.0
        t2 = (expiries[i + 1] - reference_date).days / 365.0
        if t1 <= 0 or t2 <= 0:
            continue
        tv1 = vols[i] ** 2 * t1
        tv2 = vols[i + 1] ** 2 * t2
        if tv1 > tv2 + 1e-12:
            violations.append(CalendarArbViolation(
                expiries[i], expiries[i + 1], tv1, tv2,
            ))
    return violations


# ---- Butterfly arbitrage ----

@dataclass
class ButterflyArbViolation:
    """A butterfly arbitrage violation (non-convexity in strike)."""
    strike_low: float
    strike_mid: float
    strike_high: float
    price_low: float
    price_mid: float
    price_high: float


def detect_butterfly_arb(
    strikes: list[float],
    call_prices: list[float],
) -> list[ButterflyArbViolation]:
    """Detect butterfly arbitrage: call prices must be convex in strike.

    For three consecutive strikes K₁ < K₂ < K₃:
        C(K₂) ≤ w × C(K₁) + (1-w) × C(K₃)
    where w = (K₃ - K₂) / (K₃ - K₁).
    """
    violations = []
    for i in range(len(strikes) - 2):
        k1, k2, k3 = strikes[i], strikes[i + 1], strikes[i + 2]
        c1, c2, c3 = call_prices[i], call_prices[i + 1], call_prices[i + 2]

        dk = k3 - k1
        if dk <= 0:
            continue
        w = (k3 - k2) / dk
        interpolated = w * c1 + (1 - w) * c3

        if c2 > interpolated + 1e-10:
            violations.append(ButterflyArbViolation(k1, k2, k3, c1, c2, c3))

    return violations


# ---- Combined check ----

@dataclass
class SurfaceArbReport:
    """Combined arbitrage report."""
    calendar_violations: list[CalendarArbViolation]
    butterfly_violations: list[ButterflyArbViolation]
    is_arb_free: bool


def check_surface_arbitrage(
    expiries: list[date],
    vols: list[float],
    reference_date: date,
    strikes: list[float] | None = None,
    call_prices: list[float] | None = None,
) -> SurfaceArbReport:
    """Run both calendar and butterfly arb checks."""
    cal = detect_calendar_arb(expiries, vols, reference_date)
    bfly = detect_butterfly_arb(strikes or [], call_prices or [])
    return SurfaceArbReport(cal, bfly, len(cal) == 0 and len(bfly) == 0)


# ---- Arb-free surface construction ----

def enforce_no_calendar_arb(
    expiries: list[date],
    vols: list[float],
    reference_date: date,
) -> list[float]:
    """Adjust vols to remove calendar arbitrage.

    Ensures total variance is non-decreasing by flooring each σ²T
    at the previous level.
    """
    adjusted = list(vols)
    for i in range(1, len(expiries)):
        t_prev = (expiries[i - 1] - reference_date).days / 365.0
        t_curr = (expiries[i] - reference_date).days / 365.0
        if t_prev <= 0 or t_curr <= 0:
            continue
        tv_prev = adjusted[i - 1] ** 2 * t_prev
        tv_curr = adjusted[i] ** 2 * t_curr
        if tv_curr < tv_prev:
            adjusted[i] = math.sqrt(tv_prev / t_curr)
    return adjusted


def enforce_no_butterfly_arb(
    strikes: list[float],
    call_prices: list[float],
) -> list[float]:
    """Adjust call prices to remove butterfly arbitrage.

    Ensures convexity by capping each mid-price at the linear
    interpolation of its neighbours.
    """
    adjusted = list(call_prices)
    for i in range(1, len(strikes) - 1):
        k1, k2, k3 = strikes[i - 1], strikes[i], strikes[i + 1]
        dk = k3 - k1
        if dk <= 0:
            continue
        w = (k3 - k2) / dk
        cap = w * adjusted[i - 1] + (1 - w) * adjusted[i + 1]
        if adjusted[i] > cap:
            adjusted[i] = cap
    return adjusted
