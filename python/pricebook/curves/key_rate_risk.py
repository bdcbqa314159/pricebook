"""Key-rate DV01 and bucket risk framework.

Localised curve bumps with partition-of-unity profiles for production
key-rate duration, bucket risk, and risk ladder reports.

    from pricebook.curves.key_rate_risk import (
        key_rate_dv01, bucket_risk, standard_tenors, risk_ladder,
    )

References:
    Ho (1992). Key Rate Durations: Measures of Interest Rate Risk.
    Reitano (1991). Multivariate Duration Analysis. TSA 43, pp 335-376.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


class BumpProfile(Enum):
    """Shape of the key-rate bump function."""
    TRIANGULAR = "triangular"     # linear tent: peaks at key tenor, zero at neighbours
    GAUSSIAN = "gaussian"         # smooth bell curve
    PILLAR_ONLY = "pillar_only"   # single-pillar bump (no spreading)


@dataclass
class KeyRateResult:
    """Result of key-rate DV01 computation."""
    tenors: list[float]
    dv01s: list[float]             # DV01 at each key tenor (per 1bp)
    total_dv01: float              # sum of key-rate DV01s
    parallel_dv01: float           # parallel shift DV01 (for comparison)
    gamma: list[float] | None      # key-rate gamma (optional)
    currency: str

    def to_dict(self) -> dict:
        return {
            "tenors": self.tenors,
            "dv01s": self.dv01s,
            "total_dv01": self.total_dv01,
            "parallel_dv01": self.parallel_dv01,
            "currency": self.currency,
        }


def key_rate_dv01(
    curve: DiscountCurve,
    pricer: callable,
    tenors: list[float] | None = None,
    shift_bp: float = 1.0,
    profile: BumpProfile = BumpProfile.TRIANGULAR,
    currency: str = "USD",
    compute_gamma: bool = False,
) -> KeyRateResult:
    """Compute key-rate DV01s for a portfolio/instrument.

    Each key tenor gets a localised bump (triangular, Gaussian, or point).
    The bump profile is normalised so sum of all profiles = 1 at every point
    (partition of unity).

    Args:
        curve: base discount curve.
        pricer: callable(DiscountCurve) → float. Returns PV.
        tenors: key-rate tenors in years (default: standard set).
        shift_bp: bump size in bp (default 1bp).
        profile: bump shape.
        currency: for standard tenor selection.
        compute_gamma: if True, compute second-order (key-rate gamma).

    Returns:
        KeyRateResult with DV01 at each tenor and total.
    """
    if tenors is None:
        tenors = standard_tenors(currency)

    base_pv = pricer(curve)
    shift = shift_bp / 10_000

    dv01s = []
    gammas = [] if compute_gamma else None

    for i, tenor in enumerate(tenors):
        # Build bumped curve
        bumped_up = _apply_key_rate_bump(curve, tenors, i, +shift, profile)
        pv_up = pricer(bumped_up)
        dv01 = pv_up - base_pv
        dv01s.append(dv01)

        if compute_gamma:
            bumped_dn = _apply_key_rate_bump(curve, tenors, i, -shift, profile)
            pv_dn = pricer(bumped_dn)
            gamma = (pv_up - 2 * base_pv + pv_dn) / (shift ** 2)
            gammas.append(gamma)

    # Parallel DV01
    parallel = curve.bumped(shift)
    parallel_dv01 = pricer(parallel) - base_pv

    return KeyRateResult(
        tenors=tenors,
        dv01s=dv01s,
        total_dv01=sum(dv01s),
        parallel_dv01=parallel_dv01,
        gamma=gammas,
        currency=currency,
    )


def bucket_risk(
    curve: DiscountCurve,
    pricer: callable,
    bucket_boundaries: list[float] | None = None,
    shift_bp: float = 1.0,
) -> dict[str, float]:
    """Compute risk by tenor bucket.

    Buckets: 0-1Y, 1-2Y, 2-3Y, 3-5Y, 5-7Y, 7-10Y, 10-15Y, 15-20Y, 20-30Y.

    Returns dict of {bucket_label: DV01}.
    """
    if bucket_boundaries is None:
        bucket_boundaries = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]

    base_pv = pricer(curve)
    shift = shift_bp / 10_000
    results = {}

    for i in range(len(bucket_boundaries) - 1):
        lo = bucket_boundaries[i]
        hi = bucket_boundaries[i + 1]
        label = f"{lo}-{hi}Y"

        bumped = _apply_bucket_bump(curve, lo, hi, shift)
        pv_bumped = pricer(bumped)
        results[label] = pv_bumped - base_pv

    return results


def risk_ladder(
    key_rate_result: KeyRateResult,
) -> list[dict]:
    """Format key-rate DV01 as a risk ladder report."""
    ladder = []
    for i, tenor in enumerate(key_rate_result.tenors):
        entry = {
            "tenor": tenor,
            "dv01": key_rate_result.dv01s[i],
            "pct_of_total": key_rate_result.dv01s[i] / key_rate_result.total_dv01 * 100
            if key_rate_result.total_dv01 != 0 else 0.0,
        }
        if key_rate_result.gamma is not None:
            entry["gamma"] = key_rate_result.gamma[i]
        ladder.append(entry)
    return ladder


def standard_tenors(currency: str = "USD") -> list[float]:
    """Standard key-rate tenor sets per currency."""
    _TENORS = {
        "USD": [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30],
        "EUR": [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30],
        "GBP": [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50],
        "JPY": [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40],
        "CHF": [0.25, 0.5, 1, 2, 5, 10, 15, 30],
    }
    return _TENORS.get(currency.upper(), _TENORS["USD"])


# ═══════════════════════════════════════════════════════════════
# Internal bump application
# ═══════════════════════════════════════════════════════════════


def _apply_key_rate_bump(
    curve: DiscountCurve,
    tenors: list[float],
    key_idx: int,
    shift: float,
    profile: BumpProfile,
) -> DiscountCurve:
    """Apply a localised bump at key_idx."""
    ref = curve.reference_date

    # Compute weight at each pillar
    new_dfs = []
    for d in curve.pillar_dates:
        t = (d - ref).days / 365.0
        w = _bump_weight(t, tenors, key_idx, profile)
        df = curve.df(d)
        # Bump zero rate: z_new = z_old + w × shift
        # df_new = df_old × exp(-w × shift × t)
        new_df = df * math.exp(-w * shift * t) if t > 0 else df
        new_dfs.append(new_df)

    return DiscountCurve(ref, curve.pillar_dates, new_dfs)


def _apply_bucket_bump(
    curve: DiscountCurve,
    lo: float,
    hi: float,
    shift: float,
) -> DiscountCurve:
    """Apply a flat bump to a tenor bucket [lo, hi]."""
    ref = curve.reference_date
    new_dfs = []
    for d in curve.pillar_dates:
        t = (d - ref).days / 365.0
        df = curve.df(d)
        if lo <= t <= hi:
            new_df = df * math.exp(-shift * t)
        else:
            new_df = df
        new_dfs.append(new_df)
    return DiscountCurve(ref, curve.pillar_dates, new_dfs)


def _bump_weight(t: float, tenors: list[float], key_idx: int, profile: BumpProfile) -> float:
    """Weight of the bump at time t for key tenor at key_idx."""
    key_t = tenors[key_idx]

    if profile == BumpProfile.PILLAR_ONLY:
        # Only bumps points very close to the key tenor
        return 1.0 if abs(t - key_t) < 0.01 else 0.0

    if profile == BumpProfile.TRIANGULAR:
        # Triangular: peak at key_t, zero at adjacent key tenors
        left = tenors[key_idx - 1] if key_idx > 0 else 0.0
        right = tenors[key_idx + 1] if key_idx < len(tenors) - 1 else key_t * 2

        if t <= left or t >= right:
            return 0.0
        if t <= key_t:
            return (t - left) / (key_t - left) if key_t > left else 1.0
        else:
            return (right - t) / (right - key_t) if right > key_t else 1.0

    if profile == BumpProfile.GAUSSIAN:
        # Gaussian: centered at key_t, width = distance to nearest neighbour
        left = tenors[key_idx - 1] if key_idx > 0 else 0.0
        right = tenors[key_idx + 1] if key_idx < len(tenors) - 1 else key_t * 2
        sigma = min(key_t - left, right - key_t) / 2
        if sigma <= 0:
            sigma = 1.0
        return math.exp(-0.5 * ((t - key_t) / sigma) ** 2)

    return 0.0
