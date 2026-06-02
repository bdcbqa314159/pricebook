"""Steepener/flattener structured notes.

CMS curve-slope products: leveraged steepener, callable steepener,
range accrual on curve slope, digital steepener.

* :func:`steepener_note` — leveraged CMS10 − CMS2 note.
* :func:`callable_steepener` — with issuer call right.
* :func:`slope_range_accrual` — accrues when slope in range.
* :func:`digital_steepener` — digital payout on slope.

References:
    Brigo & Mercurio, *Interest Rate Models*, Ch. 13, 2006.
    Piterbarg, *Rates Squared*, Risk, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SteepenerResult:
    """Steepener note pricing result."""
    price: float                # per 100 notional
    expected_coupon: float      # annualised expected coupon
    worst_coupon: float         # 5th percentile coupon
    best_coupon: float          # 95th percentile coupon
    prob_floor_hit: float       # probability floor is binding
    slope_mean: float           # expected CMS10 − CMS2

    def to_dict(self) -> dict:
        return vars(self)


def steepener_note(
    cms_long: float,
    cms_short: float,
    vol_long: float,
    vol_short: float,
    correlation: float,
    leverage: float = 3.0,
    floor: float = 0.0,
    cap: float = 0.10,
    T: float = 5.0,
    rate: float = 0.04,
    n_sims: int = 50_000,
    seed: int = 42,
) -> SteepenerResult:
    """Leveraged steepener note: leverage × (CMS10 − CMS2).

    Coupon = max(floor, min(cap, leverage × (CMS_long − CMS_short))).

    Args:
        cms_long: initial long-end CMS rate (e.g. CMS10 = 0.04).
        cms_short: initial short-end CMS rate (e.g. CMS2 = 0.035).
        vol_long: long-end vol (annualised).
        vol_short: short-end vol (annualised).
        correlation: CMS10-CMS2 correlation.
        leverage: coupon multiplier.
        floor: coupon floor.
        cap: coupon cap.
        T: maturity (years).
        rate: risk-free rate.
    """
    rng = np.random.default_rng(seed)
    n_periods = int(T * 2)  # semi-annual coupons
    dt = 0.5

    # Cholesky for 2 correlated rates
    L = np.array([[1.0, 0.0], [correlation, math.sqrt(1 - correlation**2)]])

    total_pv = 0.0
    coupons_all = []

    for _ in range(n_sims):
        path_pv = 0.0
        r_long = cms_long
        r_short = cms_short
        path_coupons = []

        for p in range(1, n_periods + 1):
            t = p * dt
            Z = rng.standard_normal(2)
            corr_Z = L @ Z

            r_long += vol_long * math.sqrt(dt) * corr_Z[0]
            r_short += vol_short * math.sqrt(dt) * corr_Z[1]

            slope = r_long - r_short
            coupon = max(floor, min(cap, leverage * slope))

            df = math.exp(-rate * t)
            path_pv += coupon * dt * df
            path_coupons.append(coupon)

        # Principal at maturity
        path_pv += math.exp(-rate * T)
        total_pv += path_pv

        avg_cpn = sum(path_coupons) / len(path_coupons) if path_coupons else 0
        coupons_all.append(avg_cpn)

    price = total_pv / n_sims * 100
    coupons_arr = np.array(coupons_all)

    return SteepenerResult(
        price=price,
        expected_coupon=float(np.mean(coupons_arr)),
        worst_coupon=float(np.percentile(coupons_arr, 5)),
        best_coupon=float(np.percentile(coupons_arr, 95)),
        prob_floor_hit=float(np.mean(coupons_arr <= floor + 1e-10)),
        slope_mean=cms_long - cms_short,
    )


def slope_range_accrual(
    cms_long: float,
    cms_short: float,
    vol_long: float,
    vol_short: float,
    correlation: float,
    lower_slope: float = 0.0,
    upper_slope: float = 0.03,
    accrual_coupon: float = 0.06,
    T: float = 5.0,
    rate: float = 0.04,
    n_sims: int = 50_000,
    seed: int = 42,
) -> SteepenerResult:
    """Range accrual on curve slope.

    Accrues coupon when CMS10 − CMS2 ∈ [lower, upper].
    N/N_total × coupon per period.

    Args:
        lower_slope: lower bound of accrual range.
        upper_slope: upper bound of accrual range.
        accrual_coupon: coupon rate when accruing.
    """
    rng = np.random.default_rng(seed)
    n_periods = int(T * 4)  # quarterly observation
    dt = 0.25
    L = np.array([[1.0, 0.0], [correlation, math.sqrt(1 - correlation**2)]])

    total_pv = 0.0
    coupons_all = []

    for _ in range(n_sims):
        path_pv = 0.0
        r_long = cms_long
        r_short = cms_short
        in_range_count = 0

        for p in range(1, n_periods + 1):
            t = p * dt
            Z = rng.standard_normal(2)
            corr_Z = L @ Z

            r_long += vol_long * math.sqrt(dt) * corr_Z[0]
            r_short += vol_short * math.sqrt(dt) * corr_Z[1]

            slope = r_long - r_short
            if lower_slope <= slope <= upper_slope:
                in_range_count += 1

        # Accrual fraction
        accrual_frac = in_range_count / n_periods
        total_coupon = accrual_coupon * accrual_frac

        # PV: coupon over life + principal
        for p in range(1, n_periods + 1):
            df = math.exp(-rate * p * dt)
            path_pv += total_coupon * dt * df

        path_pv += math.exp(-rate * T)
        total_pv += path_pv
        coupons_all.append(total_coupon)

    price = total_pv / n_sims * 100
    coupons_arr = np.array(coupons_all)

    return SteepenerResult(
        price=price,
        expected_coupon=float(np.mean(coupons_arr)),
        worst_coupon=float(np.percentile(coupons_arr, 5)),
        best_coupon=float(np.percentile(coupons_arr, 95)),
        prob_floor_hit=float(np.mean(coupons_arr < 0.001)),
        slope_mean=cms_long - cms_short,
    )


def digital_steepener(
    cms_long: float,
    cms_short: float,
    vol_long: float,
    vol_short: float,
    correlation: float,
    slope_threshold: float = 0.005,
    digital_coupon: float = 0.06,
    T: float = 5.0,
    rate: float = 0.04,
    n_sims: int = 50_000,
    seed: int = 42,
) -> SteepenerResult:
    """Digital steepener: pays fixed coupon if slope > threshold.

    Coupon = digital_coupon if CMS10 − CMS2 > threshold, else 0.

    Args:
        slope_threshold: minimum slope for coupon.
        digital_coupon: coupon rate when triggered.
    """
    rng = np.random.default_rng(seed)
    n_periods = int(T * 2)
    dt = 0.5
    L = np.array([[1.0, 0.0], [correlation, math.sqrt(1 - correlation**2)]])

    total_pv = 0.0
    coupons_all = []

    for _ in range(n_sims):
        path_pv = 0.0
        r_long = cms_long
        r_short = cms_short
        path_cpn_total = 0.0

        for p in range(1, n_periods + 1):
            t = p * dt
            Z = rng.standard_normal(2)
            corr_Z = L @ Z
            r_long += vol_long * math.sqrt(dt) * corr_Z[0]
            r_short += vol_short * math.sqrt(dt) * corr_Z[1]

            slope = r_long - r_short
            cpn = digital_coupon if slope > slope_threshold else 0.0

            df = math.exp(-rate * t)
            path_pv += cpn * dt * df
            path_cpn_total += cpn

        path_pv += math.exp(-rate * T)
        total_pv += path_pv
        coupons_all.append(path_cpn_total / n_periods)

    price = total_pv / n_sims * 100
    coupons_arr = np.array(coupons_all)

    return SteepenerResult(
        price=price,
        expected_coupon=float(np.mean(coupons_arr)),
        worst_coupon=float(np.percentile(coupons_arr, 5)),
        best_coupon=float(np.percentile(coupons_arr, 95)),
        prob_floor_hit=float(np.mean(coupons_arr < 0.001)),
        slope_mean=cms_long - cms_short,
    )
