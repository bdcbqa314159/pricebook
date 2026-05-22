"""Curve blending and splicing.

Combine short-end (futures) with long-end (swaps), or parametric
with bootstrapped, using smooth transition zones.

    from pricebook.curves.curve_blending import (
        splice_curves, blend_curves, BlendMethod,
    )

References:
    Hagan & West (2008). Methods for Constructing a Yield Curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


class BlendMethod(Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    STEP = "step"


def splice_curves(
    short_curve: DiscountCurve,
    long_curve: DiscountCurve,
    splice_tenor: float,
    transition_width: float = 1.0,
    method: BlendMethod = BlendMethod.SIGMOID,
    n_output_points: int = 100,
) -> DiscountCurve:
    """Splice a short-end curve with a long-end curve.

    Uses short_curve for t < splice_tenor - width/2,
    long_curve for t > splice_tenor + width/2,
    and blends in the transition zone.

    Args:
        short_curve: curve for the short end.
        long_curve: curve for the long end.
        splice_tenor: center of the transition zone (years).
        transition_width: width of transition (years).
        method: blending method in the transition.
        n_output_points: number of output pillar points.
    """
    ref = short_curve.reference_date
    dc = DayCountConvention.ACT_365_FIXED

    # Determine output time range
    all_dates = sorted(set(short_curve.pillar_dates + long_curve.pillar_dates))
    t_max = max(year_fraction(ref, d, dc) for d in all_dates)
    times = np.linspace(1.0 / 365, t_max, n_output_points)

    out_dates = []
    out_dfs = []

    for t in times:
        d = date.fromordinal(ref.toordinal() + int(t * 365))
        df_short = short_curve.df(d)
        df_long = long_curve.df(d)

        # Weight for long curve
        w = _blend_weight(t, splice_tenor, transition_width, method)
        # Blend in log-DF space for smoothness
        log_df = (1 - w) * math.log(max(df_short, 1e-15)) + w * math.log(max(df_long, 1e-15))

        out_dates.append(d)
        out_dfs.append(math.exp(log_df))

    return DiscountCurve(ref, out_dates, out_dfs)


def blend_curves(
    curves: list[DiscountCurve],
    weights: list[float],
    n_output_points: int = 100,
) -> DiscountCurve:
    """Weighted blend of multiple curves.

    Blends in log-DF space: log(df_blend) = Σ w_i × log(df_i).

    Args:
        curves: list of discount curves to blend.
        weights: weight for each curve (should sum to 1).
        n_output_points: output pillar count.
    """
    if len(curves) != len(weights):
        raise ValueError("curves and weights must have same length")
    if not curves:
        raise ValueError("At least one curve required")

    w_total = sum(weights)
    w_norm = [w / w_total for w in weights]

    ref = curves[0].reference_date
    dc = DayCountConvention.ACT_365_FIXED
    all_dates = sorted(set(d for c in curves for d in c.pillar_dates))
    t_max = max(year_fraction(ref, d, dc) for d in all_dates)
    times = np.linspace(1.0 / 365, t_max, n_output_points)

    out_dates = []
    out_dfs = []
    for t in times:
        d = date.fromordinal(ref.toordinal() + int(t * 365))
        log_df = sum(w * math.log(max(c.df(d), 1e-15)) for c, w in zip(curves, w_norm))
        out_dates.append(d)
        out_dfs.append(math.exp(log_df))

    return DiscountCurve(ref, out_dates, out_dfs)


def _blend_weight(t, center, width, method):
    """Weight for long curve at time t."""
    if width <= 0:
        return 1.0 if t >= center else 0.0

    x = (t - center) / (width / 2)  # normalized to [-1, 1]

    if method == BlendMethod.STEP:
        return 1.0 if t >= center else 0.0
    elif method == BlendMethod.LINEAR:
        return max(0.0, min(1.0, (x + 1) / 2))
    elif method == BlendMethod.SIGMOID:
        return 1.0 / (1.0 + math.exp(-3 * x))
    return 0.5
