"""Futures-based curve stripping.

Combines deposits, IR futures (with convexity adjustment), and swaps
in a single bootstrap to build a discount curve.
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.ir_futures import hw_convexity_adjustment
from pricebook.solvers import brentq


def futures_strip(
    reference_date: date,
    deposits: list[tuple[date, float]],
    futures: list[tuple[date, date, float]],
    swaps: list[tuple[date, float]],
    hw_a: float = 0.0,
    hw_sigma: float = 0.0,
    deposit_day_count: DayCountConvention = DayCountConvention.ACT_360,
    swap_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    turn_of_year: float = 0.0,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> DiscountCurve:
    """Bootstrap a discount curve from deposits + futures + swaps.

    Args:
        reference_date: curve reference date.
        deposits: list of (maturity, rate) for money market deposits.
        futures: list of (accrual_start, accrual_end, futures_rate) for IR futures.
            futures_rate is 1 - price/100 (already in decimal).
        swaps: list of (maturity, par_rate) for vanilla IRS.
        hw_a: Hull-White mean reversion for convexity adjustment.
        hw_sigma: Hull-White volatility for convexity adjustment.
        turn_of_year: additional spread (bp) for year-end funding premium.
        deposit_day_count: day count for deposits.
        swap_day_count: day count for swaps.
        interpolation: interpolation method for the curve.

    Returns:
        Bootstrapped DiscountCurve.
    """
    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []

    # Phase 1: Deposits (short end)
    for mat, rate in sorted(deposits, key=lambda x: x[0]):
        tau = year_fraction(reference_date, mat, deposit_day_count)
        df = 1.0 / (1.0 + rate * tau)
        pillar_dates.append(mat)
        pillar_dfs.append(df)

    # Phase 2: Futures (middle)
    for start, end, fut_rate in sorted(futures, key=lambda x: x[0]):
        t_start = year_fraction(reference_date, start, deposit_day_count)
        t_end = year_fraction(reference_date, end, deposit_day_count)
        tau = year_fraction(start, end, deposit_day_count)

        # Convexity adjustment: forward rate = futures rate - CA
        ca = 0.0
        if hw_sigma > 0:
            ca = hw_convexity_adjustment(hw_a, hw_sigma, 0.0, t_start, t_end)
        fwd_rate = fut_rate - ca

        # Turn-of-year: add spread if period crosses year-end
        if start.year != end.year and turn_of_year != 0:
            fwd_rate += turn_of_year

        # df(end) = df(start) / (1 + fwd * tau)
        # Need df(start) — interpolate from existing pillars
        if pillar_dates:
            temp_curve = DiscountCurve(reference_date, pillar_dates, pillar_dfs,
                                       interpolation=interpolation)
            df_start = temp_curve.df(start)
        else:
            df_start = 1.0

        df_end = df_start / (1.0 + fwd_rate * tau)
        pillar_dates.append(end)
        pillar_dfs.append(df_end)

    # Phase 3: Swaps (long end)
    from pricebook.schedule import Frequency, generate_schedule

    for mat, par_rate in sorted(swaps, key=lambda x: x[0]):
        schedule = generate_schedule(reference_date, mat, Frequency.SEMI_ANNUAL)

        def _swap_pv(df_mat, _schedule=schedule, _par=par_rate, _dc=swap_day_count):
            trial_dates = pillar_dates + [mat]
            trial_dfs = pillar_dfs + [df_mat]
            trial_curve = DiscountCurve(reference_date, trial_dates, trial_dfs,
                                         interpolation=interpolation)
            fixed_pv = 0.0
            for i in range(1, len(_schedule)):
                tau = year_fraction(_schedule[i-1], _schedule[i], _dc)
                fixed_pv += _par * tau * trial_curve.df(_schedule[i])
            float_pv = trial_curve.df(_schedule[0]) - trial_curve.df(_schedule[-1])
            return fixed_pv - float_pv

        df_mat = brentq(_swap_pv, 0.01, 2.0)
        pillar_dates.append(mat)
        pillar_dfs.append(df_mat)

    return DiscountCurve(reference_date, pillar_dates, pillar_dfs,
                         interpolation=interpolation)
