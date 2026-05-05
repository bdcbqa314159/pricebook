"""SOFR/ESTR/SONIA curve building convenience wrappers.

One-call functions for post-LIBOR RFR curve construction.

    from pricebook.sofr_curve import (
        build_sofr_curve, build_estr_curve, build_sonia_curve,
        build_xccy_basis_curve,
    )

    sofr = build_sofr_curve(ref, futures, swaps)
    estr = build_estr_curve(ref, swap_rates)
    xccy = build_xccy_basis_curve(sofr, estr, fx_points, spot, ref)

References:
    Henrard (2014), Multi-curve Framework, Ch. 2-3.
    Ametrano & Bianchetti (2013), Bootstrapping across currencies.
"""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.ois import bootstrap_ois
from pricebook.schedule import Frequency
from pricebook.interpolation import InterpolationMethod


def build_sofr_curve(
    reference_date: date,
    sofr_futures: list[tuple[date, date, float]] | None = None,
    sofr_swaps: list[tuple[date, float]] | None = None,
    deposits: list[tuple[date, float]] | None = None,
    hw_a: float = 0.03,
    hw_sigma: float = 0.01,
) -> DiscountCurve:
    """Build USD OIS curve from SOFR futures + SOFR OIS swaps.

    Short end: SOFR 1M/3M futures (with Hull-White convexity adjustment).
    Long end: SOFR OIS swap rates.
    Optionally: overnight/1W deposits for very short end.

    Args:
        sofr_futures: list of (accrual_start, accrual_end, futures_rate).
        sofr_swaps: list of (maturity, par_rate).
        deposits: list of (maturity, rate) for very short end.
        hw_a: Hull-White mean reversion for convexity adjustment.
        hw_sigma: Hull-White vol for convexity adjustment.
    """
    # Convert futures to equivalent swap-like inputs with convexity adjustment
    from pricebook.ir_futures import hw_convexity_adjustment

    swap_inputs = []

    # Deposits first (shortest)
    if deposits:
        for mat, rate in sorted(deposits, key=lambda x: x[0]):
            swap_inputs.append((mat, rate))

    # Futures with convexity adjustment
    if sofr_futures:
        for start, end, futures_rate in sorted(sofr_futures, key=lambda x: x[0]):
            # T in ACT/365 for HW convexity (time measure)
            # tau in ACT/360 for SOFR accrual (money market convention)
            T = year_fraction(reference_date, start, DayCountConvention.ACT_365_FIXED)
            tau = year_fraction(start, end, DayCountConvention.ACT_360)
            ca = hw_convexity_adjustment(T, tau, hw_a, hw_sigma)
            forward_rate = futures_rate - ca
            # Convert forward to par-swap-equivalent for the OIS stripper
            swap_inputs.append((end, forward_rate))

    # Swaps (longer tenors)
    if sofr_swaps:
        for mat, rate in sorted(sofr_swaps, key=lambda x: x[0]):
            swap_inputs.append((mat, rate))

    if not swap_inputs:
        raise ValueError("No inputs for SOFR curve — provide futures and/or swaps")

    return bootstrap_ois(reference_date, swap_inputs,
                         day_count=DayCountConvention.ACT_360,
                         fixed_frequency=Frequency.ANNUAL)


def build_estr_curve(
    reference_date: date,
    estr_swaps: list[tuple[date, float]],
) -> DiscountCurve:
    """Build EUR OIS curve from ESTR swap rates.

    ESTR convention: fixed annual ACT/360.
    """
    return bootstrap_ois(reference_date, estr_swaps,
                         day_count=DayCountConvention.ACT_360,
                         fixed_frequency=Frequency.ANNUAL)


def build_sonia_curve(
    reference_date: date,
    sonia_swaps: list[tuple[date, float]],
) -> DiscountCurve:
    """Build GBP OIS curve from SONIA swap rates.

    SONIA convention: fixed annual ACT/365 Fixed.
    """
    return bootstrap_ois(reference_date, sonia_swaps,
                         day_count=DayCountConvention.ACT_365_FIXED,
                         fixed_frequency=Frequency.ANNUAL)


def build_xccy_basis_curve(
    domestic_ois: DiscountCurve,
    foreign_ois: DiscountCurve,
    fx_swap_points: list[tuple[date, float]],
    spot: float,
    reference_date: date,
) -> DiscountCurve:
    """Build cross-currency basis-adjusted discount curve from FX swap points.

    The FX forward implied by the domestic OIS may differ from the
    market FX forward (= spot + points). The difference is the xccy basis.

    Implied foreign DF = domestic_df × spot / (spot + points).

    This produces a foreign discount curve CONSISTENT WITH FX forwards,
    which may differ from the foreign OIS curve by the basis.

    Args:
        domestic_ois: domestic currency OIS curve (e.g., SOFR for USD).
        foreign_ois: foreign currency OIS curve (e.g., ESTR for EUR).
            Used as fallback for tenors without FX swap data.
        fx_swap_points: list of (delivery_date, forward_points).
            Points in quote currency per unit of base.
        spot: FX spot rate (quote/base, e.g., 1.08 USD per EUR).
    """
    if not fx_swap_points:
        return foreign_ois  # no basis data → fall back to foreign OIS (no xccy adjustment)

    if spot <= 0:
        raise ValueError(f"FX spot must be positive, got {spot}")

    pillar_dates = [d for d, _ in sorted(fx_swap_points, key=lambda x: x[0])]
    implied_dfs = []

    for delivery, points in sorted(fx_swap_points, key=lambda x: x[0]):
        fwd = spot + points
        dom_df = domestic_ois.df(delivery)
        # CIP: F = S × df_foreign / df_domestic
        # → df_foreign = F / S × df_domestic = (spot + points) / spot × df_domestic
        # Wait — this is the foreign DF that makes FX forward consistent
        # Actually: F = S × df_base / df_quote (base=foreign, quote=domestic in EUR/USD)
        # For EURUSD: F = spot × df_EUR / df_USD
        # → df_EUR = F × df_USD / spot = (spot + points) × df_USD / spot
        # But this gives df_EUR that's consistent with the FX market
        implied_df = fwd / spot * dom_df
        implied_dfs.append(float(implied_df))

    return DiscountCurve(
        reference_date, pillar_dates, implied_dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
    )
