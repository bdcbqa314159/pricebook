"""Curve trading strategies: spreads, butterflies, carry, and roll-down.

Build DV01-neutral spread and butterfly trades. Analyse carry and
roll-down from curve shape. Compute breakeven rate moves.

    from pricebook.curve_trading import spread_trade, butterfly_trade

    legs = spread_trade(curve, start, end_short=2, end_long=10)
    assert abs(legs["net_dv01"]) < 1e-6

    bf = butterfly_trade(curve, start, end_short=2, end_belly=5, end_long=10)
"""

from __future__ import annotations

from datetime import date
from typing import Any

from dateutil.relativedelta import relativedelta

from pricebook.discount_curve import DiscountCurve
from pricebook.swap import InterestRateSwap, SwapDirection


# ---- DV01 helpers ----

def swap_dv01(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    shift: float = 0.0001,
) -> float:
    """Parallel DV01 of a swap via bump-and-reprice."""
    base = swap.pv(curve, projection_curve)
    bumped = curve.bumped(shift)
    proj_bumped = projection_curve.bumped(shift) if projection_curve else None
    return swap.pv(bumped, proj_bumped) - base


def swap_pv(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
) -> float:
    return swap.pv(curve, projection_curve)


# ---- Spread trade (e.g. 2s10s) ----

def spread_trade(
    curve: DiscountCurve,
    start: date,
    end_short_years: int,
    end_long_years: int,
    notional_long: float = 1_000_000.0,
    direction: str = "steepener",
    projection_curve: DiscountCurve | None = None,
) -> dict[str, Any]:
    """Build a DV01-neutral spread trade.

    Steepener: long short-end, short long-end (profits if curve steepens).
    Flattener: short short-end, long long-end (profits if curve flattens).

    Args:
        curve: discount curve for pricing and DV01.
        start: trade start date.
        end_short_years: short leg maturity in years from start.
        end_long_years: long leg maturity in years from start.
        notional_long: notional of the long-end leg.
        direction: "steepener" or "flattener".
        projection_curve: optional projection curve.

    Returns:
        dict with short_swap, long_swap, notionals, DV01s, net_dv01.
    """
    end_short = start + relativedelta(years=end_short_years)
    end_long = start + relativedelta(years=end_long_years)

    par_short = _par_rate(curve, start, end_short, projection_curve)
    par_long = _par_rate(curve, start, end_long, projection_curve)

    # Build at-par swaps
    long_swap = InterestRateSwap(
        start, end_long, fixed_rate=par_long,
        direction=SwapDirection.PAYER, notional=notional_long,
    )
    dv01_long = swap_dv01(long_swap, curve, projection_curve)

    # Find short notional for DV01 neutrality
    short_unit = InterestRateSwap(
        start, end_short, fixed_rate=par_short,
        direction=SwapDirection.PAYER, notional=1_000_000.0,
    )
    dv01_short_unit = swap_dv01(short_unit, curve, projection_curve)

    if abs(dv01_short_unit) < 1e-12:
        ratio = 1.0
    else:
        ratio = abs(dv01_long / dv01_short_unit)
    notional_short = ratio * 1_000_000.0

    short_swap = InterestRateSwap(
        start, end_short, fixed_rate=par_short,
        direction=SwapDirection.PAYER, notional=notional_short,
    )

    if direction == "steepener":
        # Long short-end (receiver), short long-end (payer)
        short_dir = SwapDirection.RECEIVER
        long_dir = SwapDirection.PAYER
    else:
        short_dir = SwapDirection.PAYER
        long_dir = SwapDirection.RECEIVER

    short_leg = InterestRateSwap(
        start, end_short, fixed_rate=par_short,
        direction=short_dir, notional=notional_short,
    )
    long_leg = InterestRateSwap(
        start, end_long, fixed_rate=par_long,
        direction=long_dir, notional=notional_long,
    )

    dv01_s = swap_dv01(short_leg, curve, projection_curve)
    dv01_l = swap_dv01(long_leg, curve, projection_curve)

    return {
        "type": "spread",
        "direction": direction,
        "short_leg": short_leg,
        "long_leg": long_leg,
        "short_tenor": end_short_years,
        "long_tenor": end_long_years,
        "notional_short": notional_short,
        "notional_long": notional_long,
        "dv01_short": dv01_s,
        "dv01_long": dv01_l,
        "net_dv01": dv01_s + dv01_l,
        "pv": short_leg.pv(curve, projection_curve) + long_leg.pv(curve, projection_curve),
    }


# ---- Butterfly trade (e.g. 2s5s10s) ----

def butterfly_trade(
    curve: DiscountCurve,
    start: date,
    end_short_years: int,
    end_belly_years: int,
    end_long_years: int,
    notional_belly: float = 1_000_000.0,
    projection_curve: DiscountCurve | None = None,
) -> dict[str, Any]:
    """Build a DV01-neutral butterfly: long belly, short wings.

    Profits when belly outperforms wings (curve flattening at belly).

    Args:
        curve: discount curve.
        start: trade start date.
        end_short/belly/long_years: maturities.
        notional_belly: notional of the belly leg.
        projection_curve: optional projection curve.

    Returns:
        dict with three legs, notionals, DV01s, net_dv01.
    """
    end_short = start + relativedelta(years=end_short_years)
    end_belly = start + relativedelta(years=end_belly_years)
    end_long = start + relativedelta(years=end_long_years)

    par_short = _par_rate(curve, start, end_short, projection_curve)
    par_belly = _par_rate(curve, start, end_belly, projection_curve)
    par_long = _par_rate(curve, start, end_long, projection_curve)

    # Belly: receiver (long)
    belly_swap = InterestRateSwap(
        start, end_belly, fixed_rate=par_belly,
        direction=SwapDirection.RECEIVER, notional=notional_belly,
    )
    dv01_belly = swap_dv01(belly_swap, curve, projection_curve)

    # Wings: payer (short), split DV01 equally
    short_unit = InterestRateSwap(
        start, end_short, fixed_rate=par_short,
        direction=SwapDirection.PAYER, notional=1_000_000.0,
    )
    long_unit = InterestRateSwap(
        start, end_long, fixed_rate=par_long,
        direction=SwapDirection.PAYER, notional=1_000_000.0,
    )
    dv01_short_unit = swap_dv01(short_unit, curve, projection_curve)
    dv01_long_unit = swap_dv01(long_unit, curve, projection_curve)

    # Split belly DV01 equally across wings
    half_belly_dv01 = abs(dv01_belly) / 2.0

    if abs(dv01_short_unit) < 1e-12 or abs(dv01_long_unit) < 1e-12:
        notional_short = notional_belly
        notional_long = notional_belly
    else:
        notional_short = (half_belly_dv01 / abs(dv01_short_unit)) * 1_000_000.0
        notional_long = (half_belly_dv01 / abs(dv01_long_unit)) * 1_000_000.0

    short_leg = InterestRateSwap(
        start, end_short, fixed_rate=par_short,
        direction=SwapDirection.PAYER, notional=notional_short,
    )
    long_leg = InterestRateSwap(
        start, end_long, fixed_rate=par_long,
        direction=SwapDirection.PAYER, notional=notional_long,
    )

    dv01_s = swap_dv01(short_leg, curve, projection_curve)
    dv01_l = swap_dv01(long_leg, curve, projection_curve)
    dv01_b = swap_dv01(belly_swap, curve, projection_curve)

    total_pv = (
        short_leg.pv(curve, projection_curve)
        + belly_swap.pv(curve, projection_curve)
        + long_leg.pv(curve, projection_curve)
    )

    return {
        "type": "butterfly",
        "short_leg": short_leg,
        "belly_leg": belly_swap,
        "long_leg": long_leg,
        "short_tenor": end_short_years,
        "belly_tenor": end_belly_years,
        "long_tenor": end_long_years,
        "notional_short": notional_short,
        "notional_belly": notional_belly,
        "notional_long": notional_long,
        "dv01_short": dv01_s,
        "dv01_belly": dv01_b,
        "dv01_long": dv01_l,
        "net_dv01": dv01_s + dv01_b + dv01_l,
        "pv": total_pv,
    }


# ---- Carry and roll-down ----

def _shift_curve(curve: DiscountCurve, horizon_days: int) -> DiscountCurve:
    """Shift a curve forward in time: new ref date, DFs from forward rates."""
    import math
    ref = curve.reference_date
    new_ref = date.fromordinal(ref.toordinal() + horizon_days)
    df_shift = curve.df(new_ref)

    new_dates = []
    new_dfs = []
    for d in curve.pillar_dates:
        if d > new_ref:
            new_dates.append(d)
            new_dfs.append(curve.df(d) / df_shift)

    if not new_dates:
        return DiscountCurve.flat(new_ref, 0.0)

    return DiscountCurve(new_ref, new_dates, new_dfs)


def swap_carry(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    horizon_days: int = 1,
    projection_curve: DiscountCurve | None = None,
) -> float:
    """Carry + roll-down: PV change from time passing, curve shape unchanged.

    Shifts the curve reference date forward, keeping the same forward rates,
    then re-prices the original swap. This captures both coupon income (carry)
    and the effect of rolling down the curve.
    """
    base_pv = swap.pv(curve, projection_curve)

    shifted = _shift_curve(curve, horizon_days)
    proj_shifted = _shift_curve(projection_curve, horizon_days) if projection_curve else None

    rolled_pv = swap.pv(shifted, proj_shifted)
    return rolled_pv - base_pv


def breakeven_rate_move(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    horizon_days: int = 1,
    projection_curve: DiscountCurve | None = None,
) -> float:
    """Breakeven parallel rate move: how much rates can move before carry is lost.

    breakeven = carry / |DV01|

    Positive means rates can move against you by this much (in rate units)
    before carry is wiped out.
    """
    carry = swap_carry(swap, curve, horizon_days, projection_curve)
    dv01 = swap_dv01(swap, curve, projection_curve)
    if abs(dv01) < 1e-12:
        return float("inf")
    return abs(carry / dv01)


# ---- Helpers ----

def _par_rate(
    curve: DiscountCurve,
    start: date,
    end: date,
    projection_curve: DiscountCurve | None = None,
) -> float:
    """Compute par swap rate."""
    swap = InterestRateSwap(
        start, end, fixed_rate=0.05,
        direction=SwapDirection.PAYER, notional=1_000_000.0,
    )
    return swap.par_rate(curve, projection_curve)
