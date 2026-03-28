"""Curve-based risk: bump and reprice."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod


def bump_curve(
    curve: DiscountCurve,
    pillar_dates: list[date],
    pillar_dfs: list[float],
    pillar_index: int,
    bump_bps: float = 1.0,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> DiscountCurve:
    """
    Create a new curve with one pillar's zero rate bumped by bump_bps basis points.

    The bump is applied to the continuously compounded zero rate at the
    given pillar, then converted back to a discount factor.
    """
    ref = curve.reference_date
    dc = curve.day_count
    bump = bump_bps / 10000.0

    new_dfs = list(pillar_dfs)
    t = year_fraction(ref, pillar_dates[pillar_index], dc)
    if t > 0:
        import math
        z = -math.log(pillar_dfs[pillar_index]) / t
        new_dfs[pillar_index] = math.exp(-(z + bump) * t)

    return DiscountCurve(ref, pillar_dates, new_dfs, dc, interpolation)


def parallel_bump_curve(
    curve: DiscountCurve,
    pillar_dates: list[date],
    pillar_dfs: list[float],
    bump_bps: float = 1.0,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> DiscountCurve:
    """Create a new curve with ALL pillar zero rates bumped by bump_bps."""
    import math

    ref = curve.reference_date
    dc = curve.day_count
    bump = bump_bps / 10000.0

    new_dfs = []
    for i, (d, df) in enumerate(zip(pillar_dates, pillar_dfs)):
        t = year_fraction(ref, d, dc)
        if t > 0:
            z = -math.log(df) / t
            new_dfs.append(math.exp(-(z + bump) * t))
        else:
            new_dfs.append(df)

    return DiscountCurve(ref, pillar_dates, new_dfs, dc, interpolation)


def dv01_curve(
    pricer,
    curve: DiscountCurve,
    pillar_dates: list[date],
    pillar_dfs: list[float],
    bump_bps: float = 1.0,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> float:
    """
    Parallel DV01: total price sensitivity to a parallel curve shift.

    Args:
        pricer: callable(curve) -> float, returns the price/PV given a curve.
        curve: base curve.
        pillar_dates: curve pillar dates.
        pillar_dfs: curve pillar discount factors.
        bump_bps: bump size in basis points.

    Returns:
        Price change per basis point (parallel shift).
    """
    base_price = pricer(curve)
    bumped = parallel_bump_curve(curve, pillar_dates, pillar_dfs, bump_bps, interpolation)
    return (pricer(bumped) - base_price) / bump_bps


def key_rate_durations(
    pricer,
    curve: DiscountCurve,
    pillar_dates: list[date],
    pillar_dfs: list[float],
    bump_bps: float = 1.0,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> list[tuple[date, float]]:
    """
    Key rate durations: sensitivity to each individual pillar.

    Returns a list of (pillar_date, price_change_per_bp) pairs.
    """
    base_price = pricer(curve)
    results = []
    for i in range(len(pillar_dates)):
        bumped = bump_curve(curve, pillar_dates, pillar_dfs, i, bump_bps, interpolation)
        delta = (pricer(bumped) - base_price) / bump_bps
        results.append((pillar_dates[i], delta))
    return results
