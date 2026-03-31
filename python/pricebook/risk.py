"""Curve-based risk: bump and reprice."""

from datetime import date

from pricebook.discount_curve import DiscountCurve


def dv01_curve(
    pricer,
    curve: DiscountCurve,
    bump_bps: float = 1.0,
) -> float:
    """
    Parallel DV01: total price sensitivity to a parallel curve shift.

    Args:
        pricer: callable(curve) -> float, returns the price/PV given a curve.
        curve: base curve.
        bump_bps: bump size in basis points.

    Returns:
        Price change per basis point (parallel shift).
    """
    base_price = pricer(curve)
    bumped = curve.bumped(bump_bps / 10000.0)
    return (pricer(bumped) - base_price) / bump_bps


def key_rate_durations(
    pricer,
    curve: DiscountCurve,
    bump_bps: float = 1.0,
) -> list[tuple[date, float]]:
    """
    Key rate durations: sensitivity to each individual pillar.

    Returns a list of (pillar_date, price_change_per_bp) pairs.
    """
    base_price = pricer(curve)
    pillar_dates = curve.pillar_dates
    results = []
    for i, d in enumerate(pillar_dates):
        bumped = curve.bumped_at(i, bump_bps / 10000.0)
        delta = (pricer(bumped) - base_price) / bump_bps
        results.append((d, delta))
    return results
