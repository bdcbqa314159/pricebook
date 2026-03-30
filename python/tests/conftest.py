"""Shared test fixtures and helpers."""

import math
from datetime import date

from pricebook.discount_curve import DiscountCurve


def make_flat_curve(ref: date, rate: float) -> DiscountCurve:
    """Build a flat discount curve at the given continuously compounded rate."""
    tenors_years = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors_years]
    dfs = [math.exp(-rate * t) for t in tenors_years]
    return DiscountCurve(ref, dates, dfs)
