"""Shared test fixtures and helpers."""

from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


def make_flat_curve(ref: date, rate: float) -> DiscountCurve:
    """Build a flat discount curve at the given continuously compounded rate."""
    return DiscountCurve.flat(ref, rate)


def make_flat_survival(ref: date, hazard: float) -> SurvivalCurve:
    """Build a flat survival curve at the given constant hazard rate."""
    return SurvivalCurve.flat(ref, hazard)
