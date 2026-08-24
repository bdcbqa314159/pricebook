"""L1 audit Batch C — #4: Surface interpolates TOTAL variance (no calendar arbitrage). repro_D.

Total variance `w = σ²·T` must be non-decreasing across expiry; the old linear-in-σ² scheme let it
DECREASE (negative forward variance). `at` now takes the pricer's canonical `TimeMeasure` so the
surface T-axis and the pricer's expiry-t are one source (§3d).
"""

from datetime import date

import pytest

from pricebook_ng.foundation import DayCountConvention, Interpolation, TimeMeasure
from pricebook_ng.market.vol_surface import Surface

VAL = date(2025, 1, 1)
TM = TimeMeasure(VAL, DayCountConvention.ACT_365_FIXED)
# repro_D: a downward-sloping term structure (normal in rates)
DOWN = Surface((0.40, 0.20), (date(2026, 1, 1), date(2031, 1, 1)))


def test_total_variance_is_non_decreasing() -> None:  # repro_D — no calendar arbitrage
    prev_w = -1.0
    for e in (date(2026, 1, 1), date(2027, 7, 1), date(2029, 1, 1), date(2030, 7, 1), date(2031, 1, 1)):
        sig = DOWN.at(e, 0.03, TM)
        w = sig * sig * TM.year_fraction(e)  # total variance
        assert w >= prev_w - 1e-12  # non-decreasing (was decreasing past the peak)
        prev_w = w


def test_non_linear_surface_is_rejected() -> None:
    with pytest.raises(ValueError):
        Surface((0.2, 0.25), (date(2026, 1, 1), date(2031, 1, 1)), Interpolation.CUBIC_SPLINE)


def test_flat_surface_ignores_time_measure() -> None:  # slices 1–2 byte-identical
    assert Surface.flat(0.2).at(date(2030, 6, 1), 0.03, TM) == 0.2
