"""S1 oracle — day-count year fractions against ISDA/ICMA vectors.

Each expected value is written as the convention's own defining arithmetic
(day split / denominator), not a transcribed decimal — so the oracle is the
published ISDA 2006 (s.4.16) / ICMA (Rule 251) definition itself, exact to
machine precision. Calendar-dependent BUS/252 is deferred with the schedule
/ calendar slice.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction

ABS = 1e-12


# ---- ACT/360 ------------------------------------------------------------------
@pytest.mark.parametrize("start,end,expected", [
    (date(2024, 1, 15), date(2024, 7, 15), 182 / 360),   # 182 actual days
    (date(2024, 1, 1), date(2025, 1, 1), 366 / 360),     # leap year
    (date(2024, 1, 1), date(2024, 2, 1), 31 / 360),
])
def test_act_360(start, end, expected):
    assert year_fraction(start, end, DC.ACT_360) == pytest.approx(expected, abs=ABS)


# ---- ACT/365F -----------------------------------------------------------------
@pytest.mark.parametrize("start,end,expected", [
    (date(2024, 1, 15), date(2024, 7, 15), 182 / 365),
    (date(2024, 1, 1), date(2025, 1, 1), 366 / 365),     # leap: still /365
    (date(2023, 1, 1), date(2024, 1, 1), 365 / 365),
])
def test_act_365_fixed(start, end, expected):
    assert year_fraction(start, end, DC.ACT_365_FIXED) == pytest.approx(expected, abs=ABS)


# ---- 30/360 (US bond basis) ---------------------------------------------------
@pytest.mark.parametrize("start,end,expected", [
    (date(2024, 1, 15), date(2024, 7, 15), 180 / 360),           # 6*30
    (date(2024, 1, 1), date(2025, 1, 1), 1.0),
    (date(2024, 1, 31), date(2024, 3, 31), 60 / 360),            # d1 31->30, d2 31->30
    (date(2024, 1, 15), date(2024, 3, 31), 76 / 360),            # d2 31 stays (d1!=30)
    (date(2024, 2, 29), date(2024, 8, 31), 180 / 360),           # last-day-Feb -> 30; d2 31->30
])
def test_thirty_360(start, end, expected):
    assert year_fraction(start, end, DC.THIRTY_360) == pytest.approx(expected, abs=ABS)


# ---- 30E/360 (Eurobond basis) -------------------------------------------------
@pytest.mark.parametrize("start,end,expected", [
    (date(2007, 1, 15), date(2007, 2, 15), 30 / 360),
    (date(2024, 1, 31), date(2024, 3, 31), 60 / 360),            # both 31->30 unconditionally
    (date(2006, 8, 31), date(2007, 2, 28), 178 / 360),           # 6*30 + (28-30)
])
def test_thirty_e_360(start, end, expected):
    assert year_fraction(start, end, DC.THIRTY_E_360) == pytest.approx(expected, abs=ABS)


# ---- ACT/ACT ISDA -------------------------------------------------------------
def test_act_act_isda_canonical_isda_example():
    """ISDA 2006 s.4.16(b) worked example: 2003-11-01 -> 2004-05-01.

    Split at the year boundary: 61 days in 2003 (/365) + 121 days in leap
    2004 (/366) = 0.497724380...  (the published value).
    """
    yf = year_fraction(date(2003, 11, 1), date(2004, 5, 1), DC.ACT_ACT_ISDA)
    assert yf == pytest.approx(61 / 365 + 121 / 366, abs=ABS)


@pytest.mark.parametrize("start,end,expected", [
    (date(2024, 3, 1), date(2024, 9, 1), 184 / 366),             # wholly within leap year
    (date(2023, 3, 1), date(2023, 9, 1), 184 / 365),             # wholly within non-leap
    # spans a full calendar year 2023 plus fractions either side
    (date(2022, 7, 1), date(2024, 7, 1), 184 / 365 + 1.0 + 182 / 366),
])
def test_act_act_isda(start, end, expected):
    assert year_fraction(start, end, DC.ACT_ACT_ISDA) == pytest.approx(expected, abs=ABS)


# ---- ACT/ACT ICMA (Rule 251.1) ------------------------------------------------
def test_icma_regular_semi_annual_is_exactly_half():
    """A regular semi-annual coupon period yields exactly 0.5 regardless of the
    period's actual length — the defining property of ACT/ACT ICMA."""
    for ref_start, ref_end in [
        (date(2024, 2, 15), date(2024, 8, 15)),   # 182 days
        (date(2024, 8, 15), date(2025, 2, 15)),   # 184 days
    ]:
        yf = year_fraction(
            ref_start, ref_end, DC.ACT_ACT_ICMA,
            ref_start=ref_start, ref_end=ref_end, frequency=2,
        )
        assert yf == pytest.approx(0.5, abs=ABS)


def test_icma_mid_period_accrual():
    ref_start, ref_end = date(2024, 2, 15), date(2024, 8, 15)  # 182 days
    mid = date(2024, 5, 15)                                    # 90 days in
    expected = (mid - ref_start).days / ((ref_end - ref_start).days * 2)
    yf = year_fraction(
        ref_start, mid, DC.ACT_ACT_ICMA,
        ref_start=ref_start, ref_end=ref_end, frequency=2,
    )
    assert yf == pytest.approx(expected, abs=ABS)


# ---- Debt shed: ICMA anchors are required (no silent ACT/365F fallback) --------
@pytest.mark.parametrize("kwargs", [
    {},                                                       # no anchors at all
    {"ref_start": date(2024, 2, 15), "frequency": 2},        # missing ref_end
    {"ref_start": date(2024, 2, 15), "ref_end": date(2024, 8, 15)},  # missing frequency
])
def test_icma_missing_anchors_raises(kwargs):
    with pytest.raises(ValueError):
        year_fraction(date(2024, 2, 15), date(2024, 8, 15), DC.ACT_ACT_ICMA, **kwargs)


@pytest.mark.parametrize("frequency,ref_start,ref_end", [
    (0, date(2024, 2, 15), date(2024, 8, 15)),               # frequency <= 0
    (2, date(2024, 8, 15), date(2024, 2, 15)),               # inverted period
])
def test_icma_invalid_anchors_raise(frequency, ref_start, ref_end):
    with pytest.raises(ValueError):
        year_fraction(
            date(2024, 2, 15), date(2024, 8, 15), DC.ACT_ACT_ICMA,
            ref_start=ref_start, ref_end=ref_end, frequency=frequency,
        )


# ---- Edge cases common to all -------------------------------------------------
def test_same_date_is_zero():
    d = date(2024, 6, 15)
    for conv in [DC.ACT_360, DC.ACT_365_FIXED, DC.THIRTY_360, DC.THIRTY_E_360, DC.ACT_ACT_ISDA]:
        assert year_fraction(d, d, conv) == 0.0


def test_start_after_end_raises():
    with pytest.raises(ValueError):
        year_fraction(date(2024, 7, 1), date(2024, 1, 1), DC.ACT_360)
