"""Tenor oracles (L0) — Topic 0 gate rework (S7).

`Tenor(count, unit)` is the value type behind index tenors, schedule steps and curve
pillars — so `"28D"` is parsed once, not in three modules. Replaces the ruled-away
"Tenor stays a string" (S3 overturned it).
"""

from datetime import date

import pytest

from pricebook_ng.foundation.tenor import Tenor, TenorUnit


def test_parse_and_str_round_trip():
    for s, count, unit in [("3M", 3, TenorUnit.MONTH), ("28D", 28, TenorUnit.DAY),
                           ("2W", 2, TenorUnit.WEEK), ("1Y", 1, TenorUnit.YEAR)]:
        t = Tenor.parse(s)
        assert t == Tenor(count, unit)
        assert str(t) == s


def test_months_for_month_and_year():
    assert Tenor(6, TenorUnit.MONTH).months() == 6
    assert Tenor(1, TenorUnit.YEAR).months() == 12
    with pytest.raises(ValueError):
        Tenor(28, TenorUnit.DAY).months()   # days don't convert to a fixed month count


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        Tenor.parse("3X")


# ── S7: date + Tenor (the most-used operation in curve building) ──
def test_date_plus_tenor_days_and_weeks():
    assert date(2024, 1, 1) + Tenor(28, TenorUnit.DAY) == date(2024, 1, 29)
    assert date(2024, 1, 1) + Tenor(2, TenorUnit.WEEK) == date(2024, 1, 15)


def test_date_plus_tenor_months_and_years():
    assert date(2024, 1, 15) + Tenor(3, TenorUnit.MONTH) == date(2024, 4, 15)
    assert date(2024, 2, 15) + Tenor(1, TenorUnit.YEAR) == date(2025, 2, 15)


def test_date_plus_tenor_clamps_short_month():
    # 31 Jan + 1M has no 31 Feb → clamps to the last day of the target month
    assert date(2024, 1, 31) + Tenor(1, TenorUnit.MONTH) == date(2024, 2, 29)  # 2024 leap
    assert date(2023, 1, 31) + Tenor(1, TenorUnit.MONTH) == date(2023, 2, 28)


def test_tenor_add_is_commutative_via_radd():
    # `date + Tenor` works via __radd__ (date.__add__ can't handle a Tenor)
    assert date(2024, 6, 1) + Tenor(6, TenorUnit.MONTH) == date(2024, 12, 1)
