"""Tenor oracles (L0) — Topic 0 gate rework (S7).

`Tenor(count, unit)` is the value type behind index tenors, schedule steps and curve
pillars — so `"28D"` is parsed once, not in three modules. Replaces the ruled-away
"Tenor stays a string" (S3 overturned it).
"""

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
