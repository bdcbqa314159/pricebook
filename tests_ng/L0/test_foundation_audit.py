"""Foundation-audit oracles (L0) — fix/foundation-audit findings F1–F3.

Each finding is proven by a failing test before its fix (an assertion without a red test is
an opinion). Verified against ARRC/ISDA RFR conventions and the NY Fed SOFR methodology.
Severity order: silent-wrongness (F2, F1-COMPOUNDED) > wrong-answer (F3) > crash (F1-others).
"""

from datetime import date

import pytest

from pricebook_ng.foundation.calendars import BusinessDayConvention as BDC
from pricebook_ng.foundation.day_count import Accrual
from pricebook_ng.foundation.day_count import DayCountConvention as DC
from pricebook_ng.foundation.market_calendars import get_calendar
from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.rate_index import (
    AccrualConvention,
    AccrualMethod,
    FixingHistory,
    FixingRule,
    IndexId,
    ObservationStyle,
    RateIndex,
    RfrConvention,
    accrued_rate,
)
from pricebook_ng.foundation.schedule import RollRule
from pricebook_ng.foundation.tenor import Tenor, TenorUnit

USD = Currency.USD
NY = get_calendar("NEW_YORK_SIFMA")


def _idx(name, comp, *, obs=0, lb=0, dc=DC.ACT_360):
    return RateIndex(
        IndexId(name, USD, Tenor(1, TenorUnit.DAY)),
        AccrualConvention(dc, RollRule(NY, BDC.MODIFIED_FOLLOWING, eom=False)),
        FixingRule(ObservationStyle.BACKWARD_LOOKING, comp, fixing_lag=0),
        RfrConvention(observation_shift=obs, lookback=lb),
    )


def _flat(name, val):
    days, d = {}, date(2024, 1, 1)
    while d <= date(2024, 12, 31):
        days[d] = val
        d = date.fromordinal(d.toordinal() + 1)
    return FixingHistory({name: days})


# ── F2 (silent-wrongness): observation_shift numerator/denominator over the SAME window ──
def test_f2_observation_shift_denominator_matches_numerator_window():
    # Jun 21–28 2024: the 2-business-day shift on the START crosses Juneteenth (Wed Jun 19),
    # so the shifted observation window (8 cal days) spans one more day than the interest
    # period (7). A flat 5% series must still realize ~5%, not 5%·8/7 ≈ 5.72%.
    acc = Accrual(date(2024, 6, 21), date(2024, 6, 28), DC.ACT_360)
    r = accrued_rate(_idx("F2", AccrualMethod.COMPOUNDED, obs=2), acc, _flat("F2", 0.05))
    assert r == pytest.approx(0.05, abs=1e-3)   # bug gives 0.05717


# ── F1 (silent-wrongness + crash): degenerate window RAISES cleanly (S14) ──
@pytest.mark.parametrize("comp", [AccrualMethod.COMPOUNDED, AccrualMethod.AVERAGED, AccrualMethod.EXPONENTIAL])
def test_f1_degenerate_window_raises_value_error(comp):
    # [Sat 2024-01-06, Mon 2024-01-08) is a valid accrual (start < end) but contains no
    # business day. COMPOUNDED currently returns 0.0; AVERAGED/EXPONENTIAL ZeroDivisionError.
    dc = DC.BUS_252 if comp is AccrualMethod.EXPONENTIAL else DC.ACT_360
    acc = Accrual(date(2024, 1, 6), date(2024, 1, 8), dc)
    with pytest.raises(ValueError):
        accrued_rate(_idx("F1", comp, dc=dc), acc, _flat("F1", 0.05))


# ── F3 (wrong-answer): add_business_days(d, 0) on a non-business day is undefined → raise ──
def test_f3_add_business_days_zero_on_non_business_day_raises():
    with pytest.raises(ValueError):
        NY.add_business_days(date(2024, 1, 6), 0)   # Saturday — not a business day


def test_f3_add_business_days_zero_on_business_day_returns_it():
    assert NY.add_business_days(date(2024, 1, 8), 0) == date(2024, 1, 8)   # Monday — unchanged
