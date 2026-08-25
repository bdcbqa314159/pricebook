"""L0 audit Batch D — #11 (spot_lag direction), #12 (interpolate ascending), #18 (pay-lag no calendar).

All three are no-silent-fallback guards (§2): a value that was silently wrong now resolves correctly
or raises clearly.
"""

from datetime import date

import pytest

from pricebook_ng.foundation import (
    Currency,
    CurrencyPair,
    Frequency,
    Interpolation,
    PaymentRule,
    RollRule,
    ScheduleTerms,
    build_schedule,
    interpolate,
    spot_lag,
)


def test_spot_lag_is_direction_independent() -> None:  # #11, repro_R
    forward = CurrencyPair(Currency.USD, Currency.CAD)  # "USDCAD" — the T+1 market convention
    reverse = CurrencyPair(Currency.CAD, Currency.USD)  # "CADUSD" — same pair, declared the other way
    assert spot_lag(forward) == 1
    assert spot_lag(reverse) == 1  # was 2 (the name-keyed lookup missed)


def test_interpolate_rejects_descending_xs() -> None:  # #12 — the ascending guard, not extrapolation-RAISE
    with pytest.raises(ValueError, match="ascending"):
        interpolate((3.0, 2.0, 1.0), (0.9, 0.95, 0.99), 2.5, Interpolation.LINEAR)


def test_pay_lag_without_calendar_raises() -> None:  # #18
    terms = ScheduleTerms(
        frequency=Frequency.ANNUAL,
        roll=RollRule(calendar=None),
        payment=PaymentRule(calendar=None, lag=5),  # a business-day lag with no calendar to count on
    )
    with pytest.raises(ValueError):
        build_schedule(date(2026, 1, 15), date(2029, 1, 15), terms)
