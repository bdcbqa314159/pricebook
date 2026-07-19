"""RateIndex oracles (L0) — Topic 0 gate rework (F2/F3/F4).

The decomposed index (IndexId/AccrualConvention/FixingRule/RfrConvention). **F2**: the
calendar is the index's own (`accrual.roll.calendar`), never inferred from currency — SOFR
fixes on SIFMA, and two same-currency indices on different calendars differ. Plus the
compounding oracles (compounded/exponential; lookback ≠ observation-shift; forward ≠
backward; fallback = base + spread) carried across.
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
    exponential_growth,
    get_rate_index,
    list_rate_indices,
)
from pricebook_ng.foundation.schedule import RollRule
from pricebook_ng.foundation.tenor import Tenor, TenorUnit
from pricebook_ng.foundation.underlying import AssetClass, Underlying

USD = Currency.USD


def _rfr(name="TEST_ON", cal="US_GOVERNMENT_SECURITIES", *, obs_shift=0, lookback=0, lockout=0,
         compounding=AccrualMethod.COMPOUNDED, spread=0.0, dc=DC.ACT_360):
    return RateIndex(
        IndexId(name, USD, Tenor(1, TenorUnit.DAY)),
        AccrualConvention(dc, RollRule(get_calendar(cal), BDC.MODIFIED_FOLLOWING, eom=False)),
        FixingRule(ObservationStyle.BACKWARD_LOOKING, compounding, fixing_lag=0),
        RfrConvention(observation_shift=obs_shift, lookback=lookback, lockout=lockout),
        spread_adjustment=spread,
    )


def _flat(name, value, start, end):
    days, d = {}, date(start.year, 1, 1)
    while d <= end:
        days[d] = value
        d = date.fromordinal(d.toordinal() + 1)
    return FixingHistory({name: days})


def _series(name, base, start, end):
    days, d, i = {}, date(start.year, 1, 1), 0
    while d <= end:
        days[d] = base + 0.0001 * i
        i, d = i + 1, date.fromordinal(d.toordinal() + 1)
    return FixingHistory({name: days})


# ── F2 (the regression): the calendar is the INDEX's, not the currency's ──
def test_calendar_comes_from_the_index_not_the_currency():
    # SOFR fixes on SIFMA (Columbus Day 2024-10-14 is a holiday); a same-currency index on
    # TARGET (10-14 is a business day there) observes a different set of days over the period.
    acc = Accrual(date(2024, 10, 10), date(2024, 10, 18), DC.ACT_360)   # spans Columbus Day
    fx = _series("X_ON", 0.05, date(2024, 9, 1), date(2024, 11, 1))
    sifma = _rfr("X_ON", "US_GOVERNMENT_SECURITIES")
    target = _rfr("X_ON", "TARGET")                                     # same USD, other calendar
    assert accrued_rate(sifma, acc, fx) != accrued_rate(target, acc, fx)


def test_sofr_uses_sifma():
    assert get_rate_index("SOFR").accrual.roll.calendar is get_calendar("US_GOVERNMENT_SECURITIES")


# ── F4: RateIndex is an Underlying; siblings report their asset class ──
def test_rate_index_is_an_underlying():
    sofr = get_rate_index("SOFR")
    assert isinstance(sofr, Underlying)
    assert sofr.name == "SOFR"
    assert sofr.asset_class is AssetClass.RATES


# ── compounded RFR against a hand-computed series ──
def test_compounded_flat_series():
    acc = Accrual(date(2024, 6, 10), date(2024, 6, 14), DC.ACT_360)     # 4 SIFMA biz days
    fx = _flat("TEST_ON", 0.05, date(2024, 6, 1), date(2024, 6, 30))
    assert accrued_rate(_rfr(), acc, fx) == pytest.approx(0.050010417631, abs=1e-10)


def test_lookback_differs_from_observation_shift():
    acc = Accrual(date(2024, 6, 17), date(2024, 6, 24), DC.ACT_360)
    fx = _series("TEST_ON", 0.05, date(2024, 5, 1), date(2024, 7, 1))
    assert accrued_rate(_rfr(lookback=2), acc, fx) != accrued_rate(_rfr(obs_shift=2), acc, fx)


def test_zero_zero_differs_from_two_two():
    acc = Accrual(date(2024, 6, 17), date(2024, 6, 24), DC.ACT_360)
    fx = _series("TEST_ON", 0.05, date(2024, 5, 1), date(2024, 7, 1))
    assert accrued_rate(_rfr(obs_shift=0), acc, fx) != accrued_rate(_rfr(obs_shift=2), acc, fx)


# ── forward-looking term vs backward-looking compounded ──
def test_forward_looking_term_vs_backward_compounded():
    acc = Accrual(date(2024, 6, 10), date(2024, 9, 10), DC.ACT_360)
    term = RateIndex(
        IndexId("TERM_3M", USD, Tenor.parse("3M")),
        AccrualConvention(DC.ACT_360, RollRule(get_calendar("US_GOVERNMENT_SECURITIES"), BDC.MODIFIED_FOLLOWING, eom=False)),
        FixingRule(ObservationStyle.FORWARD_LOOKING, AccrualMethod.FLAT, fixing_lag=2),
        RfrConvention.none(),
    )
    fx = _flat("TERM_3M", 0.048, date(2024, 5, 1), date(2024, 9, 30))
    rfr_fx = _series("TEST_ON", 0.05, date(2024, 5, 1), date(2024, 10, 1))
    assert accrued_rate(term, acc, fx) == pytest.approx(0.048, abs=1e-12)
    assert accrued_rate(_rfr(), acc, rfr_fx) != pytest.approx(0.048, abs=1e-6)


def test_fallback_adds_spread():
    acc = Accrual(date(2024, 6, 17), date(2024, 6, 24), DC.ACT_360)
    fx = _series("TEST_ON", 0.05, date(2024, 5, 1), date(2024, 7, 1))
    base = accrued_rate(_rfr(spread=0.0), acc, fx)
    fallback = accrued_rate(_rfr(spread=0.0026161), acc, fx)
    assert fallback - base == pytest.approx(0.0026161, abs=1e-12)


# ── Brazilian exponential (BUS/252) ──
def test_brazilian_exponential_compounding():
    cdi = get_rate_index("CDI")                       # BRL, BUS/252, EXPONENTIAL, São Paulo
    acc = Accrual(date(2024, 6, 10), date(2024, 6, 14), DC.BUS_252)
    fx = _flat("CDI", 0.10, date(2024, 6, 1), date(2024, 6, 30))
    assert accrued_rate(cdi, acc, fx) == pytest.approx(0.10, abs=1e-12)
    assert cdi.fixing.compounding is AccrualMethod.EXPONENTIAL


def test_exponential_growth_single_rate():
    assert exponential_growth(0.10, 252, DC.BUS_252) == pytest.approx(1.10, abs=1e-12)
    assert exponential_growth(0.10, 4, DC.BUS_252) == pytest.approx(1.10 ** (4 / 252), abs=1e-15)
    with pytest.raises(ValueError):
        exponential_growth(0.10, 4, DC.ACT_360)


# ── registry ──
def test_registry_and_declared_kinds():
    assert get_rate_index("SOFR").id.currency is USD
    assert get_rate_index("EURIBOR_3M").fixing.observation_style is ObservationStyle.FORWARD_LOOKING
    assert "SONIA" in list_rate_indices()
    with pytest.raises(ValueError):
        get_rate_index("NOT_A_RATE")
