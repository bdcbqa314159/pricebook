"""RateIndex / FixingHistory / accrued_rate oracles (L0) — Topic 0 Slice 5.

The declarative index identity (RateIndex is the first instance; a new index is a
DECLARATION, never code). Covers ALL rate kinds — backward-looking compounded RFR and
forward-looking term/IBOR (`observation_style`), plus `spread_adjustment` (ISDA
fallbacks). `FixingHistory` is generic over index. One generic `accrued_rate`, branching
only on `CompoundingMethod`.

Oracles: compounded RFR against a hand-computed series; lookback vs observation-shift
give DIFFERENT rates; 0/0 vs 2/2 differ; forward-looking term vs backward-looking
compounded differ; a fallback = base RFR + spread (spread not silently absorbed).
"""

from datetime import date

import pytest

from pricebook_ng.foundation.cashflow import Accrual
from pricebook_ng.foundation.day_count import DayCountConvention as DC
from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.rate_index import (
    CompoundingMethod,
    FixingHistory,
    ObservationStyle,
    RateIndex,
    accrued_rate,
    get_rate_index,
    list_rate_indices,
)

USD = Currency.USD


def _rfr(name="TEST_ON", *, obs_shift=0, lookback=0, lockout=0, compounding=CompoundingMethod.COMPOUNDED,
         spread=0.0):
    return RateIndex(
        name=name, currency=USD, tenor="ON", day_count=DC.ACT_360, fixing_lag=0,
        observation_shift=obs_shift, lookback=lookback, lockout=lockout, payment_delay=0,
        compounding=compounding, observation_style=ObservationStyle.BACKWARD_LOOKING,
        spread_adjustment=spread,
    )


def _flat(name, value, start, end):
    days, d = {}, date(start.year, 1, 1)
    while d <= end:
        days[d] = value
        d = date.fromordinal(d.toordinal() + 1)
    return FixingHistory({name: days})


def _series(name, base, start, end):
    """A rising series so a shifted observation window sees different rates."""
    days, d, i = {}, date(start.year, 1, 1), 0
    while d <= end:
        days[d] = base + 0.0001 * i
        i += 1
        d = date.fromordinal(d.toordinal() + 1)
    return FixingHistory({name: days})


# ── compounded RFR against a hand-computed series ──
def test_compounded_flat_series():
    idx = _rfr()  # 0/0 overnight, ACT/360
    acc = Accrual(date(2024, 6, 10), date(2024, 6, 14), DC.ACT_360)  # Mon..Fri, 4 biz days
    fx = _flat("TEST_ON", 0.05, date(2024, 6, 1), date(2024, 6, 30))
    assert accrued_rate(idx, acc, fx) == pytest.approx(0.050010417631, abs=1e-10)


# ── lookback vs observation-shift give DIFFERENT rates ──
def test_lookback_differs_from_observation_shift():
    acc = Accrual(date(2024, 6, 17), date(2024, 6, 24), DC.ACT_360)  # spans a weekend
    fx = _series("TEST_ON", 0.05, date(2024, 5, 1), date(2024, 7, 1))
    lb = accrued_rate(_rfr(lookback=2), acc, fx)
    os = accrued_rate(_rfr(obs_shift=2), acc, fx)
    assert lb != os


def test_zero_zero_differs_from_two_two():
    acc = Accrual(date(2024, 6, 17), date(2024, 6, 24), DC.ACT_360)
    fx = _series("TEST_ON", 0.05, date(2024, 5, 1), date(2024, 7, 1))
    assert accrued_rate(_rfr(obs_shift=0), acc, fx) != accrued_rate(_rfr(obs_shift=2), acc, fx)


# ── forward-looking term vs backward-looking compounded ──
def test_forward_looking_term_vs_backward_compounded():
    acc = Accrual(date(2024, 6, 10), date(2024, 9, 10), DC.ACT_360)
    term = RateIndex(
        name="TERM_3M", currency=USD, tenor="3M", day_count=DC.ACT_360, fixing_lag=2,
        observation_shift=0, lookback=0, lockout=0, payment_delay=0,
        compounding=CompoundingMethod.FLAT, observation_style=ObservationStyle.FORWARD_LOOKING,
    )
    fx = _flat("TERM_3M", 0.048, date(2024, 5, 1), date(2024, 9, 30))
    rfr_fx = _series("TEST_ON", 0.05, date(2024, 5, 1), date(2024, 10, 1))
    # forward term = the single fixing at the start; backward = compounded over the period
    assert accrued_rate(term, acc, fx) == pytest.approx(0.048, abs=1e-12)
    assert accrued_rate(_rfr(), acc, rfr_fx) != pytest.approx(0.048, abs=1e-6)


# ── fallback = base RFR + spread; the spread is not silently absorbed ──
def test_fallback_adds_spread():
    acc = Accrual(date(2024, 6, 17), date(2024, 6, 24), DC.ACT_360)
    fx = _series("TEST_ON", 0.05, date(2024, 5, 1), date(2024, 7, 1))
    base = accrued_rate(_rfr(spread=0.0), acc, fx)
    fallback = accrued_rate(_rfr(spread=0.0026161), acc, fx)
    assert fallback - base == pytest.approx(0.0026161, abs=1e-12)


# ── registry: a new index is a declaration; explicit construction (no import-time I/O) ──
def test_registry_has_standard_indices():
    sofr = get_rate_index("SOFR")
    assert sofr.currency is USD
    assert sofr.observation_style is ObservationStyle.BACKWARD_LOOKING
    assert get_rate_index("EURIBOR_3M").observation_style is ObservationStyle.FORWARD_LOOKING
    assert "SONIA" in list_rate_indices()


def test_unknown_index_raises():
    with pytest.raises(ValueError):
        get_rate_index("NOT_A_RATE")


# ── Brazilian exponential (BUS/252) compounding — CDI/SELIC, LTN/NTN-F/DI ──
def test_brazilian_exponential_compounding():
    # BRL rates compound EXPONENTIALLY on a business-day basis: (1+r)^(bd/252).
    # A flat rate reprices to itself EXACTLY — unlike money-market ∏(1+r·δ) compounding
    # (test_compounded_flat_series shows flat 5% → 0.05001, not 0.05).
    cdi = get_rate_index("CDI")                      # BRL, BUS/252, EXPONENTIAL
    acc = Accrual(date(2024, 6, 10), date(2024, 6, 14), DC.BUS_252)  # 4 São Paulo biz days
    fx = _flat("CDI", 0.10, date(2024, 6, 1), date(2024, 6, 30))
    assert accrued_rate(cdi, acc, fx) == pytest.approx(0.10, abs=1e-12)
    assert cdi.compounding is CompoundingMethod.EXPONENTIAL
