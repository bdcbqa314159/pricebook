"""L1 oracle — FX spot in the snapshot, directional (doc 19 §2.1, slice 6a).

The snapshot's `scalars` shape holds FX spots keyed by pair in the declared canonical order;
`fx_rate(base, quote)` resolves the declared pair, inverts internally, and raises on an
undeclared cross. A rate is never a bare, direction-ambiguous pair-scalar.
"""

from datetime import date

import pytest

from pricebook_ng.foundation import Currency, CurrencyPair, register_fx_pair
from pricebook_ng.market.curve_set import CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot, ScalarKey

EURUSD = CurrencyPair(Currency.EUR, Currency.USD)  # declared canonical: EUR base, USD quote


def _snapshot(spot: float) -> MarketSnapshot:
    return MarketSnapshot(date(2026, 1, 15), CurveSet({}), {ScalarKey(EURUSD): spot})


def test_fx_rate_round_trips_to_one() -> None:
    s = _snapshot(1.08)
    assert abs(s.fx_rate(Currency.EUR, Currency.USD) * s.fx_rate(Currency.USD, Currency.EUR) - 1.0) < 1e-15


def test_fx_rate_is_directional() -> None:
    s = _snapshot(1.08)
    assert s.fx_rate(Currency.EUR, Currency.USD) == 1.08  # canonical direction: stored value
    assert abs(s.fx_rate(Currency.USD, Currency.EUR) - 1.0 / 1.08) < 1e-15  # reverse: reciprocal


def test_undeclared_cross_raises() -> None:
    s = _snapshot(1.08)
    with pytest.raises(ValueError):
        s.fx_rate(Currency.GBP, Currency.JPY)  # never guesses a direction


def test_register_fx_pair_guards() -> None:
    with pytest.raises(ValueError):
        register_fx_pair(Currency.EUR, Currency.USD)  # already declared at import
    with pytest.raises(ValueError):
        register_fx_pair(Currency.GBP, Currency.GBP)  # a pair needs two currencies
