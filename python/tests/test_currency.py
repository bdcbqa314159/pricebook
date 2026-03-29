"""Tests for currency and currency pair."""

import pytest

from pricebook.currency import Currency, CurrencyPair


class TestCurrencyPair:

    def test_explicit_construction(self):
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        assert pair.base == Currency.EUR
        assert pair.quote == Currency.USD

    def test_repr(self):
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        assert repr(pair) == "EUR/USD"

    def test_same_currency_raises(self):
        with pytest.raises(ValueError):
            CurrencyPair(Currency.USD, Currency.USD)


class TestMarketConvention:

    def test_eur_usd(self):
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        assert pair.base == Currency.EUR
        assert pair.quote == Currency.USD

    def test_usd_eur_gives_eur_usd(self):
        """USD/EUR should flip to EUR/USD by market convention."""
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.EUR)
        assert pair.base == Currency.EUR

    def test_gbp_usd(self):
        pair = CurrencyPair.from_currencies(Currency.GBP, Currency.USD)
        assert pair.base == Currency.GBP
        assert pair.quote == Currency.USD

    def test_eur_gbp(self):
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.GBP)
        assert pair.base == Currency.EUR
        assert pair.quote == Currency.GBP

    def test_from_currencies_same_raises(self):
        with pytest.raises(ValueError):
            CurrencyPair.from_currencies(Currency.USD, Currency.USD)


class TestInvert:

    def test_invert(self):
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        inv = pair.invert()
        assert inv.base == Currency.USD
        assert inv.quote == Currency.EUR

    def test_double_invert_recovers(self):
        pair = CurrencyPair(Currency.GBP, Currency.USD)
        assert pair.invert().invert() == pair


class TestEquality:

    def test_equal(self):
        a = CurrencyPair(Currency.EUR, Currency.USD)
        b = CurrencyPair(Currency.EUR, Currency.USD)
        assert a == b

    def test_not_equal(self):
        a = CurrencyPair(Currency.EUR, Currency.USD)
        b = CurrencyPair(Currency.GBP, Currency.USD)
        assert a != b

    def test_hashable(self):
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        d = {pair: 1.10}
        assert d[pair] == 1.10
