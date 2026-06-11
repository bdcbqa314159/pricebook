"""Tests for the new `pricebook.market_data` L1 types (G1 P2 Slice 1)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime
from uuid import UUID

import pytest

from pricebook.market_data import (
    FixingHistory,
    MarketSnapshot,
    Quote,
    QuoteId,
    QuoteKind,
)


# ============================================================
# QuoteKind
# ============================================================

class TestQuoteKind:
    def test_string_values(self):
        assert QuoteKind.DEPOSIT_RATE.value == "deposit_rate"
        assert QuoteKind.SWAP_RATE.value == "swap_rate"
        assert QuoteKind.CDS_SPREAD.value == "cds_spread"
        assert QuoteKind.OTHER.value == "other"

    def test_string_enum_roundtrip(self):
        k = QuoteKind.SWAPTION_VOL
        assert QuoteKind(k.value) is k


# ============================================================
# QuoteId
# ============================================================

class TestQuoteId:
    def test_construction(self):
        qid = QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD")
        assert qid.kind is QuoteKind.SWAP_RATE
        assert qid.tenor == "5Y"
        assert qid.currency == "USD"
        assert qid.label == ""

    def test_equality(self):
        a = QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD")
        b = QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD")
        c = QuoteId(QuoteKind.SWAP_RATE, "5Y", "EUR")
        assert a == b
        assert a != c

    def test_hashable(self):
        a = QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD")
        b = QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD")
        s = {a, b}
        assert len(s) == 1

    def test_label_disambiguates(self):
        a = QuoteId(QuoteKind.BOND_PRICE, "5Y", "USD", "ISIN_X")
        b = QuoteId(QuoteKind.BOND_PRICE, "5Y", "USD", "ISIN_Y")
        assert a != b
        assert hash(a) != hash(b) or a != b  # any difference shows up in equality

    def test_string_repr(self):
        a = QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD")
        assert str(a) == "swap_rate:5Y:USD"
        b = QuoteId(QuoteKind.BOND_PRICE, "5Y", "USD", "ISIN_X")
        assert str(b) == "bond_price:5Y:USD:ISIN_X"

    def test_frozen(self):
        a = QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD")
        with pytest.raises(FrozenInstanceError):
            a.tenor = "10Y"  # type: ignore[misc]


# ============================================================
# Quote
# ============================================================

class TestQuote:
    def test_construction(self):
        q = Quote(QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD"), value=0.04)
        assert q.value == 0.04
        assert q.bid_ask_bp is None

    def test_with_bid_ask(self):
        q = Quote(QuoteId(QuoteKind.CDS_SPREAD, "5Y", "USD"), value=0.012, bid_ask_bp=2.5)
        assert q.bid_ask_bp == 2.5

    def test_frozen(self):
        q = Quote(QuoteId(QuoteKind.SWAP_RATE, "5Y", "USD"), value=0.04)
        with pytest.raises(FrozenInstanceError):
            q.value = 0.05  # type: ignore[misc]


# ============================================================
# MarketSnapshot
# ============================================================

class TestMarketSnapshot:
    def _q(self, kind: QuoteKind, tenor: str, value: float, currency: str = "USD"):
        return Quote(QuoteId(kind, tenor, currency), value=value)

    def test_new_generates_id_and_timestamp(self):
        s = MarketSnapshot.new()
        assert isinstance(s.id, UUID)
        assert isinstance(s.as_of, datetime)
        assert s.quotes == ()
        assert s.label == ""

    def test_empty_factory(self):
        s = MarketSnapshot.empty(label="EOD")
        assert len(s) == 0
        assert s.label == "EOD"

    def test_unique_id_per_new(self):
        s1 = MarketSnapshot.new()
        s2 = MarketSnapshot.new()
        assert s1.id != s2.id

    def test_with_explicit_as_of(self):
        t = datetime(2026, 6, 11, 17, 0)
        s = MarketSnapshot.new(as_of=t)
        assert s.as_of == t

    def test_quotes_preserved_as_tuple(self):
        quotes = [
            self._q(QuoteKind.SWAP_RATE, "5Y", 0.04),
            self._q(QuoteKind.SWAP_RATE, "10Y", 0.045),
        ]
        s = MarketSnapshot.new(quotes=quotes)
        assert len(s) == 2
        assert s.quotes == tuple(quotes)
        # Iteration
        assert list(s) == quotes

    def test_get_by_qid(self):
        q1 = self._q(QuoteKind.SWAP_RATE, "5Y", 0.04)
        q2 = self._q(QuoteKind.SWAP_RATE, "10Y", 0.045)
        s = MarketSnapshot.new(quotes=[q1, q2])
        assert s.get(q1.id) is q1
        assert s.get(QuoteId(QuoteKind.CDS_SPREAD, "5Y", "USD")) is None

    def test_contains(self):
        q1 = self._q(QuoteKind.SWAP_RATE, "5Y", 0.04)
        s = MarketSnapshot.new(quotes=[q1])
        assert q1.id in s
        assert QuoteId(QuoteKind.CDS_SPREAD, "5Y", "USD") not in s
        assert "not a QuoteId" not in s

    def test_filter_by_kind(self):
        q1 = self._q(QuoteKind.SWAP_RATE, "5Y", 0.04)
        q2 = self._q(QuoteKind.SWAP_RATE, "10Y", 0.045)
        q3 = self._q(QuoteKind.CDS_SPREAD, "5Y", 0.012)
        s = MarketSnapshot.new(quotes=[q1, q2, q3])
        assert s.filter(kind=QuoteKind.SWAP_RATE) == (q1, q2)
        assert s.filter(kind=QuoteKind.CDS_SPREAD) == (q3,)

    def test_filter_by_currency(self):
        q1 = self._q(QuoteKind.SWAP_RATE, "5Y", 0.04, currency="USD")
        q2 = self._q(QuoteKind.SWAP_RATE, "5Y", 0.025, currency="EUR")
        s = MarketSnapshot.new(quotes=[q1, q2])
        assert s.filter(currency="USD") == (q1,)
        assert s.filter(currency="EUR") == (q2,)

    def test_with_quote_returns_new_snapshot(self):
        q1 = self._q(QuoteKind.SWAP_RATE, "5Y", 0.04)
        q2 = self._q(QuoteKind.SWAP_RATE, "10Y", 0.045)
        s1 = MarketSnapshot.new(quotes=[q1])
        s2 = s1.with_quote(q2)
        # Original unchanged
        assert s1.quotes == (q1,)
        # New has both
        assert len(s2) == 2
        assert q2.id in s2

    def test_with_quote_fresh_id(self):
        q1 = self._q(QuoteKind.SWAP_RATE, "5Y", 0.04)
        s1 = MarketSnapshot.new(quotes=[q1])
        s2 = s1.with_quote(self._q(QuoteKind.SWAP_RATE, "10Y", 0.045))
        # New snapshot has a fresh id — it is a different snapshot
        assert s1.id != s2.id

    def test_with_quote_replaces_same_id(self):
        q1 = self._q(QuoteKind.SWAP_RATE, "5Y", 0.04)
        q1_revised = Quote(q1.id, value=0.041)  # same id, new value
        s1 = MarketSnapshot.new(quotes=[q1])
        s2 = s1.with_quote(q1_revised)
        assert len(s2) == 1
        assert s2.get(q1.id) is q1_revised
        # Original unchanged
        assert s1.get(q1.id) is q1

    def test_frozen(self):
        s = MarketSnapshot.new()
        with pytest.raises(FrozenInstanceError):
            s.label = "x"  # type: ignore[misc]


# ============================================================
# FixingHistory
# ============================================================

class TestFixingHistory:
    def test_empty_default(self):
        h = FixingHistory()
        assert h.fixings == {}
        assert h.rate_names() == set()
        assert h.get("SOFR", date(2026, 6, 11)) is None

    def test_get(self):
        d = date(2026, 6, 11)
        h = FixingHistory(fixings={("SOFR", d): 0.053})
        assert h.get("SOFR", d) == 0.053
        assert h.get("EURIBOR", d) is None

    def test_for_rate(self):
        d1, d2 = date(2026, 6, 10), date(2026, 6, 11)
        h = FixingHistory(fixings={
            ("SOFR", d1): 0.053,
            ("SOFR", d2): 0.054,
            ("EURIBOR_3M", d1): 0.025,
        })
        assert h.for_rate("SOFR") == {d1: 0.053, d2: 0.054}
        assert h.for_rate("EURIBOR_3M") == {d1: 0.025}
        assert h.for_rate("XXX") == {}

    def test_rate_names(self):
        d = date(2026, 6, 11)
        h = FixingHistory(fixings={
            ("SOFR", d): 0.053,
            ("EURIBOR_3M", d): 0.025,
        })
        assert h.rate_names() == {"SOFR", "EURIBOR_3M"}

    def test_with_fixing_returns_new(self):
        d = date(2026, 6, 11)
        h1 = FixingHistory()
        h2 = h1.with_fixing("SOFR", d, 0.053)
        assert h1.fixings == {}        # original unchanged
        assert h2.get("SOFR", d) == 0.053

    def test_with_fixing_replaces(self):
        d = date(2026, 6, 11)
        h1 = FixingHistory(fixings={("SOFR", d): 0.053})
        h2 = h1.with_fixing("SOFR", d, 0.054)   # same key, new value
        assert h2.get("SOFR", d) == 0.054
        assert h1.get("SOFR", d) == 0.053       # original unchanged

    def test_frozen(self):
        h = FixingHistory()
        with pytest.raises(FrozenInstanceError):
            h.fixings = {}  # type: ignore[misc]
