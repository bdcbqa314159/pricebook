"""Tests for treasury quoting: 32nds, reopenings, delivery options."""
from __future__ import annotations
from datetime import date
import math
import pytest
from pricebook.fixed_income.treasury_quoting import (
    to_32nds, from_32nds, tick_value, tick_value_half,
    TreasuryReopen, delivery_option_value,
)


class Test32nds:
    def test_to_32nds_exact(self):
        assert to_32nds(99.50) == "99-16"
        assert to_32nds(100.0) == "100-00"
        assert to_32nds(98.0) == "98-00"

    def test_to_32nds_plus(self):
        assert to_32nds(99.515625) == "99-16+"

    def test_from_32nds_exact(self):
        assert from_32nds("99-16") == 99.50
        assert from_32nds("100-00") == 100.0
        assert from_32nds("98-08") == 98.25

    def test_from_32nds_plus(self):
        assert from_32nds("99-16+") == pytest.approx(99.515625)
        assert from_32nds("98-08+") == pytest.approx(98.265625)

    def test_roundtrip(self):
        for price in [99.0, 99.25, 99.50, 99.75, 100.0, 100.5, 101.0,
                       99.515625, 98.265625, 102.015625]:
            quote = to_32nds(price)
            back = from_32nds(quote)
            assert back == pytest.approx(price, abs=1/128)  # within half-32nd

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            from_32nds("99.50")

    def test_invalid_ticks_raises(self):
        with pytest.raises(ValueError):
            from_32nds("99-35")

    def test_tick_value(self):
        assert tick_value(1_000_000) == pytest.approx(312.50)

    def test_tick_value_half(self):
        assert tick_value_half(1_000_000) == pytest.approx(156.25)


class TestTreasuryReopen:
    def test_premium_reopen(self):
        r = TreasuryReopen(
            original_issue=date(2024, 2, 15), reopen_date=date(2024, 5, 15),
            original_coupon=0.04, reopen_yield=0.035,
            original_outstanding=40e9, reopen_amount=20e9,
        )
        assert r.is_premium
        assert not r.is_discount
        assert r.total_outstanding == 60e9
        assert r.reopen_price_approx > 100

    def test_discount_reopen(self):
        r = TreasuryReopen(
            original_issue=date(2024, 2, 15), reopen_date=date(2024, 5, 15),
            original_coupon=0.04, reopen_yield=0.045,
            original_outstanding=40e9, reopen_amount=20e9,
        )
        assert r.is_discount
        assert r.reopen_price_approx < 100

    def test_to_dict(self):
        r = TreasuryReopen(date(2024, 1, 1), date(2024, 4, 1), 0.04, 0.04, 40e9, 20e9)
        d = r.to_dict()
        assert "coupon" in d
        assert "total_outstanding" in d


class TestDeliveryOption:
    def test_positive_total(self):
        r = delivery_option_value(0.50, 0.30)
        assert r.total_option_value > 0

    def test_net_basis(self):
        r = delivery_option_value(0.50, 0.30)
        assert r.ctd_net_basis == pytest.approx(0.20)

    def test_more_deliverables_higher_quality(self):
        r_few = delivery_option_value(0.50, 0.30, n_deliverables=3)
        r_many = delivery_option_value(0.50, 0.30, n_deliverables=10)
        assert r_many.quality_option > r_few.quality_option

    def test_higher_vol_higher_wild_card(self):
        r_low = delivery_option_value(0.50, 0.30, futures_vol=0.02)
        r_high = delivery_option_value(0.50, 0.30, futures_vol=0.08)
        assert r_high.wild_card_option > r_low.wild_card_option
