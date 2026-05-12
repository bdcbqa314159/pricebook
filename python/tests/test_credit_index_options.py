"""Tests for credit index options."""
from __future__ import annotations
import pytest
from pricebook.credit_index_options import credit_index_option, index_option_greeks


class TestCreditIndexOption:
    def test_payer_positive(self):
        r = credit_index_option(0.0080, 0.0080, 0.40, 0.25, 4.5)
        assert r.price > 0

    def test_otm_cheaper(self):
        atm = credit_index_option(0.0080, 0.0080, 0.40, 0.25, 4.5)
        otm = credit_index_option(0.0080, 0.0120, 0.40, 0.25, 4.5)
        assert otm.price < atm.price

    def test_receiver_positive(self):
        r = credit_index_option(0.0080, 0.0080, 0.40, 0.25, 4.5, is_payer=False)
        assert r.price > 0

    def test_higher_vol_higher_price(self):
        low = credit_index_option(0.0080, 0.0080, 0.30, 0.25, 4.5)
        high = credit_index_option(0.0080, 0.0080, 0.60, 0.25, 4.5)
        assert high.price > low.price


class TestCreditIndexGreeks:
    def test_delta_positive_payer(self):
        g = index_option_greeks(0.0080, 0.0080, 0.40, 0.25, 4.5)
        assert g.delta > 0  # payer benefits from wider spreads

    def test_vega_positive(self):
        g = index_option_greeks(0.0080, 0.0080, 0.40, 0.25, 4.5)
        assert g.vega > 0
