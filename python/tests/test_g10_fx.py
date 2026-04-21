"""Tests for G10 FX conventions."""

import math
from datetime import date
import pytest

from pricebook.currency import Currency, CurrencyPair, all_g10_pairs


class TestG10FXPairs:
    def test_45_unique_pairs(self):
        """10 currencies → 10×9/2 = 45 unique pairs."""
        pairs = all_g10_pairs()
        assert len(pairs) == 45

    def test_no_duplicate_pairs(self):
        pairs = all_g10_pairs()
        pair_set = set((p.base, p.quote) for p in pairs)
        assert len(pair_set) == 45

    def test_eurusd_convention(self):
        """EUR/USD: EUR is base (EUR > USD in priority)."""
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        assert pair.base == Currency.EUR
        assert pair.quote == Currency.USD

    def test_gbpusd_convention(self):
        pair = CurrencyPair.from_currencies(Currency.GBP, Currency.USD)
        assert pair.base == Currency.GBP

    def test_usdjpy_convention(self):
        """USD/JPY: USD is base (USD > JPY in priority)."""
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.JPY)
        assert pair.base == Currency.USD
        assert pair.quote == Currency.JPY

    def test_audusd_convention(self):
        pair = CurrencyPair.from_currencies(Currency.AUD, Currency.USD)
        assert pair.base == Currency.AUD

    def test_eurgbp_convention(self):
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.GBP)
        assert pair.base == Currency.EUR

    def test_nzdusd_convention(self):
        pair = CurrencyPair.from_currencies(Currency.NZD, Currency.USD)
        assert pair.base == Currency.NZD

    def test_usdnok_convention(self):
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.NOK)
        assert pair.base == Currency.USD

    def test_usdsek_convention(self):
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.SEK)
        assert pair.base == Currency.USD

    def test_usdchf_convention(self):
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.CHF)
        assert pair.base == Currency.USD

    def test_usdcad_convention(self):
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.CAD)
        assert pair.base == Currency.USD


class TestSettlement:
    def test_usdcad_t1(self):
        """USD/CAD settles T+1."""
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.CAD)
        assert pair.settlement_lag == 1

    def test_eurusd_t2(self):
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        assert pair.settlement_lag == 2

    def test_usdjpy_t2(self):
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.JPY)
        assert pair.settlement_lag == 2

    def test_all_g10_deliverable(self):
        """All G10 pairs are deliverable (not NDF)."""
        for pair in all_g10_pairs():
            assert not pair.is_ndf, f"{pair} wrongly flagged as NDF"


class TestForwardPricing:
    def test_cip_basic(self):
        """Forward = Spot × exp((r_quote − r_base) × T)."""
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        # EUR/USD = 1.10, EUR rate = 3%, USD rate = 4%
        fwd = pair.forward_rate(1.10, rate_base=0.03, rate_quote=0.04, T=1.0)
        # F = 1.10 × exp(0.04 - 0.03) = 1.10 × 1.01005 ≈ 1.1111
        assert fwd == pytest.approx(1.10 * math.exp(0.01), rel=1e-10)

    def test_forward_points(self):
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        pts = pair.forward_points(1.10, 0.03, 0.04, 1.0)
        assert pts > 0  # USD rate > EUR rate → positive points

    def test_zero_rate_differential(self):
        """Equal rates → forward = spot."""
        pair = CurrencyPair.from_currencies(Currency.USD, Currency.JPY)
        fwd = pair.forward_rate(150.0, 0.04, 0.04, 1.0)
        assert fwd == pytest.approx(150.0)

    def test_negative_points_when_base_higher(self):
        """When base rate > quote rate, forward < spot (negative points)."""
        pair = CurrencyPair.from_currencies(Currency.AUD, Currency.USD)
        pts = pair.forward_points(0.65, rate_base=0.05, rate_quote=0.04, T=1.0)
        assert pts < 0

    def test_invert_preserves_cip(self):
        """CIP holds for inverted pair too."""
        pair = CurrencyPair.from_currencies(Currency.EUR, Currency.USD)
        fwd = pair.forward_rate(1.10, 0.03, 0.04, 1.0)
        inv_fwd = pair.invert().forward_rate(1/1.10, 0.04, 0.03, 1.0)
        assert fwd * inv_fwd == pytest.approx(1.0, rel=1e-10)
