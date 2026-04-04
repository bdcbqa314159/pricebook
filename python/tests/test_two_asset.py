"""Tests for two-asset options via ADI."""

import math
import pytest

from pricebook.adi import two_asset_option
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


S1, S2, K, R, T = 100.0, 100.0, 0.0, 0.05, 1.0
VOL1, VOL2 = 0.20, 0.25


class TestSpreadOption:
    def test_positive(self):
        price = two_asset_option(S1, S2, K, R, VOL1, VOL2, rho=0.5, T=T, payoff_type="spread")
        assert price > 0

    def test_higher_correlation_lower_spread(self):
        """Higher correlation → S1-S2 less volatile → spread option cheaper."""
        p_low = two_asset_option(S1, S2, K, R, VOL1, VOL2, rho=0.3, T=T, payoff_type="spread")
        p_high = two_asset_option(S1, S2, K, R, VOL1, VOL2, rho=0.9, T=T, payoff_type="spread")
        assert p_high < p_low

    def test_spread_with_strike(self):
        price = two_asset_option(S1, S2, 10.0, R, VOL1, VOL2, rho=0.5, T=T, payoff_type="spread")
        assert price >= 0


class TestBasketOption:
    def test_positive(self):
        price = two_asset_option(S1, S2, 200.0, R, VOL1, VOL2, rho=0.5, T=T,
                                 payoff_type="basket")
        assert price > 0

    def test_equal_weight_atm(self):
        """Basket of two equal assets at ATM should be positive."""
        price = two_asset_option(100, 100, 200.0, R, VOL1, VOL2, rho=0.0, T=T,
                                 payoff_type="basket")
        assert price > 0

    def test_diversification(self):
        """Lower correlation → more diversification → lower basket vol → lower price."""
        p_high = two_asset_option(S1, S2, 200.0, R, VOL1, VOL2, rho=0.9, T=T,
                                  payoff_type="basket")
        p_low = two_asset_option(S1, S2, 200.0, R, VOL1, VOL2, rho=0.0, T=T,
                                 payoff_type="basket")
        assert p_low < p_high


class TestBestOfOption:
    def test_positive(self):
        price = two_asset_option(S1, S2, 100.0, R, VOL1, VOL2, rho=0.5, T=T,
                                 payoff_type="best_of")
        assert price > 0

    def test_geq_single_asset(self):
        """Best-of ≥ single-asset call (best of two ≥ one of them)."""
        best = two_asset_option(S1, S2, 100.0, R, VOL1, VOL2, rho=0.5, T=T,
                                payoff_type="best_of")
        single = equity_option_price(S1, 100.0, R, VOL1, T)
        assert best >= single * 0.95  # allow small grid error

    def test_low_correlation_higher_price(self):
        """Lower correlation → more dispersion → best-of more valuable."""
        p_high = two_asset_option(S1, S2, 100.0, R, VOL1, VOL2, rho=0.9, T=T,
                                  payoff_type="best_of")
        p_low = two_asset_option(S1, S2, 100.0, R, VOL1, VOL2, rho=0.0, T=T,
                                 payoff_type="best_of")
        assert p_low > p_high


class TestZeroCorrelation:
    def test_basket_symmetric(self):
        """With zero corr and equal params, basket should be symmetric in S1, S2."""
        p1 = two_asset_option(100, 110, 200.0, R, 0.20, 0.20, rho=0.0, T=T,
                              payoff_type="basket")
        p2 = two_asset_option(110, 100, 200.0, R, 0.20, 0.20, rho=0.0, T=T,
                              payoff_type="basket")
        assert p1 == pytest.approx(p2, rel=0.05)
