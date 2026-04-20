"""Tests for skew trading."""
import pytest
from pricebook.skew_trading import risk_reversal_strategy, skew_mean_reversion_signal, skew_carry_trade, cross_asset_skew_comparison

class TestRR:
    def test_put_skew(self):
        r = risk_reversal_strategy(0.18, 0.22, 0.20, 100, 0.03, 1.0)
        assert r.rr_value < 0  # put skew
        assert r.strategy == "long_put_skew"
    def test_call_skew(self):
        r = risk_reversal_strategy(0.22, 0.18, 0.20, 100, 0.03, 1.0)
        assert r.rr_value > 0

class TestSkewMR:
    def test_sell_skew(self):
        r = skew_mean_reversion_signal(-0.01, [-0.04, -0.03, -0.035, -0.03, -0.04])
        assert r.signal == "sell_skew"  # current less negative than average
    def test_neutral(self):
        r = skew_mean_reversion_signal(-0.03, [-0.04, -0.03, -0.035, -0.03, -0.025])
        assert r.signal == "neutral"

class TestSkewCarry:
    def test_basic(self):
        r = skew_carry_trade(rr_cost=0.5, daily_theta=0.02, skew_vol=2.0)
        assert r.breakeven_days == 25
        assert r.carry_ratio > 0

class TestCrossAssetSkew:
    def test_ranking(self):
        r = cross_asset_skew_comparison({"equity": -0.04, "fx": -0.02, "rates": -0.01})
        assert r.steepest_skew == "equity"  # most negative
        assert r.flattest_skew == "rates"
