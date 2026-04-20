"""Tests for vol risk premium."""
import pytest
from pricebook.vol_risk_premium import vrp_single_asset, vrp_term_structure, cross_asset_vrp_comparison, vrp_strategy_signal

class TestVRP:
    def test_positive(self):
        r = vrp_single_asset("equity", 0.20, 0.15)
        assert r.vrp_vol == pytest.approx(0.05)
        assert r.vrp_ratio > 1
    def test_negative(self):
        r = vrp_single_asset("fx", 0.10, 0.12)
        assert r.vrp_vol < 0

class TestVRPTermStructure:
    def test_normal(self):
        """Normal: VRP increases with tenor."""
        r = vrp_term_structure([0.25, 1.0, 3.0], [0.18, 0.20, 0.22], 0.15)
        assert not r.is_inverted
    def test_inverted(self):
        """Inverted: short VRP > long VRP."""
        r = vrp_term_structure([0.25, 1.0, 3.0], [0.25, 0.21, 0.19], 0.15)
        assert r.is_inverted

class TestCrossAssetVRP:
    def test_ranking(self):
        r = cross_asset_vrp_comparison({"equity": (0.20, 0.15), "fx": (0.10, 0.09), "rates": (0.005, 0.004)})
        assert r.highest_vrp == "equity"
    def test_spread(self):
        r = cross_asset_vrp_comparison({"a": (0.20, 0.10), "b": (0.15, 0.14)})
        assert r.spread > 0

class TestVRPSignal:
    def test_high(self):
        r = vrp_strategy_signal(0.08, [0.03, 0.04, 0.05, 0.04, 0.03])
        assert r.signal in ("high", "very_high")
    def test_low(self):
        r = vrp_strategy_signal(-0.02, [0.03, 0.04, 0.05, 0.04, 0.03])
        assert "low" in r.signal or r.signal == "sell"
