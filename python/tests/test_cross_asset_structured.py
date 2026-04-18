"""Tests for cross-asset structured notes."""
import math, numpy as np, pytest
from pricebook.cross_asset_structured import (
    equity_fx_fusion_note, correlation_trigger_note,
    commodity_equity_autocall, dual_asset_range_accrual,
)

def _paths(spot=100, vol=0.20, r=0.03, T=1.0, n=500, steps=50, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / steps
    p = np.full((n, steps + 1), float(spot))
    for s in range(steps):
        p[:, s+1] = p[:, s] * np.exp((r - 0.5*vol**2)*dt + vol*math.sqrt(dt)*rng.standard_normal(n))
    return p

class TestFusionNote:
    def test_basic(self):
        eq = _paths(seed=1); fx = _paths(spot=1.0, vol=0.10, seed=2)
        r = equity_fx_fusion_note(eq, fx, 0.0, 1.0, 1000, 0.97)
        assert r.price > 0
    def test_notional_scaling(self):
        eq = _paths(seed=1); fx = _paths(spot=1.0, vol=0.10, seed=2)
        s = equity_fx_fusion_note(eq, fx, 0.0, 1.0, 500, 0.97)
        b = equity_fx_fusion_note(eq, fx, 0.0, 1.0, 1000, 0.97)
        assert b.price == pytest.approx(2 * s.price, rel=0.01)

class TestCorrelationTrigger:
    def test_basic(self):
        a1 = _paths(seed=1); a2 = _paths(seed=2)
        r = correlation_trigger_note(a1, a2, 0.05, 0.5, 1000, 0.97)
        assert r.price > 0
        assert 0 <= r.coupon_probability <= 1
    def test_high_threshold_higher_probability(self):
        a1 = _paths(seed=1); a2 = _paths(seed=2)
        low = correlation_trigger_note(a1, a2, 0.05, 0.0, 1000, 0.97)
        high = correlation_trigger_note(a1, a2, 0.05, 0.9, 1000, 0.97)
        assert high.coupon_probability >= low.coupon_probability

class TestCommodityEquityAutocall:
    def test_basic(self):
        eq = _paths(seed=1); cm = _paths(spot=80, vol=0.30, seed=2)
        df = np.exp(-0.03 * np.linspace(0, 1, 51))
        r = commodity_equity_autocall(eq, cm, 1.0, 60, 30, 1000, [10, 20, 30, 40, 50], df)
        assert r.price > 0
        assert 0 <= r.commodity_ko_probability <= 1
    def test_lower_ko_more_knockouts(self):
        eq = _paths(seed=1); cm = _paths(spot=80, vol=0.30, seed=2)
        df = np.exp(-0.03 * np.linspace(0, 1, 51))
        high_ko = commodity_equity_autocall(eq, cm, 1.0, 75, 30, 1000, [10, 30, 50], df)
        low_ko = commodity_equity_autocall(eq, cm, 1.0, 50, 30, 1000, [10, 30, 50], df)
        assert high_ko.commodity_ko_probability >= low_ko.commodity_ko_probability

class TestDualRangeAccrual:
    def test_basic(self):
        a1 = _paths(vol=0.10, seed=1); a2 = _paths(vol=0.10, seed=2)
        r = dual_asset_range_accrual(a1, a2, (90, 110), (90, 110), 0.001, 0.97)
        assert 0 <= r.accrual_rate <= 1
    def test_wider_range_higher_accrual(self):
        a1 = _paths(vol=0.10, seed=1); a2 = _paths(vol=0.10, seed=2)
        narrow = dual_asset_range_accrual(a1, a2, (98, 102), (98, 102), 0.001, 0.97)
        wide = dual_asset_range_accrual(a1, a2, (50, 200), (50, 200), 0.001, 0.97)
        assert wide.accrual_rate >= narrow.accrual_rate
    def test_dual_leq_single(self):
        a1 = _paths(vol=0.10, seed=1); a2 = _paths(vol=0.10, seed=2)
        dual = dual_asset_range_accrual(a1, a2, (90, 110), (90, 110), 0.001, 0.97)
        # Single: just check asset 1 in range (ignore asset 2 by using very wide range)
        single = dual_asset_range_accrual(a1, a2, (90, 110), (0, 10000), 0.001, 0.97)
        assert dual.accrual_rate <= single.accrual_rate + 0.01
