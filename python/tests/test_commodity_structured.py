"""Tests for commodity structured products."""

import math

import numpy as np
import pytest

from pricebook.commodity_structured import (
    CommodityAutocallResult,
    CommodityLinkedBondResult,
    CommodityRangeAccrualResult,
    CommodityTARFResult,
    DualCommodityResult,
    commodity_autocallable,
    commodity_linked_bond,
    commodity_range_accrual,
    commodity_tarf,
    dual_commodity_note,
)


# ---- Helpers ----

def _gen_paths(spot=80.0, vol=0.30, drift=0.03, T=1.0,
               n_paths=1000, n_obs=12, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_obs
    paths = np.full((n_paths, n_obs + 1), spot)
    for step in range(n_obs):
        z = rng.standard_normal(n_paths)
        paths[:, step + 1] = paths[:, step] * np.exp(
            (drift - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z
        )
    df = np.exp(-drift * np.linspace(0, T, n_obs + 1))
    return paths, df


# ---- Autocallable ----

class TestCommodityAutocallable:
    def test_basic(self):
        paths, df = _gen_paths(n_obs=4)
        result = commodity_autocallable(
            paths, autocall_barrier=90, coupon=0.05,
            discount_factors=df, observation_dates=[1, 2, 3, 4],
        )
        assert isinstance(result, CommodityAutocallResult)

    def test_low_barrier_higher_autocall(self):
        paths, df = _gen_paths(spot=80, vol=0.3, n_obs=6)
        low = commodity_autocallable(paths, 82, 0.03, df, [1, 2, 3, 4, 5, 6])
        high = commodity_autocallable(paths, 120, 0.03, df, [1, 2, 3, 4, 5, 6])
        assert low.autocall_probability > high.autocall_probability

    def test_memory_coupon(self):
        paths, df = _gen_paths(n_obs=4)
        with_mem = commodity_autocallable(paths, 90, 0.05, df, [1, 2, 3, 4],
                                            has_memory=True)
        without = commodity_autocallable(paths, 90, 0.05, df, [1, 2, 3, 4],
                                           has_memory=False)
        # Prices should be in reasonable vicinity
        assert with_mem.price > 0
        assert without.price > 0

    def test_protection_barrier(self):
        paths, df = _gen_paths(n_obs=4)
        result = commodity_autocallable(paths, 100, 0.03, df, [1, 2, 3, 4],
                                          protection_barrier=60)
        assert result.price > 0


# ---- Commodity-linked bond ----

class TestCommodityLinkedBond:
    def test_basic(self):
        paths, df = _gen_paths(spot=80, n_obs=5)
        result = commodity_linked_bond(paths, base_coupon=0.03,
                                         participation=0.5,
                                         discount_factors=df,
                                         coupon_dates=[1, 2, 3, 4, 5])
        assert isinstance(result, CommodityLinkedBondResult)

    def test_floor_bounded(self):
        paths, df = _gen_paths(n_obs=5)
        result = commodity_linked_bond(paths, 0.03, 0.5, df, coupon_dates=[1, 2, 3, 4, 5])
        # Price ≥ bond floor (commodity upside ≥ 0)
        assert result.price >= result.bond_floor * 0.99
        assert result.commodity_upside >= -1e-3

    def test_higher_participation_higher_price(self):
        paths, df = _gen_paths(n_obs=5, seed=42)
        low = commodity_linked_bond(paths, 0.03, 0.1, df, coupon_dates=[1, 2, 3, 4, 5])
        high = commodity_linked_bond(paths, 0.03, 1.0, df, coupon_dates=[1, 2, 3, 4, 5])
        assert high.price >= low.price

    def test_zero_participation_equals_bond(self):
        paths, df = _gen_paths(n_obs=5)
        result = commodity_linked_bond(paths, 0.03, 0.0, df, coupon_dates=[1, 2, 3, 4, 5])
        assert result.price == pytest.approx(result.bond_floor, rel=1e-6)


# ---- Commodity TARF ----

class TestCommodityTARF:
    def test_basic(self):
        paths, df = _gen_paths(spot=80, vol=0.3, n_obs=12)
        result = commodity_tarf(paths, strike=80, target=20,
                                  discount_factors=df)
        assert isinstance(result, CommodityTARFResult)
        assert 0 <= result.prob_target_hit <= 1

    def test_target_hit_probability(self):
        paths, df = _gen_paths(n_obs=24, vol=0.3)
        low = commodity_tarf(paths, 80, target=5, discount_factors=df)
        high = commodity_tarf(paths, 80, target=50, discount_factors=df)
        assert low.prob_target_hit > high.prob_target_hit

    def test_termination_time(self):
        paths, df = _gen_paths(n_obs=24, vol=0.4)
        result = commodity_tarf(paths, 80, target=10, discount_factors=df)
        # If target is hit often, mean term time < max
        if result.prob_target_hit > 0.5:
            assert result.mean_termination_time < 23


# ---- Commodity range accrual ----

class TestCommodityRangeAccrual:
    def test_basic(self):
        paths, df = _gen_paths(n_obs=30)
        result = commodity_range_accrual(
            paths, range_low=75, range_high=85,
            coupon_per_day=0.001,
            discount_factor_T=df[-1],
        )
        assert isinstance(result, CommodityRangeAccrualResult)
        assert 0 <= result.accrual_rate <= 1

    def test_wide_range_higher_accrual(self):
        paths, df = _gen_paths(n_obs=30)
        narrow = commodity_range_accrual(paths, 78, 82, 0.001, df[-1])
        wide = commodity_range_accrual(paths, 50, 150, 0.001, df[-1])
        assert wide.accrual_rate > narrow.accrual_rate

    def test_n_observations(self):
        paths, df = _gen_paths(n_obs=30)
        result = commodity_range_accrual(paths, 75, 85, 0.001, df[-1])
        assert result.n_observations == 30


# ---- Dual-commodity note ----

class TestDualCommodityNote:
    def test_basic(self):
        long_paths, df = _gen_paths(spot=80, vol=0.3, n_obs=12, seed=1)
        short_paths, _ = _gen_paths(spot=3, vol=0.4, n_obs=12, seed=2)
        result = dual_commodity_note(
            long_paths, short_paths,
            participation=0.5, floor_return=0.0,
            discount_factor_T=df[-1],
        )
        assert isinstance(result, DualCommodityResult)
        assert result.price > 0

    def test_floor_protection(self):
        long_paths, df = _gen_paths(spot=80, vol=0.3, n_obs=12, seed=1)
        short_paths, _ = _gen_paths(spot=3, vol=0.4, n_obs=12, seed=2)
        with_floor = dual_commodity_note(long_paths, short_paths,
                                           1.0, floor_return=0.0,
                                           discount_factor_T=df[-1])
        no_floor = dual_commodity_note(long_paths, short_paths,
                                         1.0, floor_return=-1.0,
                                         discount_factor_T=df[-1])
        # Higher floor → higher price (more protection)
        assert with_floor.price >= no_floor.price

    def test_higher_participation_effect(self):
        long_paths, df = _gen_paths(spot=80, vol=0.2, drift=0.1, n_obs=12, seed=1)
        short_paths, _ = _gen_paths(spot=3, vol=0.2, drift=0.0, n_obs=12, seed=2)
        # Long expected to outperform → higher participation = higher price
        low = dual_commodity_note(long_paths, short_paths, 0.1, 0.0, df[-1])
        high = dual_commodity_note(long_paths, short_paths, 1.0, 0.0, df[-1])
        assert high.price >= low.price
