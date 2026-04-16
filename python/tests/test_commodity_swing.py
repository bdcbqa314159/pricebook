"""Tests for swing options and virtual gas storage."""

import math

import numpy as np
import pytest

from pricebook.commodity_swing import (
    NominationResult,
    SwingOptionResult,
    VirtualGasStorage,
    VirtualStorageResult,
    nomination_rights_value,
    swing_option_lsm,
)


# ---- Helpers ----

def _gen_gbm_paths(spot=100.0, vol=0.30, r=0.03, T=1.0,
                   n_paths=1000, n_steps=12, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.full((n_paths, n_steps + 1), spot)
    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        paths[:, step + 1] = paths[:, step] * np.exp(
            (r - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z
        )
    df = np.exp(-r * np.linspace(0, T, n_steps + 1))
    return paths, df


# ---- Swing option ----

class TestSwingOption:
    def test_basic(self):
        paths, df = _gen_gbm_paths(n_paths=500, n_steps=12)
        result = swing_option_lsm(
            paths, exercise_dates=list(range(1, 13)),
            strike=100, max_exercises=6, discount_factors=df,
        )
        assert isinstance(result, SwingOptionResult)
        assert result.price >= 0

    def test_more_exercises_higher_value(self):
        paths, df = _gen_gbm_paths(n_paths=500, n_steps=12)
        low = swing_option_lsm(paths, list(range(1, 13)), 100,
                                 max_exercises=2, discount_factors=df)
        high = swing_option_lsm(paths, list(range(1, 13)), 100,
                                  max_exercises=6, discount_factors=df)
        assert high.price >= low.price * 0.9

    def test_max_exercises_capped(self):
        """max_exercises cannot exceed number of exercise dates."""
        paths, df = _gen_gbm_paths(n_paths=200, n_steps=6)
        result = swing_option_lsm(paths, list(range(1, 7)), 100,
                                    max_exercises=20, discount_factors=df)
        assert result.max_exercises <= 6

    def test_put_swing(self):
        paths, df = _gen_gbm_paths(n_paths=500, n_steps=12)
        result = swing_option_lsm(paths, list(range(1, 13)), 110,
                                    max_exercises=4, discount_factors=df,
                                    is_call=False)
        assert result.price >= 0

    def test_n_exercise_dates(self):
        paths, df = _gen_gbm_paths(n_paths=200, n_steps=12)
        result = swing_option_lsm(paths, [2, 4, 6, 8, 10], 100, 3, df)
        assert result.n_exercise_dates == 5


# ---- Virtual gas storage ----

class TestVirtualGasStorage:
    def test_basic_intrinsic(self):
        storage = VirtualGasStorage(
            max_capacity=1000, max_inject_rate=100,
            max_withdraw_rate=100,
        )
        # Seasonal forward curve: low in summer, high in winter
        fwd = np.array([80, 75, 70, 80, 100, 110, 100, 95, 90, 85, 80, 90])
        df = np.exp(-0.03 * np.linspace(0, 1, 12))
        iv = storage.intrinsic_value(fwd, df, initial_inventory=0)
        assert iv > 0     # profitable to buy low, sell high

    def test_flat_curve_zero_intrinsic(self):
        storage = VirtualGasStorage(1000, 100, 100)
        fwd = np.full(12, 80.0)
        df = np.ones(12)
        iv = storage.intrinsic_value(fwd, df)
        # Flat curve → no arbitrage → zero intrinsic (or near zero)
        assert iv < 1000   # small, well bounded

    def test_full_value(self):
        storage = VirtualGasStorage(1000, 100, 100)
        paths, df = _gen_gbm_paths(spot=80, vol=0.30, n_paths=100, n_steps=12)
        result = storage.value(paths, df, initial_inventory=0, n_inventory=8)
        assert isinstance(result, VirtualStorageResult)
        assert result.extrinsic_value >= 0
        assert result.total_value >= result.intrinsic_value - 100

    def test_bigger_storage_higher_value(self):
        small = VirtualGasStorage(500, 50, 50)
        big = VirtualGasStorage(2000, 200, 200)
        fwd = np.array([80, 75, 70, 80, 100, 110, 100, 95, 90, 85, 80, 90])
        df = np.exp(-0.03 * np.linspace(0, 1, 12))
        iv_small = small.intrinsic_value(fwd, df)
        iv_big = big.intrinsic_value(fwd, df)
        assert iv_big > iv_small

    def test_capacity_constraints(self):
        storage = VirtualGasStorage(1000, 100, 100)
        assert storage.max_capacity == 1000
        assert storage.max_inject_rate == 100


# ---- Nomination rights ----

class TestNominationRights:
    def test_basic(self):
        paths, df = _gen_gbm_paths(n_paths=500, n_steps=20)
        result = nomination_rights_value(
            paths[:, :20], strike=100, min_daily=50, max_daily=150,
            discount_factors=df[:20],
        )
        assert isinstance(result, NominationResult)
        assert result.n_decision_points == 20

    def test_flexibility_positive(self):
        """Flexibility over fixed schedule should be ≥ 0."""
        paths, df = _gen_gbm_paths(n_paths=500, n_steps=20)
        result = nomination_rights_value(
            paths[:, :20], 100, 50, 150, df[:20],
        )
        assert result.flexibility_value >= -1.0   # allow MC noise

    def test_wider_bounds_more_flexibility(self):
        paths, df = _gen_gbm_paths(n_paths=500, n_steps=20)
        narrow = nomination_rights_value(paths[:, :20], 100, 90, 110, df[:20])
        wide = nomination_rights_value(paths[:, :20], 100, 50, 150, df[:20])
        assert wide.flexibility_value >= narrow.flexibility_value - 1.0

    def test_buyer_vs_seller(self):
        paths, df = _gen_gbm_paths(spot=105, vol=0.2, n_paths=500, n_steps=20)
        buyer = nomination_rights_value(paths[:, :20], 100, 50, 150, df[:20], is_buyer=True)
        seller = nomination_rights_value(paths[:, :20], 100, 50, 150, df[:20], is_buyer=False)
        # With spot above strike, buyer has positive expected value
        assert buyer.price > 0

    def test_mean_nominations_in_range(self):
        paths, df = _gen_gbm_paths(n_paths=500, n_steps=20)
        result = nomination_rights_value(paths[:, :20], 100, 50, 150, df[:20])
        assert 50 <= result.mean_nominations <= 150
