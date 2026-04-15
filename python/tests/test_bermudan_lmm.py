"""Tests for Bermudan swaption pricing under LMM."""

import math

import numpy as np
import pytest

from pricebook.bermudan_lmm import (
    BermudanBoundsResult,
    BermudanLMMResult,
    ExerciseBoundary,
    bermudan_exercise_boundary,
    bermudan_swaption_lmm,
    bermudan_upper_bound,
)


# ---- LSM under LMM ----

class TestBermudanSwaptionLMM:
    def test_basic_pricing(self):
        fwd = [0.05] * 6
        vols = [0.20] * 6
        result = bermudan_swaption_lmm(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=5_000, n_steps=50, seed=42,
        )
        assert isinstance(result, BermudanLMMResult)
        assert result.price >= 0

    def test_payer_positive_price(self):
        """ATM payer Bermudan should have positive value."""
        fwd = [0.05] * 8
        vols = [0.20] * 8
        result = bermudan_swaption_lmm(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3, 4],
            swap_end_idx=8, n_paths=10_000, n_steps=50, seed=42,
        )
        assert result.price > 0

    def test_receiver_positive_price(self):
        """ATM receiver Bermudan should have positive value."""
        fwd = [0.05] * 8
        vols = [0.20] * 8
        result = bermudan_swaption_lmm(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3, 4],
            swap_end_idx=8, is_payer=False, n_paths=10_000, n_steps=50, seed=42,
        )
        assert result.price > 0

    def test_deep_otm_near_zero(self):
        """Deep OTM swaption should be cheap."""
        fwd = [0.05] * 6
        vols = [0.20] * 6
        result = bermudan_swaption_lmm(
            fwd, vols, strike=0.15, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=10_000, n_steps=50, seed=42,
        )
        assert result.price < 0.01

    def test_more_exercise_dates_higher_value(self):
        """More exercise dates → higher value (more optionality)."""
        fwd = [0.05] * 8
        vols = [0.20] * 8
        few = bermudan_swaption_lmm(
            fwd, vols, strike=0.05, exercise_indices=[2],
            swap_end_idx=8, n_paths=10_000, n_steps=50, seed=42,
        )
        many = bermudan_swaption_lmm(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3, 4],
            swap_end_idx=8, n_paths=10_000, n_steps=50, seed=42,
        )
        assert many.price >= few.price * 0.9  # allow MC noise

    def test_exercise_rate(self):
        """Some paths should exercise."""
        fwd = [0.05] * 6
        vols = [0.20] * 6
        result = bermudan_swaption_lmm(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=10_000, n_steps=50, seed=42,
        )
        assert 0 < result.exercise_rate <= 1

    def test_n_exercise_dates(self):
        fwd = [0.05] * 6
        vols = [0.20] * 6
        result = bermudan_swaption_lmm(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=1_000, n_steps=30, seed=42,
        )
        assert result.n_exercise_dates == 3

    def test_higher_vol_higher_price(self):
        """Higher vol → higher option price."""
        fwd = [0.05] * 6
        low_vol = bermudan_swaption_lmm(
            fwd, [0.10] * 6, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=10_000, n_steps=50, seed=42,
        )
        high_vol = bermudan_swaption_lmm(
            fwd, [0.30] * 6, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=10_000, n_steps=50, seed=42,
        )
        assert high_vol.price > low_vol.price


# ---- Exercise boundary ----

class TestExerciseBoundary:
    def test_basic(self):
        fwd = [0.05] * 6
        vols = [0.20] * 6
        boundary = bermudan_exercise_boundary(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=5_000, n_steps=50, seed=42,
        )
        assert isinstance(boundary, ExerciseBoundary)
        assert len(boundary.exercise_times) == 3
        assert len(boundary.boundary_rates) == 3
        assert len(boundary.exercise_counts) == 3

    def test_some_exercise(self):
        """Some paths should exercise at each date."""
        fwd = [0.05] * 8
        vols = [0.20] * 8
        boundary = bermudan_exercise_boundary(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3, 4],
            swap_end_idx=8, n_paths=10_000, n_steps=50, seed=42,
        )
        assert boundary.exercise_counts.sum() > 0

    def test_boundary_rates_near_strike(self):
        """For ATM, boundary rates should be near the strike."""
        fwd = [0.05] * 8
        vols = [0.15] * 8
        boundary = bermudan_exercise_boundary(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3, 4],
            swap_end_idx=8, n_paths=10_000, n_steps=50, seed=42,
        )
        valid = ~np.isnan(boundary.boundary_rates)
        if valid.sum() > 0:
            for rate in boundary.boundary_rates[valid]:
                assert 0.01 < rate < 0.20


# ---- Upper bound ----

class TestBermudanUpperBound:
    def test_basic(self):
        fwd = [0.05] * 6
        vols = [0.20] * 6
        result = bermudan_upper_bound(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=5_000, n_steps=50, seed=42,
        )
        assert isinstance(result, BermudanBoundsResult)

    def test_upper_geq_lower(self):
        """Upper bound ≥ lower bound."""
        fwd = [0.05] * 6
        vols = [0.20] * 6
        result = bermudan_upper_bound(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=5_000, n_steps=50, seed=42,
        )
        assert result.upper_bound >= result.lower_bound - 1e-6

    def test_gap_positive(self):
        fwd = [0.05] * 6
        vols = [0.20] * 6
        result = bermudan_upper_bound(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3],
            swap_end_idx=6, n_paths=5_000, n_steps=50, seed=42,
        )
        assert result.gap >= -1e-6

    def test_bounds_reasonable(self):
        """Both bounds should be positive for ATM."""
        fwd = [0.05] * 8
        vols = [0.20] * 8
        result = bermudan_upper_bound(
            fwd, vols, strike=0.05, exercise_indices=[1, 2, 3, 4],
            swap_end_idx=8, n_paths=5_000, n_steps=50, seed=42,
        )
        assert result.lower_bound > 0
        assert result.upper_bound > 0
