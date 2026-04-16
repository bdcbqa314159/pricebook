"""Tests for advanced yield curve construction."""

import math

import numpy as np
import pytest

from pricebook.curve_advanced import (
    NSFitResult,
    SmoothForwardResult,
    SvenssonFitResult,
    TOYResult,
    nelson_siegel_fit,
    ns_yield_curve,
    smooth_forward_curve,
    svensson_fit,
    svensson_yield_curve,
    turn_of_year_adjustment,
)


# ---- Nelson-Siegel ----

class TestNelsonSiegel:
    def _sample_curve(self):
        """Typical upward-sloping yield curve."""
        mats = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        yields = [0.035, 0.036, 0.038, 0.040, 0.042, 0.044, 0.045, 0.046, 0.047, 0.047]
        return mats, yields

    def test_basic_fit(self):
        mats, yields = self._sample_curve()
        result = nelson_siegel_fit(mats, yields)
        assert isinstance(result, NSFitResult)
        assert result.method == "nelson_siegel"

    def test_residual_small(self):
        mats, yields = self._sample_curve()
        result = nelson_siegel_fit(mats, yields)
        assert result.residual < 0.005

    def test_long_rate_is_beta0(self):
        """β₀ should be close to the long-end yield."""
        mats, yields = self._sample_curve()
        result = nelson_siegel_fit(mats, yields)
        assert result.beta0 == pytest.approx(0.047, abs=0.005)

    def test_negative_slope_for_normal_curve(self):
        """Normal curve: short < long → β₁ < 0."""
        mats, yields = self._sample_curve()
        result = nelson_siegel_fit(mats, yields)
        assert result.beta1 < 0

    def test_evaluate_at_pillars(self):
        mats, yields = self._sample_curve()
        result = nelson_siegel_fit(mats, yields)
        fitted = ns_yield_curve(result, mats)
        np.testing.assert_allclose(fitted, yields, atol=0.005)

    def test_flat_curve(self):
        mats = [1, 2, 5, 10]
        yields = [0.05, 0.05, 0.05, 0.05]
        result = nelson_siegel_fit(mats, yields)
        assert result.residual < 0.001

    def test_tau_positive(self):
        mats, yields = self._sample_curve()
        result = nelson_siegel_fit(mats, yields)
        assert result.tau > 0


# ---- Svensson ----

class TestSvensson:
    def _sample_curve(self):
        mats = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        yields = [0.035, 0.036, 0.038, 0.041, 0.043, 0.044, 0.0445, 0.045, 0.046, 0.0465, 0.047]
        return mats, yields

    def test_basic_fit(self):
        mats, yields = self._sample_curve()
        result = svensson_fit(mats, yields)
        assert isinstance(result, SvenssonFitResult)
        assert result.method == "svensson"

    def test_residual_small(self):
        mats, yields = self._sample_curve()
        result = svensson_fit(mats, yields)
        assert result.residual < 0.005

    def test_svensson_at_least_as_good_as_ns(self):
        """Svensson has more params → should fit at least as well."""
        mats, yields = self._sample_curve()
        ns = nelson_siegel_fit(mats, yields)
        sv = svensson_fit(mats, yields)
        assert sv.residual <= ns.residual + 0.001

    def test_evaluate(self):
        mats, yields = self._sample_curve()
        result = svensson_fit(mats, yields)
        fitted = svensson_yield_curve(result, mats)
        np.testing.assert_allclose(fitted, yields, atol=0.005)

    def test_taus_positive(self):
        mats, yields = self._sample_curve()
        result = svensson_fit(mats, yields)
        assert result.tau1 > 0
        assert result.tau2 > 0


# ---- Smooth forward ----

class TestSmoothForward:
    def test_basic(self):
        pillars = [0.25, 0.5, 1, 2, 5, 10]
        rates = [0.04, 0.041, 0.042, 0.044, 0.046, 0.048]
        result = smooth_forward_curve(pillars, rates)
        assert isinstance(result, SmoothForwardResult)
        assert result.method == "monotone_hermite"

    def test_output_size(self):
        pillars = [0.5, 1, 2, 5, 10]
        rates = [0.04, 0.042, 0.044, 0.046, 0.048]
        result = smooth_forward_curve(pillars, rates, n_output=200)
        assert len(result.pillars) == 200
        assert len(result.forwards) == 200
        assert len(result.discount_factors) == 200

    def test_forwards_positive(self):
        """Forward rates should be positive for normal curve."""
        pillars = [0.25, 0.5, 1, 2, 5, 10]
        rates = [0.04, 0.041, 0.042, 0.044, 0.046, 0.048]
        result = smooth_forward_curve(pillars, rates)
        assert np.all(result.forwards > 0)

    def test_df_decreasing(self):
        """Discount factors should be decreasing for positive rates."""
        pillars = [0.25, 0.5, 1, 2, 5, 10]
        rates = [0.04, 0.041, 0.042, 0.044, 0.046, 0.048]
        result = smooth_forward_curve(pillars, rates)
        assert np.all(np.diff(result.discount_factors) < 0)

    def test_df_starts_at_one(self):
        pillars = [0.25, 0.5, 1, 2, 5, 10]
        rates = [0.04, 0.041, 0.042, 0.044, 0.046, 0.048]
        result = smooth_forward_curve(pillars, rates)
        assert result.discount_factors[0] == pytest.approx(1.0, abs=0.01)

    def test_flat_curve_constant_forward(self):
        """Flat zero curve → constant forward rate."""
        pillars = [0.5, 1, 2, 5, 10]
        rates = [0.05] * 5
        result = smooth_forward_curve(pillars, rates, n_output=50)
        assert np.std(result.forwards) < 0.005


# ---- Turn-of-year ----

class TestTurnOfYear:
    def test_basic(self):
        pillars = [0.25, 0.5, 1, 2, 3, 5]
        rates = [0.04, 0.041, 0.042, 0.043, 0.044, 0.046]
        result = turn_of_year_adjustment(pillars, rates)
        assert isinstance(result, TOYResult)

    def test_adjusted_higher(self):
        """TOY adds spread → adjusted rates >= original."""
        pillars = [0.25, 0.5, 1, 2, 3, 5]
        rates = [0.04, 0.041, 0.042, 0.043, 0.044, 0.046]
        result = turn_of_year_adjustment(pillars, rates, toy_spread=0.0020)
        assert np.all(result.adjusted_rates >= np.array(rates) - 1e-10)

    def test_zero_spread_no_change(self):
        pillars = [0.25, 0.5, 1, 2, 3]
        rates = [0.04, 0.041, 0.042, 0.043, 0.044]
        result = turn_of_year_adjustment(pillars, rates, toy_spread=0.0)
        np.testing.assert_allclose(result.adjusted_rates, rates)

    def test_larger_spread_larger_adjustment(self):
        pillars = [0.25, 0.5, 1, 2, 3]
        rates = [0.04, 0.041, 0.042, 0.043, 0.044]
        small = turn_of_year_adjustment(pillars, rates, toy_spread=0.0005)
        large = turn_of_year_adjustment(pillars, rates, toy_spread=0.0020)
        assert large.total_adjustment > small.total_adjustment

    def test_peak_near_year_end(self):
        """Largest adjustment should be near t=1 (year-end)."""
        pillars = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        rates = [0.04] * 7
        result = turn_of_year_adjustment(pillars, rates, toy_spread=0.002)
        peak_idx = np.argmax(result.toy_spreads)
        assert 0.5 <= pillars[peak_idx] <= 1.5
