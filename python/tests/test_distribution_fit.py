"""Tests for distribution fitting."""
import pytest
import numpy as np
from pricebook.statistics.distribution_fit import (
    fit_normal, fit_student_t, fit_gev, ks_test, anderson_darling, qq_data,
)


class TestFitNormal:
    def test_recovers_params(self):
        data = np.random.default_rng(42).normal(5.0, 2.0, 1000)
        result = fit_normal(data)
        assert abs(result.params["mu"] - 5.0) < 0.2
        assert abs(result.params["sigma"] - 2.0) < 0.2


class TestFitStudentT:
    def test_heavy_tails(self):
        data = np.random.default_rng(42).standard_t(4, 1000)
        result = fit_student_t(data)
        assert result.params["nu"] < 10
        assert result.params["nu"] > 2


class TestFitGEV:
    def test_gev_fit(self):
        data = np.random.default_rng(42).gumbel(0, 1, 500)
        result = fit_gev(data)
        assert "xi" in result.params  # shape parameter
        assert "mu" in result.params
        assert "sigma" in result.params


class TestKSTest:
    def test_normal_passes(self):
        data = np.random.default_rng(42).normal(0, 1, 500)
        result = ks_test(data, "normal")
        assert result.p_value > 0.05

    def test_non_normal_fails(self):
        data = np.random.default_rng(42).exponential(1, 500)
        result = ks_test(data, "normal")
        assert result.p_value < 0.05


class TestAndersonDarling:
    def test_normal_passes(self):
        data = np.random.default_rng(42).normal(0, 1, 500)
        result = anderson_darling(data)
        assert hasattr(result, "statistic")

    def test_w7_pvalue_and_no_future_warning(self):
        """W7 regression: anderson_darling must call scipy with
        method='interpolate' to avoid the scipy-1.17 FutureWarning and
        return a real pvalue (the new contract, forward-compatible to
        scipy 1.19 when the legacy critical_values attributes go away).
        """
        import warnings as _warnings
        data = np.random.default_rng(0).normal(0, 1, 200)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", FutureWarning)
            result = anderson_darling(data)
        assert 0.0 <= result.pvalue <= 1.0
        assert isinstance(result.reject_at_5pct, bool)

    def test_w7_rejects_non_normal(self):
        """Sanity: clearly non-normal data is rejected (pvalue < 0.05)."""
        data = np.random.default_rng(0).exponential(1.0, 500)
        result = anderson_darling(data, distribution="normal")
        assert result.reject_at_5pct
        assert result.pvalue < 0.05


class TestQQData:
    def test_qq_length(self):
        data = np.random.default_rng(42).normal(0, 1, 100)
        result = qq_data(data)
        assert len(result.theoretical) == 100

    def test_qq_linear_for_normal(self):
        data = np.random.default_rng(42).normal(0, 1, 200)
        result = qq_data(data)
        corr = np.corrcoef(result.theoretical, result.empirical)[0, 1]
        assert corr > 0.99
