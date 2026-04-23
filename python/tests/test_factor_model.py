"""Tests for factor models."""

import numpy as np
import pytest

from pricebook.factor_model import (
    build_factor, build_multi_asset_factors, factor_attribution,
    factor_covariance, factor_timing, zscore, percentile_rank,
)


def _returns(n=500, seed=42):
    return np.random.default_rng(seed).standard_normal(n) * 0.01


# ---- Factor construction ----

class TestBuildFactor:
    def test_momentum(self):
        f = build_factor(_returns(), "momentum", 60)
        assert len(f.values) == 500
        assert f.name == "momentum"

    def test_carry(self):
        f = build_factor(_returns(), "carry", 20)
        assert len(f.z_scores) == 500

    def test_value(self):
        f = build_factor(_returns(), "value", 60)
        assert f.name == "value"

    def test_vol(self):
        f = build_factor(_returns(), "vol", 30)
        # Low vol → high signal (negative vol → positive signal)
        assert f.values[60] <= 0  # negative of vol

    def test_quality(self):
        f = build_factor(_returns(), "quality", 60)
        assert len(f.percentiles) == 500

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_factor(_returns(), "nonsense")

    def test_multi_asset(self):
        assets = {"SPX": _returns(seed=1), "UST": _returns(seed=2)}
        factors = build_multi_asset_factors(assets, ["momentum", "carry"])
        assert "momentum" in factors
        assert "SPX" in factors["momentum"]


# ---- Z-score and percentile ----

class TestZScore:
    def test_full_sample(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = zscore(x)
        assert z.mean() == pytest.approx(0.0, abs=1e-10)
        assert z.std() == pytest.approx(1.0, rel=0.01)

    def test_percentile_rank(self):
        x = np.array([10, 20, 30, 40, 50])
        p = percentile_rank(x)
        assert p[-1] == pytest.approx(1.0)
        assert p[0] == pytest.approx(0.2)


# ---- Factor attribution ----

class TestFactorAttribution:
    def test_single_factor(self):
        rng = np.random.default_rng(42)
        factor = rng.standard_normal(500) * 0.01
        portfolio = 0.001 + 0.5 * factor + rng.standard_normal(500) * 0.002
        result = factor_attribution(portfolio, {"market": factor})
        assert result.betas["market"] == pytest.approx(0.5, rel=0.2)
        assert result.r_squared > 0.5

    def test_two_factors(self):
        rng = np.random.default_rng(42)
        f1 = rng.standard_normal(500) * 0.01
        f2 = rng.standard_normal(500) * 0.01
        port = 0.3 * f1 + 0.7 * f2 + rng.standard_normal(500) * 0.001
        result = factor_attribution(port, {"f1": f1, "f2": f2})
        assert result.betas["f1"] == pytest.approx(0.3, rel=0.3)
        assert result.betas["f2"] == pytest.approx(0.7, rel=0.3)


# ---- Factor covariance ----

class TestFactorCovariance:
    def test_sample(self):
        factors = {"A": _returns(seed=1), "B": _returns(seed=2)}
        result = factor_covariance(factors, method="sample")
        assert result.covariance.shape == (2, 2)
        assert result.condition_number > 0

    def test_shrinkage(self):
        factors = {"A": _returns(seed=1), "B": _returns(seed=2), "C": _returns(seed=3)}
        result = factor_covariance(factors, method="shrinkage")
        assert result.method == "shrinkage"
        # Shrunk matrix should be better conditioned
        sample = factor_covariance(factors, method="sample")
        assert result.condition_number <= sample.condition_number * 1.1

    def test_correlation_bounded(self):
        factors = {"A": _returns(seed=1), "B": _returns(seed=2)}
        result = factor_covariance(factors)
        assert np.all(result.correlation >= -1.01)
        assert np.all(result.correlation <= 1.01)


# ---- Factor timing ----

class TestFactorTiming:
    def test_overweight(self):
        values = np.concatenate([np.zeros(100), np.ones(10) * 3])  # spike at end
        returns = np.random.default_rng(42).standard_normal(110) * 0.01
        result = factor_timing(values, returns)
        assert result.signal == "overweight"

    def test_neutral(self):
        values = np.random.default_rng(42).standard_normal(200)
        returns = np.random.default_rng(43).standard_normal(200) * 0.01
        result = factor_timing(values, returns, z_threshold=3.0)
        assert result.signal == "neutral"
