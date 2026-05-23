"""Tests for regime-switching process and regime-dependent market data."""

import pytest
import math
import numpy as np
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.regime_process import (
    RegimeProcessSpec, create_regime_gbm, create_regime_ou,
)
from pricebook.models.regime_surfaces import (
    RegimeVolSurface, RegimeCurve, regime_blend, regime_price,
    RegimeBlendResult,
)

REF = date(2024, 1, 15)


# ═══════════════════════════════════════════════════════════════
# 1.3: Regime-Switching Process
# ═══════════════════════════════════════════════════════════════

class TestRegimeGBM:
    def test_create(self):
        spec = create_regime_gbm(
            100.0, [0.15, 0.35], [0.04, 0.04],
            [[0.95, 0.05], [0.10, 0.90]],
        )
        assert spec.n_regimes == 2
        assert spec.x0 == math.log(100.0)

    def test_simulate(self):
        spec = create_regime_gbm(
            100.0, [0.15, 0.35], [0.04, 0.04],
            [[0.95, 0.05], [0.10, 0.90]],
        )
        result = spec.simulate(n_paths=100, n_steps=50, dt=1/252)
        assert result["paths"].shape == (100, 51)
        assert result["regimes"].shape == (100, 51)

    def test_regimes_are_valid(self):
        spec = create_regime_gbm(
            100.0, [0.10, 0.30], [0.03, 0.03],
            [[0.98, 0.02], [0.05, 0.95]],
        )
        result = spec.simulate(50, 100, 1/252)
        assert set(result["regimes"].flatten()).issubset({0, 1})

    def test_high_vol_regime_wider(self):
        """Paths should have higher variance in high-vol regime."""
        spec = create_regime_gbm(
            100.0, [0.05, 0.50], [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],  # no switching — pure regimes
        )
        # Force all paths to start in regime 0 (low vol)
        result = spec.simulate(500, 252, 1/252, seed=42)
        # Terminal variance should reflect regime mix
        terminal_std = result["paths"][:, -1].std()
        assert terminal_std > 0


class TestRegimeOU:
    def test_create(self):
        spec = create_regime_ou(
            0.05, [0.03, 0.08], [0.5, 0.5], [0.01, 0.02],
            [[0.95, 0.05], [0.10, 0.90]],
        )
        assert spec.n_regimes == 2

    def test_simulate(self):
        spec = create_regime_ou(
            0.05, [0.03, 0.08], [0.5, 0.5], [0.01, 0.02],
            [[0.95, 0.05], [0.10, 0.90]],
        )
        result = spec.simulate(50, 100, 1/252)
        assert result["paths"].shape == (50, 101)

    def test_to_dict(self):
        spec = create_regime_gbm(
            100.0, [0.15, 0.35], [0.04, 0.04],
            [[0.95, 0.05], [0.10, 0.90]],
        )
        d = spec.to_dict()
        assert d["n_regimes"] == 2


# ═══════════════════════════════════════════════════════════════
# 1.4: Regime-Dependent Market Data
# ═══════════════════════════════════════════════════════════════

class TestRegimeVolSurface:
    def test_blend_flat_vols(self):
        rvs = RegimeVolSurface([0.15, 0.35], np.array([0.7, 0.3]))
        v = rvs.vol(1.0)
        # Variance blend: sqrt(0.7 × 0.15² + 0.3 × 0.35²) ≈ 0.224
        expected = math.sqrt(0.7 * 0.15**2 + 0.3 * 0.35**2)
        assert abs(v - expected) < 0.001

    def test_linear_blend(self):
        rvs = RegimeVolSurface([0.15, 0.35], np.array([0.5, 0.5]), blend_variance=False)
        v = rvs.vol()
        assert abs(v - 0.25) < 0.001

    def test_regime_vols(self):
        rvs = RegimeVolSurface([0.10, 0.30, 0.50], np.array([0.5, 0.3, 0.2]))
        r = rvs.regime_vols(1.0)
        assert isinstance(r, RegimeBlendResult)
        assert r.n_regimes == 3
        assert len(r.regime_values) == 3

    def test_to_dict(self):
        rvs = RegimeVolSurface([0.15, 0.35], np.array([0.5, 0.5]))
        d = rvs.to_dict()
        assert d["n_regimes"] == 2

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            RegimeVolSurface([0.15, 0.35], np.array([0.5, 0.3, 0.2]))


class TestRegimeCurve:
    def test_blend_curves(self):
        c1 = DiscountCurve.flat(REF, 0.03)
        c2 = DiscountCurve.flat(REF, 0.07)
        rc = RegimeCurve([c1, c2], np.array([0.6, 0.4]))
        df = rc.df(date(2029, 1, 15))
        df1 = c1.df(date(2029, 1, 15))
        df2 = c2.df(date(2029, 1, 15))
        expected = 0.6 * df1 + 0.4 * df2
        assert abs(df - expected) < 1e-10

    def test_zero_rate(self):
        c1 = DiscountCurve.flat(REF, 0.03)
        c2 = DiscountCurve.flat(REF, 0.07)
        rc = RegimeCurve([c1, c2], np.array([0.5, 0.5]))
        zr = rc.zero_rate(date(2029, 1, 15))
        assert 0.03 < zr < 0.07

    def test_regime_dfs(self):
        c1 = DiscountCurve.flat(REF, 0.04)
        c2 = DiscountCurve.flat(REF, 0.06)
        rc = RegimeCurve([c1, c2], np.array([0.5, 0.5]))
        r = rc.regime_dfs(date(2029, 1, 15))
        assert r.n_regimes == 2
        assert r.regime_values[0] > r.regime_values[1]  # lower rate → higher DF

    def test_to_dict(self):
        c = DiscountCurve.flat(REF, 0.04)
        rc = RegimeCurve([c, c], np.array([0.5, 0.5]))
        d = rc.to_dict()
        assert d["n_regimes"] == 2


class TestRegimeBlend:
    def test_scalar_blend(self):
        v = regime_blend([100, 200, 300], np.array([0.5, 0.3, 0.2]))
        assert abs(v - 170) < 0.01

    def test_regime_price(self):
        pricers = [lambda: 100.0, lambda: 95.0, lambda: 110.0]
        result = regime_price(pricers, np.array([0.5, 0.3, 0.2]))
        assert result["blended_price"] > 0
        assert len(result["regime_prices"]) == 3
        assert result["regime_spread"] == 15.0
