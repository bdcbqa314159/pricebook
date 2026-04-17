"""Tests for advanced bond RV."""
import math
import numpy as np
import pytest
from pricebook.bond_rv_advanced import (
    issuer_curve_fit, invoice_spread, ois_asw_decomposition, bond_yield_pca,
)

class TestIssuerCurveFit:
    def test_basic(self):
        bonds = ["2Y", "5Y", "10Y", "30Y"]
        mats = [2, 5, 10, 30]
        yields = [4.0, 4.2, 4.5, 4.7]
        r = issuer_curve_fit(bonds, mats, yields)
        assert r.rms_residual < 10  # < 10 bps
    def test_rich_cheap_dict(self):
        r = issuer_curve_fit(["a", "b"], [2, 10], [4.0, 4.5])
        assert "a" in r.rich_cheap and "b" in r.rich_cheap
    def test_residuals_sum_near_zero(self):
        r = issuer_curve_fit(["a", "b", "c"], [1, 5, 10], [3.5, 4.0, 4.5])
        assert abs(r.residuals.sum()) < 30  # roughly mean-zero

class TestInvoiceSpread:
    def test_basic(self):
        r = invoice_spread(bond_asw_bps=25, futures_implied_asw_bps=20)
        assert r.invoice_spread_bps == 5
    def test_negative(self):
        r = invoice_spread(15, 20)
        assert r.invoice_spread_bps == -5

class TestOISASW:
    def test_decomposition(self):
        r = ois_asw_decomposition(4.50, 4.30, 4.20)
        assert r.ibor_asw_bps == pytest.approx(20)
        assert r.ibor_ois_basis_bps == pytest.approx(10)
        assert r.ois_asw_bps == pytest.approx(30)
    def test_identity(self):
        """OIS ASW = IBOR ASW + IBOR-OIS basis."""
        r = ois_asw_decomposition(5.0, 4.8, 4.7)
        assert r.ois_asw_bps == pytest.approx(r.ibor_asw_bps + r.ibor_ois_basis_bps)

class TestBondYieldPCA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        # Simulate: level dominates
        level = rng.standard_normal(100)
        slope = rng.standard_normal(100) * 0.3
        changes = np.column_stack([
            level + slope * (-1), level + slope * 0, level + slope * 1,
            level + slope * 2, level + slope * 3,
        ])
        r = bond_yield_pca(changes, n_components=3)
        assert r.n_components == 3
        assert r.explained_variance_ratio[0] > 0.5  # level explains most
    def test_cumulative_sums(self):
        rng = np.random.default_rng(42)
        changes = rng.standard_normal((50, 5))
        r = bond_yield_pca(changes, 3)
        assert r.cumulative_explained[-1] <= 1.0 + 1e-6
    def test_three_factors_high_explanation(self):
        """Three factors typically explain >90% of bond yield changes."""
        rng = np.random.default_rng(42)
        level = rng.standard_normal(200)
        slope = rng.standard_normal(200) * 0.2
        curv = rng.standard_normal(200) * 0.05
        tenors = np.array([1, 2, 3, 5, 7, 10])
        changes = np.column_stack([
            level + slope * (t - 5) / 5 + curv * ((t - 5) / 5)**2
            for t in tenors
        ])
        r = bond_yield_pca(changes, 3)
        assert r.cumulative_explained[-1] > 0.90
