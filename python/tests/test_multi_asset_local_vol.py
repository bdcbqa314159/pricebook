"""Tests for multi-asset local vol."""
import numpy as np, pytest
from pricebook.multi_asset_local_vol import (
    dupire_2d_local_vol, multi_asset_slv_simulate, smile_consistency_check,
)

class TestDupire2D:
    def test_basic(self):
        vols1 = np.full((3, 5), 0.20); vols2 = np.full((3, 5), 0.25)
        r = dupire_2d_local_vol(vols1, vols2, np.array([0.25, 0.5, 1.0]), np.linspace(80, 120, 5))
        assert r.method == "marginal_dupire"
        assert r.vol_surface_1.shape == (3, 5)

class TestMultiAssetSLV:
    def test_basic(self):
        r = multi_asset_slv_simulate(100, 100, 0.03, 0.02, 0.02, 0.20, 0.25,
                                       0.04, 2.0, 0.04, 0.3, 0.5, -0.3,
                                       1.0, n_paths=200, seed=42)
        assert r.spot1_paths.shape == (200, 51)
        assert r.mean_terminal_1 > 0

    def test_spots_positive(self):
        r = multi_asset_slv_simulate(100, 100, 0.03, 0.02, 0.02, 0.20, 0.25,
                                       0.04, 2.0, 0.04, 0.3, 0.5, -0.3, 1.0, n_paths=100, seed=42)
        assert np.all(r.spot1_paths > 0)
        assert np.all(r.spot2_paths > 0)

class TestSmileConsistency:
    def test_consistent(self):
        r = smile_consistency_check(0.15, [0.20, 0.25], [0.5, 0.5], 0.5)
        assert r.is_consistent
        assert r.consistency_ratio <= 1.01

    def test_too_high_basket_vol(self):
        """Basket vol > weighted avg → inconsistent."""
        r = smile_consistency_check(0.30, [0.20, 0.25], [0.5, 0.5], 0.5)
        assert not r.is_consistent
