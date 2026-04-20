"""Tests for vol surface stress testing."""
import numpy as np, pytest
from pricebook.vol_stress import parallel_vol_bump, tilt_vol_bump, twist_vol_bump, vol_scenario_replay, cross_asset_vol_stress

class TestParallel:
    def test_basic(self):
        r = parallel_vol_bump([0.20, 0.20, 0.20], [100, 80, 60], bump_bps=100)
        assert r.bumped_vols[0] == pytest.approx(0.21)
        assert r.pnl_estimate == pytest.approx(240 * 0.01)
    def test_negative_bump(self):
        r = parallel_vol_bump([0.20], [100], bump_bps=-100)
        assert r.bumped_vols[0] == pytest.approx(0.19)

class TestTilt:
    def test_basic(self):
        r = tilt_vol_bump([0.25, 1.0, 3.0], [0.20, 0.20, 0.20], [100, 80, 60])
        assert r.bump_type == "tilt"
    def test_opposite_ends(self):
        r = tilt_vol_bump([0.25, 3.0], [0.20, 0.20], [100, 60], tilt_bps=100)
        # Short end goes down, long end goes up
        assert r.bumped_vols[0] < r.bumped_vols[-1]

class TestTwist:
    def test_basic(self):
        r = twist_vol_bump([0.25, 1.0, 3.0], [0.20]*3, [100]*3)
        assert r.bump_type == "twist"

class TestReplay:
    def test_basic(self):
        r = vol_scenario_replay("covid_crash", [0.05, 0.08, 0.10], [100, 80, 60])
        assert r.pnl != 0
        assert r.max_vol_change == 0.10

class TestCrossAssetStress:
    def test_basic(self):
        r = cross_asset_vol_stress(
            {"equity": [100, 80], "fx": [50, 30]},
            {"equity": 200, "fx": 100},
        )
        assert len(r.per_asset_pnl) == 2
        assert r.total_pnl != 0
    def test_diversification(self):
        r = cross_asset_vol_stress(
            {"a": [100], "b": [-80]},  # offsetting
            {"a": 100, "b": 100},
        )
        assert r.diversification_benefit > 0
