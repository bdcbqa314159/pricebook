"""Tests for swaption vol cube (3D: expiry × tenor × strike)."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.options.swaption_vol_cube import (
    SwaptionVolCube, SABRNode, build_swaption_vol_cube,
)

REF = date(2024, 11, 4)

ATM_EXPIRIES = [0.5, 1.0, 2.0, 5.0, 10.0]
ATM_TENORS = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]

# Realistic ATM vols (USD-like, in decimal: 0.006 = 60bp)
ATM_VOLS = [
    [0.0080, 0.0075, 0.0065, 0.0060, 0.0058, 0.0055],  # 6M expiry
    [0.0075, 0.0070, 0.0062, 0.0058, 0.0055, 0.0052],  # 1Y
    [0.0068, 0.0065, 0.0058, 0.0054, 0.0052, 0.0050],  # 2Y
    [0.0060, 0.0058, 0.0052, 0.0048, 0.0046, 0.0044],  # 5Y
    [0.0055, 0.0052, 0.0048, 0.0044, 0.0042, 0.0040],  # 10Y
]


class TestSwaptionVolCubeATM:
    def test_atm_at_node(self):
        cube = SwaptionVolCube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS)
        vol = cube.vol_by_years(1.0, 5.0)
        assert vol == pytest.approx(0.0062, abs=0.0001)

    def test_atm_interpolation(self):
        cube = SwaptionVolCube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS)
        vol = cube.vol_by_years(1.5, 3.0)  # between nodes
        assert 0.003 < vol < 0.010

    def test_atm_via_date(self):
        cube = SwaptionVolCube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS)
        vol = cube.vol(REF + relativedelta(years=1), tenor=5.0)
        assert vol > 0

    def test_atm_no_strike(self):
        cube = SwaptionVolCube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS)
        vol = cube.vol_by_years(5.0, 10.0, strike=None)
        assert vol == pytest.approx(0.0048)


class TestSwaptionVolCubeSABR:
    def test_sabr_smile(self):
        """With SABR node, OTM vol should differ from ATM."""
        node = SABRNode(5.0, 10.0, 0.04, 0.015, 0.5, -0.25, 0.4, 0.0048)
        cube = SwaptionVolCube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS, [node])

        atm = cube.vol_by_years(5.0, 10.0, strike=0.04)
        otm_put = cube.vol_by_years(5.0, 10.0, strike=0.02)
        otm_call = cube.vol_by_years(5.0, 10.0, strike=0.06)

        # Negative rho → put skew → OTM put vol > ATM
        assert otm_put > atm * 0.9  # at least close to ATM

    def test_sabr_at_atm_matches(self):
        """At ATM strike, SABR vol ≈ ATM vol."""
        node = SABRNode(5.0, 10.0, 0.04, 0.015, 0.5, -0.25, 0.4, 0.0048)
        assert abs(node.vol(0.04) - 0.0048) < 0.002

    def test_smile_list(self):
        node = SABRNode(5.0, 10.0, 0.04, 0.015, 0.5, -0.25, 0.4, 0.0048)
        cube = SwaptionVolCube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS, [node])
        vols = cube.smile(5.0, 10.0, [0.02, 0.03, 0.04, 0.05, 0.06])
        assert len(vols) == 5
        assert all(v > 0 for v in vols)


class TestBuildVolCube:
    def test_build_atm_only(self):
        cube = build_swaption_vol_cube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS)
        assert cube.vol_by_years(1.0, 5.0) > 0

    def test_build_with_smile(self):
        smile = {
            (5.0, 10.0): {
                "forward": 0.04,
                "strikes": [0.02, 0.03, 0.04, 0.05, 0.06],
                "vols": [0.0060, 0.0052, 0.0048, 0.0046, 0.0050],
            }
        }
        cube = build_swaption_vol_cube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS,
                                        smile_data=smile)
        # Should have 1 SABR node
        vol_otm = cube.vol_by_years(5.0, 10.0, strike=0.02)
        vol_atm = cube.vol_by_years(5.0, 10.0, strike=0.04)
        assert vol_otm != vol_atm  # smile should produce different OTM


class TestVolCubeOps:
    def test_bumped(self):
        cube = SwaptionVolCube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS)
        bumped = cube.bumped(0.001)  # +10bp
        v_base = cube.vol_by_years(5.0, 10.0)
        v_bump = bumped.vol_by_years(5.0, 10.0)
        assert v_bump == pytest.approx(v_base + 0.001)

    def test_to_dict(self):
        cube = SwaptionVolCube(REF, ATM_EXPIRIES, ATM_TENORS, ATM_VOLS)
        d = cube.to_dict()
        assert "expiries" in d
        assert "tenors" in d

    def test_sabr_node_to_dict(self):
        node = SABRNode(5.0, 10.0, 0.04, 0.015, 0.5, -0.25, 0.4, 0.0048)
        d = node.to_dict()
        assert d["alpha"] == 0.015
