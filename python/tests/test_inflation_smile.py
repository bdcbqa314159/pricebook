"""Tests for inflation smile."""
import math, numpy as np, pytest
from pricebook.inflation_smile import calibrate_inflation_sabr, InflationVolCube, zc_inflation_cap_smile
from pricebook.sabr import sabr_implied_vol

class TestInflationSABR:
    def test_basic(self):
        a, rho, nu, beta = 0.01, -0.3, 0.4, 0.5
        strikes = [0.015, 0.02, 0.025, 0.03, 0.035]
        vols = [sabr_implied_vol(0.025, k, 5.0, a, beta, rho, nu) for k in strikes]
        r = calibrate_inflation_sabr(0.025, 5.0, strikes, vols)
        assert r.residual < 0.01
    def test_atm_vol(self):
        r = calibrate_inflation_sabr(0.025, 5.0, [0.02, 0.025, 0.03], [0.03, 0.028, 0.032])
        assert r.atm_vol > 0

class TestInflationVolCube:
    def test_basic(self):
        n1 = calibrate_inflation_sabr(0.025, 2.0, [0.02, 0.025, 0.03], [0.03, 0.028, 0.032])
        n2 = calibrate_inflation_sabr(0.025, 5.0, [0.02, 0.025, 0.03], [0.035, 0.030, 0.038])
        cube = InflationVolCube([n1, n2])
        v = cube.vol(3.0, 0.025)
        assert v > 0

class TestZCCapSmile:
    def test_basic(self):
        r = zc_inflation_cap_smile(0.025, 0.04, 5.0, 0.03)
        assert len(r.strikes) == 7
        assert np.all(r.prices >= 0)
