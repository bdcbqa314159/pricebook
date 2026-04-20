"""Tests for vol term structure."""
import math, numpy as np, pytest
from pricebook.vol_term_structure import forward_vol_from_term, calendar_spread_strategy, vol_curve_shape, Bergomi2Factor

class TestForwardVol:
    def test_basic(self):
        r = forward_vol_from_term([0.25, 0.5, 1.0, 2.0], [0.22, 0.21, 0.20, 0.19], 0.5, 1.0)
        assert r.forward_vol > 0
    def test_flat_curve(self):
        r = forward_vol_from_term([0.5, 1.0], [0.20, 0.20], 0.5, 1.0)
        assert r.forward_vol == pytest.approx(0.20, abs=0.005)

class TestCalendarSpread:
    def test_basic(self):
        r = calendar_spread_strategy([0.25, 1.0, 2.0], [0.22, 0.20, 0.19], 2.0, 0.25)
        assert r.spread < 0  # long end lower vol

class TestVolCurveShape:
    def test_contango(self):
        r = vol_curve_shape([0.25, 1.0, 2.0], [0.18, 0.20, 0.22])
        assert r.shape == "contango"
    def test_backwardation(self):
        r = vol_curve_shape([0.25, 1.0, 2.0], [0.25, 0.22, 0.20])
        assert r.shape == "backwardation"
    def test_flat(self):
        r = vol_curve_shape([0.25, 1.0, 2.0], [0.20, 0.20, 0.20])
        assert r.shape == "flat"

class TestBergomi2Factor:
    def test_basic(self):
        model = Bergomi2Factor(0.04, 1.5, 0.8, rho12=0.3)
        r = model.simulate(1.0, n_paths=500, seed=42)
        assert r.vol_paths.shape == (500, 51)
        assert r.mean_terminal_vol > 0
    def test_vol_positive(self):
        model = Bergomi2Factor(0.04, 1.0, 0.5)
        r = model.simulate(1.0, n_paths=200, seed=42)
        assert np.all(r.vol_paths > 0)
