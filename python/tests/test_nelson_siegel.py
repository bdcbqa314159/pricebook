"""Tests for Nelson-Siegel and Svensson parametric curves."""

import math
import pytest
from datetime import date

from pricebook.nelson_siegel import (
    nelson_siegel_yield,
    svensson_yield,
    ns_discount_curve,
    svensson_discount_curve,
    calibrate_nelson_siegel,
    calibrate_svensson,
)


REF = date(2024, 1, 15)


class TestNelsonSiegel:
    def test_flat_curve(self):
        """beta0 only → flat curve."""
        for t in [0.5, 1.0, 5.0, 10.0, 30.0]:
            y = nelson_siegel_yield(t, beta0=0.05, beta1=0.0, beta2=0.0, tau=2.0)
            assert y == pytest.approx(0.05, abs=1e-10)

    def test_short_rate(self):
        """At t→0, yield → beta0 + beta1."""
        y = nelson_siegel_yield(0.001, 0.05, -0.02, 0.01, 2.0)
        assert y == pytest.approx(0.05 + (-0.02), abs=0.001)

    def test_long_rate(self):
        """At t→∞, yield → beta0."""
        y = nelson_siegel_yield(100.0, 0.05, -0.02, 0.01, 2.0)
        assert y == pytest.approx(0.05, abs=0.001)

    def test_normal_shape(self):
        """Upward sloping: beta1 < 0."""
        y_short = nelson_siegel_yield(0.5, 0.05, -0.02, 0.01, 2.0)
        y_long = nelson_siegel_yield(10.0, 0.05, -0.02, 0.01, 2.0)
        assert y_long > y_short

    def test_inverted_shape(self):
        """Inverted: beta1 > 0."""
        y_short = nelson_siegel_yield(0.5, 0.05, 0.02, 0.0, 2.0)
        y_long = nelson_siegel_yield(10.0, 0.05, 0.02, 0.0, 2.0)
        assert y_short > y_long


class TestSvensson:
    def test_reduces_to_ns(self):
        """beta3=0 → Svensson = NS."""
        for t in [0.5, 2.0, 10.0]:
            ns = nelson_siegel_yield(t, 0.05, -0.02, 0.01, 2.0)
            sv = svensson_yield(t, 0.05, -0.02, 0.01, 2.0, 0.0, 5.0)
            assert sv == pytest.approx(ns, abs=1e-12)

    def test_second_hump(self):
        """beta3 adds a second hump at different decay."""
        sv1 = svensson_yield(3.0, 0.05, -0.02, 0.01, 2.0, 0.0, 5.0)
        sv2 = svensson_yield(3.0, 0.05, -0.02, 0.01, 2.0, 0.005, 5.0)
        assert sv2 != sv1


class TestDiscountCurve:
    def test_ns_curve_valid(self):
        curve = ns_discount_curve(REF, 0.05, -0.02, 0.01, 2.0)
        d5y = date.fromordinal(REF.toordinal() + int(5 * 365))
        df = curve.df(d5y)
        assert 0.0 < df < 1.0

    def test_svensson_curve_valid(self):
        curve = svensson_discount_curve(REF, 0.05, -0.02, 0.01, 2.0, 0.005, 5.0)
        d10y = date.fromordinal(REF.toordinal() + int(10 * 365))
        df = curve.df(d10y)
        assert 0.0 < df < 1.0

    def test_ns_curve_matches_yield(self):
        curve = ns_discount_curve(REF, 0.05, -0.01, 0.0, 2.0)
        t = 5.0
        d = date.fromordinal(REF.toordinal() + int(t * 365))
        expected_y = nelson_siegel_yield(t, 0.05, -0.01, 0.0, 2.0)
        actual_y = curve.zero_rate(d)
        assert actual_y == pytest.approx(expected_y, rel=0.01)


class TestCalibration:
    def test_ns_calibrate(self):
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        true_params = (0.05, -0.02, 0.01, 2.0)
        yields = [nelson_siegel_yield(t, *true_params) for t in tenors]

        result = calibrate_nelson_siegel(tenors, yields)
        assert result["rmse"] < 1e-6
        assert result["beta0"] == pytest.approx(0.05, abs=0.005)

    def test_svensson_calibrate(self):
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        true_params = (0.05, -0.02, 0.01, 2.0, 0.005, 5.0)
        yields = [svensson_yield(t, *true_params) for t in tenors]

        result = calibrate_svensson(tenors, yields)
        assert result["rmse"] < 1e-4

    def test_ns_matches_bootstrap(self):
        """Calibrated NS matches market yields at pillars."""
        tenors = [1, 2, 3, 5, 7, 10]
        yields = [0.04, 0.042, 0.044, 0.046, 0.047, 0.048]

        result = calibrate_nelson_siegel(tenors, yields)
        for t, y in zip(tenors, yields):
            model_y = nelson_siegel_yield(t, result["beta0"], result["beta1"],
                                          result["beta2"], result["tau"])
            assert model_y == pytest.approx(y, abs=0.002)
