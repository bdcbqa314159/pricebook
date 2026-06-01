"""Tests for Hull-White calibration from swaption vols."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.models.hw_calibration import (
    calibrate_hull_white, HWCalibrationResult, _hw_swaption_price, _hw_implied_vol,
)

REF = date(2024, 11, 4)


def _make_curve(rate=0.04):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 35)]
    dfs = [math.exp(-rate * y) for y in range(1, 35)]
    return DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


class TestHWSwaption:
    def test_price_positive(self):
        curve = _make_curve()
        price = _hw_swaption_price(0.03, 0.01, curve, 5.0, 5.0, 0.04)
        assert price > 0

    def test_higher_vol_higher_price(self):
        curve = _make_curve()
        p1 = _hw_swaption_price(0.03, 0.005, curve, 5.0, 5.0, 0.04)
        p2 = _hw_swaption_price(0.03, 0.015, curve, 5.0, 5.0, 0.04)
        assert p2 > p1

    def test_implied_vol_positive(self):
        curve = _make_curve()
        iv = _hw_implied_vol(0.03, 0.01, curve, 5.0, 5.0, 0.04)
        assert iv > 0


class TestCalibration:
    def test_round_trip(self):
        """Generate vols from known HW, calibrate back."""
        curve = _make_curve()
        true_a, true_sigma = 0.03, 0.01

        # Generate "market" vols from known params
        grid = [(1, 5), (5, 5), (10, 5), (5, 10), (10, 10)]
        market_vols = {}
        for exp, tenor in grid:
            iv = _hw_implied_vol(true_a, true_sigma, curve, exp, tenor, 0.04, n_steps=50)
            if iv > 0:
                market_vols[(exp, tenor)] = iv

        if len(market_vols) < 3:
            pytest.skip("Not enough valid vols for calibration")

        result = calibrate_hull_white(curve, market_vols, strike=0.04, n_steps=50)

        assert result.converged
        assert result.a == pytest.approx(true_a, rel=0.30)
        assert result.sigma == pytest.approx(true_sigma, rel=0.30)

    def test_calibration_produces_hw(self):
        curve = _make_curve()
        vols = {(5, 5): 0.006, (10, 5): 0.005}
        result = calibrate_hull_white(curve, vols, strike=0.04)
        assert isinstance(result.model, type(curve).__class__) or hasattr(result.model, 'zcb_price')
        assert result.a > 0
        assert result.sigma > 0

    def test_rmse_finite(self):
        curve = _make_curve()
        vols = {(5, 5): 0.006, (10, 5): 0.005, (10, 10): 0.004}
        result = calibrate_hull_white(curve, vols, strike=0.04)
        assert math.isfinite(result.rmse_vol)

    def test_per_swaption_errors(self):
        curve = _make_curve()
        vols = {(5, 5): 0.006, (10, 10): 0.004}
        result = calibrate_hull_white(curve, vols, strike=0.04)
        assert len(result.per_swaption_errors) == 2
        assert all("error_bp" in e for e in result.per_swaption_errors)

    def test_to_dict(self):
        curve = _make_curve()
        vols = {(5, 5): 0.006}
        result = calibrate_hull_white(curve, vols, strike=0.04)
        d = result.to_dict()
        assert "a" in d
        assert "sigma" in d
        assert "rmse_vol" in d
