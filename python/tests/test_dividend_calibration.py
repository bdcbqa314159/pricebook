"""Tests for dividend term structure calibration."""

import pytest
import math
import numpy as np

from pricebook.equity.dividend_calibration import (
    calibrate_dividend_curve, calibrate_from_options,
    dividend_curve_seasonality,
)


class TestCalibrateDividendCurve:
    def _synthetic_futures(self, spot, rate, q, tenors):
        """Generate futures from known constant yield."""
        return [spot * q * T for T in tenors]

    def test_linear_roundtrip(self):
        spot, rate, q = 100, 0.05, 0.02
        tenors = [0.25, 0.5, 1.0, 2.0]
        futures = self._synthetic_futures(spot, rate, q, tenors)

        result = calibrate_dividend_curve(spot, rate, tenors, futures, method="linear")
        assert result.rmse < 0.01
        assert result.method == "linear"

    def test_optimize_roundtrip(self):
        spot, rate, q = 100, 0.05, 0.02
        tenors = [0.25, 0.5, 1.0, 2.0]
        futures = self._synthetic_futures(spot, rate, q, tenors)

        result = calibrate_dividend_curve(spot, rate, tenors, futures, method="optimize")
        assert result.rmse < 0.01
        assert result.method == "optimize"

    def test_spline_roundtrip(self):
        spot, rate, q = 100, 0.05, 0.02
        tenors = [0.25, 0.5, 1.0, 2.0]
        futures = self._synthetic_futures(spot, rate, q, tenors)

        result = calibrate_dividend_curve(spot, rate, tenors, futures, method="spline")
        assert result.rmse < 0.01
        assert result.method == "spline"

    def test_optimize_vs_linear(self):
        """Optimised should fit at least as well as linear."""
        spot, rate = 100, 0.05
        tenors = [0.25, 0.5, 1.0, 2.0, 3.0]
        # Non-constant yield → linear will have error
        futures = [0.5, 1.1, 2.5, 5.5, 9.0]

        r_lin = calibrate_dividend_curve(spot, rate, tenors, futures, method="linear")
        r_opt = calibrate_dividend_curve(spot, rate, tenors, futures, method="optimize")

        assert r_opt.rmse <= r_lin.rmse + 0.01

    def test_yields_positive(self):
        spot, rate = 100, 0.05
        tenors = [0.25, 0.5, 1.0, 2.0]
        futures = [0.5, 1.0, 2.0, 4.0]

        result = calibrate_dividend_curve(spot, rate, tenors, futures, method="optimize")
        assert all(y >= 0 for y in result.curve.implied_yields)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            calibrate_dividend_curve(100, 0.05, [1.0], [2.0], method="unknown")

    def test_to_dict(self):
        result = calibrate_dividend_curve(100, 0.05, [1.0], [2.0], method="linear")
        d = result.to_dict()
        assert "rmse" in d


class TestCalibrateFromOptions:
    def test_basic(self):
        spot, rate = 100, 0.05
        # Synthetic: PV(div) = S - K·df - (C - P)
        # If no dividends: C - P = S - K·df → PV(div) = 0
        # With 2% yield for T=1: PV(div) ≈ S·q·T·df = 100·0.02·1·0.95 ≈ 1.9
        T = 1.0
        K = 100
        df = math.exp(-rate * T)
        div_pv = 1.9
        # C - P = S - div_pv - K·df
        c_minus_p = spot - div_pv - K * df

        result = calibrate_from_options(
            spot,
            [{"T": T, "strike": K, "call": c_minus_p + 5, "put": 5}],
            rate,
        )
        assert result.curve.cumulative_dividends[0] == pytest.approx(div_pv, abs=0.01)

    def test_multi_expiry(self):
        spot, rate = 100, 0.05
        options = []
        for T in [0.25, 0.5, 1.0]:
            K = 100
            df = math.exp(-rate * T)
            div_pv = spot * 0.02 * T  # simple linear div
            c_minus_p = spot - div_pv - K * df
            options.append({"T": T, "strike": K, "call": c_minus_p + 5, "put": 5})

        result = calibrate_from_options(spot, options, rate)
        assert len(result.curve.tenors) == 3
        # Cumulative should be increasing
        cum = result.curve.cumulative_dividends
        assert cum[0] < cum[1] < cum[2]


class TestSeasonality:
    def test_uniform_seasonality(self):
        """Uniform cumulative → equal quarterly weights."""
        from pricebook.equity.dividend_advanced import DividendCurve
        curve = DividendCurve(
            tenors=np.array([0.25, 0.5, 0.75, 1.0, 2.0]),
            cumulative_dividends=np.array([0.5, 1.0, 1.5, 2.0, 4.0]),
            implied_yields=np.array([0.02, 0.02, 0.02, 0.02, 0.02]),
            method="test",
        )
        s = dividend_curve_seasonality(curve)
        assert len(s.quarterly_weights) == 4
        assert sum(s.quarterly_weights) == pytest.approx(1.0)
        # Uniform → all weights ≈ 0.25
        assert all(abs(w - 0.25) < 0.01 for w in s.quarterly_weights)

    def test_q2_heavy(self):
        """Heavy Q2 (ex-date season) → Q2 weight highest."""
        from pricebook.equity.dividend_advanced import DividendCurve
        curve = DividendCurve(
            tenors=np.array([0.25, 0.5, 0.75, 1.0]),
            cumulative_dividends=np.array([0.3, 1.5, 1.8, 2.0]),
            implied_yields=np.array([0.012, 0.030, 0.024, 0.020]),
            method="test",
        )
        s = dividend_curve_seasonality(curve)
        assert s.peak_quarter == 2  # Q2 has biggest increment

    def test_to_dict(self):
        from pricebook.equity.dividend_advanced import DividendCurve
        curve = DividendCurve(
            tenors=np.array([0.5, 1.0]),
            cumulative_dividends=np.array([1.0, 2.0]),
            implied_yields=np.array([0.02, 0.02]),
            method="test",
        )
        s = dividend_curve_seasonality(curve)
        d = s.to_dict()
        assert "quarterly_weights" in d


# ---- Canonical CalibrationResult (G1 P2 widen producers) ----

class TestDividendCanonicalResult:
    def test_builds_faithful_residuals_and_caches(self):
        spot, rate = 100.0, 0.05
        tenors = [1.0, 2.0, 3.0]
        futures = [2.0, 4.1, 6.0]
        result = calibrate_dividend_curve(spot, rate, tenors, futures, method="optimize")
        cr = result.to_calibration_result()
        assert cr.model_class == "dividend_curve"
        assert cr.optimiser.algorithm == "optimize"
        assert len(cr.residuals) == 3
        # cached: second call returns the same instance (stable id)
        assert result.to_calibration_result() is cr

    def test_persists_via_db(self):
        from pricebook.db.db import PricebookDB
        result = calibrate_dividend_curve(100.0, 0.05, [1.0, 2.0], [2.0, 4.0], method="spline")
        with PricebookDB(":memory:") as db:
            cid = db.save_calibration(result)
            assert db.load_calibration(cid) == result.to_calibration_result()
            assert db.list_calibrations(model_class="dividend_curve")[0]["calibration_id"] == cid
