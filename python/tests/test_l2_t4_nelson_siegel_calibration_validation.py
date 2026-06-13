"""Regression for L2 Wave-2 audit — `calibrate_nelson_siegel` and
`calibrate_svensson` had three contract gaps.

Pre-fix:
1. Empty ``market_yields`` raised ``IndexError`` at
   ``market_yields[-1]`` while constructing the default initial guess,
   with no diagnostic.
2. ``len(tenors) != len(market_yields)`` was silently masked by
   ``zip()`` which truncates to the shorter sequence — extra rows
   were dropped without warning.  A user passing 10 tenors and 8
   yields got a calibration on the first 8 points only.
3. The optimizer's convergence flag was discarded — the returned dict
   gave no way for the caller to detect a non-converged calibration.

Post-fix:
1, 2 → ``ValueError`` upfront with diagnostic.
3 → new ``converged: bool`` field in the result dict.
"""

from __future__ import annotations

import pytest

from pricebook.curves.nelson_siegel import (
    calibrate_nelson_siegel,
    calibrate_svensson,
)


class TestEmptyInputsRaise:
    def test_empty_tenors_raises(self):
        with pytest.raises(ValueError, match="tenors must be non-empty"):
            calibrate_nelson_siegel(tenors=[], market_yields=[0.04, 0.05])

    def test_empty_yields_raises(self):
        with pytest.raises(ValueError, match="market_yields must be non-empty"):
            calibrate_nelson_siegel(tenors=[1.0, 2.0], market_yields=[])

    def test_svensson_empty_raises(self):
        with pytest.raises(ValueError, match="tenors must be non-empty"):
            calibrate_svensson(tenors=[], market_yields=[0.04])


class TestMismatchedLengthsRaise:
    def test_ns_mismatched_raises(self):
        with pytest.raises(ValueError, match="silently truncated"):
            calibrate_nelson_siegel(
                tenors=[1.0, 2.0, 5.0, 10.0],
                market_yields=[0.03, 0.04],  # mismatched length
            )

    def test_svensson_mismatched_raises(self):
        with pytest.raises(ValueError, match="silently truncated"):
            calibrate_svensson(
                tenors=[1.0, 2.0, 5.0],
                market_yields=[0.03, 0.04, 0.045, 0.05],
            )


class TestConvergedFieldReported:
    def test_ns_result_has_converged_field(self):
        result = calibrate_nelson_siegel(
            tenors=[1.0, 2.0, 5.0, 10.0],
            market_yields=[0.030, 0.035, 0.040, 0.045],
        )
        assert "converged" in result
        assert isinstance(result["converged"], bool)

    def test_svensson_result_has_converged_field(self):
        result = calibrate_svensson(
            tenors=[1.0, 2.0, 5.0, 10.0, 20.0],
            market_yields=[0.030, 0.035, 0.040, 0.045, 0.048],
        )
        assert "converged" in result
        assert isinstance(result["converged"], bool)


class TestHealthyCalibrationPreserved:
    def test_ns_calibrates_smooth_curve(self):
        result = calibrate_nelson_siegel(
            tenors=[1.0, 2.0, 5.0, 10.0, 20.0],
            market_yields=[0.030, 0.035, 0.040, 0.045, 0.048],
        )
        # RMSE should be tiny on a smooth synthetic curve.
        assert result["rmse"] < 0.01
