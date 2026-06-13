"""Regression for L2 Wave-2 audit — `calibrate_svensson` had no
degeneracy guard for τ1 ≈ τ2.

The Svensson parameterisation has two decay constants `tau1` and
`tau2`.  When they collapse to (approximately) the same value, the
second Svensson factor becomes a linear combination of the first plus
the NS factor — so the objective surface develops a flat valley along
the (beta2, beta3) ridge.  Nelder-Mead can drift arbitrarily far in
that direction without the loss changing, producing wildly different
parameter sets that all "calibrate" equally well to the same input.

Pre-fix the calibration accepted such degenerate solutions silently.

Post-fix the objective rejects τ1 ≈ τ2 candidates (|τ1-τ2| < 0.05) by
returning a penalty sentinel, keeping the optimizer in the
identifiable region of parameter space.
"""

from __future__ import annotations

import pytest

from pricebook.curves.nelson_siegel import calibrate_svensson, svensson_yield


class TestNoTauDegeneracy:
    def test_calibrated_taus_well_separated(self):
        """A normal calibration should never land at τ1 ≈ τ2."""
        result = calibrate_svensson(
            tenors=[0.5, 1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 30.0],
            market_yields=[0.02, 0.025, 0.03, 0.035, 0.038, 0.040, 0.043, 0.045],
        )
        tau1 = result["tau1"]
        tau2 = result["tau2"]
        # The degeneracy guard kicks in at |tau1-tau2| < 0.05.
        assert abs(tau1 - tau2) >= 0.05 - 1e-6


class TestCalibrationStillWorks:
    def test_svensson_fits_smooth_curve(self):
        """The degeneracy guard must not break healthy calibration."""
        result = calibrate_svensson(
            tenors=[1.0, 2.0, 5.0, 10.0, 20.0],
            market_yields=[0.030, 0.035, 0.040, 0.045, 0.048],
        )
        assert result["rmse"] < 0.01

    def test_svensson_calibration_returns_finite_values(self):
        result = calibrate_svensson(
            tenors=[0.5, 1.0, 2.0, 5.0, 10.0],
            market_yields=[0.020, 0.025, 0.030, 0.035, 0.038],
        )
        for key in ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2"):
            import math as _m
            assert _m.isfinite(result[key])
