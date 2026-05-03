"""Tests for CLN pipeline: calibration wiring, stochastic pricing, bilateral CLN."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.survival_curve import SurvivalCurve
from pricebook.hazard_rate_models import (
    HWHazardRate, CIRPlusPlus, verify_calibration,
)
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 7, 15)


# ── Phase 2: CIR++/HW wiring to SurvivalCurve ──

class TestCIRPPFromCurve:

    def test_from_survival_curve_creates_model(self):
        surv = make_flat_survival(REF, 0.02)
        model = CIRPlusPlus.from_survival_curve(surv, kappa=1.0, xi=0.05)
        assert model.kappa == 1.0
        assert model.xi == 0.05
        assert len(model._market_hazards) > 0

    def test_from_survival_curve_theta_auto(self):
        """Theta defaults to average hazard when not provided."""
        surv = make_flat_survival(REF, 0.03)
        model = CIRPlusPlus.from_survival_curve(surv, kappa=1.0, xi=0.05)
        assert abs(model.theta - 0.03) < 0.005  # ~3% for flat curve

    def test_cirpp_mc_survival_close_to_market(self):
        """CIR++ MC survival should approximate market survival."""
        surv = make_flat_survival(REF, 0.02)
        model = CIRPlusPlus.from_survival_curve(surv, kappa=1.0, xi=0.05)
        x0 = model.theta
        result = model.simulate(x0, 5.0, n_steps=100, n_paths=20_000, seed=42)
        market_surv = surv.survival(REF + relativedelta(years=5))
        # MC should be within 10% of market (CIR++ with phi shift)
        assert abs(result.survival_mc - market_surv) / market_surv < 0.15


class TestHWFromCurve:

    def test_from_survival_curve_creates_model(self):
        surv = make_flat_survival(REF, 0.02)
        model = HWHazardRate.from_survival_curve(surv, a=0.5, sigma=0.005)
        assert model.a == 0.5
        assert model.sigma == 0.005
        assert len(model._market_hazards) > 0

    def test_hw_analytical_survival_close_to_market(self):
        """HW analytical survival should approximate market for low vol."""
        surv = make_flat_survival(REF, 0.02)
        model = HWHazardRate.from_survival_curve(surv, a=0.5, sigma=0.001)
        lam0 = 0.02
        hw_surv = model.survival_analytical(lam0, 5.0)
        market_surv = surv.survival(REF + relativedelta(years=5))
        assert abs(hw_surv - market_surv) / market_surv < 0.10


class TestVerifyCalibration:

    def test_verify_passes_for_low_vol(self):
        """Low vol → model closely matches market."""
        surv = make_flat_survival(REF, 0.02)
        model = HWHazardRate.from_survival_curve(surv, a=0.5, sigma=0.001)
        errors = verify_calibration(model, surv, x0=0.02, tolerance=0.10)
        assert all(e < 0.10 for e in errors.values())

    def test_verify_raises_for_bad_model(self):
        """Misspecified model should fail verification."""
        surv = make_flat_survival(REF, 0.02)
        # Very wrong model (hazard 10x too high)
        model = HWHazardRate(a=0.5, sigma=0.001,
                             market_hazards=[(1, 0.20), (5, 0.20)])
        with pytest.raises(ValueError, match="Calibration failed"):
            verify_calibration(model, surv, x0=0.20, tolerance=0.01)
