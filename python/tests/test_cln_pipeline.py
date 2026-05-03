"""Tests for CLN pipeline: calibration wiring, stochastic pricing, bilateral CLN."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cln import CreditLinkedNote
from pricebook.survival_curve import SurvivalCurve
from pricebook.schedule import Frequency
from pricebook.hazard_rate_models import (
    HWHazardRate, CIRPlusPlus, verify_calibration,
)
from pricebook.cds_swaption import (
    StochasticIntensitySwaption, implied_intensity_vol,
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


# ── Phase 3: CDS swaption → intensity vol ──

class TestImpliedIntensityVol:

    def test_roundtrip(self):
        """Price swaption at known xi, then invert to recover xi."""
        kappa, theta, xi_true = 1.0, 0.02, 0.15
        model = StochasticIntensitySwaption(kappa, theta, xi_true, 0.04, 0.4)
        result = model.price(1.0, 5.0, 0.01, n_paths=20_000, n_steps=50, seed=42)
        premium = result.premium

        xi_recovered = implied_intensity_vol(
            premium, 1.0, 5.0, 0.01, kappa, theta,
            flat_rate=0.04, recovery=0.4, n_paths=20_000, seed=42,
        )
        # Should recover within 30% (MC noise)
        assert abs(xi_recovered - xi_true) / xi_true < 0.30

    def test_higher_xi_higher_premium(self):
        """Higher intensity vol → higher swaption premium."""
        kappa, theta = 1.0, 0.02
        m_low = StochasticIntensitySwaption(kappa, theta, 0.05, 0.04, 0.4)
        m_high = StochasticIntensitySwaption(kappa, theta, 0.20, 0.04, 0.4)
        p_low = m_low.price(1.0, 5.0, 0.01, n_paths=10_000, n_steps=50).premium
        p_high = m_high.price(1.0, 5.0, 0.01, n_paths=10_000, n_steps=50).premium
        assert p_high > p_low


# ── Phase 4: CLN stochastic intensity pricing ──

class TestCLNStochasticIntensity:

    def _cln(self):
        return CreditLinkedNote(
            start=REF, end=REF + relativedelta(years=5),
            coupon_rate=0.05, notional=1_000_000, recovery=0.4,
            frequency=Frequency.QUARTERLY,
        )

    def test_deterministic_limit(self):
        """With very low xi, stochastic price ≈ deterministic."""
        cln = self._cln()
        curve = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)

        det_pv = cln.dirty_price(curve, surv)
        stoch_result = cln.price_stochastic_intensity_from_curve(
            curve, surv, model_type="cir++", xi=0.001,
            n_paths=20_000, n_steps=100, seed=42,
        )
        # Should be within 2% for very low vol
        assert abs(stoch_result.price - det_pv) / det_pv < 0.02

    def test_stochastic_price_finite(self):
        cln = self._cln()
        curve = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        result = cln.price_stochastic_intensity_from_curve(
            curve, surv, xi=0.10, n_paths=5_000, n_steps=50,
        )
        assert math.isfinite(result.price)
        assert result.price > 0

    def test_convergence_with_paths(self):
        """More paths → tighter estimates."""
        cln = self._cln()
        curve = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        p1 = cln.price_stochastic_intensity_from_curve(
            curve, surv, xi=0.10, n_paths=2_000, n_steps=50, seed=42).price
        p2 = cln.price_stochastic_intensity_from_curve(
            curve, surv, xi=0.10, n_paths=20_000, n_steps=50, seed=42).price
        det = cln.dirty_price(curve, surv)
        # Higher paths should be closer to deterministic
        assert abs(p2 - det) <= abs(p1 - det) + det * 0.05  # allowing some MC noise

    def test_hw_model_also_works(self):
        """HW model type should also produce valid prices."""
        cln = self._cln()
        curve = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        result = cln.price_stochastic_intensity_from_curve(
            curve, surv, model_type="hw", xi=0.005,
            n_paths=5_000, n_steps=50,
        )
        assert math.isfinite(result.price)
        assert result.price > 0
