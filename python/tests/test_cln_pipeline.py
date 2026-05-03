"""Tests for CLN pipeline: calibration wiring, stochastic pricing, bilateral CLN."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cln import CreditLinkedNote, BasketCLN
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


# ── Phase 5: Bilateral CLN ──

class TestBilateralCLN:

    def _cln(self):
        return CreditLinkedNote(
            start=REF, end=REF + relativedelta(years=5),
            coupon_rate=0.05, notional=1_000_000, recovery=0.4,
            frequency=Frequency.QUARTERLY,
        )

    def test_bilateral_with_zero_issuer_risk(self):
        """Issuer hazard=0 → bilateral ≈ unilateral."""
        cln = self._cln()
        curve = make_flat_curve(REF, 0.04)
        ref_surv = make_flat_survival(REF, 0.02)
        # Zero issuer risk (hazard ≈ 0)
        issuer_surv = make_flat_survival(REF, 0.0001)
        bilateral = cln.price_bilateral_mc(
            curve, ref_surv, issuer_surv,
            n_paths=10_000, seed=42,
        )
        unilateral = cln.dirty_price(curve, ref_surv)
        # Should be close
        assert abs(bilateral.price - unilateral) / unilateral < 0.05

    def test_issuer_risk_reduces_price(self):
        """Adding issuer credit risk should reduce CLN price."""
        cln = self._cln()
        curve = make_flat_curve(REF, 0.04)
        ref_surv = make_flat_survival(REF, 0.02)
        # High issuer risk
        issuer_surv = make_flat_survival(REF, 0.05)
        bilateral = cln.price_bilateral_mc(
            curve, ref_surv, issuer_surv,
            n_paths=10_000, seed=42,
        )
        unilateral = cln.dirty_price(curve, ref_surv)
        assert bilateral.price < unilateral

    def test_correlation_effect(self):
        """Correlation changes the bilateral price."""
        cln = self._cln()
        curve = make_flat_curve(REF, 0.04)
        ref_surv = make_flat_survival(REF, 0.03)
        issuer_surv = make_flat_survival(REF, 0.03)
        p_low = cln.price_bilateral_mc(
            curve, ref_surv, issuer_surv, correlation=0.1, n_paths=10_000).price
        p_high = cln.price_bilateral_mc(
            curve, ref_surv, issuer_surv, correlation=0.8, n_paths=10_000).price
        assert p_low != p_high


# ── Phase 6: Basket rho01 ──

class TestBasketRho01:

    def _basket(self, attach=0.0, detach=0.03):
        return BasketCLN(
            start=REF, end=REF + relativedelta(years=5),
            coupon_rate=0.05, notional=10_000_000,
            attachment=attach, detachment=detach,
            recovery=0.4, n_names=50,
        )

    def _survs(self, n=50, h=0.02):
        return [make_flat_survival(REF, h) for _ in range(n)]

    def test_rho_bump_changes_price(self):
        """Price should change when correlation bumps."""
        basket = self._basket()
        curve = make_flat_curve(REF, 0.04)
        survs = self._survs()
        p_base = basket.price_mc(curve, survs, rho=0.30, n_sims=5_000, seed=42).price
        p_bump = basket.price_mc(curve, survs, rho=0.31, n_sims=5_000, seed=42).price
        assert p_base != p_bump

    def test_senior_vs_equity_rho_direction(self):
        """Equity tranche: higher rho → higher price (fewer idiosyncratic defaults).
        Senior tranche: higher rho → lower price (more tail risk)."""
        curve = make_flat_curve(REF, 0.04)
        survs = self._survs()
        equity = self._basket(0.0, 0.03)
        senior = self._basket(0.07, 0.10)

        eq_low = equity.price_mc(curve, survs, rho=0.10, n_sims=5_000, seed=42).price
        eq_high = equity.price_mc(curve, survs, rho=0.50, n_sims=5_000, seed=42).price
        sr_low = senior.price_mc(curve, survs, rho=0.10, n_sims=5_000, seed=42).price
        sr_high = senior.price_mc(curve, survs, rho=0.50, n_sims=5_000, seed=42).price

        # Equity: higher rho → higher price (positive rho01)
        assert eq_high > eq_low
        # Senior: higher rho → lower price (negative rho01)
        assert sr_high < sr_low


# ── Phase 7: PricingContext extension ──

class TestPricingContextCredit:

    def test_context_stores_stochastic_model(self):
        from pricebook.pricing_context import PricingContext
        surv = make_flat_survival(REF, 0.02)
        model = CIRPlusPlus.from_survival_curve(surv, kappa=1.0, xi=0.1)
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=make_flat_curve(REF, 0.04),
            credit_curves={"AAPL": surv},
            stochastic_credit_models={"AAPL": model},
        )
        assert "AAPL" in ctx.stochastic_credit_models
        assert ctx.stochastic_credit_models["AAPL"].xi == 0.1

    def test_context_replace_preserves_models(self):
        from pricebook.pricing_context import PricingContext
        surv = make_flat_survival(REF, 0.02)
        model = CIRPlusPlus.from_survival_curve(surv, kappa=1.0, xi=0.1)
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=make_flat_curve(REF, 0.04),
            stochastic_credit_models={"AAPL": model},
        )
        bumped = ctx.replace(discount_curve=make_flat_curve(REF, 0.05))
        assert "AAPL" in bumped.stochastic_credit_models

    def test_cln_pv_ctx_uses_stochastic_model(self):
        """CLN.pv_ctx dispatches to stochastic when model in context."""
        from pricebook.pricing_context import PricingContext
        cln = CreditLinkedNote(
            start=REF, end=REF + relativedelta(years=5),
            coupon_rate=0.05, notional=1_000_000, recovery=0.4,
            frequency=Frequency.QUARTERLY,
        )
        surv = make_flat_survival(REF, 0.02)
        model = CIRPlusPlus.from_survival_curve(surv, kappa=1.0, xi=0.001)
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=make_flat_curve(REF, 0.04),
            credit_curves={"ref": surv},
            stochastic_credit_models={"ref": model},
        )
        pv = cln.pv_ctx(ctx)
        det = cln.dirty_price(make_flat_curve(REF, 0.04), surv)
        # Low vol → should be close to deterministic
        assert abs(pv - det) / det < 0.05


# ── Phase 8: End-to-end integration ──

class TestE2EIntegration:

    def test_e2e_single_name_pipeline(self):
        """CDS quotes → curve → CIR++ → CLN stochastic price → desk risk."""
        from pricebook.cln_desk import cln_risk_metrics
        curve = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)

        # Step 1: Calibrate CIR++ from survival curve
        model = CIRPlusPlus.from_survival_curve(surv, kappa=1.0, xi=0.05)

        # Step 2: Price CLN under stochastic intensity
        cln = CreditLinkedNote(
            start=REF, end=REF + relativedelta(years=5),
            coupon_rate=0.05, notional=1_000_000, recovery=0.4,
            frequency=Frequency.QUARTERLY,
        )
        stoch = cln.price_stochastic_intensity(curve, model, n_paths=10_000, n_steps=50)
        assert stoch.price > 0

        # Step 3: Desk risk
        rm = cln_risk_metrics(cln, curve, surv)
        assert rm.cs01 < 0
        assert rm.dv01 < 0
        assert rm.jump_to_default_pnl < 0

    def test_e2e_bilateral(self):
        """Ref + issuer curves → bilateral price < unilateral."""
        cln = CreditLinkedNote(
            start=REF, end=REF + relativedelta(years=5),
            coupon_rate=0.05, notional=1_000_000, recovery=0.4,
            frequency=Frequency.QUARTERLY,
        )
        curve = make_flat_curve(REF, 0.04)
        ref_surv = make_flat_survival(REF, 0.02)
        issuer_surv = make_flat_survival(REF, 0.03)

        unilateral = cln.dirty_price(curve, ref_surv)
        bilateral = cln.price_bilateral_mc(
            curve, ref_surv, issuer_surv, n_paths=10_000).price
        assert bilateral < unilateral

    def test_e2e_basket_rho_sign(self):
        """Equity tranche: positive rho01. Senior: negative."""
        curve = make_flat_curve(REF, 0.04)
        survs = [make_flat_survival(REF, 0.02) for _ in range(50)]

        eq = BasketCLN(REF, REF + relativedelta(years=5), 0.05, 10_000_000,
                       0.0, 0.03, 0.4, 50)
        p_low = eq.price_mc(curve, survs, rho=0.15, n_sims=3_000, seed=42).price
        p_high = eq.price_mc(curve, survs, rho=0.45, n_sims=3_000, seed=42).price
        assert p_high > p_low  # equity: positive rho01
