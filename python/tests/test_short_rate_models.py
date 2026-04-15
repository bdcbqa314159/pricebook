"""Tests for short rate model deepening."""

import math

import numpy as np
import pytest

from pricebook.short_rate_models import (
    AffineModel,
    BKRateModel,
    CIRPPRateModel,
    CheyetteModel,
)


# ---- BK rates ----

class TestBKRateModel:
    def test_always_positive(self):
        model = BKRateModel(a=0.5, sigma=0.2)
        result = model.simulate(0.05, 5.0, n_paths=5_000, seed=42)
        assert np.all(result.rate_paths > 0)

    def test_zcb_between_zero_and_one(self):
        model = BKRateModel(a=0.5, sigma=0.1)
        result = model.simulate(0.05, 5.0, n_paths=20_000, seed=42)
        assert 0 < result.zcb_price < 1

    def test_higher_rate_lower_zcb(self):
        model = BKRateModel(a=0.5, sigma=0.1)
        low = model.simulate(0.02, 5.0, n_paths=20_000, seed=42)
        high = model.simulate(0.08, 5.0, n_paths=20_000, seed=42)
        assert high.zcb_price < low.zcb_price

    def test_shape(self):
        model = BKRateModel(a=0.5, sigma=0.1)
        result = model.simulate(0.05, 1.0, n_steps=50, n_paths=100, seed=42)
        assert result.rate_paths.shape == (100, 51)


# ---- CIR++ rates ----

class TestCIRPPRateModel:
    def test_zcb_positive(self):
        model = CIRPPRateModel(kappa=2.0, theta=0.04, xi=0.1,
                               market_rates=[(1, 0.05), (5, 0.05)])
        result = model.simulate(0.05, 5.0, n_paths=20_000, seed=42)
        assert 0 < result.zcb_price < 1

    def test_analytical_matches_mc(self):
        """CIR++ analytical ZCB should be close to MC."""
        model = CIRPPRateModel(kappa=2.0, theta=0.04, xi=0.1,
                               market_rates=[(1, 0.04), (5, 0.04)])
        result = model.simulate(0.04, 3.0, n_paths=50_000, seed=42)
        assert result.zcb_price == pytest.approx(result.zcb_analytical, rel=0.15)

    def test_x_paths_non_negative(self):
        model = CIRPPRateModel(kappa=2.0, theta=0.04, xi=0.1)
        result = model.simulate(0.05, 3.0, n_paths=5_000, seed=42)
        # x component should be non-negative (CIR)
        phi_vals = np.array([model.phi(t, 0.04) for t in result.times])
        x_paths = result.rate_paths - phi_vals[np.newaxis, :]
        assert np.all(x_paths >= -1e-6)


# ---- Cheyette ----

class TestCheyetteModel:
    def test_zcb_positive(self):
        model = CheyetteModel(kappa=0.1, sigma=0.01)
        result = model.simulate(5.0, n_paths=20_000, seed=42)
        assert 0 < result.zcb_price < 1

    def test_x_starts_at_zero(self):
        model = CheyetteModel(kappa=0.1, sigma=0.01)
        result = model.simulate(3.0, n_paths=100, seed=42)
        assert np.all(result.x_paths[:, 0] == 0.0)
        assert np.all(result.y_paths[:, 0] == 0.0)

    def test_rate_near_forward(self):
        """Mean rate should be near the forward curve."""
        model = CheyetteModel(kappa=0.5, sigma=0.005,
                              forward_rates=[(5, 0.04), (10, 0.04)])
        result = model.simulate(3.0, n_paths=20_000, seed=42)
        mean_terminal = result.rate_paths[:, -1].mean()
        assert mean_terminal == pytest.approx(0.04, rel=0.15)

    def test_analytical_zcb(self):
        model = CheyetteModel(kappa=0.1, sigma=0.01,
                              forward_rates=[(10, 0.05)])
        zcb = model.zcb_analytical(5.0)
        assert zcb == pytest.approx(math.exp(-0.05 * 5), rel=0.01)


# ---- Affine (Dai-Singleton) ----

class TestAffineModel:
    def test_vasicek_is_a01(self):
        """A_0(1) = Vasicek (Gaussian, no square-root)."""
        model = AffineModel([0.5], [0.05], [0.01], m=0)
        assert model.model_class == "A_0(1)"
        result = model.zcb_price([0.05], 5.0)
        assert 0 < result.zcb_price < 1

    def test_cir_is_a11(self):
        """A_1(1) = CIR (square-root diffusion)."""
        model = AffineModel([2.0], [0.04], [0.1], m=1)
        assert model.model_class == "A_1(1)"
        result = model.zcb_price([0.05], 5.0)
        assert 0 < result.zcb_price < 1

    def test_g2pp_is_a02(self):
        """A_0(2) = G2++ (two Gaussian factors)."""
        model = AffineModel([0.5, 1.0], [0.03, 0.02], [0.01, 0.008], m=0)
        assert model.model_class == "A_0(2)"
        result = model.zcb_price([0.03, 0.02], 5.0)
        assert 0 < result.zcb_price < 1

    def test_higher_rate_lower_zcb(self):
        model = AffineModel([0.5], [0.05], [0.01], m=0)
        low = model.zcb_price([0.02], 5.0)
        high = model.zcb_price([0.08], 5.0)
        assert high.zcb_price < low.zcb_price

    def test_zcb_decreasing_in_maturity(self):
        model = AffineModel([0.5], [0.05], [0.01], m=0)
        short = model.zcb_price([0.05], 1.0)
        long = model.zcb_price([0.05], 10.0)
        assert long.zcb_price < short.zcb_price
