"""Tests for FX jumps and regime switching."""

import math

import numpy as np
import pytest

from pricebook.fx_jumps import (
    BatesFXModel,
    BatesFXResult,
    InterventionResult,
    MertonFXResult,
    RegimeSwitchingResult,
    RegimeSwitchingVol,
    fx_intervention_adjustment,
    merton_fx_price,
)


# ---- Merton FX ----

class TestMertonFX:
    def test_basic(self):
        result = merton_fx_price(1.0, 1.0, 0.02, 0.01, 0.10, 1.0,
                                   lambda_jump=0.5, jump_mean=-0.05, jump_vol=0.10)
        assert isinstance(result, MertonFXResult)
        assert result.price > 0

    def test_no_jumps_matches_bs(self):
        """λ=0 → Merton reduces to Black-Scholes."""
        from pricebook.black76 import black76_price, OptionType
        result = merton_fx_price(1.0, 1.0, 0.02, 0.01, 0.10, 1.0,
                                   lambda_jump=0.0, jump_mean=0.0, jump_vol=0.01)
        F = 1.0 * math.exp((0.02 - 0.01) * 1.0)
        df = math.exp(-0.02)
        bs = black76_price(F, 1.0, 0.10, 1.0, df, OptionType.CALL)
        assert result.price == pytest.approx(bs, rel=0.02)

    def test_negative_jumps_raise_put(self):
        """Negative mean jumps → put value rises (fat left tail)."""
        put_no_jump = merton_fx_price(1.0, 1.0, 0.02, 0.01, 0.10, 1.0,
                                       0.0, 0.0, 0.01, is_call=False)
        put_jump = merton_fx_price(1.0, 1.0, 0.02, 0.01, 0.10, 1.0,
                                    1.0, -0.15, 0.10, is_call=False)
        assert put_jump.price > put_no_jump.price

    def test_jumps_raise_otm_options(self):
        """Jumps add fat tails → OTM options more valuable."""
        otm_no_jump = merton_fx_price(1.0, 1.10, 0.02, 0.01, 0.10, 1.0,
                                       0.0, 0.0, 0.01)
        otm_jump = merton_fx_price(1.0, 1.10, 0.02, 0.01, 0.10, 1.0,
                                    1.0, 0.05, 0.15)
        assert otm_jump.price > otm_no_jump.price


# ---- Bates FX ----

class TestBatesFX:
    def test_basic(self):
        model = BatesFXModel(v0=0.02, kappa_v=1.0, theta_v=0.02, xi=0.3, rho=-0.3,
                              lambda_jump=0.3, jump_mean=-0.05, jump_vol=0.10)
        result = model.simulate_option(1.0, 1.0, 0.02, 0.01, 1.0,
                                         n_paths=2000, n_steps=50, seed=42)
        assert isinstance(result, BatesFXResult)
        assert result.price > 0

    def test_paths_positive(self):
        model = BatesFXModel(0.02, 1.0, 0.02, 0.3, -0.3, 0.3, -0.05, 0.10)
        result = model.simulate_option(1.0, 1.0, 0.02, 0.01, 1.0,
                                         n_paths=500, n_steps=30, seed=42)
        assert np.all(result.paths >= 0)
        assert np.all(result.variance_paths >= 0)

    def test_no_jumps_reduces_to_heston(self):
        """λ=0 → Bates is Heston."""
        no_jumps = BatesFXModel(0.02, 1.0, 0.02, 0.3, -0.3, 0.0, 0.0, 0.01)
        result = no_jumps.simulate_option(1.0, 1.0, 0.02, 0.01, 1.0,
                                           n_paths=1000, n_steps=30, seed=42)
        assert result.n_jumps_total == 0

    def test_jumps_accumulate(self):
        """With positive λ, some jumps should occur."""
        with_jumps = BatesFXModel(0.02, 1.0, 0.02, 0.3, -0.3, 2.0, 0.0, 0.05)
        result = with_jumps.simulate_option(1.0, 1.0, 0.02, 0.01, 1.0,
                                             n_paths=500, n_steps=30, seed=42)
        assert result.n_jumps_total > 0


# ---- Regime switching ----

class TestRegimeSwitchingVol:
    def _make_model(self):
        # 3 regimes: low (0.05), normal (0.10), crisis (0.30)
        Q = np.array([
            [-0.5, 0.4, 0.1],    # from low
            [0.3, -0.5, 0.2],    # from normal
            [0.1, 0.4, -0.5],    # from crisis
        ])
        return RegimeSwitchingVol([0.05, 0.10, 0.30], Q, initial_regime=1)

    def test_basic(self):
        model = self._make_model()
        result = model.simulate(1.0, 0.02, 0.01, 1.0,
                                 n_paths=500, n_steps=30, seed=42)
        assert isinstance(result, RegimeSwitchingResult)

    def test_shapes(self):
        model = self._make_model()
        result = model.simulate(1.0, 0.02, 0.01, 1.0,
                                 n_paths=100, n_steps=30, seed=42)
        assert result.spot_paths.shape == (100, 31)
        assert result.regime_paths.shape == (100, 31)

    def test_initial_regime(self):
        model = self._make_model()
        result = model.simulate(1.0, 0.02, 0.01, 1.0,
                                 n_paths=50, n_steps=20, seed=42)
        assert np.all(result.regime_paths[:, 0] == 1)

    def test_regime_durations(self):
        model = self._make_model()
        result = model.simulate(1.0, 0.02, 0.01, 1.0,
                                 n_paths=200, n_steps=50, seed=42)
        # All 3 regimes should be visited at least once
        assert len(result.mean_regime_duration) == 3
        # Durations should sum approximately to T
        total = sum(result.mean_regime_duration.values())
        assert total == pytest.approx(1.0, rel=0.1)


# ---- Intervention ----

class TestInterventionAdjustment:
    def test_basic(self):
        result = fx_intervention_adjustment(
            1.0, 1.0, 0.02, 0.01, 0.05, 1.0,
            break_intensity=0.1, break_jump_size=-0.20,
        )
        assert isinstance(result, InterventionResult)
        assert result.break_probability > 0

    def test_no_intervention_matches_base(self):
        """Zero break intensity → adjusted = base."""
        result = fx_intervention_adjustment(
            1.0, 1.0, 0.02, 0.01, 0.10, 1.0,
            break_intensity=0.0, break_jump_size=-0.20,
        )
        assert result.intervention_adjusted_price == pytest.approx(
            result.base_price, rel=0.02)

    def test_intervention_raises_put(self):
        """Devaluation risk → higher put value."""
        base = fx_intervention_adjustment(
            1.0, 1.0, 0.02, 0.01, 0.05, 1.0,
            break_intensity=0.0, break_jump_size=-0.20,
            is_call=False,
        )
        with_risk = fx_intervention_adjustment(
            1.0, 1.0, 0.02, 0.01, 0.05, 1.0,
            break_intensity=0.5, break_jump_size=-0.20,
            is_call=False,
        )
        assert with_risk.intervention_adjusted_price > base.intervention_adjusted_price

    def test_break_probability_bounded(self):
        result = fx_intervention_adjustment(
            1.0, 1.0, 0.02, 0.01, 0.05, 1.0,
            break_intensity=2.0, break_jump_size=-0.20,
        )
        assert 0 <= result.break_probability <= 1

    def test_expected_loss_positive(self):
        """For negative break jump → expected loss is positive."""
        result = fx_intervention_adjustment(
            1.0, 1.0, 0.02, 0.01, 0.05, 1.0,
            break_intensity=0.5, break_jump_size=-0.20,
        )
        assert result.expected_loss_from_break > 0
