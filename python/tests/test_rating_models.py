"""Tests for rating-based credit models."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from pricebook.rating_transition import RatingTransitionMatrix, standard_generator
from pricebook.rating_models import (
    CalibrationResult,
    MomentumTransitionMatrix,
    TTCPITResult,
    calibrate_generator,
    pit_to_ttc,
    time_varying_generator,
    ttc_to_pit,
)


# ---- Generator calibration ----

class TestCalibrateGenerator:
    def test_round_trip(self):
        """Calibrated Q reproduces input P via exp(Qt)."""
        base = standard_generator()
        P_1y = base.transition_prob(1.0)
        result = calibrate_generator(base.ratings, P_1y, horizon=1.0)
        P_recovered = expm(result.generator.Q * 1.0)
        np.testing.assert_allclose(P_recovered, P_1y, atol=0.02)

    def test_valid_generator(self):
        """Calibrated Q has non-negative off-diag, rows sum to 0."""
        base = standard_generator()
        P = base.transition_prob(1.0)
        result = calibrate_generator(base.ratings, P)
        Q = result.generator.Q
        # Off-diagonal non-negative
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                if i != j:
                    assert Q[i, j] >= -1e-10
        # Rows sum to ~0
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-8)

    def test_default_state_absorbing(self):
        base = standard_generator()
        P = base.transition_prob(1.0)
        result = calibrate_generator(base.ratings, P)
        assert np.allclose(result.generator.Q[-1, :], 0.0)


# ---- TTC / PIT ----

class TestTTCPIT:
    def test_neutral_cycle(self):
        """Zero cycle factor: PIT = TTC."""
        result = ttc_to_pit(0.02, 0.0)
        assert result.pit_pd == pytest.approx(0.02, rel=1e-6)
        assert result.regime == "neutral"

    def test_downturn_increases_pd(self):
        """Positive cycle factor → higher PIT PD."""
        result = ttc_to_pit(0.02, 0.5)
        assert result.pit_pd > result.ttc_pd
        assert result.regime == "contraction"

    def test_expansion_decreases_pd(self):
        """Negative cycle factor → lower PIT PD."""
        result = ttc_to_pit(0.02, -0.5)
        assert result.pit_pd < result.ttc_pd
        assert result.regime == "expansion"

    def test_round_trip(self):
        """TTC → PIT → TTC should round-trip."""
        ttc = 0.03
        factor = 0.4
        pit = ttc_to_pit(ttc, factor).pit_pd
        recovered = pit_to_ttc(pit, factor).ttc_pd
        assert recovered == pytest.approx(ttc, rel=1e-6)

    def test_extreme_pd(self):
        """Near-zero and near-one PDs should not blow up."""
        r1 = ttc_to_pit(0.0001, 1.0)
        assert 0 < r1.pit_pd < 1
        r2 = ttc_to_pit(0.99, -1.0)
        assert 0 < r2.pit_pd < 1


# ---- Momentum ----

class TestMomentumTransitionMatrix:
    def test_higher_default_rate_than_base(self):
        """Momentum should produce higher default probability than memoryless."""
        base = standard_generator()
        momentum = MomentumTransitionMatrix(base, momentum_factor=2.0)

        pd_base = base.default_probability("BBB", 5.0)
        pd_mom = momentum.default_probability_mc("BBB", 5.0, n_steps=50,
                                                   n_paths=20_000, seed=42)
        assert pd_mom > pd_base * 0.9  # should be higher (allow MC noise)

    def test_momentum_factor_one_matches_base(self):
        """momentum_factor=1.0 should match the base model."""
        base = standard_generator()
        momentum = MomentumTransitionMatrix(base, momentum_factor=1.0)
        pd_base = base.default_probability("BBB", 3.0)
        pd_mom = momentum.default_probability_mc("BBB", 3.0, n_steps=30,
                                                   n_paths=20_000, seed=42)
        assert pd_mom == pytest.approx(pd_base, rel=0.10)

    def test_default_state_absorbing(self):
        base = standard_generator()
        momentum = MomentumTransitionMatrix(base, 1.5)
        paths = momentum.simulate_paths("CCC", 5.0, 50, 100, seed=42)
        # Once defaulted, should stay defaulted
        for p in range(100):
            defaulted = False
            for i in range(51):
                if paths[p, i] == base.default_state:
                    defaulted = True
                if defaulted:
                    assert paths[p, i] == base.default_state


# ---- Time-varying generator ----

class TestTimeVaryingGenerator:
    def test_stress_increases_downgrades(self):
        base = standard_generator()
        adjusted = time_varying_generator(base, [(0.0, 0.5)])
        Q_stress = adjusted[0][1].Q

        # Downgrade intensities should be higher than base
        for i in range(base.n - 1):
            for j in range(i + 1, base.n):
                assert Q_stress[i, j] >= base.Q[i, j] - 1e-10

    def test_expansion_increases_upgrades(self):
        base = standard_generator()
        adjusted = time_varying_generator(base, [(0.0, -0.3)])
        Q_exp = adjusted[0][1].Q

        # Upgrade intensities should be higher than base
        for i in range(1, base.n - 1):
            for j in range(i):
                assert Q_exp[i, j] >= base.Q[i, j] - 1e-10

    def test_neutral_unchanged(self):
        base = standard_generator()
        adjusted = time_varying_generator(base, [(0.0, 0.0)])
        Q_neutral = adjusted[0][1].Q
        np.testing.assert_allclose(Q_neutral, base.Q, atol=1e-10)

    def test_rows_sum_to_zero(self):
        base = standard_generator()
        adjusted = time_varying_generator(base, [(0.0, 0.8)])
        Q = adjusted[0][1].Q
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-10)
