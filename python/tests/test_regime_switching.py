"""Tests for regime-switching credit model."""

import math
import pytest
import numpy as np

from pricebook.credit.regime_switching import (
    RegimeSwitchingCredit, RegimeState, calibrate_regime_model,
)


@pytest.fixture
def two_state():
    """2-state model: expansion (low hazard) and recession (high hazard)."""
    return RegimeSwitchingCredit(
        hazard_rates=[0.005, 0.03],
        transition_matrix=[[0.95, 0.05],
                           [0.15, 0.85]],
        state_names=["expansion", "recession"],
    )


@pytest.fixture
def three_state():
    """3-state: expansion, normal, recession."""
    return RegimeSwitchingCredit(
        hazard_rates=[0.002, 0.01, 0.05],
        transition_matrix=[[0.90, 0.08, 0.02],
                           [0.05, 0.90, 0.05],
                           [0.02, 0.08, 0.90]],
        state_names=["expansion", "normal", "recession"],
    )


class TestSurvival:
    def test_survival_at_zero(self, two_state):
        assert two_state.survival(0.0) == 1.0

    def test_survival_decreasing(self, two_state):
        q1 = two_state.survival(1.0)
        q5 = two_state.survival(5.0)
        q10 = two_state.survival(10.0)
        assert q1 > q5 > q10

    def test_survival_between_flat_bounds(self, two_state):
        """Regime-switching survival should be between pure-expansion and pure-recession."""
        t = 5.0
        q_low = math.exp(-0.005 * t)   # pure expansion
        q_high = math.exp(-0.03 * t)   # pure recession
        q_mix = two_state.survival(t)
        assert q_high < q_mix < q_low

    def test_survival_conditional_on_state(self, two_state):
        """Conditioning on recession state → lower survival."""
        q_exp = two_state.survival(5.0, initial_state=0)
        q_rec = two_state.survival(5.0, initial_state=1)
        assert q_exp > q_rec

    def test_survival_term_structure(self, two_state):
        times = [1, 3, 5, 7, 10]
        qs = two_state.survival_term_structure(times)
        assert len(qs) == 5
        for i in range(1, len(qs)):
            assert qs[i] < qs[i - 1]

    def test_three_state_survival(self, three_state):
        q = three_state.survival(5.0)
        assert 0 < q < 1


class TestImpliedSpread:
    def test_implied_hazard_positive(self, two_state):
        h = two_state.implied_hazard(5.0)
        assert h > 0

    def test_implied_hazard_between_states(self, two_state):
        """Implied hazard should be between the two state hazards."""
        h = two_state.implied_hazard(5.0)
        assert 0.005 < h < 0.03

    def test_implied_spread(self, two_state):
        s = two_state.implied_spread(5.0, recovery=0.40)
        assert s > 0
        # Spread = h × (1-R) ≈ between 30bp and 180bp
        assert 0.003 < s < 0.018

    def test_spread_term_structure(self, two_state):
        spreads = two_state.spread_term_structure([1, 3, 5, 10], recovery=0.40)
        assert len(spreads) == 4
        assert all(s > 0 for s in spreads)

    def test_conditional_spread_differs(self, two_state):
        s_exp = two_state.implied_spread(5.0, 0.40, initial_state=0)
        s_rec = two_state.implied_spread(5.0, 0.40, initial_state=1)
        assert s_exp < s_rec


class TestRegimeProbabilities:
    def test_sum_to_one(self, two_state):
        probs = two_state.regime_probabilities(5.0)
        assert abs(np.sum(probs) - 1.0) < 1e-10

    def test_stationary_sum(self, two_state):
        pi = two_state.stationary_distribution()
        assert abs(np.sum(pi) - 1.0) < 1e-10

    def test_expected_hazard(self, two_state):
        h = two_state.expected_hazard(5.0)
        assert 0.005 < h < 0.03

    def test_three_state_probs(self, three_state):
        probs = three_state.regime_probabilities(10.0)
        assert len(probs) == 3
        assert abs(np.sum(probs) - 1.0) < 1e-10


class TestCalibration:
    def test_calibrate_2_state(self):
        spreads = [0.008, 0.010, 0.012, 0.015]  # 80-150bp
        tenors = [1, 3, 5, 10]
        model = calibrate_regime_model(spreads, tenors, recovery=0.40, n_states=2)
        assert model.n_states == 2
        assert model.hazard_rates[0] < model.hazard_rates[1]

    def test_calibrate_3_state(self):
        spreads = [0.005, 0.008, 0.012]
        tenors = [1, 5, 10]
        model = calibrate_regime_model(spreads, tenors, recovery=0.40, n_states=3)
        assert model.n_states == 3

    def test_calibrated_reprices(self):
        """Calibrated model should produce spreads in the right ballpark."""
        spreads = [0.01, 0.012, 0.015]
        tenors = [1, 5, 10]
        model = calibrate_regime_model(spreads, tenors, recovery=0.40)
        fitted = model.spread_term_structure(tenors, recovery=0.40)
        # Order of magnitude check
        for obs, fit in zip(spreads, fitted):
            assert abs(fit - obs) / obs < 2.0  # within 2x

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            calibrate_regime_model([0.01], [5])


class TestSerialization:
    def test_to_dict(self, two_state):
        d = two_state.to_dict()
        assert d["n_states"] == 2
        assert len(d["hazard_rates"]) == 2
        assert len(d["stationary_dist"]) == 2

    def test_states(self, two_state):
        assert len(two_state.states) == 2
        assert two_state.states[0].name == "expansion"
        assert two_state.states[1].hazard_rate == 0.03
