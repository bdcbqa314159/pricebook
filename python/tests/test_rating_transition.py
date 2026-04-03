"""Tests for credit rating transition models."""

import pytest
import math
import numpy as np

from pricebook.rating_transition import (
    RatingTransitionMatrix,
    risky_zcb_jlt,
    standard_generator,
)


@pytest.fixture
def rtm():
    return standard_generator()


class TestGenerator:
    def test_rows_sum_to_zero(self, rtm):
        """Generator rows must sum to 0."""
        row_sums = rtm.Q.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 0.0, decimal=12)

    def test_off_diagonal_non_negative(self, rtm):
        for i in range(rtm.n):
            for j in range(rtm.n):
                if i != j:
                    assert rtm.Q[i, j] >= 0

    def test_absorbing_state(self, rtm):
        """Last row (default) is all zeros."""
        np.testing.assert_array_equal(rtm.Q[-1, :], 0.0)

    def test_ratings(self, rtm):
        assert rtm.ratings[0] == "AAA"
        assert rtm.ratings[-1] == "D"
        assert rtm.n == 8


class TestTransitionProbability:
    def test_rows_sum_to_one(self, rtm):
        P = rtm.transition_prob(1.0)
        row_sums = P.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=4)

    def test_identity_at_zero(self, rtm):
        P = rtm.transition_prob(0.0)
        np.testing.assert_array_almost_equal(P, np.eye(rtm.n), decimal=10)

    def test_non_negative(self, rtm):
        P = rtm.transition_prob(1.0)
        assert np.all(P >= -1e-10)

    def test_default_absorbing(self, rtm):
        """Once in default, stays in default."""
        P = rtm.transition_prob(5.0)
        d = rtm.default_state
        assert P[d, d] == pytest.approx(1.0)


class TestDefaultProbability:
    def test_aaa_less_than_bbb(self, rtm):
        pd_aaa = rtm.default_probability("AAA", 5.0)
        pd_bbb = rtm.default_probability("BBB", 5.0)
        assert pd_aaa < pd_bbb

    def test_bbb_less_than_ccc(self, rtm):
        pd_bbb = rtm.default_probability("BBB", 5.0)
        pd_ccc = rtm.default_probability("CCC", 5.0)
        assert pd_bbb < pd_ccc

    def test_increases_with_time(self, rtm):
        pd_1y = rtm.default_probability("BBB", 1.0)
        pd_5y = rtm.default_probability("BBB", 5.0)
        assert pd_5y > pd_1y

    def test_zero_at_time_zero(self, rtm):
        assert rtm.default_probability("BBB", 0.0) == pytest.approx(0.0)

    def test_default_state_always_one(self, rtm):
        assert rtm.default_probability("D", 1.0) == pytest.approx(1.0)


class TestSpreadTermStructure:
    def test_aaa_spread_less_than_bbb(self, rtm):
        s_aaa = rtm.spread_term_structure("AAA", [5.0])[0]
        s_bbb = rtm.spread_term_structure("BBB", [5.0])[0]
        assert s_aaa < s_bbb

    def test_positive_spreads(self, rtm):
        spreads = rtm.spread_term_structure("BB", [1, 3, 5, 10])
        assert all(s > 0 for s in spreads)

    def test_ordering_across_ratings(self, rtm):
        """IG spreads < HY spreads."""
        for T in [3.0, 5.0]:
            s_a = rtm.spread_term_structure("A", [T])[0]
            s_bb = rtm.spread_term_structure("BB", [T])[0]
            assert s_a < s_bb


class TestSimulation:
    def test_shape(self, rtm):
        paths = rtm.simulate_paths("BBB", T=5.0, n_steps=50, n_paths=100)
        assert paths.shape == (100, 51)

    def test_starts_at_initial(self, rtm):
        paths = rtm.simulate_paths("AA", T=1.0, n_steps=10, n_paths=100)
        idx = rtm.rating_index("AA")
        np.testing.assert_array_equal(paths[:, 0], idx)

    def test_default_probability_matches(self, rtm):
        """Simulated default rate ≈ analytical."""
        T = 5.0
        n_paths = 50_000
        paths = rtm.simulate_paths("BBB", T, n_steps=50, n_paths=n_paths)
        sim_pd = (paths[:, -1] == rtm.default_state).mean()
        ana_pd = rtm.default_probability("BBB", T)
        assert sim_pd == pytest.approx(ana_pd, abs=0.01)

    def test_absorbing(self, rtm):
        """Once defaulted, stays defaulted."""
        paths = rtm.simulate_paths("CCC", T=10.0, n_steps=100, n_paths=1000)
        for p in range(paths.shape[0]):
            defaulted = False
            for i in range(paths.shape[1]):
                if paths[p, i] == rtm.default_state:
                    defaulted = True
                if defaulted:
                    assert paths[p, i] == rtm.default_state

    def test_default_times(self, rtm):
        dt = rtm.simulate_default_times("BBB", T=10.0, n_steps=100, n_paths=1000)
        # Some should default (finite), some not (inf)
        assert np.any(np.isfinite(dt))
        assert np.any(np.isinf(dt))
        # Default times should be positive
        assert np.all(dt[np.isfinite(dt)] > 0)


class TestJLT:
    def test_risky_less_than_riskfree(self, rtm):
        risky = risky_zcb_jlt(rtm, "BBB", 5.0, 0.05, recovery=0.4)
        riskfree = math.exp(-0.05 * 5)
        assert risky < riskfree

    def test_aaa_near_riskfree(self, rtm):
        risky = risky_zcb_jlt(rtm, "AAA", 5.0, 0.05, recovery=0.4)
        riskfree = math.exp(-0.05 * 5)
        assert risky == pytest.approx(riskfree, rel=0.02)

    def test_full_recovery_equals_riskfree(self, rtm):
        """Recovery=1 → risky = risk-free (no loss on default)."""
        risky = risky_zcb_jlt(rtm, "CCC", 5.0, 0.05, recovery=1.0)
        riskfree = math.exp(-0.05 * 5)
        assert risky == pytest.approx(riskfree, rel=1e-10)

    def test_ordering(self, rtm):
        p_aaa = risky_zcb_jlt(rtm, "AAA", 5.0, 0.05)
        p_bbb = risky_zcb_jlt(rtm, "BBB", 5.0, 0.05)
        p_ccc = risky_zcb_jlt(rtm, "CCC", 5.0, 0.05)
        assert p_aaa > p_bbb > p_ccc
