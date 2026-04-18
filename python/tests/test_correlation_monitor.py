"""Tests for correlation monitoring."""
import numpy as np, pytest
from pricebook.correlation_monitor import (
    implied_vs_realised_correlation, correlation_term_structure,
    correlation_stress_matrix, multi_asset_smile_arb_check,
)

class TestImpliedVsRealised:
    def test_basic(self):
        r = implied_vs_realised_correlation(0.6, 0.4)
        assert r.spread == pytest.approx(0.2)
    def test_z_score_with_history(self):
        r = implied_vs_realised_correlation(0.6, 0.4, [0.1, 0.15, 0.2, 0.25, 0.18])
        assert isinstance(r.z_score, float)
    def test_no_history(self):
        r = implied_vs_realised_correlation(0.5, 0.5)
        assert r.z_score == 0.0

class TestCorrTermStructure:
    def test_normal(self):
        r = correlation_term_structure([0.25, 1.0, 5.0], [0.5, 0.55, 0.6])
        assert not r.is_inverted
        assert r.slope > 0
    def test_inverted(self):
        r = correlation_term_structure([0.25, 1.0, 5.0], [0.6, 0.55, 0.4])
        assert r.is_inverted
        assert r.slope < 0

class TestCorrStress:
    def test_uniform_up(self):
        base = np.array([[1.0, 0.5], [0.5, 1.0]])
        r = correlation_stress_matrix(base, "uniform_up", 0.2)
        assert r.stressed_correlation[0, 1] == pytest.approx(0.7)
    def test_uniform_down(self):
        base = np.array([[1.0, 0.5], [0.5, 1.0]])
        r = correlation_stress_matrix(base, "uniform_down", 0.2)
        assert r.stressed_correlation[0, 1] == pytest.approx(0.3)
    def test_tail_stress(self):
        base = np.array([[1.0, 0.3], [0.3, 1.0]])
        r = correlation_stress_matrix(base, "tail_stress", 0.3)
        assert r.stressed_correlation[0, 1] > 0.3
    def test_decorrelation(self):
        base = np.array([[1.0, 0.8], [0.8, 1.0]])
        r = correlation_stress_matrix(base, "sector_decorrelation", 0.5)
        assert r.stressed_correlation[0, 1] < 0.8

class TestSmileArbCheck:
    def test_consistent(self):
        """Use basket vol matching the model at ρ=0.5."""
        r = multi_asset_smile_arb_check(0.195, [0.20, 0.25], [0.5, 0.5], 0.5)
        assert r.is_arbitrage_free
    def test_inconsistent(self):
        r = multi_asset_smile_arb_check(0.50, [0.20, 0.25], [0.5, 0.5], 0.5)
        assert not r.is_arbitrage_free
