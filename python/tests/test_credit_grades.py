"""Tests for CreditGrades model."""

import math
import pytest

from pricebook.credit.credit_grades import (
    CreditGrades, CreditGradesResult,
    credit_grades_survival, credit_grades_spread,
)


@pytest.fixture
def ig_model():
    """Investment-grade: low vol, low leverage."""
    return CreditGrades(asset_vol=0.20, leverage=0.30, recovery_mean=0.50, recovery_vol=0.20)


@pytest.fixture
def hy_model():
    """High-yield: high vol, high leverage."""
    return CreditGrades(asset_vol=0.40, leverage=0.60, recovery_mean=0.35, recovery_vol=0.30)


class TestSurvival:
    def test_at_zero(self, ig_model):
        assert ig_model.survival(0.0) == 1.0

    def test_decreasing(self, ig_model):
        q1 = ig_model.survival(1.0)
        q5 = ig_model.survival(5.0)
        q10 = ig_model.survival(10.0)
        assert 1.0 > q1 > q5 > q10 > 0

    def test_ig_high_survival(self, ig_model):
        """IG issuer should have high 5Y survival."""
        q = ig_model.survival(5.0)
        assert q > 0.90

    def test_hy_lower_survival(self, hy_model):
        """HY issuer should have lower survival."""
        q = hy_model.survival(5.0)
        assert q < 0.90

    def test_higher_leverage_lower_survival(self):
        low_lev = CreditGrades(0.30, 0.30)
        high_lev = CreditGrades(0.30, 0.70)
        assert high_lev.survival(5.0) < low_lev.survival(5.0)

    def test_higher_vol_lower_survival(self):
        low_vol = CreditGrades(0.15, 0.40)
        high_vol = CreditGrades(0.40, 0.40)
        assert high_vol.survival(5.0) < low_vol.survival(5.0)


class TestSpread:
    def test_ig_spread_reasonable(self, ig_model):
        """IG 5Y spread should be < 200bp."""
        s = ig_model.cds_spread(5.0)
        assert 0 < s < 0.02  # < 200bp

    def test_hy_spread_wider(self, hy_model, ig_model):
        s_ig = ig_model.cds_spread(5.0)
        s_hy = hy_model.cds_spread(5.0)
        assert s_hy > s_ig

    def test_spread_term_structure(self, ig_model):
        spreads = ig_model.spread_term_structure([1, 3, 5, 7, 10])
        assert len(spreads) == 5
        assert all(s > 0 for s in spreads)

    def test_custom_recovery(self, ig_model):
        s1 = ig_model.cds_spread(5.0, recovery=0.30)
        s2 = ig_model.cds_spread(5.0, recovery=0.60)
        assert s1 > s2  # lower recovery → wider spread


class TestDistanceToDefault:
    def test_dd_positive_ig(self, ig_model):
        dd = ig_model.distance_to_default()
        assert dd > 0

    def test_dd_higher_for_ig(self, ig_model, hy_model):
        assert ig_model.distance_to_default() > hy_model.distance_to_default()


class TestEvaluate:
    def test_evaluate(self, ig_model):
        r = ig_model.evaluate(5.0)
        assert isinstance(r, CreditGradesResult)
        assert abs(r.survival_prob + r.default_prob - 1.0) < 1e-10
        assert r.cds_spread > 0
        assert r.distance_to_default > 0

    def test_to_dict(self, ig_model):
        d = ig_model.evaluate(5.0).to_dict()
        assert "survival_prob" in d
        assert "cds_spread" in d


class TestConvenience:
    def test_credit_grades_survival(self):
        q = credit_grades_survival(0.25, 0.40, 5.0)
        assert 0 < q < 1

    def test_credit_grades_spread(self):
        s = credit_grades_spread(0.25, 0.40, 5.0)
        assert s > 0


class TestEdgeCases:
    def test_invalid_leverage(self):
        with pytest.raises(ValueError, match="leverage"):
            CreditGrades(0.30, 0.0)
        with pytest.raises(ValueError, match="leverage"):
            CreditGrades(0.30, 1.0)

    def test_very_low_leverage(self):
        """Near-zero leverage → near-zero default prob."""
        m = CreditGrades(0.20, 0.01)
        q = m.survival(5.0)
        assert q > 0.999

    def test_high_leverage(self):
        """Near-1 leverage → high default prob."""
        m = CreditGrades(0.30, 0.95)
        q = m.survival(5.0)
        assert q < 0.50

    def test_model_to_dict(self, ig_model):
        d = ig_model.to_dict()
        assert d["asset_vol"] == 0.20
        assert d["leverage"] == 0.30
