"""Tests for survival curve."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.survival_curve import SurvivalCurve
from pricebook.day_count import DayCountConvention


REF = date(2024, 1, 15)


def _flat_survival_curve(ref: date, hazard_rate: float = 0.02) -> SurvivalCurve:
    """Build a survival curve with constant hazard rate."""
    tenors = [1, 2, 3, 5, 7, 10]
    dates = [ref + relativedelta(years=t) for t in tenors]
    survs = [math.exp(-hazard_rate * t) for t in tenors]
    return SurvivalCurve(ref, dates, survs)


class TestSurvival:

    def test_survival_at_reference_is_one(self):
        curve = _flat_survival_curve(REF)
        assert curve.survival(REF) == 1.0

    def test_survival_before_reference_is_one(self):
        curve = _flat_survival_curve(REF)
        assert curve.survival(REF - relativedelta(years=1)) == 1.0

    def test_survival_at_pillar(self):
        h = 0.02
        curve = _flat_survival_curve(REF, hazard_rate=h)
        expected = math.exp(-h * 5)
        assert curve.survival(REF + relativedelta(years=5)) == pytest.approx(expected, rel=1e-6)

    def test_survival_decreasing(self):
        curve = _flat_survival_curve(REF)
        dates = [REF + relativedelta(years=t) for t in [1, 3, 5, 10]]
        survs = [curve.survival(d) for d in dates]
        for i in range(1, len(survs)):
            assert survs[i] < survs[i - 1]

    def test_survival_between_zero_and_one(self):
        curve = _flat_survival_curve(REF)
        for t in [1, 2, 3, 5, 7, 10]:
            s = curve.survival(REF + relativedelta(years=t))
            assert 0 < s < 1


class TestHazardRate:

    def test_constant_hazard_rate_recovered(self):
        h = 0.03
        curve = _flat_survival_curve(REF, hazard_rate=h)
        # Mid-way through various periods
        for t in [1, 3, 5]:
            d = REF + relativedelta(years=t, months=6)
            assert curve.hazard_rate(d) == pytest.approx(h, rel=5e-3)

    def test_hazard_rate_positive(self):
        curve = _flat_survival_curve(REF)
        assert curve.hazard_rate(REF + relativedelta(years=2)) > 0

    def test_hazard_rate_at_reference_is_zero(self):
        curve = _flat_survival_curve(REF)
        assert curve.hazard_rate(REF) == 0.0


class TestDefaultProb:

    def test_default_prob_positive(self):
        curve = _flat_survival_curve(REF)
        dp = curve.default_prob(REF, REF + relativedelta(years=5))
        assert dp > 0

    def test_default_prob_equals_survival_diff(self):
        curve = _flat_survival_curve(REF)
        d1 = REF + relativedelta(years=1)
        d2 = REF + relativedelta(years=3)
        dp = curve.default_prob(d1, d2)
        expected = curve.survival(d1) - curve.survival(d2)
        assert dp == pytest.approx(expected, rel=1e-10)

    def test_default_prob_sums_to_one_minus_survival(self):
        """Total default prob from ref to T = 1 - Q(T)."""
        curve = _flat_survival_curve(REF)
        mat = REF + relativedelta(years=10)
        dp = curve.default_prob(REF, mat)
        assert dp == pytest.approx(1 - curve.survival(mat), rel=1e-6)

    def test_default_prob_d1_after_d2_raises(self):
        curve = _flat_survival_curve(REF)
        with pytest.raises(ValueError):
            curve.default_prob(REF + relativedelta(years=3), REF + relativedelta(years=1))


class TestValidation:

    def test_survival_above_one_raises(self):
        with pytest.raises(ValueError):
            SurvivalCurve(REF, [REF + relativedelta(years=1)], [1.1])

    def test_survival_zero_raises(self):
        with pytest.raises(ValueError):
            SurvivalCurve(REF, [REF + relativedelta(years=1)], [0.0])

    def test_survival_negative_raises(self):
        with pytest.raises(ValueError):
            SurvivalCurve(REF, [REF + relativedelta(years=1)], [-0.5])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            SurvivalCurve(REF, [REF + relativedelta(years=1)], [0.99, 0.95])
