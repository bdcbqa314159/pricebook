"""Regression for L2 phase-2 audit of `risk.model_selection`:

Pre-fix `model_committee_price` computed a *weighted* mean but an
*unweighted* std — inconsistent.  For BMA-weighted committees where
one model dominates, the unweighted std treated all models equally,
overstating uncertainty.

Example: weights = (0.9, 0.05, 0.05), prices = (100, 200, 200).
Weighted mean = 100 + 10 = 110.
Unweighted std (pre-fix) = sqrt((100−167)² + (200−167)² + (200−167)²)/3 ≈ 47.
Weighted std (post-fix) = sqrt(0.9·(100−110)² + 0.05·(200−110)² + 0.05·(200−110)²)
                        = sqrt(90 + 405 + 405) = sqrt(900) = 30.

The post-fix figure correctly accounts for the dominant model's high weight.
"""

from __future__ import annotations

import math

import pytest

from pricebook.risk.model_selection import (
    ModelCandidate, model_committee_price,
)


class TestWeightedStd:
    def test_dominant_model_lowers_std(self):
        """One model with 90% weight + 2 with 5% each → std much smaller than unweighted."""
        models = [
            ModelCandidate("dominant", lambda: 100.0, weight=0.9),
            ModelCandidate("rare_high", lambda: 200.0, weight=0.05),
            ModelCandidate("rare_high2", lambda: 200.0, weight=0.05),
        ]
        result = model_committee_price(models)
        # Weighted mean = 0.9·100 + 0.05·200 + 0.05·200 = 90 + 10 + 10 = 110.
        assert result.price == pytest.approx(110.0, abs=1e-9)
        # Weighted std = sqrt(0.9·(100−110)² + 0.05·(200−110)² + 0.05·(200−110)²)
        #              = sqrt(90 + 405 + 405) = sqrt(900) = 30.
        expected_std = 30.0
        assert result.price_std == pytest.approx(expected_std, abs=1e-9)
        # Pre-fix unweighted std would be ~47.
        assert result.price_std < 40.0

    def test_equal_weights_matches_population_std(self):
        """When all weights equal, weighted std = population std (ddof=0)."""
        models = [
            ModelCandidate("A", lambda: 10.0, weight=1.0),
            ModelCandidate("B", lambda: 11.0, weight=1.0),
            ModelCandidate("C", lambda: 12.0, weight=1.0),
        ]
        result = model_committee_price(models)
        # Mean = 11.  Weighted var = (1/3)·(1²+0²+1²) = 2/3. Std = sqrt(2/3) ≈ 0.8165.
        assert result.price == pytest.approx(11.0)
        assert result.price_std == pytest.approx(math.sqrt(2/3), abs=1e-9)


class TestCommitteeStdEdgeCases:
    def test_single_model_zero_std(self):
        models = [ModelCandidate("only", lambda: 100.0)]
        result = model_committee_price(models)
        assert result.price_std == 0.0

    def test_identical_prices_zero_std(self):
        models = [
            ModelCandidate("A", lambda: 50.0, weight=2.0),
            ModelCandidate("B", lambda: 50.0, weight=1.0),
        ]
        result = model_committee_price(models)
        assert result.price_std == pytest.approx(0.0, abs=1e-12)


class TestCommitteeRangeUnchanged:
    """Range (min, max) is unaffected by the std fix."""

    def test_range_uses_unweighted_extremes(self):
        models = [
            ModelCandidate("A", lambda: 5.0, weight=0.01),
            ModelCandidate("B", lambda: 15.0, weight=0.99),
        ]
        result = model_committee_price(models)
        # Range is unweighted min/max.
        assert result.price_range == (5.0, 15.0)
        # Reserve = (max - min) / 2.
        assert result.model_uncertainty_reserve == pytest.approx(5.0)
