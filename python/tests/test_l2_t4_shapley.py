"""Regression for L2 phase-2 audit of `risk.shapley`:

`shapley_capital_allocation` built an ``enriched_values`` dict
containing per-desk ``{shapley_allocation, standalone_risk,
diversification_benefit}`` then immediately discarded it by returning
the raw ``ShapleyResult``.  The function docstring promised
diversification reporting that the caller never received.

Fix: added ``diversification: dict | None`` field on ``ShapleyResult``
and populate it in ``shapley_capital_allocation``.
"""

from __future__ import annotations

import pytest

from pricebook.risk.shapley import (
    ShapleyResult, shapley_capital_allocation, shapley_value,
)


class TestShapleyCapitalAllocationDiversification:
    def test_diversification_field_populated(self):
        """Pre-fix: result.diversification = None (field didn't exist).
        Post-fix: dict with one entry per desk."""
        desks = ["A", "B"]
        standalone = {"A": 100.0, "B": 80.0}

        def portfolio_risk(S):
            # Subadditive: diversified risk < sum of standalones.
            return sum(standalone[d] for d in S) * 0.8

        result = shapley_capital_allocation(desks, standalone, portfolio_risk)
        assert result.diversification is not None
        assert set(result.diversification.keys()) == {"A", "B"}

    def test_diversification_benefit_correct(self):
        """diversification_benefit = standalone_risk - shapley_allocation."""
        desks = ["A", "B"]
        standalone = {"A": 100.0, "B": 80.0}

        def portfolio_risk(S):
            return sum(standalone[d] for d in S) * 0.5

        result = shapley_capital_allocation(desks, standalone, portfolio_risk)
        for desk in desks:
            entry = result.diversification[desk]
            assert entry["standalone_risk"] == standalone[desk]
            assert entry["shapley_allocation"] == result.values[desk]
            assert entry["diversification_benefit"] == pytest.approx(
                standalone[desk] - result.values[desk], abs=1e-12,
            )

    def test_diversification_in_to_dict(self):
        desks = ["A", "B"]
        standalone = {"A": 50.0, "B": 50.0}
        result = shapley_capital_allocation(
            desks, standalone, lambda S: sum(standalone[d] for d in S)
        )
        d = result.to_dict()
        assert "diversification" in d
        assert d["diversification"] is not None


class TestShapleyResultBackwardsCompat:
    def test_default_diversification_is_none(self):
        """For other Shapley callers, diversification stays None."""
        result = shapley_value(lambda S: float(len(S)), ["A", "B"])
        assert result.diversification is None
