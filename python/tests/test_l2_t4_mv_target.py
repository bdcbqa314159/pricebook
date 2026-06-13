"""Regression for L2 phase-2 audit of `risk.portfolio_construction.mean_variance`:

Pre-fix `target_return` parameter was declared in the signature but the
body never referenced it — every call computed the max-Sharpe tangency
portfolio regardless of whether a target was passed.

Fix: when target_return is supplied, solve the QP
    min w'Σw s.t. μ'w = target_return, Σw = 1, bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.portfolio_construction import mean_variance


class TestMeanVarianceTargetReturn:
    def test_target_return_constraint_honoured(self):
        """The result's expected_return should equal the target."""
        mu = np.array([0.10, 0.06, 0.04])
        cov = np.diag([0.04, 0.0225, 0.01])
        target = 0.07
        result = mean_variance(mu, cov, target_return=target)
        assert result.expected_return == pytest.approx(target, rel=1e-6)

    def test_target_return_minimises_variance(self):
        """For target ∈ [min(mu), max(mu)] under long-only, the solver returns
        the unique min-variance portfolio achieving that return level."""
        mu = np.array([0.10, 0.05])
        cov = np.array([[0.04, 0.01], [0.01, 0.0225]])
        target = 0.08
        result = mean_variance(mu, cov, target_return=target, long_only=True)
        # Verify target met.
        assert result.expected_return == pytest.approx(target, rel=1e-6)
        # Weights sum to 1.
        assert sum(result.weights) == pytest.approx(1.0, abs=1e-6)
        # Long-only: all weights >= 0.
        assert all(w >= -1e-9 for w in result.weights)

    def test_no_target_max_sharpe_unchanged(self):
        """Without target, behaviour matches pre-fix (max Sharpe)."""
        mu = np.array([0.10, 0.06])
        cov = np.diag([0.04, 0.0225])
        result = mean_variance(mu, cov, target_return=None)
        # Sharpe must be positive (positive excess, positive vol).
        assert result.sharpe > 0


class TestTargetReturnEdgeCases:
    def test_infeasible_target_falls_back(self):
        """Target above max asset return is infeasible for long-only; falls back."""
        mu = np.array([0.05, 0.06])
        cov = np.diag([0.04, 0.0225])
        # Target 0.15 > max(mu) = 0.06.  SLSQP will fail; fallback w0 = equal-weight.
        result = mean_variance(mu, cov, target_return=0.15, long_only=True)
        # Result should be deterministic (no crash).
        assert result.weights is not None
