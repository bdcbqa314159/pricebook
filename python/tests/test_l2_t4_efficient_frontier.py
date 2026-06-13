"""Regression for L2 phase-2 audit of `risk.efficient_frontier`:

Pre-fix ``minimum_variance_portfolio`` always returned
``FrontierPoint(0, vol, 0, w)`` — ``expected_return`` and
``sharpe_ratio`` were hardcoded to 0 because the function had no access
to ``mu``.  Inside ``efficient_frontier`` this was patched up by
mutating the returned dataclass; but users calling the function
*directly* (as `test_portfolio_game_theory.py` does) got bogus fields.

Fix: added optional ``mu`` parameter so callers can populate the
fields correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.efficient_frontier import minimum_variance_portfolio


class TestMinVarianceWithMu:
    def test_mu_supplied_populates_return_and_sharpe(self):
        cov = np.diag([0.04, 0.04])
        mu = np.array([0.10, 0.06])
        mv = minimum_variance_portfolio(cov, mu=mu, risk_free_rate=0.02)
        # Equal-weighted MV for identical-variance assets → ret = 0.08.
        assert mv.expected_return == pytest.approx(0.08, abs=1e-9)
        # vol = sqrt(0.5² * 0.04 + 0.5² * 0.04) = sqrt(0.02) ≈ 0.1414.
        # sharpe = (0.08 − 0.02) / 0.1414 ≈ 0.424.
        assert mv.sharpe_ratio == pytest.approx(0.06 / mv.volatility, rel=1e-6)

    def test_mu_omitted_returns_zero_legacy(self):
        cov = np.diag([0.04, 0.04])
        mv = minimum_variance_portfolio(cov)
        assert mv.expected_return == 0.0
        assert mv.sharpe_ratio == 0.0
        # Volatility still computed correctly.
        assert mv.volatility > 0

    def test_long_only_with_mu(self):
        cov = np.array([[0.04, 0.02], [0.02, 0.025]])
        mu = np.array([0.08, 0.05])
        mv = minimum_variance_portfolio(cov, long_only=True, mu=mu)
        # mu @ w must match expected_return field exactly.
        assert mv.expected_return == pytest.approx(float(mu @ mv.weights), rel=1e-9)

    def test_mu_shape_mismatch_raises(self):
        cov = np.eye(3)
        with pytest.raises(ValueError, match="mu length"):
            minimum_variance_portfolio(cov, mu=np.array([0.1, 0.2]))


class TestMinVarianceVolUnchanged:
    """The actual portfolio weights and vol must not change due to the fix."""

    def test_unconstrained_two_assets(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.0225]])
        mv = minimum_variance_portfolio(cov, long_only=False)
        # Analytical: w = Σ⁻¹·1 / (1'·Σ⁻¹·1).
        inv = np.linalg.inv(cov)
        ones = np.ones(2)
        expected = inv @ ones / (ones @ inv @ ones)
        np.testing.assert_allclose(mv.weights, expected, atol=1e-10)
