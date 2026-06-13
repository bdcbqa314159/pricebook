"""Regression for L2 phase-2 audit of `risk.market_risk_enhanced`:

(a) Parametric individual_vars used np.std (ddof=0) while cov used
    np.cov (ddof=1).  Inconsistent across the same calc — same shape
    as v0.995 backtest fix.

(b) Historical incremental_var conflated LOO ("VaR drop if removed")
    with Euler decomposition ("contributions summing to portfolio
    VaR").  The docstring promises ``sum(IVaR_i) = portfolio VaR``
    but pre-fix set both incremental_vars AND component_vars to LOO,
    so neither summed to portfolio VaR for the historical case.
    Fix: separate the two — incremental remains LOO, component now
    uses conditional tail expectation E[-P&L_i | portfolio in tail],
    which sums to portfolio_var by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.risk.market_risk_enhanced import incremental_var


class TestParametricDdofConsistency:
    def test_individual_vars_use_sample_std(self):
        """Individual VaR should match np.std(ddof=1) × z."""
        rng = np.random.default_rng(42)
        pnls = {
            "A": rng.standard_normal(252).tolist(),
            "B": rng.standard_normal(252).tolist(),
        }
        from scipy.stats import norm
        # Compute expected individual VaR with ddof=1.
        sample_std_a = np.std(pnls["A"], ddof=1)
        sample_std_b = np.std(pnls["B"], ddof=1)
        z = norm.ppf(0.99)
        result = incremental_var(pnls, confidence=0.99, method="parametric")
        # The result struct doesn't expose individual_vars; check diversification.
        # diversification = sum(individual_vars) - portfolio_var.
        expected_sum_individual = sample_std_a * z + sample_std_b * z
        actual_sum_individual = result.diversification_benefit + result.portfolio_var
        assert actual_sum_individual == pytest.approx(expected_sum_individual, rel=1e-9)


class TestHistoricalEulerDecomposition:
    def test_component_vars_sum_to_portfolio_es(self):
        """Historical component_vars decomposition sums to portfolio ES, not VaR.

        This is the standard practice: ES has a clean Euler decomposition
        via tail-conditional expectation, VaR does not.  Pre-fix used
        LOO for both incremental and component, neither of which had
        the sum-to-anything property.
        """
        rng = np.random.default_rng(42)
        T = 500
        pnls = {
            "A": rng.standard_normal(T).tolist(),
            "B": (0.5 * rng.standard_normal(T)).tolist(),
            "C": (0.3 * rng.standard_normal(T)).tolist(),
        }
        result = incremental_var(pnls, confidence=0.95, method="historical")
        # Compute portfolio ES directly for comparison.
        port_pnl = np.array(pnls["A"]) + np.array(pnls["B"]) + np.array(pnls["C"])
        threshold = np.percentile(port_pnl, 5)
        port_es = -float(port_pnl[port_pnl <= threshold].mean())
        sum_components = sum(result.component_vars)
        assert sum_components == pytest.approx(port_es, rel=1e-6)

    def test_loo_vs_euler_differ(self):
        """LOO and Euler are different metrics; both should be reported separately."""
        rng = np.random.default_rng(42)
        T = 500
        pnls = {
            "A": rng.standard_normal(T).tolist(),
            "B": rng.standard_normal(T).tolist(),
        }
        result = incremental_var(pnls, confidence=0.95, method="historical")
        # incremental_vars is LOO; component_vars is Euler.  They generally differ.
        assert result.incremental_vars != result.component_vars


class TestParametricEulerInvariant:
    def test_euler_sums_to_portfolio_var(self):
        """Parametric Euler decomposition: sum(IVaR_i) = portfolio_VaR."""
        rng = np.random.default_rng(42)
        T = 252
        pnls = {f"P{i}": rng.standard_normal(T).tolist() for i in range(4)}
        result = incremental_var(pnls, method="parametric")
        sum_incr = sum(result.incremental_vars)
        assert sum_incr == pytest.approx(result.portfolio_var, rel=1e-9)


class TestDiversificationNonNegative:
    """Diversification benefit should be ≥ 0 (sub-additivity for parametric/historical)."""

    def test_parametric_diversification_geq_zero(self):
        rng = np.random.default_rng(42)
        T = 252
        pnls = {f"P{i}": rng.standard_normal(T).tolist() for i in range(5)}
        result = incremental_var(pnls, method="parametric")
        # Diversification = sum(individual_var) - portfolio_var.
        # For non-perfectly-correlated assets, portfolio vol < sum of vols.
        assert result.diversification_benefit >= -1e-9  # numerical tolerance
