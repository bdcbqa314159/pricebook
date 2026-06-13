"""Regression for L2 phase-2 audit of `risk.portfolio_analytics.tracking_metrics`:

Pre-fix mixed ddof conventions: ``np.cov(...)`` defaults to ddof=1
(sample covariance) but ``np.var(...)`` defaults to ddof=0 (population
variance).  The ratio gives a beta off by factor (n-1)/n — material
for small samples.

Fix: both use ddof=1 (sample-statistics convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.portfolio_analytics import tracking_metrics


class TestTrackingMetricsBeta:
    def test_perfect_correlation_beta_one(self):
        """Identical returns → beta = 1."""
        rng = np.random.default_rng(42)
        bench = rng.standard_normal(252)
        port = bench.copy()
        result = tracking_metrics(port, bench)
        assert result["beta"] == pytest.approx(1.0, rel=1e-9)

    def test_beta_matches_ddof_1_formula(self):
        """Beta = cov_ddof1(p,b) / var_ddof1(b)."""
        rng = np.random.default_rng(42)
        bench = rng.standard_normal(100)
        port = 1.5 * bench + 0.3 * rng.standard_normal(100)

        expected_beta = float(np.cov(port, bench, ddof=1)[0, 1]
                              / np.var(bench, ddof=1))
        result = tracking_metrics(port, bench)
        assert result["beta"] == pytest.approx(expected_beta, rel=1e-12)

    def test_pre_fix_was_off_by_n_minus_1_factor(self):
        """Pre-fix value would differ by (n-1)/n; post-fix uses ddof=1 consistently."""
        rng = np.random.default_rng(7)
        n = 30  # small sample for visible discrepancy
        bench = rng.standard_normal(n)
        port = 1.2 * bench + 0.1 * rng.standard_normal(n)

        pre_fix_beta = float(np.cov(port, bench)[0, 1] / np.var(bench))  # ddof=1/ddof=0 mix
        post_fix_beta = float(np.cov(port, bench, ddof=1)[0, 1]
                              / np.var(bench, ddof=1))
        result = tracking_metrics(port, bench)
        assert result["beta"] == pytest.approx(post_fix_beta, rel=1e-12)
        # The two estimators should differ by ~(n-1)/n = 29/30.
        assert pre_fix_beta != pytest.approx(post_fix_beta, rel=1e-3)
