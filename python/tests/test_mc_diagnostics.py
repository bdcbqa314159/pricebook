"""Tests for MC convergence diagnostics."""

import pytest
import numpy as np

from pricebook.models.mc_diagnostics import (
    batch_means, effective_sample_size, convergence_table,
)


class TestBatchMeans:
    def test_iid_normal(self):
        """Batch means SE ≈ naive SE for iid samples."""
        rng = np.random.default_rng(42)
        values = rng.normal(10.0, 2.0, size=100_000)
        bm = batch_means(values)
        naive_se = values.std(ddof=1) / np.sqrt(len(values))
        assert bm.se == pytest.approx(naive_se, rel=0.5)

    def test_mean_correct(self):
        rng = np.random.default_rng(42)
        values = rng.normal(5.0, 1.0, size=50_000)
        bm = batch_means(values)
        assert bm.mean == pytest.approx(5.0, abs=0.05)

    def test_to_dict(self):
        values = np.array([1.0, 2.0, 3.0, 4.0] * 100)
        bm = batch_means(values, n_batches=4)
        d = bm.to_dict()
        assert "se" in d


class TestEffectiveSampleSize:
    def test_iid_ess_equals_n(self):
        """ESS ≈ N for iid samples."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=10_000)
        ess = effective_sample_size(values)
        assert ess > 8_000  # close to N for iid

    def test_correlated_ess_less_than_n(self):
        """AR(1) correlated samples should have ESS < N."""
        rng = np.random.default_rng(42)
        n = 10_000
        phi = 0.9  # strong autocorrelation
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i-1] + rng.normal()
        ess = effective_sample_size(x)
        assert ess < n * 0.5  # significantly less than N

    def test_constant_returns_n(self):
        """Constant values → ESS = N (zero variance edge case)."""
        values = np.ones(1000)
        ess = effective_sample_size(values)
        assert ess == 1000


class TestConvergenceTable:
    def test_default_checkpoints(self):
        rng = np.random.default_rng(42)
        values = rng.normal(10.0, 1.0, size=50_000)
        table = convergence_table(values)
        assert len(table) >= 3
        # SE should decrease with N
        ses = [e.se for e in table]
        assert ses[-1] < ses[0]

    def test_custom_checkpoints(self):
        values = np.random.default_rng(42).normal(0, 1, 5000)
        table = convergence_table(values, checkpoints=[100, 500, 5000])
        assert len(table) == 3
        assert table[0].n_samples == 100
        assert table[-1].n_samples == 5000

    def test_to_dict(self):
        values = np.random.default_rng(42).normal(0, 1, 1000)
        table = convergence_table(values, checkpoints=[1000])
        d = table[0].to_dict()
        assert "relative_error_pct" in d
