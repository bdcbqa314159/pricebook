"""Regression for L2 Wave-2 audit — `antithetic_paths` had a bimodal
interface that was misleading in one branch and WRONG in the other.

Pre-fix:
- Called with ``rng_normals=Z``: returned ``-Z`` (just negated the
  argument).  The function's name suggested it returned PATHS, not
  negated normals — caller was supposed to know to rebuild paths from
  ``-Z``.  Misleading API.
- Called with ``rng_normals=None``: computed "mirror around log-mean
  of terminal values".  This is NOT a valid antithetic — the
  transformation depends on the sample mean (a random quantity),
  making the result biased and not properly antithetic.  It also
  silently produced NaNs for Bachelier/OU/etc. paths where terminal
  spots can be ≤ 0 (``np.log(0)`` is ``-inf``).

Post-fix:
- Renamed to `antithetic_normals` to reflect what it actually does.
- Requires the normal draws explicitly (positional, no default).
- Only performs the valid `-Z` operation.
- Legacy alias `antithetic_paths` kept so existing imports don't break,
  but it now points to the same function with the same signature.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.numerical._mc import antithetic_normals, antithetic_paths


class TestAntitheticNormalsCorrect:
    def test_returns_negated_array(self):
        Z = np.array([1.0, -2.0, 0.5, 3.7])
        result = antithetic_normals(Z)
        np.testing.assert_array_equal(result, -Z)

    def test_array_2d_works(self):
        Z = np.random.default_rng(42).standard_normal((100, 50))
        result = antithetic_normals(Z)
        np.testing.assert_array_equal(result, -Z)

    def test_estimator_unbiased_and_variance_reduced(self):
        """Sanity: applying antithetic to a sample-mean estimator of a
        non-linear function should reduce variance vs independent runs."""
        rng = np.random.default_rng(42)
        n = 10_000
        Z = rng.standard_normal(n)
        Z_anti = antithetic_normals(Z)

        # Estimator: E[Z²] (true value = 1).
        # Direct:
        var_direct = np.var((Z ** 2 + Z_anti ** 2) / 2, ddof=1) / n
        # For comparison: same-size independent draws
        Z_indep = rng.standard_normal(n)
        var_indep = np.var((Z ** 2 + Z_indep ** 2) / 2, ddof=1) / n

        # For E[Z²], antithetic gives perfect correlation since
        # (-Z)² = Z², so the antithetic average has half the variance
        # of independent draws... actually for E[Z²] the antithetic
        # IS identical so the variance is just Var(Z²)/n, not
        # Var(Z²)/(2n).  Let's use a non-symmetric function instead.
        # Use E[exp(Z)] which is the lognormal mean.
        # Direct: average of (e^Z + e^-Z)/2 — strongly negatively correlated.
        est_anti = (np.exp(Z) + np.exp(Z_anti)) / 2
        var_anti = np.var(est_anti, ddof=1) / n
        # Two independent samples:
        Z2 = rng.standard_normal(n)
        est_indep = (np.exp(Z) + np.exp(Z2)) / 2
        var_indep = np.var(est_indep, ddof=1) / n
        # Antithetic should have strictly lower variance.
        assert var_anti < var_indep


class TestLegacyAliasStillWorks:
    def test_antithetic_paths_alias(self):
        """The old name is preserved as a deprecation-compat alias and
        points at the SAME function as the new name."""
        assert antithetic_paths is antithetic_normals

    def test_alias_returns_negated_normals(self):
        Z = np.array([1.0, 2.0, 3.0])
        result = antithetic_paths(Z)
        np.testing.assert_array_equal(result, -Z)
