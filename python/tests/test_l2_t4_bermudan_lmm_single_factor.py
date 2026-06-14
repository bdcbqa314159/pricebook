"""Regression for L2 T4 audit of `options.bermudan_lmm._simulate_lmm_paths`:

Pre-fix the simulator carried a mutually inconsistent drift and
diffusion specification:

- **Drift** used the single-factor (ρ_jk = 1) terminal-measure formula
  ``μ_j = -σ_j · Σ_{k>j} σ_k · τ · F_k / (1+τF_k)``.
- **Diffusion** drew an independent Brownian increment ``dW_j`` for
  each forward — algebraically multi-factor with ρ_jk = δ_jk.

Under truly independent factors the terminal-measure drift collapses
to zero for all forwards (cross-correlation terms vanish), so the
non-zero drift in the code was unjustifiable.  Under truly
single-factor LMM all forwards must share the same ``dW``.

Fix: use a single shared ``dW`` per step (single-factor LMM,
consistent with the drift formula).

These tests pin the qualitative consequence: with the proper
single-factor structure, forwards co-move (positive correlation
between adjacent forwards on a path), which the broken implementation
destroyed by drawing independent shocks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.options.bermudan_lmm import _simulate_lmm_paths


class TestForwardsAreCoMoving:
    def test_adjacent_forwards_positively_correlated(self):
        """In a single-factor LMM all forwards are driven by a common
        Brownian, so any two forwards at any time should have positive
        path-wise correlation (above zero) of their LOG changes.

        Pre-fix the independent-shock structure produced near-zero
        empirical correlation; post-fix the correlation should be near
        +1 (single-factor implies perfect correlation in the diffusion
        term, modulo small drift-difference effects)."""
        n_fwd = 5
        forward_rates = np.full(n_fwd, 0.04)
        inst_vols = np.full(n_fwd, 0.20)
        T = 1.0
        n_steps = 100
        n_paths = 5_000

        rng = np.random.default_rng(seed=42)
        F = _simulate_lmm_paths(
            forward_rates, inst_vols, T, n_steps, n_paths,
            dt_tenor=0.5, rng=rng,
        )

        # Compute LOG forward changes per path between t=0 and t=T.
        log_F0 = np.log(F[:, 0, :])
        log_FT = np.log(F[:, -1, :])
        delta = log_FT - log_F0   # (n_paths, n_fwd)

        # Pearson correlation between forward 0 and forward 2 across paths.
        corr_02 = np.corrcoef(delta[:, 0], delta[:, 2])[0, 1]
        corr_13 = np.corrcoef(delta[:, 1], delta[:, 3])[0, 1]

        # Single-factor → correlation should be very high (well above
        # zero).  Pre-fix the independent shocks gave correlation ≈ 0
        # (only the drift coupling provided minimal correlation).
        assert corr_02 > 0.90, f"corr(log_F0, log_F2) = {corr_02:.3f} — single-factor coupling broken"
        assert corr_13 > 0.90, f"corr(log_F1, log_F3) = {corr_13:.3f} — single-factor coupling broken"


class TestVarianceMatchesSingleFactor:
    def test_log_variance_grows_as_sigma_sq_T(self):
        """For a SINGLE forward (no drift coupling for j = n_fwd - 1),
        the log-variance after time T should be σ²·T.

        Sanity check that the new shared-Brownian implementation
        preserves the per-forward marginal distribution."""
        forward_rates = np.array([0.04, 0.04])
        sigma = 0.20
        inst_vols = np.array([sigma, sigma])
        T = 1.0
        n_steps = 200
        n_paths = 10_000

        rng = np.random.default_rng(seed=42)
        F = _simulate_lmm_paths(
            forward_rates, inst_vols, T, n_steps, n_paths,
            dt_tenor=0.5, rng=rng,
        )

        # Terminal forward (j = 1, last index) has NO drift correction
        # (no k > j), so it's pure GBM with σ.
        log_FT_terminal = np.log(F[:, -1, 1])
        log_F0_terminal = np.log(F[:, 0, 1])
        empirical_var = np.var(log_FT_terminal - log_F0_terminal, ddof=1)
        # Expected: σ² · T = 0.04.  Allow some MC noise.
        assert empirical_var == pytest.approx(sigma**2 * T, rel=0.05)
