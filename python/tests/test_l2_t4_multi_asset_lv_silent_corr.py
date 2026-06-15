"""Regression for L2 T4 audit of `options.multi_asset_local_vol`:

Two findings in this slice.

T4-MALV1 — ``smile_consistency_check`` had ``correlation`` as a
silent-no-op API param.  The function locally computed the linearised
model basket variance using ``correlation``, then DISCARDED the result:
``is_consistent`` and ``consistency_ratio`` both used only the
trivial upper bound ``weighted = Σ w_i σ_i``, which doesn't depend on
``correlation``.  Calling with ρ = -1 vs ρ = +1 produced bit-identical
output.

Fix: the result fields now use the correlation-aware
``model_basket_vol``; a new ``model_basket_vol`` field exposes it.

T4-MALV2 — ``multi_asset_slv_simulate`` returned ``sqrt(v)`` for BOTH
``vol1_paths`` and ``vol2_paths`` — the same bare stochastic-vol array
twice.  The asset-specific effective vols
``eff_i = mixing · lv_i + (1 - mixing) · √v`` (which actually drive
each spot) were lost.

Fix: track per-asset effective vol traces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.options.multi_asset_local_vol import (
    smile_consistency_check, multi_asset_slv_simulate,
)


class TestSmileConsistencyUsesCorrelation:
    def test_correlation_affects_consistency(self):
        """ρ = -1 (strongly negative) gives a much lower model basket
        vol than ρ = +1 (perfect positive).  ``is_consistent`` and
        ``consistency_ratio`` must reflect this.  Pre-fix both were
        independent of ρ."""
        # Basket vol of 0.20, components both at 0.25, equal weights.
        kwargs = dict(basket_vol=0.20, component_vols=[0.25, 0.25],
                      weights=[0.5, 0.5])
        r_pos = smile_consistency_check(correlation=1.0, **kwargs)
        r_neg = smile_consistency_check(correlation=-1.0, **kwargs)

        # ρ = +1: model_var = (Σ w σ)² = 0.0625 → model_vol = 0.25.
        # ρ = -1: model_var = Σ w² σ² − (cross terms) = small → model_vol ≈ 0.
        assert r_pos.model_basket_vol == pytest.approx(0.25, rel=1e-9)
        assert r_neg.model_basket_vol < 0.01

        # basket_vol = 0.20 is consistent under ρ = +1 (≤ 0.25 × 1.05).
        assert r_pos.is_consistent
        # Under ρ = -1, model_vol ≈ 0, so basket_vol = 0.20 is inconsistent.
        assert not r_neg.is_consistent
        # Consistency ratios must differ.
        assert r_pos.consistency_ratio != pytest.approx(r_neg.consistency_ratio, abs=1e-3)


class TestVolPathsAreAssetSpecific:
    def test_vol_paths_differ_when_lvs_differ(self):
        """Pre-fix ``vol1_paths`` and ``vol2_paths`` were both
        ``sqrt(v)`` (the shared stochastic vol), so they were identical
        regardless of ``lv1`` and ``lv2``.  Post-fix they hold the
        asset-specific effective vol ``mixing·lv_i + (1-mixing)·√v``
        which differs whenever ``lv1 ≠ lv2``."""
        r = multi_asset_slv_simulate(
            spot1=100, spot2=100, rate=0.03, div1=0.02, div2=0.02,
            lv1=0.20, lv2=0.40,   # very different LVs
            heston_v0=0.04, heston_kappa=2.0, heston_theta=0.04,
            heston_xi=0.3, rho_assets=0.5, rho_vol=-0.3,
            T=1.0, n_paths=200, mixing=0.5, seed=42,
        )
        # Mean over paths at a mid step — both arrays must reflect
        # different effective vols.
        mid_step = r.vol1_paths.shape[1] // 2
        v1_mean = float(r.vol1_paths[:, mid_step].mean())
        v2_mean = float(r.vol2_paths[:, mid_step].mean())
        # Difference should be roughly mixing × (lv2 - lv1) = 0.5 × 0.20 = 0.10.
        assert abs(v2_mean - v1_mean) == pytest.approx(0.10, abs=2e-2), (
            f"vol1≈{v1_mean:.3f} vol2≈{v2_mean:.3f} — vol paths should be "
            f"asset-specific (lv1=0.20 vs lv2=0.40)"
        )
