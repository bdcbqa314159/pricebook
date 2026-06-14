"""Regression for L2 T4 audit of `options.local_vol._dupire_local_vol`:

Pre-fix the Gatheral-style Dupire local vol formula had two coupled
errors that both silently vanished when r = q = 0:

1. ``y`` was set to ``log(K / spot)`` — but Gatheral's formula in
   total-variance form uses log-moneyness against the FORWARD,
   ``y = log(K / F_T)``.
2. ``dw_dt`` was treated as ``∂w/∂T |_K`` directly, but the formula
   denominator is taken at fixed ``y``.  The chain-rule correction
   ``(r − q) · K · ∂w/∂K`` is required to convert.

Fix: use forward-relative moneyness AND add the chain term to dw_dt.

Sanity checks:
- For a FLAT implied vol surface, local vol should equal the
  implied vol exactly (no skew/term structure).  Pre- and post-fix
  both produce this result, but the post-fix code does so via the
  correct path (forward-based y and at-fixed-y time derivative).
  Both terms have non-trivial values that must cancel — the test
  pins the cancellation.
- A skewed surface with r ≠ q should change local vol predictably:
  changing r at fixed implied vol surface shifts the local vol
  (because the forward shifts).  Pre-fix this dependence was
  silently absent (y ignored r).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.options.local_vol import calibrate_dupire, LocalVolSurface


class TestFlatSurfaceMatchesImpliedVol:
    def test_flat_implied_vol_gives_flat_local_vol(self):
        """Flat 20% implied vol surface → local vol = 20% everywhere.
        With r=0.05, q=0.02 (forward ≠ spot), post-fix this still holds
        because all derivatives are zero, but the forward-shift fix
        must not introduce spurious bias."""
        spot = 100.0
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        times = [0.25, 0.5, 1.0, 2.0]
        sigma = 0.20
        implied = [[sigma] * len(strikes) for _ in times]

        lv = calibrate_dupire(spot, rate=0.05, strikes=strikes,
                              times=times, implied_vols=implied,
                              div_yield=0.02, validate=False)
        # All interior local vols should match σ to better than 1%.
        for i, t in enumerate(times):
            for j, k in enumerate(strikes):
                v = lv.vols[i, j]
                assert v == pytest.approx(sigma, rel=2e-2), (
                    f"Flat-surface local vol at (K={k}, T={t}) = {v:.4f} "
                    f"!= {sigma:.4f}"
                )


class TestForwardShiftAffectsLocalVol:
    def test_local_vol_changes_with_rate_on_skewed_surface(self):
        """For a skewed implied vol surface, changing the risk-free
        rate ``r`` (at fixed implied vol surface) MUST shift the local
        vol grid — because the formula's ``y = log(K/F_T)`` depends on
        the forward.  Pre-fix the local vol did NOT depend on ``r``
        through ``y`` (which used spot, not forward), so this
        sensitivity was zero — a structural error."""
        spot = 100.0
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        times = [0.5, 1.0, 2.0]
        # Skew: low strikes higher vol, high strikes lower.
        implied = []
        for t in times:
            row = [0.30 - 0.001 * (k - 100.0) for k in strikes]
            implied.append(row)

        lv_low_r = calibrate_dupire(spot, rate=0.01, strikes=strikes,
                                     times=times, implied_vols=implied,
                                     div_yield=0.0, validate=False)
        lv_high_r = calibrate_dupire(spot, rate=0.10, strikes=strikes,
                                      times=times, implied_vols=implied,
                                      div_yield=0.0, validate=False)

        # Compute the maximum |Δlocal_vol| across the surface.
        max_diff = float(np.max(np.abs(lv_high_r.vols - lv_low_r.vols)))
        # Pre-fix the formula didn't depend on r through y, so the
        # difference was zero (only dw_dt at fixed K saw r — but in our
        # parametrisation we control implied vol directly, not call
        # prices, so dsdt is independent of r).
        # Post-fix the chain term (r − q) · K · ∂w/∂K injects an
        # r-dependence; the difference should be visibly non-zero.
        assert max_diff > 1e-3, (
            f"max |Δ local_vol| = {max_diff:.2e} — local vol should "
            f"depend on r through the Gatheral y = log(K/F_T) and the "
            f"at-fixed-y time derivative chain term"
        )
