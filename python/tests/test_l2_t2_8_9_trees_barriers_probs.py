"""Regression for L2 Tier-2 T2.8 / T2.9 — trees: knock-in barriers + trinomial
probability renormalisation.

* T2.8 — `_apply_barrier` had a literal `pass` for `DOWN_IN` and `UP_IN`, so
  the barrier condition was checked but never enforced.  Knock-in barriers
  were silently priced as the corresponding vanilla.  Post-fix: `solve()`
  intercepts knock-in types and applies in-out parity:
      V_knock_in = V_vanilla − V_knock_out
  using two side-pricers and combining Greeks by linearity.

* T2.9 — `_trinomial_params` clamped `p_u` and `p_d` to [0,1] then set
  `p_m = max(0, 1 − p_u − p_d)`.  When the raw `p_u` or `p_d` was negative
  (large drift relative to volatility), the clamp shifted mass into `p_m`
  WITHOUT renormalising, breaking the risk-neutral moments silently.
  Post-fix: clamp each leg ≥ 0 then renormalise the triple to sum = 1.
"""

from __future__ import annotations

import math

import pytest

from pricebook.numerical._trees import (
    BarrierType,
    ExerciseType,
    TreeMethod,
    TreeSolver,
)


# ============================================================
# T2.8 — knock-in via in-out parity
# ============================================================


class TestKnockInBarriers:
    def _solve(self, spot, K, r, sigma, T, *, barrier_type=None, barrier_level=None,
               is_call=True, method=TreeMethod.CRR, n_steps=200):
        return TreeSolver(
            method=method, n_steps=n_steps,
            exercise=ExerciseType.EUROPEAN,
            barrier_type=barrier_type, barrier_level=barrier_level,
        ).solve(spot, K, r, sigma, T, is_call=is_call)

    def test_in_out_parity_down(self):
        """V(DOWN_IN) + V(DOWN_OUT) = V(vanilla).  Pre-fix DOWN_IN ≡ vanilla,
        so the sum exceeded vanilla by the DOWN_OUT amount."""
        spot, K, r, sigma, T, B = 100.0, 100.0, 0.05, 0.20, 1.0, 80.0
        van = self._solve(spot, K, r, sigma, T, is_call=True)
        ko = self._solve(spot, K, r, sigma, T, is_call=True,
                         barrier_type=BarrierType.DOWN_OUT, barrier_level=B)
        ki = self._solve(spot, K, r, sigma, T, is_call=True,
                         barrier_type=BarrierType.DOWN_IN, barrier_level=B)
        assert abs(ko.price + ki.price - van.price) < 1e-10, (
            f"In-out parity failed: vanilla={van.price:.4f}, "
            f"DOWN_OUT={ko.price:.4f}, DOWN_IN={ki.price:.4f}"
        )
        # DOWN_IN must be strictly less than vanilla (pre-fix: equal to vanilla).
        assert ki.price < van.price - 1e-6, (
            f"DOWN_IN {ki.price:.4f} not < vanilla {van.price:.4f} — T2.8 still broken"
        )

    def test_in_out_parity_up(self):
        spot, K, r, sigma, T, B = 100.0, 100.0, 0.05, 0.20, 1.0, 120.0
        van = self._solve(spot, K, r, sigma, T, is_call=False)  # put
        ko = self._solve(spot, K, r, sigma, T, is_call=False,
                         barrier_type=BarrierType.UP_OUT, barrier_level=B)
        ki = self._solve(spot, K, r, sigma, T, is_call=False,
                         barrier_type=BarrierType.UP_IN, barrier_level=B)
        assert abs(ko.price + ki.price - van.price) < 1e-10
        assert ki.price < van.price - 1e-6

    def test_knock_in_greeks_linearity(self):
        """V_ki = V_van − V_ko ⇒ Greeks combine linearly.  Smoke test:
        delta of DOWN_IN matches delta_van − delta_ko."""
        spot, K, r, sigma, T, B = 100.0, 100.0, 0.05, 0.20, 1.0, 80.0
        van = self._solve(spot, K, r, sigma, T, is_call=True)
        ko = self._solve(spot, K, r, sigma, T, is_call=True,
                         barrier_type=BarrierType.DOWN_OUT, barrier_level=B)
        ki = self._solve(spot, K, r, sigma, T, is_call=True,
                         barrier_type=BarrierType.DOWN_IN, barrier_level=B)
        assert abs(ki.delta - (van.delta - ko.delta)) < 1e-10
        assert abs(ki.gamma - (van.gamma - ko.gamma)) < 1e-10


# ============================================================
# T2.9 — trinomial probability triple sums to 1 after clamp
# ============================================================


class TestTrinomialProbabilityRenorm:
    def test_probs_sum_to_one_high_drift(self):
        """With large drift (r=20%, σ=10%), raw p_u or p_d may fall outside
        [0,1].  Post-fix the renormalisation guarantees probs sum to 1.
        Pre-fix the clamp without renorm could yield a triple that didn't
        match the risk-neutral moments and could even sum to <1 if both
        p_u and p_d were negative."""
        from pricebook.numerical._trees import _trinomial_params
        # Extreme drift: r-q-0.5σ² = 0.20 - 0 - 0.005 = 0.195 → big nu.
        u, d, p_u, p_m, p_d, disc = _trinomial_params(
            r=0.20, q=0.0, vol=0.10, dt=0.01,
        )
        s = p_u + p_m + p_d
        assert abs(s - 1.0) < 1e-12, (
            f"Trinomial probs sum to {s:.10f}, expected 1.0"
        )
        assert p_u >= 0 and p_m >= 0 and p_d >= 0

    def test_probs_unchanged_normal_regime(self):
        """For normal-drift regimes (small (r-q)·sqrt(dt) / σ), the raw probs
        are already in [0,1] and the renormalisation is a no-op."""
        from pricebook.numerical._trees import _trinomial_params
        u, d, p_u, p_m, p_d, disc = _trinomial_params(
            r=0.05, q=0.02, vol=0.20, dt=0.01,
        )
        # Raw probabilities should already be in [0,1] for these params.
        # Sum to 1 always.
        assert abs(p_u + p_m + p_d - 1.0) < 1e-12

    def test_trinomial_pricing_stable_under_high_drift(self):
        """Indirect test: a trinomial call with high drift should still
        produce a finite, reasonable price (not NaN or negative)."""
        # High-drift extreme case where the clamp activates.
        res = TreeSolver(
            method=TreeMethod.TRINOMIAL, n_steps=100,
            exercise=ExerciseType.EUROPEAN,
        ).solve(spot=100.0, strike=100.0, rate=0.30, vol=0.10, T=1.0,
                is_call=True)
        assert math.isfinite(res.price)
        assert res.price > 0
