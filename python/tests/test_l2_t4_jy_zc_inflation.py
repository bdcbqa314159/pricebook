"""Regression for L2 T4 audit of `fixed_income.jarrow_yildirim`:

Two coupled bugs in ``jy_zc_inflation_swap`` (T4-JY1):

1. **HW ZCB formula wrong** — the inner ``hw_zcb`` helper used
   ``A = -(σ²/(2a²))·(T - B - aB²/2)`` and ``exp(A - B·r₀)``, which:
   - replaced ``-T·r₀`` with ``-B·r₀`` (missing the dominant
     ``-(T-B)·r₀`` rate-only piece),
   - flipped the sign of the ``(T-B)·σ²/(2a²)`` term,
   - flipped the sign of the ``σ²·B²/(4a)`` term.
   For a=0.05, σ=0.01, r₀=0.04, T=5 it returned ZCB ≈ 0.836 vs the
   correct 0.820 — a ~2% bias that propagated to both P_n and P_r
   with different magnitudes so the ratio did not cancel.

2. **Inflation-forward ratio inverted** — the fair-rate formula
   computed ``P_n / P_r`` rather than ``P_r / P_n``.  The JY ZC
   inflation forward (Mercurio 2005) is
       I_fwd(0,T)/I(0) = P_r/P_n · exp(conv_adj)
   so the code produced ``fair_rate ≈ exp(-(r_n - r_r)·T) − 1`` —
   negative for the typical ``r_n > r_r`` setup (expected positive
   inflation breakeven).  At σ=0, the test below pre-fix would have
   reported ``fair_rate ≈ -0.095`` instead of ``+0.105``.

The existing ``TestJYZCSwap`` tests were too loose to catch either
defect: ``test_breakeven_sign`` only checks ``math.isfinite``;
``test_convexity_nonzero`` only checks ``!= 0``.
"""

from __future__ import annotations

import math

import pytest

from pricebook.fixed_income.jarrow_yildirim import (
    JYParams, jy_zc_inflation_swap,
)


def _zero_vol_params() -> JYParams:
    """Tiny-vol params so the analytical ZCB collapses to the flat curve."""
    return JYParams(
        a_n=0.05, sigma_n=1e-10, a_r=0.03, sigma_r=1e-10,
        sigma_I=1e-10, rho_nr=0.0, rho_nI=0.0, rho_rI=0.0,
    )


class TestZCBCollapsesToFlat:
    """At σ → 0, the JY HW ZCB must collapse to the flat-rate ZCB
    ``exp(-r₀·T)`` exactly."""

    def test_nominal_zcb_zero_vol(self):
        result = jy_zc_inflation_swap(_zero_vol_params(), r_n0=0.04, r_r0=0.02, T=5.0)
        assert result.nominal_zcb == pytest.approx(math.exp(-0.04 * 5.0), rel=1e-10)

    def test_real_zcb_zero_vol(self):
        result = jy_zc_inflation_swap(_zero_vol_params(), r_n0=0.04, r_r0=0.02, T=5.0)
        assert result.real_zcb == pytest.approx(math.exp(-0.02 * 5.0), rel=1e-10)


class TestFairRateMatchesInflation:
    """At σ → 0 (no convexity correction), the JY ZC fair rate must equal
    ``exp((r_n - r_r)·T) − 1`` — the deterministic-rate inflation
    breakeven."""

    def test_fair_rate_zero_vol_positive_inflation(self):
        result = jy_zc_inflation_swap(_zero_vol_params(), r_n0=0.04, r_r0=0.02, T=5.0)
        expected = math.exp((0.04 - 0.02) * 5.0) - 1.0
        # Pre-fix: -0.0952 (inverted ratio).
        assert result.fair_rate == pytest.approx(expected, rel=1e-8)
        assert result.fair_rate > 0.0  # positive inflation

    def test_fair_rate_sign_follows_rate_differential(self):
        """If r_n > r_r, fair_rate > 0; if r_n < r_r, fair_rate < 0."""
        p = _zero_vol_params()
        r_high_n = jy_zc_inflation_swap(p, r_n0=0.05, r_r0=0.02, T=3.0)
        r_low_n = jy_zc_inflation_swap(p, r_n0=0.01, r_r0=0.03, T=3.0)
        assert r_high_n.fair_rate > 0.0
        assert r_low_n.fair_rate < 0.0


class TestRealisticParams:
    """Sensible behaviour at a realistic vol level — fair rate is close to
    the deterministic-rate breakeven with small convexity correction."""

    def test_fair_rate_close_to_deterministic_with_small_vol(self):
        p = JYParams(
            a_n=0.05, sigma_n=0.01, a_r=0.03, sigma_r=0.005,
            sigma_I=0.02, rho_nr=0.3, rho_nI=-0.2, rho_rI=0.1,
        )
        result = jy_zc_inflation_swap(p, r_n0=0.04, r_r0=0.02, T=5.0)
        deterministic = math.exp((0.04 - 0.02) * 5.0) - 1.0  # ~0.105
        # Within 5% of the deterministic value — small convexity bump.
        assert abs(result.fair_rate - deterministic) < 0.05 * deterministic
        assert result.fair_rate > 0.0
