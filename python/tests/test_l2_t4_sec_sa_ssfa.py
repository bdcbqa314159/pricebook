"""Regression for L2 T4 audit of `regulatory.securitization.calculate_sec_sa_rw`:

Pre-fix the SSFA formula had two errors:
1. ``ksa_adj × (...)`` spurious multiplier on the entire-above branch.
2. ``(K_a - A)`` missing /(D-A) and ``K_a ×`` spurious on the straddle branch.

Per Basel III CRE40.53 / US Fed 12 CFR 217.43(b)(5):
    Entirely above (A >= K_a):
        K_SSFA = (e^(au) - e^(al)) / (a · (u - l))
        where u = D - K_a, l = A - K_a → (u - l) = D - A.
    Straddle (A < K_a < D):
        K_SSFA = (K_a - A)/(D - A) + (e^(au) - 1) / (a · (D - A))
        where u = D - K_a, l = 0.

We pin both formulas by computing them directly from the integral form and
comparing to the library output, choosing tranches that DON'T hit the 15%
floor so the formula error is observable.
"""

from __future__ import annotations

import math

import pytest

from pricebook.regulatory.securitization import calculate_sec_sa_rw


def _ssfa_direct(ksa: float, A: float, D: float, p: float) -> float:
    """Direct integral-form K_SSFA per CRE40.53."""
    K = ksa
    a = -1.0 / (p * K)
    if D <= K:
        return 1.0
    if A >= K:
        u = D - K
        l = A - K
        return (math.exp(a * u) - math.exp(a * l)) / (a * (u - l))
    # straddle
    u = D - K
    return (K - A) / (D - A) + (math.exp(a * u) - 1.0) / (a * (D - A))


class TestSSFAEntirelyAbove:
    def test_matches_spec_just_above(self):
        """K_a=0.08, A=0.09, D=0.20 (mezz just above K_a) — formula error
        was floored pre-fix; post-fix gives ~178% RW.
        """
        ksa, A, D = 0.08, 0.09, 0.20
        # p with n=50, lgd=0.5 → max(0.3, 0.25) = 0.3
        p = 0.3
        expected_k = _ssfa_direct(ksa, A, D, p)
        expected_rw_pct = min(max(expected_k * 12.5 * 100, 15), 1250)
        rw = calculate_sec_sa_rw(ksa, A, D, n=50, lgd=0.50)
        assert rw == pytest.approx(expected_rw_pct, rel=1e-6)
        # Sanity: not floored — must be strictly > 15.
        assert rw > 15.0
        # Sanity: in the ~150-200% RW range for this just-above mezzanine.
        assert 100 < rw < 250

    def test_matches_spec_high_above(self):
        """K_a=0.08, A=0.20, D=1.0 — typical super-senior; floored at 15% but
        formula must still be self-consistent."""
        ksa, A, D = 0.08, 0.20, 1.0
        p = 0.3
        expected_k = _ssfa_direct(ksa, A, D, p)
        expected_rw_pct = min(max(expected_k * 12.5 * 100, 15), 1250)
        rw = calculate_sec_sa_rw(ksa, A, D, n=50, lgd=0.50)
        assert rw == pytest.approx(expected_rw_pct, rel=1e-6)


class TestSSFAStraddle:
    def test_matches_spec_straddle(self):
        """K_a=0.08, A=0.05, D=0.10 (tranche straddles K_a).
        Pre-fix code gave ~65% RW; spec gives ~1089% RW. ~17× under-report."""
        ksa, A, D = 0.08, 0.05, 0.10
        p = 0.3
        expected_k = _ssfa_direct(ksa, A, D, p)
        expected_rw_pct = min(max(expected_k * 12.5 * 100, 15), 1250)
        rw = calculate_sec_sa_rw(ksa, A, D, n=50, lgd=0.50)
        assert rw == pytest.approx(expected_rw_pct, rel=1e-6)
        # Sanity: spans-K tranche must have substantial RW (well above 100%).
        assert rw > 500

    def test_matches_spec_straddle_with_w(self):
        """Same straddle with delinquency parameter w = 0.05 (K_a_adj = 0.076)."""
        ksa, A, D = 0.08, 0.05, 0.10
        w = 0.05
        ksa_adj = ksa * (1 - w)
        p = 0.3
        expected_k = _ssfa_direct(ksa_adj, A, D, p)
        expected_rw_pct = min(max(expected_k * 12.5 * 100, 15), 1250)
        rw = calculate_sec_sa_rw(ksa, A, D, n=50, lgd=0.50, w=w)
        assert rw == pytest.approx(expected_rw_pct, rel=1e-6)


class TestSSFAContinuity:
    def test_above_to_straddle_continuous(self):
        """K_SSFA should be continuous as A crosses K_a from above to straddle."""
        ksa = 0.08
        D = 0.20
        # Just above K_a.
        rw_above = calculate_sec_sa_rw(ksa, attachment=0.08 + 1e-6, detachment=D, n=50, lgd=0.5)
        # Just below (straddle).
        rw_straddle = calculate_sec_sa_rw(ksa, attachment=0.08 - 1e-6, detachment=D, n=50, lgd=0.5)
        # The straddle tranche includes a tiny full-loss strip → marginally higher.
        # Both should be close (within 1% relative).
        assert abs(rw_above - rw_straddle) / max(rw_above, 1e-3) < 0.01
