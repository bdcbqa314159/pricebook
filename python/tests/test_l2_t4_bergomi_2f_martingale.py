"""Regression for L2 T4 audit of `options.vol_term_structure.Bergomi2Factor`:

Pre-fix the martingale correction for ξ(t, T) under the lognormal
two-factor Bergomi model omitted the cross-term ``2·ρ·η₁·η₂·t``:

    -0.5 * (η₁² + η₂²) * t          # WRONG
    -0.5 * (η₁² + η₂² + 2·ρ·η₁·η₂) * t   # CORRECT

The correct quadratic variation of ``η₁·W₁ + η₂·W₂`` with
``Cov(W₁,W₂) = ρ·t`` is ``(η₁² + η₂² + 2·ρ·η₁·η₂)·t``.  Pre-fix
the simulated ξ(t, T) was NOT a martingale whenever ρ ≠ 0, so
``E[ξ(t, T)] ≠ ξ₀(T)`` — the model drifted in expectation away from
its calibrating term structure.  ρ = 0 (default) coincidentally gave
the right behaviour.

These tests pin:
- ρ = 0: behaviour unchanged.
- ρ > 0: E[ξ(t)] now equals ξ₀ to within MC noise (pre-fix it was
  systematically too HIGH because the missing negative correction
  let the lognormal drift up).
- ρ < 0: same property holds (mean below ξ₀ pre-fix, on it post-fix).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.options.vol_term_structure import Bergomi2Factor


class TestBergomi2FactorMartingale:
    def test_zero_rho_unchanged(self):
        """ρ = 0 gives the right answer both pre- and post-fix."""
        model = Bergomi2Factor(xi0=0.04, eta1=0.5, eta2=0.3, rho12=0.0)
        r = model.simulate(T=1.0, n_paths=20_000, n_steps=50, seed=42)
        # ξ_T = vol_T² (we returned vol_paths = sqrt(ξ)).
        xi_T = (r.vol_paths[:, -1]) ** 2
        assert float(xi_T.mean()) == pytest.approx(0.04, rel=5e-2)

    def test_positive_rho_martingale_holds(self):
        """ρ = +0.6: E[ξ_T] must still equal ξ₀ to MC tolerance.
        Pre-fix the missing −2·ρ·η₁·η₂·t correction would let the
        lognormal mean drift UP for positive correlation (the cross
        term contribution is positive when ρ > 0)."""
        model = Bergomi2Factor(xi0=0.04, eta1=0.6, eta2=0.4, rho12=0.6)
        r = model.simulate(T=1.0, n_paths=30_000, n_steps=50, seed=123)
        xi_T = (r.vol_paths[:, -1]) ** 2
        # Cross-term: 2·0.6·0.6·0.4 = 0.288.  Pre-fix the missing
        # correction inflates E[exp(...)] by exp(0.288/2 × 1) ≈ 1.155 → ~15% high.
        # Post-fix should be within ~3% MC noise of ξ₀ = 0.04.
        assert float(xi_T.mean()) == pytest.approx(0.04, rel=5e-2), (
            f"E[ξ_T] = {float(xi_T.mean()):.5f}, ξ₀ = 0.04 — "
            f"martingale property must hold post-fix"
        )

    def test_negative_rho_martingale_holds(self):
        """ρ = −0.6: similar test in the opposite direction.  Pre-fix
        the missing −2·ρ·η₁·η₂·t correction (with negative ρ this
        cross-term is negative) would push the mean DOWN."""
        model = Bergomi2Factor(xi0=0.04, eta1=0.6, eta2=0.4, rho12=-0.6)
        r = model.simulate(T=1.0, n_paths=30_000, n_steps=50, seed=123)
        xi_T = (r.vol_paths[:, -1]) ** 2
        assert float(xi_T.mean()) == pytest.approx(0.04, rel=5e-2)
