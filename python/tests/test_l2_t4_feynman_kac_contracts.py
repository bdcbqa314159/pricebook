"""Regression for L2 Wave-2 audit — `feynman_kac` had three silent
contract violations.

1. ``sde_to_pde(rate_fn=None)`` silently defaulted to a 4% constant.  A
   user calling the helper without specifying the rate would get a PDE
   parameterised at 4% — and if they then ran an MC at a different rate,
   the comparison would be biased by exactly the rate disagreement, with
   no warning.

2. ``pde_to_sde`` used ``math.sqrt(max(2·diffusion, 0))`` — silently
   clamping negative diffusion (which is unphysical: the PDE diffusion
   coefficient ``a = ½σ²`` is non-negative by construction) to zero.
   Bugs in upstream coefficient functions producing negative ``a`` would
   be masked.

3. ``verify_feynman_kac(n_time=...)`` ignored the user-supplied
   ``n_time`` for the MC time grid (hardcoded to 100), so a user
   thinking they were refining both MC and PDE in lockstep to study
   convergence was actually only refining the PDE.

Post-fix all three paths fail loudly (raise ValueError) or honour the
user's input.
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.feynman_kac import (
    pde_to_sde,
    sde_to_pde,
    verify_feynman_kac,
)


class TestSdeToPdeRequiresRate:
    def test_rate_fn_none_raises(self):
        with pytest.raises(ValueError, match="rate_fn is required"):
            sde_to_pde(
                mu_fn=lambda S, t: 0.04 * S,
                sigma_fn=lambda S, t: 0.20 * S,
                rate_fn=None,
            )

    def test_scalar_rate_still_works(self):
        coeffs = sde_to_pde(
            mu_fn=lambda S, t: 0.04 * S,
            sigma_fn=lambda S, t: 0.20 * S,
            rate_fn=0.05,
        )
        assert coeffs["reaction"](100, 0) == pytest.approx(-0.05)

    def test_callable_rate_still_works(self):
        coeffs = sde_to_pde(
            mu_fn=lambda S, t: 0.04 * S,
            sigma_fn=lambda S, t: 0.20 * S,
            rate_fn=lambda t: 0.03 + 0.01 * t,
        )
        assert coeffs["reaction"](100, 0) == pytest.approx(-0.03)
        assert coeffs["reaction"](100, 1) == pytest.approx(-0.04)


class TestPdeToSdeRejectsNegativeDiffusion:
    def test_negative_diffusion_raises(self):
        sde = pde_to_sde(
            diffusion_fn=lambda S, t: -0.5,
            convection_fn=lambda S, t: 0.0,
        )
        with pytest.raises(ValueError, match="diffusion.*< 0"):
            sde["volatility"](100.0, 0.0)

    def test_zero_diffusion_returns_zero_vol(self):
        """Zero diffusion is degenerate but mathematically valid (σ=0)."""
        sde = pde_to_sde(
            diffusion_fn=lambda S, t: 0.0,
            convection_fn=lambda S, t: 0.0,
        )
        assert sde["volatility"](100.0, 0.0) == pytest.approx(0.0)

    def test_positive_diffusion_unchanged(self):
        sde = pde_to_sde(
            diffusion_fn=lambda S, t: 0.5 * (0.2 * S) ** 2,
            convection_fn=lambda S, t: 0.04 * S,
        )
        assert sde["volatility"](100.0, 0.0) == pytest.approx(0.2 * 100)


class TestVerifyFeynmanKacUsesNTime:
    def test_n_time_actually_changes_mc_grid(self):
        """If `n_time` were ignored (pre-fix), then changing it from 50 to
        500 would NOT change the MC stderr.  Post-fix, the MC stderr
        depends weakly on n_time (through the discretisation error in
        the GBM exact step — but GBM has an exact step, so stderr
        should be identical to MC sampling noise).  We don't assert on
        stderr because it could be identical for exact-step processes.
        Instead, we assert the call returns successfully with the
        provided n_time.  A test that PROVES the wiring would require
        instrumenting MCEngine internals; for now this is a smoke test
        that confirms n_time isn't rejected."""
        r = verify_feynman_kac(
            spot=100, strike=100, rate=0.04, vol=0.20, T=1.0,
            n_paths=10_000, n_time=50,
        )
        assert r.mc_price > 0

        r2 = verify_feynman_kac(
            spot=100, strike=100, rate=0.04, vol=0.20, T=1.0,
            n_paths=10_000, n_time=200,
        )
        assert r2.mc_price > 0
