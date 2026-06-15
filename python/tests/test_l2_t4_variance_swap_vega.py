"""Regression for L2 T4 audit of `options.vol_derivatives_advanced.variance_swap_greeks`:

Pre-fix the vega formula was

    vega = 2 * vol * T * notional_var * remaining

— with a spurious ``T`` factor and no discount-factor scaling.  The
canonical variance-swap vega convention is

    vega = 2 · σ · notional_var · DF · remaining

so that ``vega ≈ vega_notional`` at ATM inception regardless of T
(by design of ``notional_var = vega_notional / (2·√strike_var)``).

Pre-fix consequence: a 2y var swap was reported with 2× the vega of
a 1y var swap at the same ``vega_notional`` — wrong, since the
notional is already calibrated to vega.

Existing tests in ``test_vol_derivatives_advanced`` use ``T = 1.0``
throughout, so the bug never surfaced.
"""

from __future__ import annotations

import math

import pytest

from pricebook.options.vol_derivatives_advanced import variance_swap_greeks


class TestVegaIndependentOfT:
    def test_atm_vega_matches_vega_notional(self):
        """At ATM inception (strike_var = vol²) with r=0, vega should
        equal vega_notional regardless of T."""
        vega_notional = 100_000.0
        vol = 0.20
        strike_var = vol ** 2

        for T in [0.25, 1.0, 2.0, 5.0]:
            g = variance_swap_greeks(
                spot=100.0, strike_var=strike_var, T=T, vol=vol,
                notional_vega=vega_notional, r=0.0,
            )
            assert g.vega == pytest.approx(vega_notional, rel=1e-9), (
                f"T={T}: vega={g.vega:.0f}, expected={vega_notional}"
            )

    def test_long_dated_with_rate_includes_discount(self):
        """Vega should include the remaining-time DF factor.  At T=5y
        and r=5%, the discount factor is exp(-0.25) ≈ 0.779, so vega
        ≈ 0.779 × vega_notional at inception."""
        vega_notional = 100_000.0
        vol = 0.20
        T = 5.0
        r = 0.05
        g = variance_swap_greeks(
            spot=100.0, strike_var=vol ** 2, T=T, vol=vol,
            notional_vega=vega_notional, r=r,
        )
        expected = vega_notional * math.exp(-r * T)
        assert g.vega == pytest.approx(expected, rel=1e-9), (
            f"vega = {g.vega:.0f}, expected {expected:.0f} "
            f"(= vega_notional × exp(-rT))"
        )


class TestVegaScalesWithRemaining:
    def test_half_elapsed_halves_vega(self):
        """As time elapses, vega scales by ``remaining`` (only future
        variance is sensitive to the implied vol)."""
        vega_notional = 100_000.0
        vol = 0.20
        g_fresh = variance_swap_greeks(
            100.0, vol ** 2, 1.0, vol, vega_notional, r=0.0,
            elapsed_fraction=0.0,
        )
        g_half = variance_swap_greeks(
            100.0, vol ** 2, 1.0, vol, vega_notional, r=0.0,
            elapsed_fraction=0.5, realised_var_so_far=vol ** 2,
        )
        assert g_half.vega == pytest.approx(0.5 * g_fresh.vega, rel=1e-9)
