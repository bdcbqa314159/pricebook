"""Regression for L2 T4 audit — remaining ``_phi`` date-rounding duplicates:

Two more modules carried the same finite-difference defect that
``G2PPTree._fwd_rate`` had (T4-G2T1):

* ``fixed_income.callable_floater_g2pp._phi_at`` — used for G2++ FRN
  tree (and a sister ``r0`` derivation at the bottom of the module
  that estimated the initial short rate).
* ``structured.cms_spread_g2pp._fwd`` — used in MC path simulation
  for CMS-spread structured products.

Both used ``eps = 1e-5`` years for the forward-rate finite difference;
``date_from_year_fraction``'s day rounding turned the result into
~0 or ~137·r per call across the time grid.

Fix (T4-G2T3): delegate to ``DiscountCurve.instantaneous_forward(t)``
(the same one-line collapse applied in T4-G2T1 / T4-BSWG1).

This test pins the smoothness of the helpers on a flat curve.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.fixed_income.callable_floater_g2pp import _phi_at
from pricebook.models.vasicek import G2PlusPlus


REF = date(2026, 1, 15)


@pytest.fixture
def g2pp() -> G2PlusPlus:
    curve = DiscountCurve.flat(REF, 0.04)
    return G2PlusPlus(a=0.1, b=0.05, sigma1=0.01, sigma2=0.008,
                       rho=-0.3, curve=curve)


class TestCallableFloaterG2PPPhi:
    """``_phi_at`` must produce a smooth phi(t) ≈ short rate on a flat
    curve.  Pre-fix it alternated between ~0 and ~5.48 across the grid."""

    def test_phi_at_smooth_on_flat_curve(self, g2pp):
        for k in range(31):
            t = k * 5.0 / 30.0
            phi = _phi_at(g2pp, t)
            assert 0.03 < phi < 0.05, (
                f"phi({t:.4f}) = {phi:.4f} — expected ≈ 0.04 + O(σ²)"
            )


class TestCMSSpreadG2PPPriceFinite:
    """``cms_spread_g2pp_mc`` runs to a finite, positive price post-fix.

    Pre-fix the buggy ``_fwd`` inflated phi(t) on a deterministic ~1/8
    of the simulation grid, producing arbitrarily-biased MC discount
    factors.  Smoke-test that the price is finite and ≥ 0 on a flat
    curve.
    """

    def test_mc_price_finite(self, g2pp):
        from pricebook.structured.cms_spread_g2pp import cms_spread_option_g2pp

        result = cms_spread_option_g2pp(
            g2pp,
            cms_long_tenor=10.0,
            cms_short_tenor=2.0,
            strike=0.0,
            T=1.0,
            option_type="call",
            notional=1.0,
            n_paths=2000,
            seed=42,
        )
        # Price is a sane positive number (not NaN, not absurdly large).
        # Pre-fix the buggy _fwd inflated phi(t) on a deterministic
        # ~1/8 of the simulation grid, producing arbitrarily-biased
        # MC discount factors.
        assert result.price >= 0.0
        assert result.price < 1.0   # bounded by notional
        assert result.price == result.price   # not NaN
