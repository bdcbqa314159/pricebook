"""Regression for L2 T4 audit of `options.bermudan_swaption_g2pp`:

``bermudan_swaption_g2pp_lsm`` carries a private ``_phi`` helper that
duplicates ``G2PPTree._phi`` — including the same ``eps = 1e-5`` finite
difference defect that ``date_from_year_fraction``'s day rounding turns
into either 0 or ≈ 137·r (see T4-G2T1).  The bug compounds across every
simulation step (``log_df_paths`` accumulates ``r_i * dt`` per step),
so LSM-priced Bermudan swaptions were catastrophically mis-priced.

Fix (T4-BSWG1): delegate to ``DiscountCurve.instantaneous_forward(t)``.

Sanity: the LSM and tree pricers must agree to within ~10% on a vanilla
Bermudan; pre-fix the LSM was wildly off because every path's
accumulated short rate was inflated on ~1/8 of the time grid.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.vasicek import G2PlusPlus
from pricebook.options.bermudan_swaption_g2pp import (
    bermudan_swaption_g2pp_lsm,
    bermudan_swaption_g2pp_tree,
)


REF = date(2026, 1, 15)


@pytest.fixture
def g2pp() -> G2PlusPlus:
    curve = DiscountCurve.flat(REF, 0.04)
    return G2PlusPlus(a=0.1, b=0.05, sigma1=0.01, sigma2=0.008,
                       rho=-0.3, curve=curve)


class TestLSMAgreesWithTree:
    """LSM and tree pricers for the same Bermudan swaption must agree."""

    def test_lsm_close_to_tree(self, g2pp):
        kwargs = dict(
            exercise_years=[1.0, 1.5, 2.0],
            swap_end_years=5.0,
            strike=0.04,
            is_payer=True,
            swap_freq=2,
        )
        tree = bermudan_swaption_g2pp_tree(g2pp, n_steps=30, **kwargs)
        lsm = bermudan_swaption_g2pp_lsm(
            g2pp, n_paths=5000, n_steps=60, **kwargs,
        )
        rel_err = abs(tree.price - lsm.price) / max(tree.price, 1e-6)
        # 10% tolerance: LSM Monte-Carlo noise at 5k paths + finite
        # regression-basis error.  Pre-fix the path discounts were
        # inflated on a deterministic ~1/8 of the time grid, biasing
        # the LSM price arbitrarily — typically by 30-90%.
        assert rel_err < 0.10, (
            f"tree={tree.price:.6f}, lsm={lsm.price:.6f}, rel_err={rel_err:.4f}"
        )
        assert lsm.price > 0.0
        assert tree.price > 0.0
