"""Regression for L2 T4 audit of `options.capfloor.CapFloor.pv_ctx`:

Pre-fix ``_capfloor_pv_ctx`` called ``vol_surface.vol()`` with NO
arguments and built a single ``Black76Model(vol)`` for the whole cap.
This works only for ``FlatVol`` surfaces — every other surface type
(VolTermStructure, smile cube) has ``vol(expiry, strike)`` with no
default for ``expiry`` and either errors or silently returns the
first-pillar / lowest-expiry vol for every caplet.

Either way the long caplets get the wrong vol — silently for the
soft-failure path, loudly for the strict surfaces.

Fix: loop per caplet and call ``vol_surface.vol(accrual_start, strike)``
at each caplet's accrual start.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.core.schedule import Frequency
from pricebook.core.day_count import DayCountConvention
from pricebook.models.black76 import OptionType
from pricebook.options.capfloor import CapFloor
from pricebook.options.vol_surface import FlatVol, VolTermStructure


REF = date(2026, 4, 28)


def _cap(maturity_years=3):
    return CapFloor(
        start=REF, end=REF + timedelta(days=365 * maturity_years),
        strike=0.03, option_type=OptionType.CALL,
        notional=1_000_000.0, frequency=Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
    )


def _curve():
    return DiscountCurve.flat(REF, 0.03)


class TestVolTermStructureRouted:
    def test_term_structure_surface_used_per_caplet(self):
        """Using a vol term-structure surface, the pv_ctx must route
        each caplet through its own expiry vol — NOT collapse to a
        single vol.  Build two cases that share only short-dated vols
        and differ on long-dated vols; the prices must differ."""
        # Term structure where vols vary materially with expiry.
        expiries = [REF + timedelta(days=365 * k) for k in (1, 2, 3)]
        surface_low_long = VolTermStructure(
            reference_date=REF, expiries=expiries,
            vols=[0.30, 0.30, 0.30],
        )
        surface_high_long = VolTermStructure(
            reference_date=REF, expiries=expiries,
            vols=[0.30, 0.30, 0.60],  # long-end vol doubles
        )

        ctx_low = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            vol_surfaces={"ir": surface_low_long},
        )
        ctx_high = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            vol_surfaces={"ir": surface_high_long},
        )

        cap = _cap(maturity_years=3)
        pv_low = cap.pv_ctx(ctx_low)
        pv_high = cap.pv_ctx(ctx_high)
        # Long-end vol bump must raise the cap PV (positive vega on long
        # caplets).  Pre-fix, both used surface.vol() (no args) → either
        # errored or returned the same vol → identical prices.
        assert pv_high > pv_low, (
            f"Long-end vol from 30% to 60% should raise cap PV "
            f"(low={pv_low:.0f}, high={pv_high:.0f})"
        )


class TestFlatVolUnchanged:
    def test_flat_vol_still_works(self):
        """FlatVol surface — both pre-fix and post-fix should give the
        same number (all caplets see the same vol)."""
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            vol_surfaces={"ir": FlatVol(vol=0.30)},
        )
        cap = _cap(maturity_years=3)
        pv = cap.pv_ctx(ctx)
        assert math.isfinite(pv) and pv > 0


class TestMissingSurfaceRaises:
    def test_no_vol_surface_raises(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            vol_surfaces={},
        )
        cap = _cap(maturity_years=1)
        with pytest.raises(ValueError, match="IR vol surface"):
            cap.pv_ctx(ctx)
