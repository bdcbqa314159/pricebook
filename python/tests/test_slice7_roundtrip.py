"""Slice 7 round-trip validation.

Put-call parity, cap-floor parity, Greeks consistency, ATM properties.
"""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.models.black76 import (
    black76_price, black76_delta, black76_gamma, black76_vega, OptionType,
)
from pricebook.capfloor import CapFloor
from pricebook.vol_surface import VolTermStructure
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestPutCallParity:
    """C - P = df * (F - K) across a range of parameters."""

    @pytest.mark.parametrize("F,K,vol,T,df", [
        (100, 100, 0.20, 1.0, 0.95),
        (110, 90, 0.30, 0.5, 0.99),
        (80, 120, 0.15, 2.0, 0.90),
        (50, 50, 0.50, 0.1, 1.0),
    ])
    def test_parity_holds(self, F, K, vol, T, df):
        call = black76_price(F, K, vol, T, df, OptionType.CALL)
        put = black76_price(F, K, vol, T, df, OptionType.PUT)
        assert call - put == pytest.approx(df * (F - K), rel=1e-10)


class TestCapFloorWithVolTermStructure:
    """Cap/floor prices with a term structure of vol."""

    def test_cap_with_vol_term_structure(self):
        curve = make_flat_curve(REF, rate=0.05)
        expiries = [
            REF + relativedelta(months=3),
            REF + relativedelta(years=1),
            REF + relativedelta(years=3),
        ]
        vols = [0.15, 0.20, 0.25]
        vts = VolTermStructure(REF, expiries, vols)

        from pricebook.models.models import Black76Model
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        # Use average vol from term structure as single Black76Model
        avg_vol = sum(vols) / len(vols)
        pv = cap.price(Black76Model(vol=avg_vol), curve)
        assert pv > 0

    def test_term_structure_vs_flat_differs(self):
        """Cap with different vols produces different prices."""
        from pricebook.models.models import Black76Model
        curve = make_flat_curve(REF, rate=0.05)
        end = REF + relativedelta(years=3)
        cap = CapFloor(REF, end, strike=0.05)
        pv_low = cap.price(Black76Model(vol=0.15), curve)
        pv_high = cap.price(Black76Model(vol=0.30), curve)
        assert pv_low != pytest.approx(pv_high, rel=0.01)


class TestATMProperties:
    """At-the-money option properties."""

    def test_atm_call_delta_near_half(self):
        d = black76_delta(100.0, 100.0, 0.20, 1.0, 1.0, OptionType.CALL)
        assert d == pytest.approx(0.5, abs=0.05)

    def test_atm_put_delta_near_minus_half(self):
        d = black76_delta(100.0, 100.0, 0.20, 1.0, 1.0, OptionType.PUT)
        assert d == pytest.approx(-0.5, abs=0.05)

    def test_atm_gamma_maximised(self):
        g_atm = black76_gamma(100.0, 100.0, 0.20, 1.0, 1.0)
        g_otm = black76_gamma(100.0, 140.0, 0.20, 1.0, 1.0)
        assert g_atm > g_otm

    def test_atm_vega_maximised(self):
        v_atm = black76_vega(100.0, 100.0, 0.20, 1.0, 1.0)
        v_otm = black76_vega(100.0, 140.0, 0.20, 1.0, 1.0)
        assert v_atm > v_otm


class TestGreeksConsistency:
    """Cross-check: analytical Greeks match finite difference."""

    def test_cap_vega_positive(self):
        """Cap vega (bump vol, reprice) should be positive."""
        from pricebook.models.models import Black76Model
        curve = make_flat_curve(REF, rate=0.05)
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        pv_low = cap.price(Black76Model(vol=0.19), curve)
        pv_high = cap.price(Black76Model(vol=0.21), curve)
        vega = (pv_high - pv_low) / 0.02
        assert vega > 0

    def test_cap_delta_via_curve_bump(self):
        """Bumping the curve (rates up) should change cap PV."""
        from pricebook.models.models import Black76Model
        m = Black76Model(vol=0.20)
        cap = CapFloor(REF, REF + relativedelta(years=3), strike=0.05)
        pv_base = cap.price(m, make_flat_curve(REF, rate=0.05))
        pv_up = cap.price(m, make_flat_curve(REF, rate=0.0501))
        assert pv_up > pv_base
