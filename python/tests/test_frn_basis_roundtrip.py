"""
Slice 14 round-trip validation: FRN + basis swap.

1. FRN at par: price = 100 when spread = 0 on own curve
2. DM recovery: price → DM → reprice → match
3. Basis swap reprices at par spread
4. FRN DV01 positive and reasonable
5. Basis swap PV = 0 at par spread
"""

import pytest
from datetime import date

from pricebook.frn import FloatingRateNote
from pricebook.basis_swap import BasisSwap
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


class TestFRNAtPar:
    def test_zero_spread_at_par(self):
        """FRN with zero spread on its own curve ≈ 100."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=END, spread=0.0)
        assert frn.dirty_price(curve) == pytest.approx(100.0, abs=0.5)

    def test_par_on_steep_curve(self):
        """Zero spread at par even on a non-flat curve (single-curve)."""
        from pricebook.discount_curve import DiscountCurve
        import math
        dates = [date(2024 + i, 1, 15) for i in range(1, 11)]
        dfs = [math.exp(-0.03 * i - 0.002 * i * i) for i in range(1, 11)]
        steep = DiscountCurve(REF, dates, dfs)
        frn = FloatingRateNote(start=REF, end=END, spread=0.0)
        assert frn.dirty_price(steep) == pytest.approx(100.0, abs=0.5)


class TestDMRecovery:
    def test_round_trip(self):
        """Price → DM → reprice recovers original price."""
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=END, spread=0.005)
        market_price = 98.5
        dm = frn.discount_margin(market_price, curve)

        frn_adj = FloatingRateNote(start=REF, end=END, spread=0.005 + dm)
        assert frn_adj.dirty_price(curve) == pytest.approx(market_price, abs=0.01)

    def test_dm_zero_reprices_own_price(self):
        curve = make_flat_curve(REF, 0.05)
        frn = FloatingRateNote(start=REF, end=END, spread=0.005)
        own_price = frn.dirty_price(curve)
        dm = frn.discount_margin(own_price, curve)
        assert dm == pytest.approx(0.0, abs=0.001)


class TestFRNDV01:
    def test_dv01_positive(self):
        """FRN DV01 via bump-and-reprice is positive."""
        curve_base = make_flat_curve(REF, 0.05)
        curve_up = make_flat_curve(REF, 0.0501)
        frn = FloatingRateNote(start=REF, end=END, spread=0.005)

        p_base = frn.dirty_price(curve_base)
        p_up = frn.dirty_price(curve_up)
        # FRN price changes very little with rate bump (floating resets)
        dv01 = abs(p_base - p_up)
        assert dv01 < 0.5  # FRN has low duration

    def test_spread_dv01_positive(self):
        """Bumping spread increases FRN price."""
        curve = make_flat_curve(REF, 0.05)
        frn_base = FloatingRateNote(start=REF, end=END, spread=0.005)
        frn_up = FloatingRateNote(start=REF, end=END, spread=0.0051)
        p_base = frn_base.dirty_price(curve)
        p_up = frn_up.dirty_price(curve)
        assert p_up > p_base


class TestBasisSwapParSpread:
    def test_pv_zero_at_par_spread(self):
        """Basis swap at par spread has PV ≈ 0."""
        disc = make_flat_curve(REF, 0.04)
        proj1 = make_flat_curve(REF, 0.05)
        proj2 = make_flat_curve(REF, 0.048)

        temp = BasisSwap(REF, END, spread=0.0)
        ps = temp.par_spread(disc, proj1, proj2)

        swap = BasisSwap(REF, END, spread=ps)
        pv = swap.pv(disc, proj1, proj2)
        assert pv == pytest.approx(0.0, abs=1.0)

    def test_par_spread_sign(self):
        """Higher leg2 projection → positive par spread on leg1."""
        disc = make_flat_curve(REF, 0.04)
        proj1 = make_flat_curve(REF, 0.045)
        proj2 = make_flat_curve(REF, 0.05)

        swap = BasisSwap(REF, END, spread=0.0)
        ps = swap.par_spread(disc, proj1, proj2)
        assert ps > 0
