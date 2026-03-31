"""Tests for basis swap."""

import pytest
from datetime import date

from pricebook.basis_swap import BasisSwap
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


class TestConstruction:
    def test_basic(self):
        swap = BasisSwap(REF, END, spread=0.001)
        assert swap.spread == 0.001


class TestPV:
    def test_pv_zero_when_same_curve(self):
        """Both legs on the same curve with zero spread → PV ≈ 0."""
        curve = make_flat_curve(REF, 0.05)
        swap = BasisSwap(REF, END, spread=0.0)
        pv = swap.pv(curve, curve, curve)
        assert pv == pytest.approx(0.0, abs=1.0)

    def test_pv_positive_with_spread(self):
        """Positive spread on leg 1 when same projection → positive PV."""
        curve = make_flat_curve(REF, 0.05)
        swap = BasisSwap(REF, END, spread=0.005)
        pv = swap.pv(curve, curve, curve)
        assert pv > 0

    def test_pv_changes_with_different_projections(self):
        """Different projection curves → non-zero PV even without spread."""
        disc = make_flat_curve(REF, 0.04)
        proj1 = make_flat_curve(REF, 0.05)
        proj2 = make_flat_curve(REF, 0.045)
        swap = BasisSwap(REF, END, spread=0.0)
        pv = swap.pv(disc, proj1, proj2)
        # leg1 (higher rate) > leg2 → positive PV
        assert pv > 0


class TestParSpread:
    def test_par_spread_zero_same_curve(self):
        """Same curve for both → par spread ≈ 0."""
        curve = make_flat_curve(REF, 0.05)
        swap = BasisSwap(REF, END, spread=0.0)
        ps = swap.par_spread(curve, curve, curve)
        assert ps == pytest.approx(0.0, abs=0.0001)

    def test_par_spread_positive_when_leg2_higher(self):
        """Leg 2 projects higher rate → par spread positive (leg 1 needs extra)."""
        disc = make_flat_curve(REF, 0.04)
        proj1 = make_flat_curve(REF, 0.045)
        proj2 = make_flat_curve(REF, 0.05)
        swap = BasisSwap(REF, END, spread=0.0)
        ps = swap.par_spread(disc, proj1, proj2)
        assert ps > 0

    def test_par_spread_reprices_to_zero(self):
        """Swap at par spread has PV ≈ 0."""
        disc = make_flat_curve(REF, 0.04)
        proj1 = make_flat_curve(REF, 0.045)
        proj2 = make_flat_curve(REF, 0.05)

        temp = BasisSwap(REF, END, spread=0.0)
        ps = temp.par_spread(disc, proj1, proj2)

        swap = BasisSwap(REF, END, spread=ps)
        pv = swap.pv(disc, proj1, proj2)
        assert pv == pytest.approx(0.0, abs=1.0)

    def test_par_spread_different_frequencies(self):
        """3M vs 6M with different projections."""
        disc = make_flat_curve(REF, 0.04)
        proj_3m = make_flat_curve(REF, 0.05)
        proj_6m = make_flat_curve(REF, 0.048)
        swap = BasisSwap(
            REF, END, spread=0.0,
            leg1_frequency=Frequency.QUARTERLY,
            leg2_frequency=Frequency.SEMI_ANNUAL,
        )
        ps = swap.par_spread(disc, proj_3m, proj_6m)
        # Should be a small number (basis in bp range)
        assert abs(ps) < 0.02
