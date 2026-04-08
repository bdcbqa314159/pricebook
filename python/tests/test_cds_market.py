"""Tests for CDS market making tools."""

import pytest
from datetime import date

from pricebook.cds_market import (
    build_cds_curve, reprice_spreads,
    spread_to_upfront, upfront_to_spread,
    pricing_ladder, mark_to_market, roll_pnl,
)
from pricebook.cds import CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


REF = date(2024, 1, 15)


def _dc(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _spreads():
    return {1: 0.005, 3: 0.008, 5: 0.01, 7: 0.012, 10: 0.015}


def _cds(spread=0.01, notional=10_000_000):
    return CDS(REF, date(2029, 1, 15), spread, notional=notional)


# ---- Curve building ----

class TestBuildCurve:
    def test_roundtrip_spreads(self):
        """Bootstrapped curve reprices input par spreads."""
        dc = _dc()
        spreads = _spreads()
        curve = build_cds_curve(REF, spreads, dc)
        repriced = reprice_spreads(REF, curve, dc, tenors=list(spreads.keys()))
        for tenor, s in spreads.items():
            assert repriced[tenor] == pytest.approx(s, rel=0.01)

    def test_survival_decreasing(self):
        dc = _dc()
        curve = build_cds_curve(REF, _spreads(), dc)
        s5 = curve.survival(date(2029, 1, 15))
        s10 = curve.survival(date(2034, 1, 15))
        assert 0 < s10 < s5 < 1

    def test_single_tenor(self):
        dc = _dc()
        curve = build_cds_curve(REF, {5: 0.01}, dc)
        repriced = reprice_spreads(REF, curve, dc, tenors=[5])
        assert repriced[5] == pytest.approx(0.01, rel=0.01)


# ---- Upfront / running conversion ----

class TestUpfrontConversion:
    def test_at_par_zero_upfront(self):
        """CDS at par spread has zero upfront."""
        dc = _dc()
        curve = build_cds_curve(REF, {5: 0.01}, dc)
        upfront = spread_to_upfront(REF, 5, 0.01, 0.01, dc, curve)
        assert upfront == pytest.approx(0.0, abs=1e-8)

    def test_off_par_nonzero(self):
        dc = _dc()
        curve = build_cds_curve(REF, {5: 0.01}, dc)
        upfront = spread_to_upfront(REF, 5, 0.015, 0.01, dc, curve)
        assert upfront > 0  # market wider than coupon → buyer pays upfront

    def test_roundtrip(self):
        dc = _dc()
        curve = build_cds_curve(REF, {5: 0.01}, dc)
        upfront = spread_to_upfront(REF, 5, 0.015, 0.01, dc, curve)
        spread_back = upfront_to_spread(REF, 5, upfront, 0.01, dc, curve)
        assert spread_back == pytest.approx(0.015, rel=0.01)


# ---- Pricing ladder ----

class TestPricingLadder:
    def test_ladder_length(self):
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        ladder = pricing_ladder(_cds(), dc, sc)
        assert len(ladder) > 5

    def test_ladder_sorted(self):
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        ladder = pricing_ladder(_cds(), dc, sc)
        spreads = [r.spread for r in ladder]
        assert spreads == sorted(spreads)

    def test_center_rung_near_market(self):
        """The zero-bump rung should have spread = CDS spread."""
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        cds = _cds(spread=0.01)
        ladder = pricing_ladder(cds, dc, sc, bumps_bps=[0])
        assert len(ladder) == 1
        assert ladder[0].spread == pytest.approx(0.01)

    def test_wider_spread_higher_pv(self):
        """For protection buyer, wider spread = higher PV."""
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        ladder = pricing_ladder(_cds(), dc, sc)
        # Find narrow and wide rungs
        pvs = {r.spread: r.pv for r in ladder}
        spreads = sorted(pvs.keys())
        # Lower spread = protection buyer pays less premium = higher PV
        assert pvs[spreads[0]] > pvs[spreads[-1]]


# ---- Mark-to-market ----

class TestMTM:
    def test_at_par(self):
        """CDS at par has near-zero PV."""
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        cds = _cds()
        par = cds.par_spread(dc, sc)
        at_par = CDS(REF, date(2029, 1, 15), par, notional=10_000_000)
        mtm = mark_to_market(at_par, dc, sc)
        assert abs(mtm["pv"]) < 100
        assert abs(mtm["spread_to_par"]) < 1e-8

    def test_off_market(self):
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        cds = _cds(spread=0.02)  # above par
        mtm = mark_to_market(cds, dc, sc)
        assert mtm["pv"] != 0
        assert mtm["spread_to_par"] != 0

    def test_rpv01_positive(self):
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        mtm = mark_to_market(_cds(), dc, sc)
        assert mtm["rpv01"] > 0


# ---- Roll P&L ----

class TestRollPnL:
    def test_at_par_zero_roll(self):
        """Rolling an at-par CDS to a new at-par contract → near-zero roll P&L."""
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        par = CDS(REF, date(2029, 1, 15), 0.01, notional=10_000_000).par_spread(dc, sc)
        old = CDS(REF, date(2029, 1, 15), par, notional=10_000_000)
        result = roll_pnl(old, date(2034, 1, 15), dc, sc)
        # Old is at par (PV≈0), new is at par (PV≈0)
        assert abs(result["roll_pnl"]) < 100

    def test_off_market_roll(self):
        """Off-market CDS has nonzero roll P&L."""
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        old = CDS(REF, date(2029, 1, 15), 0.02, notional=10_000_000)
        result = roll_pnl(old, date(2034, 1, 15), dc, sc)
        assert result["roll_pnl"] != 0

    def test_roll_extends_maturity(self):
        dc = _dc()
        sc = SurvivalCurve.flat(REF, 0.02)
        old = _cds()
        result = roll_pnl(old, date(2034, 1, 15), dc, sc)
        assert result["new_end"] == date(2034, 1, 15)
        assert result["old_end"] == date(2029, 1, 15)
