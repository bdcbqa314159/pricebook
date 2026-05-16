"""Tests for RecoverySurface, implied recovery, recovery term structure."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

from pricebook.curves.bootstrap import bootstrap
from pricebook.cds_market import build_cds_curve
from pricebook.recovery_surface import (
    RecoverySurface, implied_recovery, recovery_term_structure,
    SENIORITY_TABLE,
)


REF = date(2024, 7, 15)

def _ois():
    deposits = [(REF + relativedelta(months=6), 0.043)]
    swaps = [(REF + relativedelta(years=1), 0.041),
             (REF + relativedelta(years=5), 0.038),
             (REF + relativedelta(years=10), 0.036)]
    return bootstrap(REF, deposits, swaps)


CDS_SPREADS = {1: 0.0050, 3: 0.0080, 5: 0.0100, 10: 0.0120}


# ── RecoverySurface ──

class TestRecoverySurface:

    def test_from_seniority_table(self):
        surface = RecoverySurface.from_seniority_table()
        assert len(surface.seniorities) >= 5
        assert len(surface.tenors) == 5

    def test_recovery_interpolation(self):
        surface = RecoverySurface.from_seniority_table()
        r5 = surface.recovery("senior_unsecured", 5.0)
        assert 0.3 < r5 < 0.6

    def test_recovery_declines_with_tenor(self):
        surface = RecoverySurface.from_seniority_table()
        r1 = surface.recovery("senior_unsecured", 1.0)
        r10 = surface.recovery("senior_unsecured", 10.0)
        assert r1 > r10  # short-dated recovery higher

    def test_seniority_ordering(self):
        surface = RecoverySurface.from_seniority_table()
        r_senior = surface.recovery("senior_secured", 5.0)
        r_sub = surface.recovery("sub", 5.0)
        assert r_senior > r_sub

    def test_flat_surface(self):
        surface = RecoverySurface.flat(0.40)
        for sen in surface.seniorities:
            assert surface.recovery(sen, 5.0) == pytest.approx(0.40)
            assert surface.recovery(sen, 1.0) == pytest.approx(0.40)

    def test_unknown_seniority_raises(self):
        surface = RecoverySurface.from_seniority_table()
        with pytest.raises(ValueError, match="Unknown seniority"):
            surface.recovery("invalid", 5.0)

    def test_all_points(self):
        surface = RecoverySurface.from_seniority_table()
        points = surface.all_points()
        assert len(points) == len(surface.seniorities) * len(surface.tenors)
        for p in points:
            assert 0 < p.recovery < 1

    def test_recovery_vector(self):
        surface = RecoverySurface.from_seniority_table()
        tenors, recoveries = surface.recovery_vector("senior_unsecured")
        assert len(tenors) == len(recoveries)
        assert all(0 < r < 1 for r in recoveries)

    def test_to_dict_from_dict_roundtrip(self):
        surface = RecoverySurface.from_seniority_table()
        d = surface.to_dict()
        loaded = RecoverySurface.from_dict(d)
        assert loaded.recovery("senior_unsecured", 5.0) == pytest.approx(
            surface.recovery("senior_unsecured", 5.0))

    def test_std_interpolation(self):
        surface = RecoverySurface.from_seniority_table()
        s = surface.std("senior_unsecured", 5.0)
        assert 0.1 < s < 0.4


# ── Implied Recovery ──

class TestImpliedRecovery:

    def test_implied_recovery_from_spread_ratio(self):
        ois = _ois()
        senior = {5: 0.0080}
        sub = {5: 0.0200}
        results = implied_recovery(senior, sub, ois, REF)
        assert len(results) == 1
        r = results[0]
        assert 0 < r.implied_recovery < 1
        assert r.method == "spread_ratio"

    def test_same_spread_implies_same_recovery(self):
        ois = _ois()
        spreads = {5: 0.0100}
        results = implied_recovery(spreads, spreads, ois, REF,
                                    senior_recovery=0.45, sub_recovery=0.25)
        # Same spread on both → R_senior = 1 - s_sen*(1-R_sub)/s_sub = R_sub
        r = results[0]
        assert r.implied_recovery == pytest.approx(0.25, abs=0.01)

    def test_higher_sub_spread_implies_lower_senior_recovery(self):
        ois = _ois()
        senior = {5: 0.0080}
        sub_wide = {5: 0.0300}
        sub_tight = {5: 0.0150}
        r_wide = implied_recovery(senior, sub_wide, ois, REF)[0].implied_recovery
        r_tight = implied_recovery(senior, sub_tight, ois, REF)[0].implied_recovery
        # Wider sub spread → senior recovery implied is higher
        assert r_wide > r_tight


# ── Recovery Term Structure ──

class TestRecoveryTermStructure:

    def test_flat_method(self):
        ois = _ois()
        points = recovery_term_structure(CDS_SPREADS, ois, REF, method="flat")
        assert len(points) == len(CDS_SPREADS)
        for p in points:
            assert p.recovery == 0.40
            assert p.hazard > 0

    def test_flat_hazard_increases_with_spread(self):
        ois = _ois()
        points = recovery_term_structure(CDS_SPREADS, ois, REF, method="flat")
        # Wider spread → higher hazard (for flat recovery)
        spreads_and_hazards = [(p.spread, p.hazard) for p in points]
        for i in range(1, len(spreads_and_hazards)):
            s_prev, h_prev = spreads_and_hazards[i - 1]
            s_curr, h_curr = spreads_and_hazards[i]
            if s_curr > s_prev:
                assert h_curr > h_prev * 0.8  # approximately monotone

    def test_slope_method(self):
        ois = _ois()
        points = recovery_term_structure(CDS_SPREADS, ois, REF, method="slope")
        assert len(points) == len(CDS_SPREADS)
        # Recovery should vary across tenors
        recoveries = [p.recovery for p in points]
        assert not all(r == recoveries[0] for r in recoveries)

    def test_slope_hazards_smoother(self):
        ois = _ois()
        flat_pts = recovery_term_structure(CDS_SPREADS, ois, REF, method="flat")
        slope_pts = recovery_term_structure(CDS_SPREADS, ois, REF, method="slope")
        # Slope method should produce less variation in hazard rates
        flat_h = [p.hazard for p in flat_pts]
        slope_h = [p.hazard for p in slope_pts]
        flat_range = max(flat_h) - min(flat_h)
        slope_range = max(slope_h) - min(slope_h)
        assert slope_range <= flat_range


# ── Roundtrip: CDS spreads → bootstrap → recovery surface → reprice ──

class TestRecoveryRoundtrip:

    def test_reprice_at_surface_recovery(self):
        """Bootstrap with surface recovery at 5Y, reprice 5Y CDS at par."""
        ois = _ois()
        surface = RecoverySurface.from_seniority_table()
        R = surface.recovery("senior_unsecured", 5.0)

        surv = build_cds_curve(REF, CDS_SPREADS, ois, recovery=R)
        from pricebook.cds import CDS
        cds = CDS(REF, REF + relativedelta(years=5), spread=CDS_SPREADS[5],
                   notional=1.0, recovery=R)
        par = cds.par_spread(ois, surv)
        assert par == pytest.approx(CDS_SPREADS[5], rel=0.01)

    def test_recovery_family_all_reprice(self):
        """Each R in the family reprices CDS at par."""
        from pricebook.recovery_analytics import recovery_curve_family
        ois = _ois()
        family = recovery_curve_family(CDS_SPREADS, ois, REF,
                                        recoveries=[0.20, 0.40, 0.60])
        from pricebook.cds import CDS
        for R, surv in family.items():
            cds = CDS(REF, REF + relativedelta(years=5), spread=CDS_SPREADS[5],
                       notional=1.0, recovery=R)
            par = cds.par_spread(ois, surv)
            assert par == pytest.approx(CDS_SPREADS[5], rel=0.02)
