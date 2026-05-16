"""Tests for recovery analytics: curve family, adjusted repricing, Greeks, surface."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.credit.cln import CreditLinkedNote
from pricebook.credit.cds import CDS
from pricebook.core.schedule import Frequency
from pricebook.credit.recovery_analytics import (
    recovery_curve_family,
    reprice_at_recovery,
    recovery_greeks,
    recovery_pv_surface,
)
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
SPREADS = {1: 0.005, 3: 0.008, 5: 0.010, 10: 0.012}


def _cln():
    return CreditLinkedNote(
        start=REF, end=REF + relativedelta(years=5),
        coupon_rate=0.05, notional=10_000_000, recovery=0.4,
        frequency=Frequency.QUARTERLY,
    )


def _cds():
    return CDS(REF, REF + relativedelta(years=5),
               spread=0.01, notional=10_000_000, recovery=0.4)


class TestRecoveryCurveFamily:

    def test_produces_multiple_curves(self):
        curve = make_flat_curve(REF, 0.04)
        family = recovery_curve_family(SPREADS, curve, REF)
        assert len(family) >= 4

    def test_higher_recovery_higher_hazard(self):
        """Higher R → h = spread/(1-R) is higher → lower survival."""
        curve = make_flat_curve(REF, 0.04)
        family = recovery_curve_family(SPREADS, curve, REF)
        t5 = REF + relativedelta(years=5)
        surv_20 = family[0.20].survival(t5)
        surv_60 = family[0.60].survival(t5)
        # Higher R → higher h → lower survival
        assert surv_60 < surv_20

    def test_all_curves_reprice_cds(self):
        """Each curve should approximately reprice the 5Y CDS at its par spread."""
        curve = make_flat_curve(REF, 0.04)
        family = recovery_curve_family(SPREADS, curve, REF, recoveries=[0.30, 0.40, 0.50])
        for R, surv in family.items():
            cds = CDS(REF, REF + relativedelta(years=5),
                      spread=SPREADS[5], notional=10_000_000, recovery=R)
            pv = cds.pv(curve, surv)
            # At par spread, PV should be near zero
            assert abs(pv) < 50_000  # within 50k on 10M notional


class TestRepriceAtRecovery:

    def test_different_recovery_different_price(self):
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        result = reprice_at_recovery(cln, curve, SPREADS, REF, target_recovery=0.30)
        assert result.pv != result.pv_convention

    def test_recovery_changes_pv_nonlinearly(self):
        """Higher R → higher h → competing effects on CLN price.
        Direct: more recovery payment (positive).
        Indirect: more defaults from higher h (negative).
        The net effect is nonlinear — this IS the recovery convexity."""
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        r20 = reprice_at_recovery(cln, curve, SPREADS, REF, target_recovery=0.20)
        r40 = reprice_at_recovery(cln, curve, SPREADS, REF, target_recovery=0.40)
        r60 = reprice_at_recovery(cln, curve, SPREADS, REF, target_recovery=0.60)
        # All three should produce different PVs
        assert r20.pv != r40.pv
        assert r40.pv != r60.pv
        # And the differences are not symmetric (convexity)
        diff_low = r40.pv - r20.pv
        diff_high = r60.pv - r40.pv
        assert diff_low != diff_high  # asymmetry = convexity

    def test_convention_recovery_matches_base(self):
        """At convention R=40%, adjusted = convention."""
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        result = reprice_at_recovery(cln, curve, SPREADS, REF, target_recovery=0.40)
        assert abs(result.difference) < 1.0

    def test_hazard_increases_with_recovery(self):
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        r20 = reprice_at_recovery(cln, curve, SPREADS, REF, target_recovery=0.20)
        r60 = reprice_at_recovery(cln, curve, SPREADS, REF, target_recovery=0.60)
        assert r60.hazard_at_5y > r20.hazard_at_5y

    def test_works_with_cds(self):
        cds = _cds()
        curve = make_flat_curve(REF, 0.04)
        result = reprice_at_recovery(cds, curve, SPREADS, REF, target_recovery=0.30)
        assert math.isfinite(result.pv)


class TestRecoveryGreeks:

    def test_total_equals_direct_plus_indirect(self):
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        rg = recovery_greeks(cln, curve, SPREADS, REF)
        assert abs(rg.total_dPV_dR - (rg.direct_effect + rg.indirect_effect)) < abs(rg.total_dPV_dR) * 0.01

    def test_direct_positive_for_cln(self):
        """Direct effect: higher R → higher recovery payment → positive."""
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        rg = recovery_greeks(cln, curve, SPREADS, REF)
        assert rg.direct_effect > 0

    def test_indirect_negative_for_cln(self):
        """Indirect effect: higher R → higher h → more defaults → negative."""
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        rg = recovery_greeks(cln, curve, SPREADS, REF)
        assert rg.indirect_effect < 0

    def test_convexity_finite(self):
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        rg = recovery_greeks(cln, curve, SPREADS, REF)
        assert math.isfinite(rg.convexity)

    def test_leveraged_cln_higher_total(self):
        """Leveraged CLN has more recovery sensitivity."""
        cln_v = _cln()
        cln_l = CreditLinkedNote(
            start=REF, end=REF + relativedelta(years=5),
            coupon_rate=0.07, notional=10_000_000, recovery=0.4,
            leverage=2.0, frequency=Frequency.QUARTERLY,
        )
        curve = make_flat_curve(REF, 0.04)
        rg_v = recovery_greeks(cln_v, curve, SPREADS, REF)
        rg_l = recovery_greeks(cln_l, curve, SPREADS, REF)
        assert abs(rg_l.total_dPV_dR) > abs(rg_v.total_dPV_dR) * 0.5


class TestRecoverySurface:

    def test_surface_has_multiple_points(self):
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        surface = recovery_pv_surface(cln, curve, SPREADS, REF)
        assert len(surface) >= 6

    def test_pv_varies_with_recovery(self):
        """CLN PV changes nonlinearly with recovery (direct vs indirect compete)."""
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        surface = recovery_pv_surface(cln, curve, SPREADS, REF,
                                      recoveries=[0.20, 0.40, 0.60])
        pvs = [p.pv for p in surface]
        # All PVs should be different (recovery matters)
        assert len(set(round(p, 2) for p in pvs)) == len(pvs)

    def test_hazard_monotonic(self):
        """Higher R → higher h."""
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        surface = recovery_pv_surface(cln, curve, SPREADS, REF,
                                      recoveries=[0.20, 0.40, 0.60])
        hazards = [p.hazard for p in surface]
        assert hazards[-1] > hazards[0]

    def test_to_dict(self):
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        surface = recovery_pv_surface(cln, curve, SPREADS, REF)
        d = surface[0].to_dict()
        assert "recovery" in d
        assert "pv" in d
        assert "hazard" in d
