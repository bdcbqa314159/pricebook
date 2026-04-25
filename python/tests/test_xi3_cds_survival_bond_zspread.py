"""XI3: CDS → Survival Curve → Risky Bond → Z-spread integration chain.

CDS spreads → bootstrap survival → price risky bond → extract Z-spread →
verify Z-spread reprices the bond. Verify risky < risk-free.

Bug hotspots:
- day count mismatch between CDS premium leg (ACT_360) and bond (THIRTY_360)
- recovery rate consistency across CDS bootstrap and risky bond pricing
- z-spread must exactly reprice the bond (round-trip)
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.bootstrap import bootstrap
from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.risky_bond import RiskyBond, z_spread
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve


# ---- Helpers ----

REF = date(2026, 4, 25)


def _risk_free_curve(ref: date) -> DiscountCurve:
    deposits = [
        (ref + timedelta(days=30), 0.040),
        (ref + timedelta(days=91), 0.040),
        (ref + timedelta(days=182), 0.039),
    ]
    swaps = [
        (ref + timedelta(days=365), 0.038),
        (ref + timedelta(days=730), 0.037),
        (ref + timedelta(days=1095), 0.036),
        (ref + timedelta(days=1825), 0.035),
        (ref + timedelta(days=3650), 0.034),
    ]
    return bootstrap(ref, deposits, swaps)


def _cds_spreads(ref: date) -> list[tuple[date, float]]:
    """CDS par spreads in bps (converted to decimal)."""
    return [
        (ref + timedelta(days=365), 0.0050),     # 1Y 50bp
        (ref + timedelta(days=730), 0.0075),     # 2Y 75bp
        (ref + timedelta(days=1095), 0.0100),    # 3Y 100bp
        (ref + timedelta(days=1825), 0.0125),    # 5Y 125bp
        (ref + timedelta(days=3650), 0.0150),    # 10Y 150bp
    ]


RECOVERY = 0.4


# ---- R1: Chain test — CDS → survival → risky bond → z-spread ----

class TestXI3R1Chain:
    """End-to-end: bootstrap survival → risky bond → z-spread round-trip."""

    def test_bootstrap_survival_curve(self):
        """Bootstrap survival curve from CDS spreads."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        assert surv.survival(REF) == pytest.approx(1.0)
        # Survival probabilities should decrease over time
        s1 = surv.survival(REF + timedelta(days=365))
        s5 = surv.survival(REF + timedelta(days=1825))
        s10 = surv.survival(REF + timedelta(days=3650))
        assert 1.0 > s1 > s5 > s10 > 0

    def test_cds_par_spread_round_trip(self):
        """CDS priced at par spread should have PV ≈ 0."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        for mat, spread in spreads:
            cds = CDS(REF, mat, spread=spread, recovery=RECOVERY)
            pv = cds.pv(disc, surv)
            assert pv == pytest.approx(0.0, abs=100.0), (
                f"CDS at par spread should have PV≈0, got {pv:.2f} for mat={mat}"
            )

    def test_risky_price_less_than_risk_free(self):
        """Risky bond price must be less than risk-free price."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        bond = RiskyBond(REF, REF + timedelta(days=1825),
                         coupon_rate=0.04, recovery=RECOVERY)
        risky = bond.dirty_price(disc, surv)
        riskfree = bond.risk_free_price(disc)
        assert risky < riskfree

    def test_z_spread_round_trip(self):
        """z-spread extracted from risky price must reprice the bond."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        bond = RiskyBond(REF, REF + timedelta(days=1825),
                         coupon_rate=0.04, recovery=RECOVERY)
        risky_price = bond.dirty_price(disc, surv)

        zs = z_spread(bond, risky_price, disc)

        # Reprice with z-spread: bump the curve by z-spread
        bumped = disc.bumped(zs)
        repriced = bond.risk_free_price(bumped)
        assert repriced == pytest.approx(risky_price, rel=0.01)

    def test_z_spread_positive(self):
        """Z-spread should be positive for a risky bond."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        bond = RiskyBond(REF, REF + timedelta(days=1825),
                         coupon_rate=0.04, recovery=RECOVERY)
        risky_price = bond.dirty_price(disc, surv)
        zs = z_spread(bond, risky_price, disc)
        assert zs > 0


# ---- R2: Handoff audits ----

class TestXI3R2Handoffs:
    """Day count, recovery, and curve handoff consistency."""

    def test_recovery_consistency(self):
        """Same recovery in CDS bootstrap and risky bond pricing."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)

        for rec in [0.2, 0.4, 0.6]:
            surv = bootstrap_credit_curve(REF, spreads, disc, recovery=rec)
            bond = RiskyBond(REF, REF + timedelta(days=1825),
                             coupon_rate=0.04, recovery=rec)
            price = bond.dirty_price(disc, surv)
            assert price > 0
            # Higher recovery → higher risky price (less loss on default)
            if rec == 0.2:
                low_rec_price = price
            elif rec == 0.6:
                assert price > low_rec_price

    def test_wider_spread_lower_survival(self):
        """Wider CDS spreads → lower survival probabilities."""
        disc = _risk_free_curve(REF)
        t5y = REF + timedelta(days=1825)

        narrow = [(t5y, 0.0050)]
        wide = [(t5y, 0.0200)]
        surv_n = bootstrap_credit_curve(REF, narrow, disc, recovery=RECOVERY)
        surv_w = bootstrap_credit_curve(REF, wide, disc, recovery=RECOVERY)

        assert surv_w.survival(t5y) < surv_n.survival(t5y)

    def test_wider_spread_lower_risky_price(self):
        """Wider CDS → lower survival → lower risky bond price."""
        disc = _risk_free_curve(REF)

        spreads_tight = _cds_spreads(REF)
        spreads_wide = [(d, s * 2) for d, s in _cds_spreads(REF)]

        surv_t = bootstrap_credit_curve(REF, spreads_tight, disc, recovery=RECOVERY)
        surv_w = bootstrap_credit_curve(REF, spreads_wide, disc, recovery=RECOVERY)

        bond = RiskyBond(REF, REF + timedelta(days=1825),
                         coupon_rate=0.04, recovery=RECOVERY)
        price_t = bond.dirty_price(disc, surv_t)
        price_w = bond.dirty_price(disc, surv_w)
        assert price_w < price_t

    def test_zero_spread_survival_one(self):
        """Zero CDS spread → survival ≈ 1 → risky ≈ risk-free."""
        disc = _risk_free_curve(REF)
        t5y = REF + timedelta(days=1825)
        spreads_zero = [(t5y, 1e-6)]  # near-zero spread
        surv = bootstrap_credit_curve(REF, spreads_zero, disc, recovery=RECOVERY)

        assert surv.survival(t5y) > 0.999

        bond = RiskyBond(REF, t5y, coupon_rate=0.04, recovery=RECOVERY)
        risky = bond.dirty_price(disc, surv)
        riskfree = bond.risk_free_price(disc)
        assert risky == pytest.approx(riskfree, rel=0.01)

    def test_hazard_rate_positive(self):
        """Hazard rates should be positive for positive CDS spreads."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        for d_offset in [365, 730, 1825]:
            h = surv.hazard_rate(REF + timedelta(days=d_offset))
            assert h >= 0


# ---- R3/R4: Edge cases and cross-checks ----

class TestXI3R3EdgeCases:
    """Edge cases and additional cross-checks."""

    def test_short_dated_bond(self):
        """1Y risky bond: z-spread round-trip."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        bond = RiskyBond(REF, REF + timedelta(days=365),
                         coupon_rate=0.04, recovery=RECOVERY)
        price = bond.dirty_price(disc, surv)
        zs = z_spread(bond, price, disc)
        assert zs > 0
        repriced = bond.risk_free_price(disc.bumped(zs))
        assert repriced == pytest.approx(price, rel=0.01)

    def test_high_coupon_bond(self):
        """High coupon bond still works through the chain."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        bond = RiskyBond(REF, REF + timedelta(days=1825),
                         coupon_rate=0.10, recovery=RECOVERY)
        price = bond.dirty_price(disc, surv)
        riskfree = bond.risk_free_price(disc)
        assert price < riskfree
        assert price > 0

    def test_zero_coupon_risky_bond(self):
        """Zero coupon risky bond: price = notional × df × survival + recovery piece."""
        disc = _risk_free_curve(REF)
        spreads = _cds_spreads(REF)
        surv = bootstrap_credit_curve(REF, spreads, disc, recovery=RECOVERY)

        bond = RiskyBond(REF, REF + timedelta(days=1825),
                         coupon_rate=0.0, recovery=RECOVERY)
        price = bond.dirty_price(disc, surv)
        riskfree = bond.risk_free_price(disc)
        assert 0 < price < riskfree

    def test_z_spread_increases_with_credit_risk(self):
        """Higher CDS spreads → higher z-spread."""
        disc = _risk_free_curve(REF)
        bond = RiskyBond(REF, REF + timedelta(days=1825),
                         coupon_rate=0.04, recovery=RECOVERY)

        spreads_tight = _cds_spreads(REF)
        spreads_wide = [(d, s * 3) for d, s in _cds_spreads(REF)]

        surv_t = bootstrap_credit_curve(REF, spreads_tight, disc, recovery=RECOVERY)
        surv_w = bootstrap_credit_curve(REF, spreads_wide, disc, recovery=RECOVERY)

        price_t = bond.dirty_price(disc, surv_t)
        price_w = bond.dirty_price(disc, surv_w)

        zs_t = z_spread(bond, price_t, disc)
        zs_w = z_spread(bond, price_w, disc)
        assert zs_w > zs_t

    def test_flat_curve_z_spread(self):
        """Z-spread on a flat curve should still round-trip."""
        flat = DiscountCurve.flat(REF, 0.04)
        spreads = [(REF + timedelta(days=1825), 0.0100)]
        surv = bootstrap_credit_curve(REF, spreads, flat, recovery=RECOVERY)

        bond = RiskyBond(REF, REF + timedelta(days=1825),
                         coupon_rate=0.04, recovery=RECOVERY)
        price = bond.dirty_price(flat, surv)
        zs = z_spread(bond, price, flat)
        repriced = bond.risk_free_price(flat.bumped(zs))
        assert repriced == pytest.approx(price, rel=0.01)
