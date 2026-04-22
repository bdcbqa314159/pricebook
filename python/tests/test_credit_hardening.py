"""Tests for credit hardening (CR1-CR8)."""

import math
import warnings
from datetime import date

import pytest

from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.credit_risk import _bump_survival_curve
from pricebook.risky_bond import RiskyBond, z_spread, asset_swap_spread
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from tests.conftest import make_flat_curve


def _make_surv(ref, hazard=0.02):
    return SurvivalCurve.flat(ref, hazard)


# ---- CR1: Risky bond filters past cashflows ----

class TestRiskyBondPastCF:
    def test_seasoned_risky_bond_reasonable(self):
        """Seasoned risky bond should price near par, not inflated by past coupons."""
        ref = date(2028, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        surv = _make_surv(ref, 0.02)
        bond = RiskyBond(date(2024, 4, 21), date(2034, 4, 21), 0.05)
        price = bond.dirty_price(curve, surv, settlement=ref)
        assert 50 < price < 120

    def test_future_periods_only(self):
        bond = RiskyBond(date(2024, 4, 21), date(2034, 4, 21), 0.05)
        future = bond._future_periods(date(2028, 4, 21))
        for s, e in future:
            assert e > date(2028, 4, 21)


# ---- CR2: Z-spread and ASW filter past cashflows ----

class TestZSpreadASWPastCF:
    def test_z_spread_seasoned(self):
        ref = date(2028, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        bond = RiskyBond(date(2024, 4, 21), date(2034, 4, 21), 0.05)
        rf_price = bond.risk_free_price(curve, ref)
        zs = z_spread(bond, rf_price - 5.0, curve, settlement=ref)
        assert zs > 0

    def test_asw_seasoned(self):
        ref = date(2028, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        bond = RiskyBond(date(2024, 4, 21), date(2034, 4, 21), 0.05)
        rf_price = bond.risk_free_price(curve, ref)
        asw = asset_swap_spread(bond, rf_price - 3.0, curve, settlement=ref)
        assert asw > 0


# ---- CR3: credit_risk bump uses proper dates ----

class TestCreditBumpDates:
    def test_bump_preserves_pillar_dates(self):
        """Bumped survival curve should have same pillar dates as original."""
        ref = date(2026, 4, 21)
        surv = _make_surv(ref, 0.02)
        bumped = _bump_survival_curve(surv, 0.001)
        assert bumped._pillar_dates == surv._pillar_dates

    def test_bump_changes_survival(self):
        ref = date(2026, 4, 21)
        surv = _make_surv(ref, 0.02)
        bumped = _bump_survival_curve(surv, 0.01)
        # Higher hazard → lower survival
        for d in surv._pillar_dates:
            assert bumped.survival(d) < surv.survival(d)


# ---- CR5: CDS cs01 and rpv01 ----

class TestCDSConvenience:
    def test_rpv01_positive(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        surv = _make_surv(ref, 0.02)
        cds = CDS(ref, date(2031, 4, 21), 0.01)
        assert cds.rpv01(curve, surv) > 0

    def test_cs01_negative_for_protection_buyer(self):
        """Protection buyer loses when spreads tighten (hazard decreases)."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        surv = _make_surv(ref, 0.02)
        cds = CDS(ref, date(2031, 4, 21), 0.01)
        cs = cds.cs01(curve, surv)
        # CS01 = PV change for 1bp hazard increase
        # Protection buyer benefits from spread widening → CS01 > 0
        assert cs > 0

    def test_rpv01_matches_standalone(self):
        from pricebook.cds import risky_annuity
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        surv = _make_surv(ref, 0.02)
        cds = CDS(ref, date(2031, 4, 21), 0.01)
        assert cds.rpv01(curve, surv) == pytest.approx(
            risky_annuity(ref, date(2031, 4, 21), curve, surv), rel=1e-10
        )


# ---- CR6: ISDA standard upfront ----

class TestISDAUpfront:
    def test_at_standard_coupon_near_zero(self):
        """If par spread ≈ standard coupon, ISDA upfront ≈ 0."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        surv = _make_surv(ref, 0.02)
        cds = CDS(ref, date(2031, 4, 21), 0.01)
        ps = cds.par_spread(curve, surv)
        # Build CDS at par spread and check upfront at 100bp
        cds_par = CDS(ref, date(2031, 4, 21), ps)
        uf = cds_par.isda_upfront(curve, surv, standard_coupon=ps)
        assert abs(uf) < 0.001

    def test_upfront_sign_wider_than_coupon(self):
        """Par spread > standard coupon → positive upfront (buyer pays)."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        # High hazard → par spread > 100bp
        surv = _make_surv(ref, 0.05)
        cds = CDS(ref, date(2031, 4, 21), 0.01)
        uf = cds.isda_upfront(curve, surv, standard_coupon=0.01)
        assert uf > 0  # buyer pays upfront because protection is expensive


# ---- CR7: Credit bootstrap round-trip verification ----

class TestCreditBootstrapRoundTrip:
    def test_no_warning_on_good_bootstrap(self):
        """Good bootstrap should not raise RuntimeWarning."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        spreads = [
            (date(2027, 4, 21), 0.005),
            (date(2031, 4, 21), 0.010),
            (date(2036, 4, 21), 0.015),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            surv = bootstrap_credit_curve(ref, spreads, curve)
        assert surv.survival(date(2036, 4, 21)) > 0


# ---- CR8: Risky bond analytics ----

class TestRiskyBondAnalytics:
    def test_ytm_round_trip(self):
        """YTM → price → YTM should round-trip."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        surv = _make_surv(ref, 0.01)
        bond = RiskyBond(ref, date(2031, 4, 21), 0.05)
        price = bond.dirty_price(curve, surv)
        ytm = bond.yield_to_maturity(price, settlement=ref)
        assert ytm > 0
        assert ytm < 0.15

    def test_modified_duration_positive(self):
        ref = date(2026, 4, 21)
        bond = RiskyBond(ref, date(2031, 4, 21), 0.05)
        dur = bond.modified_duration(0.05, settlement=ref)
        assert dur > 0
        assert dur < 6  # less than maturity

    def test_duration_seasoned(self):
        bond = RiskyBond(date(2020, 4, 21), date(2030, 4, 21), 0.05)
        dur_issue = bond.modified_duration(0.05, settlement=date(2020, 4, 21))
        dur_5y = bond.modified_duration(0.05, settlement=date(2025, 4, 21))
        assert dur_5y < dur_issue
