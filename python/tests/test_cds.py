"""Tests for CDS."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.cds import protection_leg_pv, premium_leg_pv, CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.day_count import DayCountConvention


REF = date(2024, 1, 15)


def _flat_discount(ref: date, rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
    dfs = [math.exp(-rate * t) for t in tenors]
    return DiscountCurve(ref, dates, dfs)


def _flat_survival(ref: date, hazard: float = 0.02) -> SurvivalCurve:
    tenors = [1, 2, 3, 5, 7, 10]
    dates = [ref + relativedelta(years=t) for t in tenors]
    survs = [math.exp(-hazard * t) for t in tenors]
    return SurvivalCurve(ref, dates, survs)


class TestProtectionLeg:

    def test_pv_positive(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        pv = protection_leg_pv(REF, REF + relativedelta(years=5), dc, sc)
        assert pv > 0

    def test_pv_scales_with_notional(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        end = REF + relativedelta(years=5)
        pv1 = protection_leg_pv(REF, end, dc, sc, notional=1_000_000.0)
        pv2 = protection_leg_pv(REF, end, dc, sc, notional=2_000_000.0)
        assert pv2 == pytest.approx(2 * pv1, rel=1e-6)

    def test_pv_scales_with_lgd(self):
        """Higher LGD (lower recovery) -> higher protection PV."""
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        end = REF + relativedelta(years=5)
        pv_low_recovery = protection_leg_pv(REF, end, dc, sc, recovery=0.2)
        pv_high_recovery = protection_leg_pv(REF, end, dc, sc, recovery=0.6)
        assert pv_low_recovery > pv_high_recovery

    def test_pv_zero_at_full_recovery(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        pv = protection_leg_pv(REF, REF + relativedelta(years=5), dc, sc, recovery=1.0)
        assert pv == pytest.approx(0.0, abs=0.01)

    def test_pv_increases_with_hazard_rate(self):
        """Higher default risk -> higher protection PV."""
        dc = _flat_discount(REF)
        end = REF + relativedelta(years=5)
        sc_low = _flat_survival(REF, hazard=0.01)
        sc_high = _flat_survival(REF, hazard=0.05)
        pv_low = protection_leg_pv(REF, end, dc, sc_low)
        pv_high = protection_leg_pv(REF, end, dc, sc_high)
        assert pv_high > pv_low

    def test_longer_maturity_higher_pv(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        pv_3y = protection_leg_pv(REF, REF + relativedelta(years=3), dc, sc)
        pv_10y = protection_leg_pv(REF, REF + relativedelta(years=10), dc, sc)
        assert pv_10y > pv_3y

    def test_analytical_cross_check(self):
        """
        For flat hazard h and flat rate r, the continuous protection PV is:
            (1-R) * h / (h+r) * (1 - exp(-(h+r)*T))
        Our discretised version should be close.
        """
        r = 0.04
        h = 0.02
        R = 0.4
        T = 5.0
        dc = _flat_discount(REF, rate=r)
        sc = _flat_survival(REF, hazard=h)
        notional = 1_000_000.0

        analytical = notional * (1 - R) * h / (h + r) * (1 - math.exp(-(h + r) * T))
        numerical = protection_leg_pv(
            REF, REF + relativedelta(years=5), dc, sc,
            recovery=R, notional=notional, steps_per_year=12,
        )
        assert numerical == pytest.approx(analytical, rel=0.01)


class TestPremiumLeg:

    def test_pv_positive(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        pv = premium_leg_pv(REF, REF + relativedelta(years=5), 0.01, dc, sc)
        assert pv > 0

    def test_pv_scales_with_spread(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        end = REF + relativedelta(years=5)
        pv1 = premium_leg_pv(REF, end, 0.01, dc, sc)
        pv2 = premium_leg_pv(REF, end, 0.02, dc, sc)
        assert pv2 == pytest.approx(2 * pv1, rel=1e-6)

    def test_pv_scales_with_notional(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        end = REF + relativedelta(years=5)
        pv1 = premium_leg_pv(REF, end, 0.01, dc, sc, notional=1_000_000.0)
        pv2 = premium_leg_pv(REF, end, 0.01, dc, sc, notional=2_000_000.0)
        assert pv2 == pytest.approx(2 * pv1, rel=1e-6)

    def test_pv_decreases_with_higher_hazard(self):
        """Higher hazard -> less likely to survive to pay premiums."""
        dc = _flat_discount(REF)
        end = REF + relativedelta(years=5)
        sc_low = _flat_survival(REF, hazard=0.01)
        sc_high = _flat_survival(REF, hazard=0.10)
        pv_low = premium_leg_pv(REF, end, 0.01, dc, sc_low)
        pv_high = premium_leg_pv(REF, end, 0.01, dc, sc_high)
        assert pv_low > pv_high

    def test_longer_maturity_higher_pv(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        pv_3y = premium_leg_pv(REF, REF + relativedelta(years=3), 0.01, dc, sc)
        pv_10y = premium_leg_pv(REF, REF + relativedelta(years=10), 0.01, dc, sc)
        assert pv_10y > pv_3y

    def test_zero_spread_gives_zero_pv(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        pv = premium_leg_pv(REF, REF + relativedelta(years=5), 0.0, dc, sc)
        assert pv == pytest.approx(0.0, abs=0.01)


class TestCDS:

    def test_pv_zero_at_par_spread(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(dc, sc)
        cds_par = CDS(REF, REF + relativedelta(years=5), spread=par)
        assert cds_par.pv(dc, sc) == pytest.approx(0.0, abs=10.0)

    def test_par_spread_positive(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        assert cds.par_spread(dc, sc) > 0

    def test_par_spread_increases_with_hazard(self):
        dc = _flat_discount(REF)
        sc_low = _flat_survival(REF, hazard=0.01)
        sc_high = _flat_survival(REF, hazard=0.05)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par_low = cds.par_spread(dc, sc_low)
        par_high = cds.par_spread(dc, sc_high)
        assert par_high > par_low

    def test_protection_buyer_positive_when_spread_below_par(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF, hazard=0.03)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(dc, sc)
        cds_cheap = CDS(REF, REF + relativedelta(years=5), spread=par * 0.5)
        assert cds_cheap.pv(dc, sc) > 0

    def test_protection_buyer_negative_when_spread_above_par(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF, hazard=0.03)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(dc, sc)
        cds_expensive = CDS(REF, REF + relativedelta(years=5), spread=par * 2.0)
        assert cds_expensive.pv(dc, sc) < 0

    def test_upfront_at_par_is_zero(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(dc, sc)
        cds_par = CDS(REF, REF + relativedelta(years=5), spread=par)
        assert cds_par.upfront(dc, sc) == pytest.approx(0.0, abs=1e-5)

    def test_pv_scales_with_notional(self):
        dc = _flat_discount(REF)
        sc = _flat_survival(REF)
        end = REF + relativedelta(years=5)
        cds1 = CDS(REF, end, spread=0.01, notional=1_000_000.0)
        cds2 = CDS(REF, end, spread=0.01, notional=2_000_000.0)
        assert cds2.pv(dc, sc) == pytest.approx(2 * cds1.pv(dc, sc), rel=1e-6)


class TestCDSValidation:

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            CDS(REF, REF + relativedelta(years=5), spread=0.01, notional=-1.0)

    def test_recovery_above_one_raises(self):
        with pytest.raises(ValueError):
            CDS(REF, REF + relativedelta(years=5), spread=0.01, recovery=1.5)

    def test_recovery_negative_raises(self):
        with pytest.raises(ValueError):
            CDS(REF, REF + relativedelta(years=5), spread=0.01, recovery=-0.1)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            CDS(REF + relativedelta(years=5), REF, spread=0.01)
