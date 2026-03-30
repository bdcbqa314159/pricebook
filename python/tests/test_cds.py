"""Tests for CDS."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.cds import protection_leg_pv, premium_leg_pv, CDS, bootstrap_credit_curve
from pricebook.day_count import DayCountConvention
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)


class TestProtectionLeg:

    def test_pv_positive(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        pv = protection_leg_pv(REF, REF + relativedelta(years=5), dc, sc)
        assert pv > 0

    def test_pv_scales_with_notional(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        end = REF + relativedelta(years=5)
        pv1 = protection_leg_pv(REF, end, dc, sc, notional=1_000_000.0)
        pv2 = protection_leg_pv(REF, end, dc, sc, notional=2_000_000.0)
        assert pv2 == pytest.approx(2 * pv1, rel=1e-6)

    def test_pv_scales_with_lgd(self):
        """Higher LGD (lower recovery) -> higher protection PV."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        end = REF + relativedelta(years=5)
        pv_low_recovery = protection_leg_pv(REF, end, dc, sc, recovery=0.2)
        pv_high_recovery = protection_leg_pv(REF, end, dc, sc, recovery=0.6)
        assert pv_low_recovery > pv_high_recovery

    def test_pv_zero_at_full_recovery(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        pv = protection_leg_pv(REF, REF + relativedelta(years=5), dc, sc, recovery=1.0)
        assert pv == pytest.approx(0.0, abs=0.01)

    def test_pv_increases_with_hazard_rate(self):
        """Higher default risk -> higher protection PV."""
        dc = make_flat_curve(REF, 0.04)
        end = REF + relativedelta(years=5)
        sc_low = make_flat_survival(REF, hazard=0.01)
        sc_high = make_flat_survival(REF, hazard=0.05)
        pv_low = protection_leg_pv(REF, end, dc, sc_low)
        pv_high = protection_leg_pv(REF, end, dc, sc_high)
        assert pv_high > pv_low

    def test_longer_maturity_higher_pv(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
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
        dc = make_flat_curve(REF, rate=r)
        sc = make_flat_survival(REF, hazard=h)
        notional = 1_000_000.0

        analytical = notional * (1 - R) * h / (h + r) * (1 - math.exp(-(h + r) * T))
        numerical = protection_leg_pv(
            REF, REF + relativedelta(years=5), dc, sc,
            recovery=R, notional=notional, steps_per_year=12,
        )
        assert numerical == pytest.approx(analytical, rel=0.01)


class TestPremiumLeg:

    def test_pv_positive(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        pv = premium_leg_pv(REF, REF + relativedelta(years=5), 0.01, dc, sc)
        assert pv > 0

    def test_pv_scales_with_spread(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        end = REF + relativedelta(years=5)
        pv1 = premium_leg_pv(REF, end, 0.01, dc, sc)
        pv2 = premium_leg_pv(REF, end, 0.02, dc, sc)
        assert pv2 == pytest.approx(2 * pv1, rel=1e-6)

    def test_pv_scales_with_notional(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        end = REF + relativedelta(years=5)
        pv1 = premium_leg_pv(REF, end, 0.01, dc, sc, notional=1_000_000.0)
        pv2 = premium_leg_pv(REF, end, 0.01, dc, sc, notional=2_000_000.0)
        assert pv2 == pytest.approx(2 * pv1, rel=1e-6)

    def test_pv_decreases_with_higher_hazard(self):
        """Higher hazard -> less likely to survive to pay premiums."""
        dc = make_flat_curve(REF, 0.04)
        end = REF + relativedelta(years=5)
        sc_low = make_flat_survival(REF, hazard=0.01)
        sc_high = make_flat_survival(REF, hazard=0.10)
        pv_low = premium_leg_pv(REF, end, 0.01, dc, sc_low)
        pv_high = premium_leg_pv(REF, end, 0.01, dc, sc_high)
        assert pv_low > pv_high

    def test_longer_maturity_higher_pv(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        pv_3y = premium_leg_pv(REF, REF + relativedelta(years=3), 0.01, dc, sc)
        pv_10y = premium_leg_pv(REF, REF + relativedelta(years=10), 0.01, dc, sc)
        assert pv_10y > pv_3y

    def test_zero_spread_gives_zero_pv(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        pv = premium_leg_pv(REF, REF + relativedelta(years=5), 0.0, dc, sc)
        assert pv == pytest.approx(0.0, abs=0.01)


class TestCDS:

    def test_pv_zero_at_par_spread(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(dc, sc)
        cds_par = CDS(REF, REF + relativedelta(years=5), spread=par)
        assert cds_par.pv(dc, sc) == pytest.approx(0.0, abs=10.0)

    def test_par_spread_positive(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        assert cds.par_spread(dc, sc) > 0

    def test_par_spread_increases_with_hazard(self):
        dc = make_flat_curve(REF, 0.04)
        sc_low = make_flat_survival(REF, hazard=0.01)
        sc_high = make_flat_survival(REF, hazard=0.05)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par_low = cds.par_spread(dc, sc_low)
        par_high = cds.par_spread(dc, sc_high)
        assert par_high > par_low

    def test_protection_buyer_positive_when_spread_below_par(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, hazard=0.03)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(dc, sc)
        cds_cheap = CDS(REF, REF + relativedelta(years=5), spread=par * 0.5)
        assert cds_cheap.pv(dc, sc) > 0

    def test_protection_buyer_negative_when_spread_above_par(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, hazard=0.03)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(dc, sc)
        cds_expensive = CDS(REF, REF + relativedelta(years=5), spread=par * 2.0)
        assert cds_expensive.pv(dc, sc) < 0

    def test_upfront_at_par_is_zero(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0)
        par = cds.par_spread(dc, sc)
        cds_par = CDS(REF, REF + relativedelta(years=5), spread=par)
        assert cds_par.upfront(dc, sc) == pytest.approx(0.0, abs=1e-5)

    def test_pv_scales_with_notional(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
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


# Market data for bootstrap tests
CDS_SPREADS = [
    (REF + relativedelta(years=1), 0.0060),   # 60bp
    (REF + relativedelta(years=3), 0.0085),   # 85bp
    (REF + relativedelta(years=5), 0.0110),   # 110bp
    (REF + relativedelta(years=7), 0.0125),   # 125bp
    (REF + relativedelta(years=10), 0.0140),  # 140bp
]


class TestCreditCurveBootstrap:

    def _build(self):
        dc = make_flat_curve(REF, rate=0.04)
        return dc, bootstrap_credit_curve(REF, CDS_SPREADS, dc)

    def test_round_trip_reprices(self):
        """Bootstrapped credit curve reprices all input CDS at par."""
        dc, sc = self._build()
        for mat, spread in CDS_SPREADS:
            cds = CDS(REF, mat, spread=spread)
            pv = cds.pv(dc, sc)
            assert abs(pv) < 50.0, \
                f"CDS {mat} not at par: PV={pv:.2f}"

    def test_par_spreads_recovered(self):
        dc, sc = self._build()
        for mat, spread in CDS_SPREADS:
            cds = CDS(REF, mat, spread=0.0)
            recovered = cds.par_spread(dc, sc)
            assert recovered == pytest.approx(spread, rel=0.05), \
                f"CDS {mat}: input={spread:.4f}, recovered={recovered:.4f}"

    def test_survival_probs_decreasing(self):
        dc, sc = self._build()
        survs = [sc.survival(d) for d, _ in CDS_SPREADS]
        for i in range(1, len(survs)):
            assert survs[i] < survs[i - 1]

    def test_survival_probs_in_range(self):
        dc, sc = self._build()
        for d, _ in CDS_SPREADS:
            s = sc.survival(d)
            assert 0 < s < 1

    def test_hazard_rates_positive(self):
        dc, sc = self._build()
        for d, _ in CDS_SPREADS:
            assert sc.hazard_rate(d) > 0

    def test_unsorted_raises(self):
        dc = make_flat_curve(REF, 0.04)
        bad = [(REF + relativedelta(years=5), 0.01), (REF + relativedelta(years=1), 0.005)]
        with pytest.raises(ValueError):
            bootstrap_credit_curve(REF, bad, dc)
