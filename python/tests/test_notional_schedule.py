"""Cross-instrument notional schedule tests.

Verifies the notional schedule refactoring across all instrument types:
- Scalar notional produces identical results to pre-refactor
- Uniform list produces identical results to scalar
- AmortisingSwap (deprecated) matches InterestRateSwap with same schedule
- Variable notional flows correctly through PV, par rate, risk metrics
- api_desk.analyse() exposes schedule for IRS, CDS, CLN
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.fixed_income.swap import InterestRateSwap
from pricebook.cds import CDS
from pricebook.cln import CreditLinkedNote
from pricebook.fixed_income.bond import FixedRateBond
from pricebook.schedule import Frequency
from pricebook.desks.api_desk import analyse
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
END = date(2029, 7, 15)


def _curve():
    return make_flat_curve(REF, 0.04)


def _surv():
    return SurvivalCurve.flat(REF, 0.02)


# ── Identity: uniform list == scalar ──

class TestIdentity:
    """IRS/CDS/CLN with notional=[N,N,...,N] must match notional=N."""

    def test_irs_uniform_list_matches_scalar(self):
        curve = _curve()
        scalar = InterestRateSwap(REF, END, 0.04, notional=1_000_000)
        uniform = InterestRateSwap(REF, END, 0.04, notional=[1_000_000] * 20)
        assert scalar.pv(curve) == pytest.approx(uniform.pv(curve), abs=1e-6)
        assert scalar.par_rate(curve) == pytest.approx(uniform.par_rate(curve), abs=1e-10)
        assert scalar.notional == uniform.notional

    def test_cds_uniform_list_matches_scalar(self):
        curve, surv = _curve(), _surv()
        scalar = CDS(REF, END, 0.01, notional=1_000_000)
        uniform = CDS(REF, END, 0.01, notional=[1_000_000] * 20)
        assert scalar.pv(curve, surv) == pytest.approx(uniform.pv(curve, surv), abs=1e-6)
        assert scalar.par_spread(curve, surv) == pytest.approx(uniform.par_spread(curve, surv), abs=1e-10)

    def test_cln_uniform_list_matches_scalar(self):
        curve, surv = _curve(), _surv()
        scalar = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=1_000_000)
        uniform = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=[1_000_000] * 20)
        assert scalar.dirty_price(curve, surv) == pytest.approx(uniform.dirty_price(curve, surv), abs=1e-6)

    def test_bond_uniform_list_matches_scalar(self):
        curve = _curve()
        scalar = FixedRateBond(REF, END, 0.04, face_value=100.0)
        uniform = FixedRateBond(REF, END, 0.04, face_value=[100.0] * 20)
        assert scalar.dirty_price(curve) == pytest.approx(uniform.dirty_price(curve), abs=1e-6)


# ── Factory classmethod equivalence ──

class TestFactoryEquivalence:
    """Factory classmethods produce valid swaps matching direct construction."""

    def test_amortising_factory_matches_direct(self):
        curve = _curve()
        factory = InterestRateSwap.amortising(REF, END, 0.04, 1_000_000)
        # Direct construction with same schedule
        direct = InterestRateSwap(REF, END, 0.04,
                                  notional=factory.notional_schedule,
                                  fixed_frequency=Frequency.SEMI_ANNUAL,
                                  float_frequency=Frequency.QUARTERLY)
        assert factory.pv(curve) == pytest.approx(direct.pv(curve), abs=1.0)

    def test_accreting_factory_matches_direct(self):
        curve = _curve()
        factory = InterestRateSwap.accreting(REF, END, 0.04, 500_000, 1_000_000)
        direct = InterestRateSwap(REF, END, 0.04,
                                  notional=factory.notional_schedule,
                                  fixed_frequency=Frequency.SEMI_ANNUAL,
                                  float_frequency=Frequency.QUARTERLY)
        assert factory.pv(curve) == pytest.approx(direct.pv(curve), abs=1.0)

    def test_roller_coaster_factory_matches_direct(self):
        curve = _curve()
        schedule = [1e6, 2e6, 1e6, 2e6, 1e6]
        factory = InterestRateSwap.roller_coaster(REF, END, 0.04, schedule)
        direct = InterestRateSwap(REF, END, 0.04,
                                  notional=schedule,
                                  fixed_frequency=Frequency.SEMI_ANNUAL,
                                  float_frequency=Frequency.QUARTERLY)
        assert factory.pv(curve) == pytest.approx(direct.pv(curve), abs=1.0)


# ── Variable notional properties ──

class TestVariableNotional:
    """Verify notional, notional_schedule, average_notional are correct."""

    def test_irs_schedule_stored(self):
        schedule = [50e6, 40e6, 30e6, 20e6, 10e6]
        swap = InterestRateSwap(REF, END, 0.04, notional=schedule)
        assert swap.notional == 50e6  # first period, always float
        assert isinstance(swap.notional, float)
        assert len(swap.notional_schedule) >= 5
        assert swap.notional_schedule[0] == 50e6
        assert swap.average_notional < 50e6

    def test_cds_schedule_stored(self):
        schedule = [10e6, 8e6, 6e6]
        cds = CDS(REF, END, 0.01, notional=schedule)
        assert cds.notional == 10e6
        assert isinstance(cds.notional, float)
        assert len(cds.notional_schedule) > 0
        assert cds.average_notional < 10e6

    def test_cln_schedule_stored(self):
        schedule = [5e6, 4e6, 3e6]
        cln_inst = CreditLinkedNote(REF, END, coupon_rate=0.05, notional=schedule)
        assert cln_inst.notional == 5e6
        assert isinstance(cln_inst.notional, float)
        assert len(cln_inst.notional_schedule) > 0

    def test_irs_amortising_dv01_less_than_bullet(self):
        curve = _curve()
        bullet = InterestRateSwap(REF, END, 0.04, notional=1e6)
        amort = InterestRateSwap(REF, END, 0.04,
                                 notional=[1e6 * (1 - i/10) for i in range(10)])
        bullet_dv01 = abs(bullet.pv(curve.bumped(0.0001)) - bullet.pv(curve))
        amort_dv01 = abs(amort.pv(curve.bumped(0.0001)) - amort.pv(curve))
        assert amort_dv01 < bullet_dv01

    def test_schedule_extension(self):
        """Short schedule extended with last value."""
        swap = InterestRateSwap(REF, END, 0.04, notional=[50e6, 30e6])
        # Should have more than 2 periods, with 30e6 replicated
        assert len(swap.notional_schedule) > 2
        assert all(n == 30e6 for n in swap.notional_schedule[1:])


# ── API desk integration ──

class TestApiDeskSchedule:
    """analyse() exposes notional_schedule for all instruments."""

    def test_irs_list_notional(self):
        result = analyse("irs", curve=_curve(), tenor="5Y", rate=0.04,
                         notional=[50e6, 40e6, 30e6, 20e6, 10e6])
        assert result["type"] == "irs"
        assert "notional_schedule" in result
        assert "average_notional" in result
        assert result["notional"] == 50e6
        assert math.isfinite(result["dv01"])

    def test_irs_profile_amortising(self):
        result = analyse("irs", curve=_curve(), tenor="5Y", rate=0.04,
                         notional=50e6, notional_profile="amortising")
        assert "notional_schedule" in result
        assert result["notional_schedule"][0] > result["notional_schedule"][-1]

    def test_cds_list_notional(self):
        result = analyse("cds", curve=_curve(), tenor="5Y", spread=0.01,
                         hazard=0.02, notional=[10e6, 8e6, 6e6])
        assert "notional_schedule" in result
        assert result["notional"] == pytest.approx(10e6)

    def test_cds_profile_amortising(self):
        result = analyse("cds", curve=_curve(), tenor="5Y", spread=0.01,
                         hazard=0.02, notional=10e6, notional_profile="amortising")
        assert "notional_schedule" in result

    def test_cln_list_notional(self):
        result = analyse("cln", curve=_curve(), tenor="5Y", coupon=0.05,
                         hazard=0.02, notional=[5e6, 4e6, 3e6])
        assert "notional_schedule" in result
        assert result["pv"] != 0

    def test_cln_profile_amortising(self):
        result = analyse("cln", curve=_curve(), tenor="5Y", coupon=0.05,
                         hazard=0.02, notional=5e6, notional_profile="amortising")
        assert "notional_schedule" in result

    def test_scalar_no_schedule_in_output(self):
        """Scalar notional should NOT include notional_schedule in output."""
        result = analyse("irs", curve=_curve(), tenor="5Y", rate=0.04,
                         notional=50e6)
        assert "notional_schedule" not in result


# ── Edge cases ──

class TestEdgeCases:

    def test_single_element_list(self):
        swap = InterestRateSwap(REF, END, 0.04, notional=[1e6])
        assert swap.notional == 1e6
        assert len(swap.notional_schedule) > 1  # extended with last value

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            InterestRateSwap(REF, END, 0.04, notional=[])

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError, match="positive"):
            InterestRateSwap(REF, END, 0.04, notional=[-1e6])

    def test_zero_notional_raises(self):
        with pytest.raises(ValueError, match="positive"):
            InterestRateSwap(REF, END, 0.04, notional=0)
