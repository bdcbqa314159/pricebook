"""Tests for unitranche, FOLO, DDTL, and direct lending economics."""

import math
import pytest
from datetime import date

from pricebook.unitranche import (
    FOLO, FOLORecoveryResult, CallProtectionSchedule, DirectLendingYield,
    Unitranche, DelayedDrawTermLoan,
    folo_recovery_split, unitranche_blended_spread,
    direct_lending_economics, hold_to_maturity_yield,
)
from pricebook.discount_curve import DiscountCurve


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(date(2024, 1, 1), 0.05)


# ═══════════════════════════════════════════════════════════════
# FOLO
# ═══════════════════════════════════════════════════════════════

class TestFOLO:
    def test_basic(self):
        folo = FOLO(60_000_000, 40_000_000, 0.04, 0.075)
        assert folo.total_notional == 100_000_000
        assert folo.first_out_pct == 0.60

    def test_recovery_full(self):
        folo = FOLO(60, 40, 0.04, 0.07)
        r = folo_recovery_split(1.0, folo)
        assert r.first_out_recovery_pct == 1.0
        assert r.last_out_recovery_pct == 1.0
        assert r.first_out_loss == 0.0
        assert r.last_out_loss == 0.0

    def test_recovery_partial_fo_full(self):
        """70% recovery: FO (60%) fully recovered, LO gets remaining 10/40."""
        folo = FOLO(60, 40, 0.04, 0.07)
        r = folo_recovery_split(0.70, folo)
        assert r.first_out_recovery_pct == 1.0
        assert abs(r.last_out_recovery_pct - 10 / 40) < 1e-10
        assert r.first_out_loss == 0.0
        assert abs(r.last_out_loss - 30) < 1e-10

    def test_recovery_zero(self):
        folo = FOLO(60, 40, 0.04, 0.07)
        r = folo_recovery_split(0.0, folo)
        assert r.first_out_recovery_pct == 0.0
        assert r.last_out_recovery_pct == 0.0
        assert r.first_out_loss == 60
        assert r.last_out_loss == 40

    def test_recovery_half(self):
        """50% recovery on 100 total: FO gets 50 (cap at 60), LO gets 0."""
        folo = FOLO(60, 40, 0.04, 0.07)
        r = folo_recovery_split(0.50, folo)
        assert abs(r.first_out_recovery_pct - 50 / 60) < 1e-10
        assert r.last_out_recovery_pct == 0.0

    def test_to_dict(self):
        d = FOLO(60, 40, 0.04, 0.07).to_dict()
        assert "first_out_pct" in d
        assert "total_notional" in d


# ═══════════════════════════════════════════════════════════════
# Call Protection
# ═══════════════════════════════════════════════════════════════

class TestCallProtection:
    def test_nc_period(self):
        cp = CallProtectionSchedule(
            par_dates=[date(2026, 1, 1), date(2027, 1, 1)],
            premiums=[0.02, 0.01, 0.0],
        )
        assert not cp.is_callable(date(2025, 6, 1))
        assert cp.call_price(date(2025, 6, 1)) == float('inf')

    def test_step_down(self):
        cp = CallProtectionSchedule(
            par_dates=[date(2026, 1, 1), date(2027, 1, 1)],
            premiums=[0.02, 0.01, 0.0],
        )
        assert cp.is_callable(date(2026, 6, 1))
        assert cp.call_price(date(2026, 6, 1)) == 102.0

    def test_par_call(self):
        cp = CallProtectionSchedule(
            par_dates=[date(2026, 1, 1), date(2027, 1, 1)],
            premiums=[0.02, 0.01, 0.0],
        )
        assert cp.call_price(date(2028, 1, 1)) == 100.0


# ═══════════════════════════════════════════════════════════════
# Direct Lending Economics
# ═══════════════════════════════════════════════════════════════

class TestDirectLending:
    def test_basic_yield(self):
        dl = direct_lending_economics(spread=0.055, base_rate=0.05, oid=0.02, upfront_fee=0.01, wal=4.0)
        assert abs(dl.coupon_yield - 0.105) < 1e-10
        assert abs(dl.oid_amortisation - 0.005) < 1e-10
        assert abs(dl.upfront_fee_amort - 0.0025) < 1e-10
        assert abs(dl.all_in_yield - 0.1125) < 1e-10

    def test_no_oid(self):
        dl = direct_lending_economics(spread=0.04, base_rate=0.05)
        assert dl.oid_amortisation == 0.0
        assert dl.all_in_yield == 0.09

    def test_to_dict(self):
        d = direct_lending_economics(0.05, 0.04).to_dict()
        assert "all_in_yield" in d


class TestBlendedSpread:
    def test_equal_split(self):
        assert unitranche_blended_spread(0.04, 0.08, 0.50) == 0.06

    def test_full_fo(self):
        assert unitranche_blended_spread(0.04, 0.08, 1.0) == 0.04

    def test_full_lo(self):
        assert unitranche_blended_spread(0.04, 0.08, 0.0) == 0.08


# ═══════════════════════════════════════════════════════════════
# Unitranche
# ═══════════════════════════════════════════════════════════════

class TestUnitranche:
    def test_is_term_loan(self, flat_curve):
        u = Unitranche(date(2024, 1, 1), date(2029, 1, 1), spread=0.055, notional=50_000_000)
        assert isinstance(u, Unitranche)
        # Should price like a TermLoan
        pv = u.pv(flat_curve)
        assert pv > 0

    def test_blended_spread_from_folo(self):
        folo = FOLO(30_000_000, 20_000_000, 0.04, 0.075)
        u = Unitranche(date(2024, 1, 1), date(2029, 1, 1), spread=0.055, notional=50_000_000, folo=folo)
        expected = 0.04 * 0.60 + 0.075 * 0.40
        assert abs(u.blended_spread() - expected) < 1e-10

    def test_oid_proceeds(self):
        u = Unitranche(date(2024, 1, 1), date(2029, 1, 1), notional=50_000_000, oid=0.02)
        assert u.proceeds() == 49_000_000

    def test_folo_recovery(self):
        folo = FOLO(30, 20, 0.04, 0.075)
        u = Unitranche(date(2024, 1, 1), date(2029, 1, 1), notional=50, folo=folo)
        r = u.folo_recovery(0.70)
        assert r.first_out_recovery_pct == 1.0  # 35/30 > 1, capped at 1
        assert r.last_out_recovery_pct > 0

    def test_no_folo_error(self):
        u = Unitranche(date(2024, 1, 1), date(2029, 1, 1), notional=50)
        with pytest.raises(ValueError):
            u.folo_recovery(0.70)

    def test_to_dict(self):
        folo = FOLO(30, 20, 0.04, 0.075)
        d = Unitranche(date(2024, 1, 1), date(2029, 1, 1), notional=50, folo=folo).to_dict()
        assert "folo" in d
        assert "blended_spread" in d


# ═══════════════════════════════════════════════════════════════
# Delayed Draw Term Loan
# ═══════════════════════════════════════════════════════════════

class TestDDTL:
    def test_ticking_fee_before_draw(self, flat_curve):
        ddtl = DelayedDrawTermLoan(
            start=date(2024, 1, 1), end=date(2029, 1, 1),
            spread=0.05, notional=10_000_000,
            draw_date=date(2025, 1, 1), ticking_fee=0.01,
        )
        cfs = ddtl.cashflows(flat_curve)
        # First 4 periods (quarterly in 2024) should be ticking fee only
        for d, interest, principal in cfs[:4]:
            assert principal == 0.0  # no principal before draw
            assert interest > 0      # ticking fee

    def test_coupon_after_draw(self, flat_curve):
        ddtl = DelayedDrawTermLoan(
            start=date(2024, 1, 1), end=date(2029, 1, 1),
            spread=0.05, notional=10_000_000,
            draw_date=date(2025, 1, 1), ticking_fee=0.01,
        )
        cfs = ddtl.cashflows(flat_curve)
        # Later periods should have higher interest (coupon > ticking fee)
        late_interest = cfs[-2][1]  # second to last period
        early_interest = cfs[0][1]  # ticking fee period
        assert late_interest > early_interest

    def test_pv_positive(self, flat_curve):
        ddtl = DelayedDrawTermLoan(
            start=date(2024, 1, 1), end=date(2029, 1, 1),
            notional=10_000_000, draw_date=date(2025, 1, 1),
        )
        assert ddtl.pv(flat_curve) > 0

    def test_to_dict(self):
        d = DelayedDrawTermLoan(
            date(2024, 1, 1), date(2029, 1, 1), draw_date=date(2025, 1, 1)
        ).to_dict()
        assert "draw_date" in d
        assert "ticking_fee" in d


# ═══════════════════════════════════════════════════════════════
# Hold-to-Maturity Yield
# ═══════════════════════════════════════════════════════════════

class TestHTMYield:
    def test_par_price(self, flat_curve):
        """At par price, solving back should give a consistent yield."""
        from pricebook.loan import TermLoan
        loan = TermLoan(date(2024, 1, 1), date(2029, 1, 1), spread=0.03, notional=1_000_000)
        price = loan.dirty_price(flat_curve)
        y = hold_to_maturity_yield(loan, price, flat_curve)
        assert y > 0  # positive yield
        # Verify roundtrip: discounting at y should recover the price
        total = 0.0
        from pricebook.day_count import DayCountConvention, year_fraction
        for d, interest, principal in loan.cashflows(flat_curve):
            t = year_fraction(loan.start, d, DayCountConvention.ACT_365_FIXED)
            df = 1.0 / (1 + y) ** t
            total += df * (interest + principal)
        roundtrip_price = total / loan.notional * 100
        assert abs(roundtrip_price - price) < 0.01

    def test_discount_price_higher_yield(self, flat_curve):
        """At below par, HTM yield > coupon yield."""
        from pricebook.loan import TermLoan
        loan = TermLoan(date(2024, 1, 1), date(2029, 1, 1), spread=0.03, notional=1_000_000)
        y = hold_to_maturity_yield(loan, 95.0, flat_curve)
        assert y > 0.08  # higher yield when buying at discount
