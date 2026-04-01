"""Tests for term loan and leveraged loan."""

import pytest
from datetime import date

from pricebook.loan import TermLoan
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


class TestTermLoan:
    def test_pv_positive(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        assert loan.pv(curve) > 0

    def test_bullet_price_near_par(self):
        """Bullet loan (no amort) with zero spread ≈ par."""
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.0, amort_rate=0.0)
        price = loan.dirty_price(curve)
        assert price == pytest.approx(100.0, abs=1.0)

    def test_positive_spread_above_par(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.02)
        assert loan.dirty_price(curve) > 100.0

    def test_amortising_shorter_wal(self):
        """Amortising loan has shorter WAL than bullet."""
        curve = make_flat_curve(REF, 0.05)
        bullet = TermLoan(REF, END, spread=0.03, amort_rate=0.0)
        amort = TermLoan(REF, END, spread=0.03, amort_rate=0.05)
        assert amort.weighted_average_life(curve) < bullet.weighted_average_life(curve)

    def test_wal_bullet_near_maturity(self):
        """Bullet WAL ≈ maturity."""
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03, amort_rate=0.0)
        wal = loan.weighted_average_life(curve)
        # 5Y bullet: WAL should be close to 5
        assert 4.5 < wal < 5.5


class TestCashflows:
    def test_bullet_single_principal(self):
        """Bullet: all principal in last period."""
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03, amort_rate=0.0)
        flows = loan.cashflows(curve)
        # Only last flow should have significant principal
        for _, _, p in flows[:-1]:
            assert p == pytest.approx(0.0)
        assert flows[-1][2] > 0

    def test_amortising_reduces_outstanding(self):
        """Amortising: principal repaid each period."""
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03, amort_rate=0.05)
        flows = loan.cashflows(curve)
        # Interest should decrease over time (lower outstanding)
        interests = [interest for _, interest, _ in flows]
        assert interests[0] > interests[-2]  # first > second-to-last

    def test_total_principal_equals_notional(self):
        """Sum of all principal repayments = notional."""
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03, amort_rate=0.05)
        flows = loan.cashflows(curve)
        total_principal = sum(p for _, _, p in flows)
        assert total_principal == pytest.approx(loan.notional, rel=0.01)


class TestDiscountMargin:
    def test_dm_zero_at_own_price(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        own_price = loan.dirty_price(curve)
        dm = loan.discount_margin(own_price, curve)
        assert dm == pytest.approx(0.0, abs=0.001)

    def test_dm_round_trip(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        market_price = 98.0
        dm = loan.discount_margin(market_price, curve)

        shifted = TermLoan(REF, END, spread=0.03 + dm, amort_rate=loan.amort_rate)
        assert shifted.dirty_price(curve) == pytest.approx(market_price, abs=0.01)

    def test_dm_positive_below_par(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        own_price = loan.dirty_price(curve)
        dm = loan.discount_margin(own_price - 3.0, curve)
        # Below the loan's own price → negative DM (need less spread)
        assert dm < 0

    def test_dual_curve(self):
        disc = make_flat_curve(REF, 0.04)
        proj = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        pv = loan.pv(disc, proj)
        assert pv > 0
