"""Tests for exotic loan features."""

import pytest
from datetime import date

from pricebook.exotic_loan import (
    cpr_to_smm,
    psa_cpr,
    prepay_adjusted_loan,
    prepay_adjusted_wal,
    CovenantLoan,
)
from pricebook.loan import TermLoan
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


class TestPrepayment:
    def test_cpr_to_smm(self):
        smm = cpr_to_smm(0.06)
        # SMM = 1 - (1-0.06)^(1/12) ≈ 0.00514
        assert smm == pytest.approx(0.00514, rel=0.01)

    def test_psa_ramp(self):
        assert psa_cpr(1) == pytest.approx(0.002)
        assert psa_cpr(15) == pytest.approx(0.03)
        assert psa_cpr(30) == pytest.approx(0.06)
        assert psa_cpr(60) == pytest.approx(0.06)  # flat after 30

    def test_psa_speed(self):
        assert psa_cpr(30, psa_speed=1.5) == pytest.approx(0.09)

    def test_prepay_shortens_wal(self):
        """Prepayment → shorter WAL."""
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03, amort_rate=0.0)  # bullet
        wal_no = loan.weighted_average_life(curve)
        wal_prepay = prepay_adjusted_wal(loan, cpr=0.10, projection_curve=curve)
        assert wal_prepay < wal_no

    def test_zero_prepay_unchanged(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03, amort_rate=0.0)
        wal_no = loan.weighted_average_life(curve)
        wal_zero = prepay_adjusted_wal(loan, cpr=0.0, projection_curve=curve)
        assert wal_zero == pytest.approx(wal_no, rel=0.01)

    def test_total_principal_equals_notional(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        flows = prepay_adjusted_loan(loan, cpr=0.05, projection_curve=curve)
        total = sum(p for _, _, p in flows)
        assert total == pytest.approx(loan.notional, rel=0.01)


class TestCovenantLoan:
    def test_expected_maturity_shorter(self):
        """Covenant triggers reduce expected maturity."""
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        cl = CovenantLoan(loan, breach_prob_per_period=0.05)
        em = cl.expected_maturity(curve)
        assert em < 5.0  # shorter than 5Y bullet

    def test_zero_breach_full_maturity(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        cl = CovenantLoan(loan, breach_prob_per_period=0.0)
        em = cl.expected_maturity(curve)
        assert em == pytest.approx(5.0, rel=0.05)

    def test_pv_positive(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        cl = CovenantLoan(loan, breach_prob_per_period=0.02)
        assert cl.pv(curve) > 0

    def test_pv_close_to_base_when_no_breach(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        cl = CovenantLoan(loan, breach_prob_per_period=0.0)
        base_pv = loan.pv(curve)
        assert cl.pv(curve) == pytest.approx(base_pv, rel=0.01)

    def test_higher_breach_changes_pv(self):
        curve = make_flat_curve(REF, 0.05)
        loan = TermLoan(REF, END, spread=0.03)
        cl_low = CovenantLoan(loan, breach_prob_per_period=0.01)
        cl_high = CovenantLoan(loan, breach_prob_per_period=0.10)
        # Different breach probabilities → different PV
        assert cl_low.pv(curve) != pytest.approx(cl_high.pv(curve), rel=0.01)
