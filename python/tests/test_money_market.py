"""Tests for pricebook.fixed_income.money_market."""

import pytest
from datetime import date

from tests.conftest import make_flat_curve
from pricebook.fixed_income.money_market import (
    BankersAcceptance,
    CertificateOfDeposit,
    CommercialPaper,
    RepoRate,
)

# Common dates
SETTLE = date(2025, 1, 2)
MAT_90 = date(2025, 4, 2)   # ~90 days
MAT_180 = date(2025, 7, 2)  # ~180 days


# ---------------------------------------------------------------------------
# CertificateOfDeposit
# ---------------------------------------------------------------------------

def test_cd_maturity_cashflow_formula():
    """Cashflow = face × (1 + rate × yf) as per ACT/360 convention."""
    cd = CertificateOfDeposit(settlement=SETTLE, maturity=MAT_90,
                               face_value=1_000_000.0, coupon_rate=0.05)
    days = (MAT_90 - SETTLE).days
    tau = days / 360.0
    expected = 1_000_000.0 * (1.0 + 0.05 * tau)
    assert cd.maturity_cashflow == pytest.approx(expected, rel=1e-10)


def test_cd_maturity_cashflow_gt_face():
    """Maturity cashflow exceeds face value for positive coupon."""
    cd = CertificateOfDeposit(settlement=SETTLE, maturity=MAT_90,
                               face_value=100.0, coupon_rate=0.05)
    assert cd.maturity_cashflow > 100.0


def test_cd_from_yield_roundtrip():
    """from_yield builds a CD; yield_to_maturity(face) recovers the yield."""
    ytm = 0.045
    cd = CertificateOfDeposit.from_yield(SETTLE, MAT_90, ytm=ytm, face_value=1_000_000.0)
    # The CD at par: price = face, so ytm should roundtrip
    recovered_ytm = cd.yield_to_maturity(cd.face_value)
    assert recovered_ytm == pytest.approx(ytm, rel=1e-4)


def test_cd_dirty_price_close_to_face_at_par():
    """Pricing a par CD with its own rate gives price close to face value."""
    rate = 0.05
    cd = CertificateOfDeposit.from_yield(SETTLE, MAT_90, ytm=rate, face_value=100.0)
    # Discount curve at the same rate implies PV ≈ 100 for a par CD
    curve = make_flat_curve(SETTLE, rate)
    price = cd.dirty_price(curve)
    # Should be near 100 (small rounding due to ACT/360 vs continuous discounting)
    assert 95.0 < price < 105.0


# ---------------------------------------------------------------------------
# CommercialPaper
# ---------------------------------------------------------------------------

def test_cp_price_from_discount_lt_face():
    """CP price (discount basis) is always less than face value."""
    cp = CommercialPaper(settlement=SETTLE, maturity=MAT_90, face_value=100.0)
    price = cp.price_from_discount(discount_rate=0.05)
    assert price < 100.0


def test_cp_discount_rate_roundtrip():
    """Back out discount rate from price should recover original rate."""
    cp = CommercialPaper(settlement=SETTLE, maturity=MAT_90, face_value=100.0)
    d = 0.045
    price = cp.price_from_discount(d)
    assert cp.discount_rate(price) == pytest.approx(d, rel=1e-9)


def test_cp_price_from_yield_lt_face():
    """CP price from add-on yield is always below face."""
    cp = CommercialPaper(settlement=SETTLE, maturity=MAT_90, face_value=1000.0)
    assert cp.price_from_yield(0.05) < 1000.0


def test_cp_credit_spread_positive_when_yield_gt_rf():
    """Credit spread > 0 when corporate discount rate exceeds risk-free."""
    cp = CommercialPaper(settlement=SETTLE, maturity=MAT_90, face_value=100.0)
    corporate_price = cp.price_from_discount(0.06)  # corporate rate
    spread = cp.credit_spread(corporate_price, risk_free_rate=0.05)
    assert spread > 0.0


# ---------------------------------------------------------------------------
# BankersAcceptance
# ---------------------------------------------------------------------------

def test_ba_all_in_cost_gt_discount_rate():
    """All-in cost always exceeds the discount rate when acceptance fee > 0."""
    ba = BankersAcceptance(settlement=SETTLE, maturity=MAT_90, face_value=100.0)
    price = ba.price_from_discount(0.05)
    d_rate = ba.discount_rate(price)
    all_in = ba.all_in_cost(price, acceptance_fee=0.005)
    assert all_in > d_rate


def test_ba_price_from_discount_matches_cp_formula():
    """BA and CP share the same discount price formula."""
    ba = BankersAcceptance(settlement=SETTLE, maturity=MAT_90, face_value=100.0)
    cp = CommercialPaper(settlement=SETTLE, maturity=MAT_90, face_value=100.0)
    assert ba.price_from_discount(0.05) == pytest.approx(cp.price_from_discount(0.05), rel=1e-10)


# ---------------------------------------------------------------------------
# RepoRate
# ---------------------------------------------------------------------------

def test_repo_implied_repo_roundtrip():
    """implied_repo round-trips through purchase/sale prices."""
    purchase = 99.50
    repo_rate = 0.053
    days = 30
    sale = purchase * (1.0 + repo_rate * days / 360.0)
    assert RepoRate.implied_repo(purchase, sale, days) == pytest.approx(repo_rate, rel=1e-9)


def test_repo_haircut_adjusted_rate_gt_repo_rate():
    """Haircut-adjusted effective rate is greater than quoted repo rate."""
    repo = 0.05
    haircut = 0.02
    effective = RepoRate.haircut_adjusted_rate(repo, haircut)
    assert effective > repo


def test_repo_haircut_adjusted_rate_formula():
    """effective = repo / (1 - haircut)."""
    repo = 0.04
    haircut = 0.05
    assert RepoRate.haircut_adjusted_rate(repo, haircut) == pytest.approx(repo / (1.0 - haircut), rel=1e-10)
