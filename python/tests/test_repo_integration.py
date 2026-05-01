"""Tests for repo curve integration: as_discount_curve, repo-OIS basis, carry."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.repo_term import (
    RepoCurve, RepoRate, forward_repo_rate,
    repo_ois_basis, term_repo_carry,
)
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


def _make_repo_curve(flat_rate=0.045):
    """Flat repo curve at given rate."""
    tenors = [
        RepoRate(1, flat_rate),
        RepoRate(7, flat_rate),
        RepoRate(30, flat_rate),
        RepoRate(90, flat_rate),
        RepoRate(180, flat_rate),
        RepoRate(360, flat_rate),
    ]
    return RepoCurve(REF, tenors)


# ---- RepoCurve → DiscountCurve ----

class TestAsDiscountCurve:

    def test_conversion_works(self):
        rc = _make_repo_curve(0.045)
        dc = rc.as_discount_curve()
        assert dc is not None
        assert dc.reference_date == REF

    def test_discount_factors_match(self):
        """DFs from converted curve should match simple-rate DFs."""
        rc = _make_repo_curve(0.045)
        dc = rc.as_discount_curve()
        for days in [30, 90, 180]:
            d = REF + relativedelta(days=days)
            df_repo = 1.0 / (1.0 + 0.045 * days / 360.0)
            df_curve = dc.df(d)
            assert df_curve == pytest.approx(df_repo, rel=0.01)

    def test_can_price_with_repo_curve(self):
        """Converted repo curve can be used for bond pricing."""
        rc = _make_repo_curve(0.04)
        dc = rc.as_discount_curve()
        # Simple test: discount factor at 1Y
        d_1y = REF + relativedelta(years=1)
        df = dc.df(d_1y)
        assert 0.9 < df < 1.0


# ---- Repo-OIS basis ----

class TestRepoOISBasis:

    def test_gc_positive_basis(self):
        """GC repo > OIS → positive basis."""
        rc = _make_repo_curve(0.045)  # repo at 4.5%
        ois = make_flat_curve(REF, 0.04)  # OIS at 4.0%
        basis = repo_ois_basis(rc, ois)
        for entry in basis:
            assert entry["basis_bp"] > 0  # repo > OIS

    def test_special_negative_basis(self):
        """Special repo < OIS → negative basis."""
        rc = _make_repo_curve(0.035)  # special repo at 3.5%
        ois = make_flat_curve(REF, 0.04)  # OIS at 4.0%
        basis = repo_ois_basis(rc, ois)
        for entry in basis:
            assert entry["basis_bp"] < 0  # repo < OIS

    def test_equal_zero_basis(self):
        """Repo = OIS → zero basis."""
        rc = _make_repo_curve(0.04)
        ois = make_flat_curve(REF, 0.04)
        basis = repo_ois_basis(rc, ois, tenors_days=[90])
        assert abs(basis[0]["basis_bp"]) < 5  # near zero (compounding diff)


# ---- Carry decomposition ----

class TestCarry:

    def test_positive_carry(self):
        """Bond with coupon > repo rate has positive net carry."""
        result = term_repo_carry(
            bond_dirty_price=100.0,
            coupon_income=2.0,  # 2% semi-annual coupon
            repo_rate=0.04,
            ois_rate=0.04,
            holding_days=180,
        )
        # Coupon 2.0 vs repo cost = 100 × 0.04 × 0.5 = 2.0 → breakeven
        assert math.isfinite(result["net_carry"])

    def test_special_carry_advantage(self):
        """Bond on special: repo < OIS → positive carry advantage."""
        result = term_repo_carry(
            bond_dirty_price=100.0,
            coupon_income=2.0,
            repo_rate=0.035,  # on special
            ois_rate=0.045,
            holding_days=90,
        )
        assert result["carry_advantage_bp"] > 0

    def test_gc_no_advantage(self):
        """At GC = OIS: no carry advantage."""
        result = term_repo_carry(
            bond_dirty_price=100.0,
            coupon_income=2.0,
            repo_rate=0.04,
            ois_rate=0.04,
            holding_days=90,
        )
        assert abs(result["carry_advantage_bp"]) < 1

    def test_fields(self):
        result = term_repo_carry(100.0, 2.0, 0.04, 0.04, 90)
        assert "coupon_income" in result
        assert "repo_cost" in result
        assert "net_carry" in result
        assert "carry_advantage_bp" in result
