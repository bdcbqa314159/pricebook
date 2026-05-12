"""Tests for repo curve: term structure, forward rates, special-GC spread."""
from __future__ import annotations
from datetime import date
import pytest
from dateutil.relativedelta import relativedelta
from pricebook.repo_curve import (
    build_repo_curve, forward_repo_rate, special_gc_spread,
    repo_carry_from_curve, RepoCurve,
)

REF = date(2024, 7, 15)

def _gc_curve():
    return build_repo_curve(REF, {
        "ON": 0.0530, "1W": 0.0528, "1M": 0.0525,
        "3M": 0.0520, "6M": 0.0510, "1Y": 0.0490,
    })


class TestRepoCurve:
    def test_build(self):
        c = _gc_curve()
        assert len(c.tenors_days) == 6
        assert c.curve_type == "GC"

    def test_interpolation(self):
        c = _gc_curve()
        r_2w = c.rate_at(14)
        assert 0.05 < r_2w < 0.054

    def test_discount_factor(self):
        c = _gc_curve()
        df = c.discount_factor(REF + relativedelta(months=3))
        assert 0.98 < df < 1.0


class TestForwardRepo:
    def test_forward_positive(self):
        c = _gc_curve()
        fwd = forward_repo_rate(c, REF + relativedelta(months=1), REF + relativedelta(months=3))
        assert fwd.forward_rate > 0

    def test_forward_between_spot_rates(self):
        c = _gc_curve()
        fwd = forward_repo_rate(c, REF + relativedelta(months=1), REF + relativedelta(months=6))
        # Forward should be between 1M and 6M spot rates (inverted curve: lower forward)
        assert fwd.forward_rate < c.rate_at(30) + 0.005

    def test_invalid_dates_raises(self):
        c = _gc_curve()
        with pytest.raises(ValueError):
            forward_repo_rate(c, REF + relativedelta(months=6), REF + relativedelta(months=1))


class TestSpecialGCSpread:
    def test_on_special(self):
        gc = _gc_curve()
        r = special_gc_spread("UST 10Y", 0.0480, gc, tenor_days=1)
        assert r.spread_bps > 25
        assert r.is_special

    def test_not_special(self):
        gc = _gc_curve()
        r = special_gc_spread("UST 2Y", 0.0528, gc, tenor_days=1)
        assert not r.is_special


class TestRepoCarry:
    def test_positive_carry(self):
        c = _gc_curve()
        # Bond yielding 4.25% financed at ~5.2% → negative carry (inverted curve)
        r = repo_carry_from_curve(100.0, 0.0425, c, hold_days=30)
        assert r.net_carry < 0  # negative carry in inverted environment

    def test_carry_components(self):
        c = _gc_curve()
        r = repo_carry_from_curve(100.0, 0.0425, c, hold_days=30)
        assert r.coupon_income > 0
        assert r.repo_cost > 0
        assert abs(r.net_carry - (r.coupon_income - r.repo_cost)) < 1e-10
