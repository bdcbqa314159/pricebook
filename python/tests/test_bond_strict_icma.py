"""Regression for A.1 B1 Slice 1 — bond.py is strict-icma-safe.

Pre-fix, `FixedRateBond.accrued_interest` and `_ytm_time_to` called
``year_fraction(...)`` with ACT/ACT ICMA but without the required
``ref_start`` / ``ref_end`` / ``frequency`` arguments. The fallback inside
``_act_act_icma`` silently degraded to ACT/365F — observable as a ~0.1-0.4%
per-period bias on UST / Bund / Gilt / OAT accrued interest and yield-to-
maturity.

Slice 1 routes the accrual computation through the proper ICMA anchors and
makes the intentional fallbacks in ``_ytm_time_to`` use ``ACT_365_FIXED``
explicitly (so flipping the global default to ``strict_icma=True`` in
Slice 3 doesn't break them).

The two regression checks:

1. **strict-icma-safe smoke**: monkey-patch ``year_fraction``'s default
   ``strict_icma`` to True and verify the bond's accrued + ytm round-trip
   doesn't raise.
2. **proper-ICMA value**: at mid-period the new accrued should match the
   exact ICMA formula (rational fraction of the coupon), not the silent
   ACT/365F approximation.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import partial

import pytest

from pricebook.core import day_count as day_count_module
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.schedule import Frequency
from pricebook.fixed_income.bond import FixedRateBond


REF = date(2026, 1, 1)


def _ust5y() -> FixedRateBond:
    """5-year US-Treasury-like bond: ACT/ACT ICMA, semi-annual, 3% coupon."""
    return FixedRateBond(
        issue_date=REF,
        maturity=REF + timedelta(days=365 * 5),
        coupon_rate=0.03,
        frequency=Frequency.SEMI_ANNUAL,
        face_value=100.0,
        day_count=DayCountConvention.ACT_ACT_ICMA,
    )


# ── Check 1: strict-icma-safe smoke ────────────────────────────────


def test_bond_strict_icma_smoke(monkeypatch):
    """With strict_icma flipped to True, no bond method should silently
    fall back inside year_fraction (i.e., none should raise ValueError)."""
    bond = _ust5y()

    # Monkeypatch year_fraction so its default `strict_icma=True`.
    strict_yf = partial(year_fraction, strict_icma=True)
    monkeypatch.setattr(
        day_count_module, "year_fraction", strict_yf, raising=True,
    )
    # Patch the import inside bond.py too, since it captured the original
    # reference at import time.
    import pricebook.fixed_income.bond as bond_module
    monkeypatch.setattr(bond_module, "year_fraction", strict_yf, raising=True)

    # Mid-period settlement (90 days after issue).
    settle = REF + timedelta(days=90)

    # These three should NOT raise — i.e., every internal year_fraction call
    # has proper refs (or uses an explicit non-ICMA day-count fallback).
    accrued = bond.accrued_interest(settle)
    assert accrued > 0  # mid-period, before ex-div

    t = bond._ytm_time_to(settle, bond.maturity)
    assert t > 0

    # End-to-end: yield_to_maturity exercises _ytm_time_to which goes
    # through the strict path. Should not raise.
    ytm = bond.yield_to_maturity(market_price=100.0, settlement=settle)
    assert 0 < ytm < 1  # sane yield


# ── Check 2: proper-ICMA accrued value ─────────────────────────────


def test_bond_accrued_uses_proper_icma_not_act365f():
    """At a known mid-period date, the ICMA accrued is exactly
    ``coupon × (days_into_period / period_days) / coupons_per_year``.
    The pre-fix silent fallback gave ``coupon × days_into_period / 365``,
    which differs by a small but observable bias for non-365-day periods.
    """
    bond = _ust5y()

    # First coupon period: 2026-01-01 → 2026-07-01 (or thereabouts;
    # depends on schedule generation).  Pick a settlement 60 days in.
    first_cf = bond.coupon_leg.cashflows[0]
    period_start = first_cf.accrual_start
    period_end = first_cf.accrual_end
    period_days = (period_end - period_start).days
    settle = period_start + timedelta(days=60)
    days_into = (settle - period_start).days

    accrued_per_100 = bond.accrued_interest(settle)

    # Proper ICMA: coupon × (days_into / period_days) × 100 / coupons_per_year
    cpy = 2  # semi-annual
    expected_icma = bond.coupon_rate * (days_into / period_days) * 100.0 / cpy

    # ACT/365F fallback (the buggy pre-fix value):
    bad_act365f = bond.coupon_rate * (days_into / 365.0) * 100.0

    assert accrued_per_100 == pytest.approx(expected_icma, rel=1e-12)
    # And the two values are observably different (bias is real):
    assert abs(accrued_per_100 - bad_act365f) > 1e-6
