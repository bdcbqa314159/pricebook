"""Regression for L2 Wave-2 audit — `InterestRateSwap.cashflow_schedule`
floating row was internally inconsistent.

Pre-fix the floating row reported ``rate`` as the pure forward (excluding
spread) but ``amount`` already INCLUDED the spread.  A downstream consumer
verifying ``amount = rate · year_frac · notional`` got the wrong number
by exactly ``spread · year_frac · notional``.  On a 1mm notional swap
with 50 bp spread and a 3-month period that's ~$1,236 of "missing"
amount per coupon.

Post-fix each row carries explicit ``spread`` and ``notional`` fields so
the relation

    (rate + spread) · year_frac · notional == amount

holds exactly for both legs.  Existing ``rate`` / ``amount`` semantics
are unchanged (no break for callers that only read those).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency
from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection


REF = date(2024, 1, 1)


def _flat_curve(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.5, 1, 2, 5, 10]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(REF, dates, [math.exp(-rate * t) for t in tenors])


class TestCashflowScheduleConsistency:
    def test_row_is_self_consistent_with_spread(self):
        swap = InterestRateSwap(
            start=REF, end=REF + timedelta(days=365 * 2),
            fixed_rate=0.04, direction=SwapDirection.PAYER,
            notional=1_000_000.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
            float_frequency=Frequency.QUARTERLY,
            spread=0.0050,
        )
        rows = swap.cashflow_schedule(_flat_curve())
        for r in rows:
            implied = (r["rate"] + r["spread"]) * r["year_frac"] * r["notional"]
            assert implied == pytest.approx(r["amount"], rel=1e-9), \
                f"{r['leg']} row inconsistent: (rate+spread)·yf·N={implied}, amount={r['amount']}"

    def test_floating_row_carries_explicit_spread(self):
        """Spread must be reported explicitly so the downstream consumer
        doesn't have to know about it via side-channel."""
        swap = InterestRateSwap(
            start=REF, end=REF + timedelta(days=365 * 2),
            fixed_rate=0.04, notional=1_000_000.0, spread=0.0075,
        )
        rows = swap.cashflow_schedule(_flat_curve())
        for r in rows:
            if r["leg"] == "float":
                assert "spread" in r
                assert r["spread"] == pytest.approx(0.0075)

    def test_fixed_row_has_zero_spread(self):
        """Fixed leg has no spread — the field must be present and 0.0
        so consumers can treat the schema uniformly."""
        swap = InterestRateSwap(
            start=REF, end=REF + timedelta(days=365 * 2),
            fixed_rate=0.04, notional=1_000_000.0,
        )
        rows = swap.cashflow_schedule(_flat_curve())
        for r in rows:
            if r["leg"] == "fixed":
                assert "spread" in r
                assert r["spread"] == 0.0

    def test_notional_field_present_on_both_legs(self):
        swap = InterestRateSwap(
            start=REF, end=REF + timedelta(days=365 * 2),
            fixed_rate=0.04, notional=2_500_000.0,
        )
        rows = swap.cashflow_schedule(_flat_curve())
        for r in rows:
            assert "notional" in r
            assert r["notional"] == pytest.approx(2_500_000.0)

    def test_amortising_notional_in_row(self):
        """For amortising swaps, the row's notional must reflect the
        period (not the initial scalar)."""
        swap = InterestRateSwap.amortising(
            start=REF, end=REF + timedelta(days=365 * 5),
            fixed_rate=0.04, initial_notional=1_000_000.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
        )
        rows = swap.cashflow_schedule(_flat_curve())
        # Fixed-leg notionals decrease over time.
        fixed_notionals = [r["notional"] for r in rows if r["leg"] == "fixed"]
        assert len(fixed_notionals) > 1
        for a, b in zip(fixed_notionals, fixed_notionals[1:]):
            assert b < a, f"amortising fixed notional should decrease: {fixed_notionals}"


class TestSchemaCompatibility:
    """Existing tests only check for {payment_date, amount, df, pv} —
    those must remain present and unchanged."""

    def test_all_legacy_fields_still_present(self):
        swap = InterestRateSwap(
            start=REF, end=REF + timedelta(days=365 * 2),
            fixed_rate=0.04, notional=1_000_000.0,
        )
        rows = swap.cashflow_schedule(_flat_curve())
        legacy = {"leg", "accrual_start", "accrual_end", "payment_date",
                  "rate", "year_frac", "amount", "df", "pv"}
        for r in rows:
            for k in legacy:
                assert k in r, f"legacy field {k!r} missing from row"
