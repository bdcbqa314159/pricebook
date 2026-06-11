"""Characterisation of UST ACT/ACT ICMA mispricing (A.1 B1 Slice 2).

These tests document the bug surfaced by L0 audit A.1 B1 — `FixedRateBond.treasury_note`
declares ACT/ACT ICMA day-count but the cashflow-builder in `FixedLeg` calls
`year_fraction(accrual_start, accrual_end, ACT_ACT_ICMA)` *without* passing
`ref_start`, `ref_end`, `frequency`. ICMA silently falls back to ACT/365F, so:

- A regular 182-day semi-annual coupon period gives `yf = 182/365 ≈ 0.49863`
  instead of the correct ICMA value `yf = 1/2 = 0.50000` exactly.
- A regular 184-day semi-annual coupon period gives `yf = 184/365 ≈ 0.50411`
  instead of `0.50000`.
- Per-coupon error: ±~0.4-0.8% of the coupon amount; ~30-80 bp per 100 face.
- Par yield round-trip lands at 99.9998 (5y) or 99.9995 (30y) instead of exactly 100.

UST is quoted in 32nds (~3.1 bp resolution). The error is **observable in market quotes**.

These tests are marked `xfail(strict=True)` so:
- They MUST fail today (current bug).
- They MUST pass in A.1 B1 Slice 3 (the fix).
- `strict=True` catches the case where the fix is partial / accidentally passes here.
"""

from __future__ import annotations

from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.core.day_count import DayCountConvention
from pricebook.fixed_income.bond import FixedRateBond


# ============================================================
# UST par-coupon characterisation
# ============================================================

class TestParUSTCharacterisation:
    """Per ICMA 251.1: every semi-annual coupon period on a regular UST
    schedule must yield `year_fraction = 0.5` *exactly*, regardless of how
    many actual days the period has (182 or 184). That's the *point* of
    ACT/ACT ICMA — the denominator absorbs period-length variation."""

    @pytest.fixture
    def par_5y_ust(self):
        """5y UST at 4% coupon, issued 2024-02-15, matures 2029-02-15.
        Regular schedule: coupons on Feb 15 and Aug 15 each year.
        At par yield (ytm == coupon), price should be exactly 100.
        """
        return FixedRateBond.treasury_note(
            issue_date=date(2024, 2, 15),
            maturity=date(2029, 2, 15),
            coupon_rate=0.04,
            face_value=100.0,
        )

    def test_bond_is_acta_acta_icma(self, par_5y_ust):
        """Smoke check — confirm we're testing the right convention."""
        assert par_5y_ust.day_count == DayCountConvention.ACT_ACT_ICMA

    @pytest.mark.xfail(
        strict=True,
        reason="A.1 B1 — ICMA silent fallback to ACT/365F (will be fixed in Slice 3)",
    )
    def test_every_regular_coupon_is_exactly_half_year(self, par_5y_ust):
        """Every regular coupon period should have year_frac == 0.5 exactly."""
        for cf in par_5y_ust.coupon_leg.cashflows:
            assert cf.year_frac == pytest.approx(0.5, abs=1e-12), (
                f"Coupon {cf.accrual_start} → {cf.accrual_end}: year_frac="
                f"{cf.year_frac} (expected exactly 0.5 per ICMA 251.1)"
            )

    @pytest.mark.xfail(
        strict=True,
        reason="A.1 B1 — coupon amounts wrong because year_frac is wrong",
    )
    def test_every_coupon_amount_is_exactly_two(self, par_5y_ust):
        """At 4% coupon, face=100, semi-annual: each coupon should pay
        exactly 100 × 0.04 × 0.5 = 2.0000."""
        for cf in par_5y_ust.coupon_leg.cashflows:
            assert cf.amount == pytest.approx(2.0, abs=1e-10), (
                f"Coupon at {cf.accrual_end}: amount={cf.amount} "
                f"(expected exactly 2.0)"
            )

    @pytest.mark.xfail(
        strict=True,
        reason="A.1 B1 — par-yield round-trip drifts due to year_frac error",
    )
    def test_par_yield_round_trip_is_exact_100(self, par_5y_ust):
        """At ytm == coupon_rate, price (off internal YTM machinery) should
        be exactly 100. Today it lands at 99.99xx because of accumulated
        per-coupon year_frac error."""
        # Price the bond at its own coupon rate as YTM (the par case)
        # using the internal YTM-based pricer.
        settle = par_5y_ust.issue_date
        price = par_5y_ust._price_from_ytm(par_5y_ust.coupon_rate, settle)
        assert price == pytest.approx(100.0, abs=1e-8)


# ============================================================
# Long-bond convergence — error compounds with maturity
# ============================================================

class TestLongMaturityCharacterisation:
    """The per-coupon error doesn't cancel — it accumulates. A 30y UST has
    60 coupons, so the par-yield round-trip drift is larger than 5y."""

    @pytest.fixture
    def par_30y_ust(self):
        return FixedRateBond.treasury_note(
            issue_date=date(2024, 2, 15),
            maturity=date(2054, 2, 15),
            coupon_rate=0.04,
            face_value=100.0,
        )

    @pytest.mark.xfail(
        strict=True,
        reason="A.1 B1 — par-yield drift compounds across 60 coupons",
    )
    def test_30y_par_yield_round_trip_is_exact_100(self, par_30y_ust):
        settle = par_30y_ust.issue_date
        price = par_30y_ust._price_from_ytm(par_30y_ust.coupon_rate, settle)
        assert price == pytest.approx(100.0, abs=1e-8)


# ============================================================
# Diagnostic — print the actual current values
# ============================================================

def test_diagnostic_show_actual_year_fracs(capsys):
    """Diagnostic test (always passes) — prints today's actual per-coupon
    year_fracs so the magnitude of the bug is visible in test logs.

    Will be removed/updated in Slice 3 once year_fracs land at 0.5 exact.
    """
    bond = FixedRateBond.treasury_note(
        issue_date=date(2024, 2, 15),
        maturity=date(2029, 2, 15),
        coupon_rate=0.04,
        face_value=100.0,
    )
    with capsys.disabled():
        print()
        print("Pre-fix UST par-coupon diagnostics:")
        print(f"  day_count = {bond.day_count}")
        for i, cf in enumerate(bond.coupon_leg.cashflows):
            days = (cf.accrual_end - cf.accrual_start).days
            print(
                f"  coupon {i+1}: {cf.accrual_start} → {cf.accrual_end} "
                f"({days}d): year_frac={cf.year_frac:.6f}, amount={cf.amount:.6f}"
            )
        settle = bond.issue_date
        par_price = bond._price_from_ytm(bond.coupon_rate, settle)
        print(f"  par-yield round-trip: {par_price:.6f}  (should be 100.0 exactly)")
    # Test always passes — purely diagnostic.
    assert True
