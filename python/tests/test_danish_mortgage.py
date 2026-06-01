"""Tests for Danish mortgage bonds (realkreditobligationer)."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.fixed_income.danish_mortgage import (
    DanishMortgageBond, prepayment_model, psa_curve,
    synthetic_mortgage_quotes, MortgageBondResult,
)

REF = date(2024, 11, 4)


def _make_curve(rate=0.035):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 35)]
    dfs = [math.exp(-rate * y) for y in range(1, 35)]
    return DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


class TestPrepaymentModel:
    def test_base_cpr(self):
        """No refi incentive, fully seasoned → base CPR only."""
        cpr = prepayment_model(0.03, 0.04, seasoning_years=3.0)  # coupon < rate → no refi
        assert cpr == pytest.approx(0.03, abs=0.01)

    def test_refi_incentive_increases_cpr(self):
        """When rates fall below coupon, CPR should increase."""
        cpr_no_refi = prepayment_model(0.03, 0.04)
        cpr_refi = prepayment_model(0.04, 0.02)  # 200bp incentive
        assert cpr_refi > cpr_no_refi

    def test_seasoning_ramp(self):
        """New bonds have lower CPR (seasoning ramp-up)."""
        cpr_new = prepayment_model(0.04, 0.02, seasoning_years=0.5)
        cpr_seasoned = prepayment_model(0.04, 0.02, seasoning_years=3.0)
        assert cpr_seasoned > cpr_new

    def test_cpr_capped(self):
        """CPR should be capped at 80%."""
        cpr = prepayment_model(0.10, 0.01, seasoning_years=5.0, refi_sensitivity=20.0)
        assert cpr <= 0.80


class TestPSACurve:
    def test_ramps_up(self):
        assert psa_curve(1) < psa_curve(15) < psa_curve(30)

    def test_flat_after_30(self):
        assert psa_curve(30) == pytest.approx(psa_curve(60))

    def test_200_psa_faster(self):
        assert psa_curve(15, 200) > psa_curve(15, 100)


class TestDanishMortgageBond:
    def test_bullet_positive_price(self):
        curve = _make_curve()
        bond = DanishMortgageBond(REF, REF + relativedelta(years=10), 0.03, "bullet")
        r = bond.price(REF, curve, cpr=0.05)
        assert r.dirty_price > 0

    def test_callable_le_noncallable(self):
        """Callable bond price ≤ non-callable (prepayment costs the bondholder)."""
        curve = _make_curve(0.02)  # rates below coupon → prepayment
        bond = DanishMortgageBond(REF, REF + relativedelta(years=20), 0.04, "bullet")

        callable_px = bond.price(REF, curve, current_mortgage_rate=0.02).dirty_price
        noncallable_px = bond._price_noncallable(REF, curve)

        assert callable_px <= noncallable_px + 0.5

    def test_higher_cpr_shorter_wal(self):
        """Higher CPR → shorter weighted average life."""
        curve = _make_curve()
        bond = DanishMortgageBond(REF, REF + relativedelta(years=20), 0.03, "pass_through")

        r_low = bond.price(REF, curve, cpr=0.03)
        r_high = bond.price(REF, curve, cpr=0.15)

        assert r_high.wal < r_low.wal

    def test_pass_through_shorter_wal_than_bullet(self):
        """Pass-through (amortising) has shorter WAL than bullet."""
        curve = _make_curve()
        bullet = DanishMortgageBond(REF, REF + relativedelta(years=20), 0.03, "bullet")
        pt = DanishMortgageBond(REF, REF + relativedelta(years=20), 0.03, "pass_through")

        r_bullet = bullet.price(REF, curve, cpr=0.05)
        r_pt = pt.price(REF, curve, cpr=0.05)

        assert r_pt.wal < r_bullet.wal

    def test_oas_positive_for_callable(self):
        """OAS should be positive when bond is callable with high CPR."""
        curve = _make_curve(0.02)
        bond = DanishMortgageBond(REF - relativedelta(years=3),  # seasoned
                                   REF + relativedelta(years=10), 0.04, "bullet")
        r = bond.price(REF, curve, current_mortgage_rate=0.02)
        assert r.oas > 0

    def test_callable_value_positive(self):
        curve = _make_curve(0.02)
        bond = DanishMortgageBond(REF, REF + relativedelta(years=10), 0.04, "bullet")
        r = bond.price(REF, curve, current_mortgage_rate=0.02)
        assert r.callable_value >= 0

    def test_effective_duration(self):
        curve = _make_curve()
        bond = DanishMortgageBond(REF, REF + relativedelta(years=10), 0.03, "bullet")
        r = bond.price(REF, curve, cpr=0.05)
        assert r.effective_duration > 0

    def test_to_dict(self):
        bond = DanishMortgageBond(REF, REF + relativedelta(years=10), 0.03, "pass_through")
        d = bond.to_dict()
        assert d["type"] == "danish_mortgage"
        assert d["structure"] == "pass_through"

    def test_synthetic_quotes(self):
        quotes = synthetic_mortgage_quotes(REF)
        assert len(quotes) == 5
        assert any(q["structure"] == "pass_through" for q in quotes)
        assert any(q["structure"] == "bullet" for q in quotes)
