"""Regression for L2 T4 audit of `structured.longevity.mortality_bond_price`:

Same int(T) truncation pattern as v1.035 (cat_bond_price).  Pre-fix
``range(1, int(T) + 1)`` silently dropped non-integer T:

- T = 0.5 (6-month bond): empty loop → zero coupon PV.
- T = 3.5: 3 full annual coupons; missed 3.5-year accrual.

Fix: add a fractional final accrual when T - int(T) > 0.
"""

from __future__ import annotations

import pytest

from pricebook.structured.longevity import mortality_bond_price


class TestFractionalT:
    def test_six_month_bond_has_coupon_pv(self):
        """T=0.5: pre-fix gave 0 coupon PV; post-fix gives 0.5y accrual."""
        r = mortality_bond_price(
            notional=1000.0, coupon=0.06, risk_free_rate=0.04,
            attachment=1.5, exhaustion=2.5,
            expected_mortality=1.0, mortality_vol=0.10, T=0.5,
        )
        # Expected coupon ≈ 0.06 × 1000 × 0.5 × df(0.5) ≈ 30 × 0.98 ≈ 29.4.
        assert r["coupon_pv"] > 20.0

    def test_integer_T_unchanged(self):
        """T=3 integer case has no remainder branch and matches pre-fix."""
        r = mortality_bond_price(
            notional=1000.0, coupon=0.06, risk_free_rate=0.04,
            attachment=1.5, exhaustion=2.5,
            expected_mortality=1.0, mortality_vol=0.10, T=3.0,
        )
        # 3 coupons × 60 × discount factors ≈ 60 × (e^-0.04 + e^-0.08 + e^-0.12) ≈ 60 × 2.715 ≈ 162.9.
        assert r["coupon_pv"] == pytest.approx(166.25, rel=1e-2)

    def test_fractional_T_includes_remainder(self):
        """T=3.5: 3 integer coupons + 0.5y final accrual."""
        r_int = mortality_bond_price(
            notional=1000.0, coupon=0.06, risk_free_rate=0.04,
            attachment=1.5, exhaustion=2.5,
            expected_mortality=1.0, mortality_vol=0.10, T=3.0,
        )
        r_frac = mortality_bond_price(
            notional=1000.0, coupon=0.06, risk_free_rate=0.04,
            attachment=1.5, exhaustion=2.5,
            expected_mortality=1.0, mortality_vol=0.10, T=3.5,
        )
        # Fractional should add ~0.5 × 60 × df(3.5) ≈ 26 to integer case.
        delta = r_frac["coupon_pv"] - r_int["coupon_pv"]
        assert delta > 20.0
        assert delta < 35.0
