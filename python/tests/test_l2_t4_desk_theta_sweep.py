"""Regression for L2 T4 audit of desk-level theta computations.

Same bug pattern as v1.026 (`swap_desk` theta) found in:
- `swaption_trading_desk.swaption_risk_metrics`
- `trs_desk` daily P&L theta

In all cases the theta lambda discounted with the rolled curve but
projected forwards from the original-t=0 curve, so the floating
forward rates were stale (1 day behind the discount).

Fix: under single-curve, pass ``None`` (or rolled ``c``) for projection
so it uses the rolled discount curve; under dual-curve, pre-roll the
projection by 1 day.

Note: `cln_desk` has the same defect on the survival curve but
``SurvivalCurve`` has no ``roll_down()`` yet — tracked as a known
approximation rather than fixed in this slice.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve


def _build_flat_curve(ref: date, rate: float = 0.05) -> DiscountCurve:
    pillars = [date(ref.year + y, ref.month, ref.day) for y in (1, 2, 3, 5, 7, 10)]
    dfs = [math.exp(-rate * y) for y in (1, 2, 3, 5, 7, 10)]
    return DiscountCurve(ref, pillars, dfs, DayCountConvention.ACT_365_FIXED)


class TestSwaptionTradingDeskTheta:
    def test_theta_uses_rolled_projection(self):
        """Single-curve swaption theta must equal consistent roll PV diff."""
        from pricebook.options.swaption import Swaption, SwaptionType
        from pricebook.desks.swaption_trading_desk import swaption_risk_metrics
        from pricebook.models.models import Black76Model

        ref = date(2026, 1, 15)
        curve = _build_flat_curve(ref)
        swaption = Swaption(
            expiry=date(2027, 1, 15),
            swap_end=date(2032, 1, 15),
            strike=0.05, notional=1_000_000.0,
            swaption_type=SwaptionType.PAYER,
        )
        vol = 0.30
        rm = swaption_risk_metrics(swaption, curve, vol)
        # Independent consistent-roll: roll curve, use it for both.
        rolled = curve.roll_down(1)
        model = Black76Model(vol=vol)
        expected = swaption.price(model, rolled, rolled) - swaption.price(model, curve, curve)
        assert rm.theta == pytest.approx(expected, rel=1e-9, abs=1e-6)


class TestTrsDeskTheta:
    def test_trs_theta_uses_rolled_projection_single_curve(self):
        """TRS theta under single-curve must use rolled c for projection."""
        from pricebook.equity.trs import TotalReturnSwap, FundingLegSpec
        from pricebook.desks.trs_desk import trs_daily_pnl

        ref = date(2026, 1, 15)
        curve = _build_flat_curve(ref)
        curve_t1 = _build_flat_curve(date(2026, 1, 16))
        trs = TotalReturnSwap(
            underlying=100.0, notional=1_000_000,
            start=ref, end=date(2026, 7, 15),
            funding=FundingLegSpec(spread=0.005),
            repo_spread=0.01, haircut=0.05,
            initial_price=100.0, sigma=0.20,
        )
        pnl = trs_daily_pnl(trs, curve, curve_t1, date(2026, 1, 16))
        # Theta should be a finite number (not NaN).
        assert pnl.theta_pnl == pnl.theta_pnl
        # Independent consistent-roll check (single-curve → None projection).
        rolled = curve.roll_down(1)
        expected_theta = trs.price(rolled, None).value - trs.price(curve, None).value
        assert pnl.theta_pnl == pytest.approx(expected_theta, rel=1e-6, abs=1e-6)
