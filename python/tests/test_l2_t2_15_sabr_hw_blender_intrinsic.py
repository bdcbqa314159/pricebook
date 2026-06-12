"""Regression for L2 Tier-2 T2.15 — `price_swaption_sabr_hw` returns intrinsic
value at T=0 instead of unconditionally zero.

Pre-fix:

    if T <= 0:
        return 0.0

This returned 0.0 even when the swaption had positive intrinsic value
(e.g. a payer with strike below the current forward swap rate, evaluated
at expiry).  Correct expiry payoff:

    payer:    annuity · max(fwd − K, 0)
    receiver: annuity · max(K − fwd, 0)
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.options.swaption import (
    Swaption,
    SwaptionType,
    price_swaption_sabr_hw,
)


REF = date(2024, 1, 1)


def _flat_curve(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    dfs = [math.exp(-rate * t) for t in tenors]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


class _FakeSABRCube:
    def vol(self, expiry, tenor, strike):
        return 0.20


class _FakeHWModel:
    a = 0.05
    sigma = 0.01


class TestSABRHWBlenderAtExpiry:
    def test_payer_at_expiry_returns_intrinsic(self):
        """At T=0, a deep-ITM payer (strike well below fwd) must return
        positive intrinsic, not 0."""
        curve = _flat_curve(rate=0.06)
        # Build a swaption whose expiry == reference_date → T = 0.
        # The 4y swap starts at REF and ends 4y later; fwd ≈ 6%, K=2% → deep ITM.
        swaption = Swaption(
            expiry=REF,
            swap_end=REF + timedelta(days=4 * 365),
            strike=0.02,
            swaption_type=SwaptionType.PAYER,
            notional=1_000_000.0,
        )
        price = price_swaption_sabr_hw(
            swaption, _FakeSABRCube(), _FakeHWModel(), curve,
        )
        # Intrinsic ≈ annuity × (fwd − 0.02) × notional ≈ very positive.
        ann = swaption.annuity(curve)
        fwd = swaption.forward_swap_rate(curve)
        intrinsic = ann * max(fwd - 0.02, 0.0) * 1_000_000.0
        assert price > 0, f"price = {price} (must be > 0 at T=0 for ITM)"
        assert abs(price - intrinsic) < 1e-6, (
            f"price = {price:.4f}, intrinsic = {intrinsic:.4f}"
        )

    def test_receiver_at_expiry_returns_intrinsic(self):
        """Symmetric: at T=0 a deep-ITM receiver returns positive intrinsic."""
        curve = _flat_curve(rate=0.02)
        swaption = Swaption(
            expiry=REF,
            swap_end=REF + timedelta(days=4 * 365),
            strike=0.06,
            swaption_type=SwaptionType.RECEIVER,
            notional=1_000_000.0,
        )
        price = price_swaption_sabr_hw(
            swaption, _FakeSABRCube(), _FakeHWModel(), curve,
        )
        ann = swaption.annuity(curve)
        fwd = swaption.forward_swap_rate(curve)
        intrinsic = ann * max(0.06 - fwd, 0.0) * 1_000_000.0
        assert price > 0
        assert abs(price - intrinsic) < 1e-6

    def test_out_of_money_at_expiry_returns_zero(self):
        """Sanity: OTM at expiry → intrinsic = 0."""
        curve = _flat_curve(rate=0.04)
        swaption = Swaption(
            expiry=REF,
            swap_end=REF + timedelta(days=4 * 365),
            strike=0.10,
            swaption_type=SwaptionType.PAYER,  # strike >> fwd ⇒ OTM
            notional=1_000_000.0,
        )
        price = price_swaption_sabr_hw(
            swaption, _FakeSABRCube(), _FakeHWModel(), curve,
        )
        assert price == 0.0

    def test_atm_post_expiry_positive_T_unchanged(self):
        """Sanity: a normal forward-dated swaption still prices via the
        Black-76 path (positive T branch unchanged by the fix)."""
        curve = _flat_curve(rate=0.04)
        swaption = Swaption(
            expiry=REF + timedelta(days=2 * 365),
            swap_end=REF + timedelta(days=5 * 365),
            strike=0.04,
            swaption_type=SwaptionType.PAYER,
            notional=1_000_000.0,
        )
        price = price_swaption_sabr_hw(
            swaption, _FakeSABRCube(), _FakeHWModel(), curve,
        )
        # Should be positive (ATM has positive time value).
        assert price > 0
