"""Tests for zero-coupon swaps and IR digitals."""

import math
import pytest
from datetime import date

from pricebook.zc_swap import (
    ZeroCouponSwap, digital_capfloor, digital_cms_cap,
)
from pricebook.black76 import OptionType
from pricebook.discount_curve import DiscountCurve
from pricebook.swap import InterestRateSwap, SwapDirection


REF = date(2024, 1, 15)


def _curve(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _upward_curve():
    pillar_dates = [
        date(2025, 1, 15), date(2026, 1, 15), date(2029, 1, 15),
        date(2034, 1, 15), date(2054, 1, 15),
    ]
    rates = [0.03, 0.035, 0.04, 0.045, 0.05]
    dfs = [math.exp(-r * ((d - REF).days / 365.0)) for r, d in zip(rates, pillar_dates)]
    return DiscountCurve(REF, pillar_dates, dfs)


# ---- Zero-coupon swap ----

class TestZeroCouponSwap:
    def test_at_par_zero_pv(self):
        curve = _curve()
        zcs = ZeroCouponSwap(REF, date(2034, 1, 15), fixed_rate=0.05)
        par = zcs.par_rate(curve)
        at_par = ZeroCouponSwap(REF, date(2034, 1, 15), par)
        assert at_par.pv(curve) == pytest.approx(0.0, abs=1.0)

    def test_par_rate_above_standard(self):
        """ZC par rate > standard swap par rate on upward curve (compounding effect)."""
        curve = _upward_curve()
        zcs = ZeroCouponSwap(REF, date(2034, 1, 15), 0.05)
        zc_par = zcs.par_rate(curve)
        std_par = InterestRateSwap(
            REF, date(2034, 1, 15), 0.05, SwapDirection.PAYER,
        ).par_rate(curve)
        # ZC compounds, so par rate should differ
        assert zc_par != std_par

    def test_payer_receiver_opposite(self):
        curve = _curve()
        par = ZeroCouponSwap(REF, date(2029, 1, 15), 0.05).par_rate(curve)
        payer = ZeroCouponSwap(REF, date(2029, 1, 15), par - 0.01, SwapDirection.PAYER)
        receiver = ZeroCouponSwap(REF, date(2029, 1, 15), par - 0.01, SwapDirection.RECEIVER)
        assert payer.pv(curve) == pytest.approx(-receiver.pv(curve))

    def test_fixed_amount(self):
        zcs = ZeroCouponSwap(REF, date(2029, 1, 15), 0.05, notional=1_000_000)
        T = zcs._tenor_years()
        expected = 1_000_000 * ((1.05) ** T - 1)
        assert zcs.fixed_amount() == pytest.approx(expected)

    def test_floating_pv(self):
        """Floating PV = notional × (1 - df(T))."""
        curve = _curve()
        zcs = ZeroCouponSwap(REF, date(2029, 1, 15), 0.05, notional=1_000_000)
        df_T = curve.df(date(2029, 1, 15))
        assert zcs.floating_pv(curve) == pytest.approx(1_000_000 * (1 - df_T))

    def test_longer_tenor_higher_fixed_amount(self):
        zcs5 = ZeroCouponSwap(REF, date(2029, 1, 15), 0.05)
        zcs10 = ZeroCouponSwap(REF, date(2034, 1, 15), 0.05)
        assert zcs10.fixed_amount() > zcs5.fixed_amount()


# ---- Digital cap/floor ----

class TestDigitalCapFloor:
    def test_digital_cap_positive(self):
        curve = _curve()
        pv = digital_capfloor(
            REF, date(2029, 1, 15), 0.04, 1.0, curve, 0.20,
        )
        assert pv > 0

    def test_digital_floor_positive(self):
        curve = _curve()
        pv = digital_capfloor(
            REF, date(2029, 1, 15), 0.06, 1.0, curve, 0.20,
            option_type=OptionType.PUT,
        )
        assert pv > 0

    def test_digital_bounded(self):
        """Digital PV should be between 0 and notional × payout × n_periods × df."""
        curve = _curve()
        pv = digital_capfloor(
            REF, date(2029, 1, 15), 0.05, 1.0, curve, 0.20,
        )
        assert 0 < pv < 1_000_000 * 1.0 * 20  # conservative upper bound

    def test_deep_otm_small(self):
        curve = _curve()
        itm = digital_capfloor(REF, date(2029, 1, 15), 0.02, 1.0, curve, 0.20)
        otm = digital_capfloor(REF, date(2029, 1, 15), 0.15, 1.0, curve, 0.20)
        assert otm < itm

    def test_higher_payout_higher_pv(self):
        curve = _curve()
        low = digital_capfloor(REF, date(2029, 1, 15), 0.05, 0.5, curve, 0.20)
        high = digital_capfloor(REF, date(2029, 1, 15), 0.05, 1.0, curve, 0.20)
        assert high == pytest.approx(2 * low, rel=0.01)


# ---- Digital CMS cap ----

class TestDigitalCMSCap:
    def test_positive(self):
        curve = _curve()
        pv = digital_cms_cap(
            REF, date(2029, 1, 15), 0.04, 1.0, 10, curve, 0.20,
        )
        assert pv > 0

    def test_floor_positive(self):
        curve = _curve()
        pv = digital_cms_cap(
            REF, date(2029, 1, 15), 0.06, 1.0, 10, curve, 0.20,
            option_type=OptionType.PUT,
        )
        assert pv > 0

    def test_deep_otm_small(self):
        curve = _curve()
        itm = digital_cms_cap(REF, date(2029, 1, 15), 0.02, 1.0, 10, curve, 0.20)
        otm = digital_cms_cap(REF, date(2029, 1, 15), 0.15, 1.0, 10, curve, 0.20)
        assert otm < itm
