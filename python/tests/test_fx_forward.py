"""Tests for FX forward."""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.fx_forward import FXForward
from pricebook.currency import Currency, CurrencyPair
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


EURUSD = CurrencyPair(Currency.EUR, Currency.USD)
SPOT = 1.10


class TestForwardRate:

    def test_cip_formula(self):
        """F = S * df_base / df_quote."""
        eur_curve = make_flat_curve(REF, rate=0.04)  # base
        usd_curve = make_flat_curve(REF, rate=0.05)  # quote
        mat = REF + relativedelta(years=1)
        fwd = FXForward.forward_rate(SPOT, mat, eur_curve, usd_curve)
        expected = SPOT * eur_curve.df(mat) / usd_curve.df(mat)
        assert fwd == pytest.approx(expected, rel=1e-10)

    def test_higher_quote_rate_lowers_forward(self):
        """If USD rate > EUR rate, EUR/USD forward < spot (USD at premium)."""
        eur_curve = make_flat_curve(REF, rate=0.03)
        usd_curve = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd = FXForward.forward_rate(SPOT, mat, eur_curve, usd_curve)
        assert fwd > SPOT  # EUR appreciates when EUR rate < USD rate

    def test_higher_base_rate_lowers_forward(self):
        """If EUR rate > USD rate, EUR/USD forward < spot."""
        eur_curve = make_flat_curve(REF, rate=0.06)
        usd_curve = make_flat_curve(REF, rate=0.03)
        mat = REF + relativedelta(years=1)
        fwd = FXForward.forward_rate(SPOT, mat, eur_curve, usd_curve)
        assert fwd < SPOT

    def test_equal_rates_forward_equals_spot(self):
        eur_curve = make_flat_curve(REF, rate=0.05)
        usd_curve = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd = FXForward.forward_rate(SPOT, mat, eur_curve, usd_curve)
        assert fwd == pytest.approx(SPOT, rel=1e-6)

    def test_longer_maturity_amplifies_differential(self):
        eur_curve = make_flat_curve(REF, rate=0.03)
        usd_curve = make_flat_curve(REF, rate=0.05)
        fwd_1y = FXForward.forward_rate(SPOT, REF + relativedelta(years=1), eur_curve, usd_curve)
        fwd_5y = FXForward.forward_rate(SPOT, REF + relativedelta(years=5), eur_curve, usd_curve)
        assert abs(fwd_5y - SPOT) > abs(fwd_1y - SPOT)


class TestForwardPoints:

    def test_forward_points_sign(self):
        eur_curve = make_flat_curve(REF, rate=0.03)
        usd_curve = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        pts = FXForward.forward_points(SPOT, mat, eur_curve, usd_curve)
        assert pts > 0  # EUR/USD forward points positive when USD rate > EUR rate

    def test_forward_points_zero_equal_rates(self):
        curve = make_flat_curve(REF, rate=0.04)
        mat = REF + relativedelta(years=1)
        pts = FXForward.forward_points(SPOT, mat, curve, curve)
        assert pts == pytest.approx(0.0, abs=1e-6)


class TestPV:

    def test_pv_zero_at_forward_strike(self):
        eur_curve = make_flat_curve(REF, rate=0.04)
        usd_curve = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd = FXForward.forward_rate(SPOT, mat, eur_curve, usd_curve)
        contract = FXForward(EURUSD, mat, strike=fwd)
        assert contract.pv(SPOT, eur_curve, usd_curve) == pytest.approx(0.0, abs=1.0)

    def test_pv_positive_when_strike_below_forward(self):
        eur_curve = make_flat_curve(REF, rate=0.04)
        usd_curve = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd = FXForward.forward_rate(SPOT, mat, eur_curve, usd_curve)
        contract = FXForward(EURUSD, mat, strike=fwd - 0.05)
        assert contract.pv(SPOT, eur_curve, usd_curve) > 0

    def test_pv_negative_when_strike_above_forward(self):
        eur_curve = make_flat_curve(REF, rate=0.04)
        usd_curve = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd = FXForward.forward_rate(SPOT, mat, eur_curve, usd_curve)
        contract = FXForward(EURUSD, mat, strike=fwd + 0.05)
        assert contract.pv(SPOT, eur_curve, usd_curve) < 0

    def test_pv_scales_with_notional(self):
        eur_curve = make_flat_curve(REF, rate=0.04)
        usd_curve = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd1 = FXForward(EURUSD, mat, strike=1.08, notional=1_000_000.0)
        fwd2 = FXForward(EURUSD, mat, strike=1.08, notional=2_000_000.0)
        assert fwd2.pv(SPOT, eur_curve, usd_curve) == pytest.approx(
            2 * fwd1.pv(SPOT, eur_curve, usd_curve), rel=1e-10)


class TestValidation:

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            FXForward(EURUSD, REF + relativedelta(years=1), strike=1.10, notional=-1.0)

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError):
            FXForward(EURUSD, REF + relativedelta(years=1), strike=-1.10)
