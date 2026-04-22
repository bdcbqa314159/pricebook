"""Tests for unified API layer."""

from datetime import date

import pytest

import pricebook.api as pb
from pricebook.greeks import Greeks
from tests.conftest import make_flat_curve


REF = date(2026, 4, 21)


class TestTenorParsing:
    def test_years(self):
        assert pb._parse_tenor(REF, "5Y") == date(2031, 4, 21)

    def test_months(self):
        assert pb._parse_tenor(REF, "6M") == date(2026, 10, 21)

    def test_weeks(self):
        assert pb._parse_tenor(REF, "2W") == date(2026, 5, 5)

    def test_days(self):
        assert pb._parse_tenor(REF, "30D") == date(2026, 5, 21)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            pb._parse_tenor(REF, "5X")


class TestConventions:
    def test_usd(self):
        conv = pb.conventions("USD")
        assert conv.settlement_days == 2
        assert isinstance(conv, pb.CcyConventions)

    def test_gbp(self):
        assert pb.conventions("GBP").settlement_days == 0

    def test_unknown_defaults_to_usd(self):
        assert pb.conventions("ZAR").settlement_days == 2

    def test_frozen(self):
        conv = pb.conventions("USD")
        with pytest.raises(AttributeError):
            conv.settlement_days = 99


class TestCurveBuilding:
    def test_build_curve(self):
        curve = pb.build_curve("USD", REF,
                               deposits=[(date(2026, 7, 21), 0.043)],
                               swaps=[(date(2031, 4, 21), 0.039)])
        assert 0 < curve.df(date(2031, 4, 21)) < 1

    def test_ois_curve(self):
        curve = pb.build_curve("USD", REF,
                               ois_rates=[(date(2027, 4, 21), 0.04),
                                          (date(2031, 4, 21), 0.042)])
        assert curve.df(date(2031, 4, 21)) > 0

    def test_credit_curve(self):
        curve = make_flat_curve(REF, 0.04)
        surv = pb.build_credit_curve(REF, {"1Y": 0.005, "5Y": 0.01}, curve)
        assert 0 < surv.survival(date(2031, 4, 21)) < 1


class TestIRS:
    def test_at_par_near_zero(self):
        curve = make_flat_curve(REF, 0.04)
        pr = pb.par_rate(REF, "5Y", curve)
        pv = pb.irs(REF, "5Y", pr, curve)
        assert abs(pv) < 10

    def test_payer_positive_below_market(self):
        curve = make_flat_curve(REF, 0.04)
        assert pb.irs(REF, "5Y", 0.03, curve) > 0

    def test_receiver_negative(self):
        curve = make_flat_curve(REF, 0.04)
        assert pb.irs(REF, "5Y", 0.03, curve, direction="receiver") < 0

    def test_dv01(self):
        curve = make_flat_curve(REF, 0.04)
        dv01 = pb.swap_dv01(REF, "5Y", 0.04, curve)
        assert dv01 != 0.0


class TestFRA:
    def test_fra_nonzero(self):
        """FRA with strike below forward has positive PV."""
        curve = make_flat_curve(REF, 0.04)
        pv = pb.fra(date(2026, 7, 21), "3M", 0.03, curve)
        assert pv > 0


class TestBond:
    def test_par_near_100(self):
        curve = make_flat_curve(REF, 0.04)
        assert 99 < pb.bond(REF, "10Y", 0.04, curve) < 101

    def test_tenor_string(self):
        curve = make_flat_curve(REF, 0.04)
        assert pb.bond(REF, "5Y", 0.04, curve) > 0

    def test_duration(self):
        dur = pb.bond_duration(REF, "10Y", 0.04, 100.0, settlement=REF)
        assert 0 < dur < 10


class TestCDS:
    def test_near_par(self):
        curve = make_flat_curve(REF, 0.04)
        surv = pb.build_credit_curve(REF, {"5Y": 0.01}, curve)
        pv = pb.cds(REF, "5Y", 0.01, curve, surv)
        assert abs(pv) < 1000


class TestFXForward:
    def test_basic(self):
        eur = make_flat_curve(REF, 0.03)
        usd = make_flat_curve(REF, 0.04)
        fwd = pb.fx_forward_rate("EUR/USD", 1.10, "1Y", eur, usd, REF)
        assert fwd > 1.10


class TestSwaption:
    def test_positive_price(self):
        curve = make_flat_curve(REF, 0.04)
        assert pb.swaption(REF, "1Y", "5Y", 0.04, 0.30, curve) > 0

    def test_greeks(self):
        curve = make_flat_curve(REF, 0.04)
        g = pb.swaption(REF, "1Y", "5Y", 0.04, 0.30, curve, return_greeks=True)
        assert isinstance(g, Greeks)
        assert g.price > 0
        assert g.vega > 0


class TestCapFloor:
    def test_cap_positive(self):
        curve = make_flat_curve(REF, 0.04)
        assert pb.cap(REF, "5Y", 0.04, 0.30, curve) > 0

    def test_floor_positive(self):
        curve = make_flat_curve(REF, 0.04)
        assert pb.floor(REF, "5Y", 0.04, 0.30, curve) > 0

    def test_cap_floor_parity(self):
        """Cap - Floor ≈ Swap PV (approximately)."""
        curve = make_flat_curve(REF, 0.04)
        c = pb.cap(REF, "5Y", 0.04, 0.30, curve)
        f = pb.floor(REF, "5Y", 0.04, 0.30, curve)
        # At ATM: cap ≈ floor (put-call parity on each caplet)
        assert abs(c - f) < c * 0.2  # within 20%
