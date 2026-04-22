"""Tests for unified API layer."""

from datetime import date

import pytest

import pricebook.api as pb
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
        assert conv["settlement"] == 2

    def test_gbp(self):
        conv = pb.conventions("GBP")
        assert conv["settlement"] == 0

    def test_unknown_defaults_to_usd(self):
        conv = pb.conventions("ZAR")
        assert conv["settlement"] == 2


class TestCurveBuilding:
    def test_bootstrap(self):
        curve = pb.curves("USD", REF,
                          deposits=[(date(2026, 7, 21), 0.043)],
                          swaps=[(date(2031, 4, 21), 0.039)])
        assert curve.df(date(2031, 4, 21)) > 0
        assert curve.df(date(2031, 4, 21)) < 1

    def test_ois_bootstrap(self):
        curve = pb.curves("USD", REF,
                          ois_rates=[(date(2027, 4, 21), 0.04),
                                     (date(2031, 4, 21), 0.042)])
        assert curve.df(date(2031, 4, 21)) > 0


class TestIRS:
    def test_at_par_near_zero(self):
        curve = make_flat_curve(REF, 0.04)
        pr = pb.par_rate(REF, "5Y", curve)
        pv = pb.irs(REF, "5Y", pr, curve)
        assert abs(pv) < 10  # near zero on 1M notional

    def test_payer_positive_below_market(self):
        curve = make_flat_curve(REF, 0.04)
        pv = pb.irs(REF, "5Y", 0.03, curve)  # fixed < market → payer gains
        assert pv > 0

    def test_receiver(self):
        curve = make_flat_curve(REF, 0.04)
        pv = pb.irs(REF, "5Y", 0.03, curve, direction="receiver")
        assert pv < 0


class TestBond:
    def test_par_near_100(self):
        curve = make_flat_curve(REF, 0.04)
        px = pb.bond(REF, "10Y", 0.04, curve)
        assert 99 < px < 101

    def test_tenor_string(self):
        curve = make_flat_curve(REF, 0.04)
        px = pb.bond(REF, "5Y", 0.04, curve)
        assert px > 0


class TestCDS:
    def test_at_par_zero(self):
        """CDS without survival curve → 0 (no credit risk)."""
        curve = make_flat_curve(REF, 0.04)
        pv = pb.cds(REF, "5Y", 0.01, curve)
        assert pv == 0.0

    def test_with_survival(self):
        curve = make_flat_curve(REF, 0.04)
        surv = pb.credit_curve(REF, {"5Y": 0.01}, curve)
        pv = pb.cds(REF, "5Y", 0.01, curve, surv)
        assert abs(pv) < 1000  # near par


class TestFXForward:
    def test_basic(self):
        eur = make_flat_curve(REF, 0.03)
        usd = make_flat_curve(REF, 0.04)
        fwd = pb.fx_forward("EUR/USD", 1.10, "1Y", eur, usd, REF)
        assert fwd > 1.10  # USD rate > EUR → forward > spot


class TestSwaption:
    def test_positive_price(self):
        curve = make_flat_curve(REF, 0.04)
        pv = pb.swaption(REF, "1Y", "5Y", 0.04, 0.30, curve)
        assert pv > 0

    def test_greeks(self):
        curve = make_flat_curve(REF, 0.04)
        g = pb.swaption_greeks(REF, "1Y", "5Y", 0.04, 0.30, curve)
        assert g.price > 0
        assert g.vega > 0


class TestCapFloor:
    def test_cap_positive(self):
        curve = make_flat_curve(REF, 0.04)
        pv = pb.cap(REF, "5Y", 0.04, 0.30, curve)
        assert pv > 0

    def test_floor_positive(self):
        curve = make_flat_curve(REF, 0.04)
        pv = pb.floor(REF, "5Y", 0.04, 0.30, curve)
        assert pv > 0
