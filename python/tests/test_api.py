"""Tests for unified API layer."""

from datetime import date, timedelta

import pytest

import pricebook.api as pb
from pricebook.commodity import CommodityForwardCurve
from pricebook.greeks import Greeks


REF = date(2026, 4, 21)


# ---- Tenor parsing ----

class TestTenorParsing:
    def test_years(self):
        assert pb._parse_tenor(REF, "5Y") == date(2031, 4, 21)

    def test_months(self):
        assert pb._parse_tenor(REF, "6M") == date(2026, 10, 21)

    def test_weeks(self):
        assert pb._parse_tenor(REF, "2W") == date(2026, 5, 5)

    def test_days(self):
        assert pb._parse_tenor(REF, "30D") == date(2026, 5, 21)

    def test_pass_through_date(self):
        d = date(2030, 1, 15)
        assert pb._parse_tenor(REF, d) == d

    def test_lowercase(self):
        assert pb._parse_tenor(REF, "5y") == date(2031, 4, 21)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            pb._parse_tenor(REF, "5X")


# ---- Conventions ----

class TestConventions:
    def test_usd(self):
        conv = pb.conventions("USD")
        assert conv.settlement_days == 2
        assert isinstance(conv, pb.CcyConventions)

    def test_eur_annual_fixed(self):
        assert pb.conventions("EUR").fixed_freq.value == 12

    def test_frozen(self):
        with pytest.raises(AttributeError):
            pb.conventions("USD").settlement_days = 99


# ---- Curve building ----

class TestCurveBuilding:
    def test_flat_curve(self):
        curve = pb.flat_curve(0.04, REF)
        assert 0 < curve.df(date(2031, 4, 21)) < 1

    def test_flat_curve_no_date(self):
        curve = pb.flat_curve(0.03)
        assert curve.reference_date is not None

    def test_build_curve_tenor_strings(self):
        curve = pb.build_curve("USD", REF,
                               deposits={"3M": 0.043, "6M": 0.042},
                               swaps={"2Y": 0.039, "5Y": 0.038})
        assert 0 < curve.df(date(2031, 4, 21)) < 1

    def test_build_curve_dates(self):
        curve = pb.build_curve("USD", REF,
                               deposits=[(date(2026, 7, 21), 0.043)],
                               swaps=[(date(2031, 4, 21), 0.039)])
        assert curve.df(date(2031, 4, 21)) > 0

    def test_ois_tenor_strings(self):
        curve = pb.build_curve("USD", REF, ois_rates={"1Y": 0.04, "5Y": 0.042})
        assert curve.df(date(2031, 4, 21)) > 0

    def test_credit_curve(self):
        curve = pb.flat_curve(0.04, REF)
        surv = pb.build_credit_curve({"1Y": 0.005, "5Y": 0.01}, curve)
        assert 0 < surv.survival(date(2031, 4, 21)) < 1


# ---- IR ----

class TestIRS:
    def test_at_par(self):
        curve = pb.flat_curve(0.04, REF)
        pr = pb.par_rate("5Y", curve, start=REF)
        assert abs(pb.irs("5Y", pr, curve, start=REF)) < 10

    def test_payer(self):
        curve = pb.flat_curve(0.04, REF)
        assert pb.irs("5Y", 0.03, curve, start=REF) > 0

    def test_receiver(self):
        curve = pb.flat_curve(0.04, REF)
        assert pb.irs("5Y", 0.03, curve, start=REF, direction="RECEIVER") < 0

    def test_dv01(self):
        curve = pb.flat_curve(0.04, REF)
        assert pb.swap_dv01("5Y", 0.04, curve, start=REF) != 0.0

    def test_defaults_to_curve_ref(self):
        curve = pb.flat_curve(0.04, REF)
        assert isinstance(pb.irs("5Y", 0.04, curve), float)


class TestFRA:
    def test_below_forward(self):
        curve = pb.flat_curve(0.04, REF)
        assert pb.fra("3M", "6M", 0.03, curve, reference_date=REF) > 0


class TestCapFloor:
    def test_cap(self):
        curve = pb.flat_curve(0.04, REF)
        assert pb.cap("5Y", 0.04, 0.30, curve, start=REF) > 0

    def test_floor(self):
        curve = pb.flat_curve(0.04, REF)
        assert pb.floor("5Y", 0.04, 0.30, curve, start=REF) > 0


class TestSwaption:
    def test_price(self):
        curve = pb.flat_curve(0.04, REF)
        assert pb.swaption("1Y", "5Y", 0.04, 0.30, curve) > 0

    def test_greeks(self):
        curve = pb.flat_curve(0.04, REF)
        g = pb.swaption("1Y", "5Y", 0.04, 0.30, curve, return_greeks=True)
        assert isinstance(g, Greeks)
        assert g.vega > 0


# ---- Bonds ----

class TestBond:
    def test_par(self):
        curve = pb.flat_curve(0.04, REF)
        assert 99 < pb.bond("10Y", 0.04, curve, start=REF) < 101

    def test_ytm(self):
        assert 0.03 < pb.bond_ytm("10Y", 0.04, 100.0, start=REF, settlement=REF) < 0.05

    def test_duration(self):
        assert 0 < pb.bond_duration("10Y", 0.04, 100.0, start=REF, settlement=REF) < 10


# ---- Credit ----

class TestCDS:
    def test_near_par(self):
        curve = pb.flat_curve(0.04, REF)
        surv = pb.build_credit_curve({"5Y": 0.01}, curve)
        assert abs(pb.cds("5Y", 0.01, curve, surv, start=REF)) < 1000

    def test_par_spread(self):
        curve = pb.flat_curve(0.04, REF)
        surv = pb.build_credit_curve({"5Y": 0.01}, curve)
        assert pb.cds_par_spread("5Y", curve, surv, start=REF) > 0


# ---- FX ----

class TestFX:
    def test_forward_rate(self):
        eur = pb.flat_curve(0.03, REF)
        usd = pb.flat_curve(0.04, REF)
        assert pb.fx_forward_rate(1.10, "1Y", eur, usd, REF) > 1.10


# ---- Equity ----

class TestEquity:
    def test_forward(self):
        curve = pb.flat_curve(0.04, REF)
        assert pb.equity_forward(100, "1Y", curve, div_yield=0.02, reference_date=REF) > 100


# ---- Commodity ----

class TestCommodity:
    def test_swap(self):
        dates = [REF + timedelta(days=30 * i) for i in range(1, 13)]
        fwds = [70 + 0.5 * i for i in range(1, 13)]
        fwd_curve = CommodityForwardCurve(REF, dates, fwds, spot=70.0)
        disc = pb.flat_curve(0.04, REF)
        assert isinstance(pb.commodity_swap_pv(72.0, fwd_curve, disc, "1Y", start=REF), float)


# ---- Inflation ----

class TestInflation:
    def test_breakeven(self):
        from pricebook.inflation import CPICurve
        cpi = CPICurve.from_breakevens(REF, 300.0,
                                       [date(2031, 4, 21)], [0.025])
        be = pb.inflation_breakeven("5Y", cpi, REF)
        assert be == pytest.approx(0.025, rel=0.01)
