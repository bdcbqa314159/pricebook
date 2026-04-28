"""Comprehensive serialisation tests: all instrument types, curves, round-trips."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.serialization import (
    instrument_to_dict, instrument_from_dict,
    discount_curve_to_dict, discount_curve_from_dict,
    survival_curve_to_dict, survival_curve_from_dict,
    trade_to_dict, trade_from_dict,
    portfolio_to_dict, portfolio_from_dict,
    to_json, from_json, list_instruments,
)

REF = date(2026, 4, 28)
END_5Y = REF + timedelta(days=1825)
END_10Y = REF + timedelta(days=3650)


# ---- Instrument round-trips ----

class TestIRSRoundTrip:
    def test_round_trip(self):
        from pricebook.swap import InterestRateSwap
        irs = InterestRateSwap(REF, END_5Y, fixed_rate=0.035, notional=10_000_000)
        d = instrument_to_dict(irs)
        assert d["type"] == "irs"
        irs2 = instrument_from_dict(d)
        assert irs2.fixed_rate == 0.035
        assert irs2.notional == 10_000_000

    def test_json_round_trip(self):
        from pricebook.swap import InterestRateSwap
        irs = InterestRateSwap(REF, END_5Y, fixed_rate=0.035)
        s = to_json(irs)
        irs2 = from_json(s)
        assert irs2.fixed_rate == 0.035


class TestOISSwapRoundTrip:
    def test_round_trip(self):
        from pricebook.ois import OISSwap
        ois = OISSwap(REF, END_5Y, fixed_rate=0.03, notional=1_000_000)
        d = instrument_to_dict(ois)
        assert d["type"] == "ois_swap"
        ois2 = instrument_from_dict(d)
        assert ois2.notional == 1_000_000
        assert ois2.start == REF


class TestBasisSwapRoundTrip:
    def test_round_trip(self):
        from pricebook.basis_swap import BasisSwap
        bs = BasisSwap(REF, END_5Y, spread=0.001, notional=5_000_000)
        d = instrument_to_dict(bs)
        assert d["type"] == "basis_swap"
        bs2 = instrument_from_dict(d)
        assert bs2.spread == 0.001
        assert bs2.notional == 5_000_000


class TestDepositRoundTrip:
    def test_round_trip(self):
        from pricebook.deposit import Deposit
        dep = Deposit(REF, REF + timedelta(days=91), rate=0.03)
        d = instrument_to_dict(dep)
        assert d["type"] == "deposit"
        dep2 = instrument_from_dict(d)
        assert dep2.rate == 0.03


class TestBondRoundTrip:
    def test_round_trip(self):
        from pricebook.bond import FixedRateBond
        bond = FixedRateBond(face_value=100, coupon_rate=0.05, maturity=END_10Y,
                             issue_date=REF)
        d = instrument_to_dict(bond)
        assert d["type"] == "bond"
        bond2 = instrument_from_dict(d)
        assert bond2.coupon_rate == 0.05


class TestFRARoundTrip:
    def test_round_trip(self):
        from pricebook.fra import FRA
        fra = FRA(REF + timedelta(days=91), REF + timedelta(days=182),
                  strike=0.035, notional=1_000_000)
        d = instrument_to_dict(fra)
        assert d["type"] == "fra"
        fra2 = instrument_from_dict(d)
        assert fra2.strike == 0.035


class TestCDSRoundTrip:
    def test_round_trip(self):
        from pricebook.cds import CDS
        cds = CDS(REF, END_5Y, spread=0.01, notional=10_000_000)
        d = instrument_to_dict(cds)
        assert d["type"] == "cds"
        cds2 = instrument_from_dict(d)
        assert cds2.spread == 0.01


class TestCLNRoundTrip:
    def test_round_trip(self):
        from pricebook.cln import CreditLinkedNote
        cln = CreditLinkedNote(REF, END_5Y, coupon_rate=0.06, notional=1_000_000,
                               leverage=2.0)
        d = instrument_to_dict(cln)
        assert d["type"] == "cln"
        cln2 = instrument_from_dict(d)
        assert cln2.coupon_rate == 0.06
        assert cln2.leverage == 2.0


class TestTermLoanRoundTrip:
    def test_round_trip(self):
        from pricebook.loan import TermLoan
        loan = TermLoan(REF, END_5Y, spread=0.03, notional=10_000_000)
        d = instrument_to_dict(loan)
        assert d["type"] == "term_loan"
        loan2 = instrument_from_dict(d)
        assert loan2.spread == 0.03


class TestRevolverRoundTrip:
    def test_round_trip(self):
        from pricebook.loan import RevolvingFacility
        rev = RevolvingFacility(REF, END_5Y, max_commitment=50_000_000,
                                drawn_amount=30_000_000, drawn_spread=0.025)
        d = instrument_to_dict(rev)
        assert d["type"] == "revolver"
        rev2 = instrument_from_dict(d)
        assert rev2.max_commitment == 50_000_000
        assert rev2.drawn_amount == 30_000_000


class TestFXForwardRoundTrip:
    def test_round_trip(self):
        from pricebook.fx_forward import FXForward
        from pricebook.currency import CurrencyPair, Currency
        pair = CurrencyPair(Currency.EUR, Currency.USD)
        fwd = FXForward(pair, maturity=END_5Y, strike=1.10, notional=1_000_000)
        d = instrument_to_dict(fwd)
        assert d["type"] == "fx_forward"


class TestSwaptionRoundTrip:
    def test_round_trip(self):
        from pricebook.swaption import Swaption
        sw = Swaption(expiry=REF + timedelta(days=365), swap_end=END_5Y,
                      strike=0.035, notional=10_000_000)
        d = instrument_to_dict(sw)
        assert d["type"] == "swaption"
        sw2 = instrument_from_dict(d)
        assert sw2.strike == 0.035


# ---- TRS (polymorphic underlying) ----

class TestTRSRoundTrip:
    def test_equity_trs(self):
        from pricebook.trs import TotalReturnSwap
        trs = TotalReturnSwap(underlying=100.0, notional=10_000_000,
                              start=REF, end=REF + timedelta(days=365))
        d = instrument_to_dict(trs)
        assert d["type"] == "trs"
        assert d["params"]["underlying"]["type"] == "equity_spot"
        trs2 = instrument_from_dict(d)
        assert float(trs2.underlying) == 100.0
        assert trs2.notional == 10_000_000

    def test_bond_trs(self):
        from pricebook.trs import TotalReturnSwap
        from pricebook.bond import FixedRateBond
        bond = FixedRateBond(face_value=100, coupon_rate=0.05, maturity=END_10Y,
                             issue_date=REF)
        trs = TotalReturnSwap(underlying=bond, notional=50_000_000,
                              start=REF, end=REF + timedelta(days=365))
        d = instrument_to_dict(trs)
        assert d["params"]["underlying"]["type"] == "bond"
        trs2 = instrument_from_dict(d)
        assert type(trs2.underlying).__name__ == "FixedRateBond"

    def test_loan_trs(self):
        from pricebook.trs import TotalReturnSwap
        from pricebook.loan import TermLoan
        loan = TermLoan(REF, END_5Y, spread=0.03, notional=10_000_000)
        trs = TotalReturnSwap(underlying=loan, notional=10_000_000,
                              start=REF, end=REF + timedelta(days=365))
        d = instrument_to_dict(trs)
        assert d["params"]["underlying"]["type"] == "term_loan"
        trs2 = instrument_from_dict(d)
        assert type(trs2.underlying).__name__ == "TermLoan"

    def test_json_round_trip(self):
        from pricebook.trs import TotalReturnSwap
        trs = TotalReturnSwap(underlying=100.0, notional=5_000_000,
                              start=REF, end=REF + timedelta(days=365),
                              repo_spread=0.002, sigma=0.25)
        s = to_json(trs)
        d = json.loads(s)
        assert d["type"] == "trs"
        trs2 = from_json(s)
        assert trs2.sigma == 0.25


# ---- Curves ----

class TestCurveRoundTrip:
    def test_discount_curve(self):
        from pricebook.discount_curve import DiscountCurve
        curve = DiscountCurve.flat(REF, 0.03)
        d = discount_curve_to_dict(curve)
        curve2 = discount_curve_from_dict(d)
        assert curve2.df(END_5Y) == pytest.approx(curve.df(END_5Y))

    def test_survival_curve(self):
        from pricebook.survival_curve import SurvivalCurve
        sc = SurvivalCurve.flat(REF, 0.02)
        d = survival_curve_to_dict(sc)
        sc2 = survival_curve_from_dict(d)
        assert sc2.survival(END_5Y) == pytest.approx(sc.survival(END_5Y))


# ---- Trade/Portfolio ----

class TestTradePortfolioRoundTrip:
    def test_trade(self):
        from pricebook.swap import InterestRateSwap
        from pricebook.trade import Trade
        irs = InterestRateSwap(REF, END_5Y, fixed_rate=0.035)
        trade = Trade(irs, trade_id="T1", counterparty="ACME")
        d = trade_to_dict(trade)
        trade2 = trade_from_dict(d)
        assert trade2.trade_id == "T1"
        assert trade2.instrument.fixed_rate == 0.035

    def test_portfolio(self):
        from pricebook.swap import InterestRateSwap
        from pricebook.trade import Trade, Portfolio
        trades = [
            Trade(InterestRateSwap(REF, END_5Y, fixed_rate=0.03), trade_id="T1"),
            Trade(InterestRateSwap(REF, END_10Y, fixed_rate=0.035), trade_id="T2"),
        ]
        port = Portfolio(trades=trades, name="test_book")
        d = portfolio_to_dict(port)
        port2 = portfolio_from_dict(d)
        assert port2.name == "test_book"
        assert len(port2.trades) == 2


# ---- Registry ----

class TestRegistry:
    def test_list_instruments(self):
        types = list_instruments()
        assert "irs" in types
        assert "cln" in types
        assert "term_loan" in types
        assert "revolver" in types
        assert "deposit" in types
        assert "fx_forward" in types
        assert len(types) >= 11

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            instrument_from_dict({"type": "nonexistent", "params": {}})

    def test_unregistered_class_raises(self):
        class FakeInstrument:
            pass
        with pytest.raises(ValueError, match="No serialization"):
            instrument_to_dict(FakeInstrument())
