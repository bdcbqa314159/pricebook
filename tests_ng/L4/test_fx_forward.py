"""FX oracle — FX forward by covered interest parity (L4).

FX market data (base-currency curves + spots) now lives in the `MarketSnapshot`
(ruling §5.1, promoted from the model), keyed by currency. The FX forward is a
linear product priced with a `DiscountingModel` over that snapshot.
    F(T)  = spot * DF_base(T) / DF_quote(T)
    PV_buy = base_notional*spot*DF_base(T) - quote_notional*DF_quote(T)   (quote ccy)
"""

from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.engine.fx_forward import FXForwardEngine
from pricebook_ng.products.fx_forward import FXForward, fx_forward
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel

EUR, USD = Currency.EUR, Currency.USD
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)
NOTIONAL = 1_000_000.0
SPOT = 1.10
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=DC.ACT_365_FIXED)


def _market(quote_rate=0.03, base_rate=0.01, spot=SPOT):
    return MarketSnapshot(
        valuation_date=D0, discount_curve=_curve(quote_rate),   # USD (quote/home) curve
        fx_curves={EUR: _curve(base_rate)}, fx_spots={EUR: spot},
    )


def _price(fwd, market):
    return FXForwardEngine().price(fwd, DiscountingModel(market), NUM)


def _fwd_rate(market, maturity=MATURITY):
    return market.fx_spots[EUR] * market.fx_curves[EUR].df(maturity) / market.discount_curve.df(maturity)


def _fwd(strike, maturity=MATURITY):
    return fx_forward(Money(NOTIONAL, EUR), quote_ccy=USD, strike=strike, maturity=maturity)


def test_par_fx_forward_prices_to_zero():
    market = _market()
    result = _price(_fwd(_fwd_rate(market)), market)
    assert isinstance(result, PricingResult)
    assert result.pv.currency is USD
    assert result.pv.amount == pytest.approx(0.0, abs=1e-6)


def test_pv_matches_covered_interest_parity():
    market = _market()
    strike = 1.15
    expected = (NOTIONAL * SPOT * market.fx_curves[EUR].df(MATURITY)
                - NOTIONAL * strike * market.discount_curve.df(MATURITY))
    assert _price(_fwd(strike), market).pv.amount == pytest.approx(expected, abs=1e-6)


def test_sell_is_negative_buy():
    market = _market()
    buy = _fwd(1.15)
    sell = replace(buy, buy_base=False)
    assert _price(sell, market).pv.amount == pytest.approx(-_price(buy, market).pv.amount, abs=1e-9)


def test_missing_fx_market_is_a_failure():
    bare = MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.03))  # no fx data
    assert isinstance(_price(_fwd(1.15), bare), PricingFailure)


def test_matured_forward_is_settled():
    assert _price(_fwd(1.15, maturity=date(2025, 1, 5)), _market()).pv.amount == pytest.approx(
        0.0, abs=1e-9
    )


def test_is_pure_data_no_pricing():
    fwd = _fwd(1.10)
    assert isinstance(fwd, FXForward)
    assert not hasattr(fwd, "pv")
