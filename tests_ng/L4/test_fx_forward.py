"""FX oracle — FX forward by covered interest parity (L4).

An FX forward exchanges `base` (base ccy) for `quote` (quote ccy) at maturity.
Valued in the quote currency off two curves + spot:
    F(T)  = spot * DF_base(T) / DF_quote(T)                 (covered interest parity)
    PV_buy = base_notional*spot*DF_base(T) - quote_notional*DF_quote(T)   (in quote ccy)

The base curve + spot ride on the `FXForwardModel` for now (like the CDS survival
curve did before §5.1); promoting FX market data into the snapshot + FX greeks is
the follow-up.
"""

from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.engine.fx_forward import FXForwardEngine
from pricebook_ng.products.fx_forward import FXForward, fx_forward
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.fx_model import FXForwardModel

EUR, USD = Currency.EUR, Currency.USD
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)     # ~1Y
NOTIONAL = 1_000_000.0
SPOT = 1.10                      # USD per EUR
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=DC.ACT_365_FIXED)


def _model(quote_rate=0.03, base_rate=0.01, spot=SPOT):
    market = MarketSnapshot(valuation_date=D0, discount_curve=_curve(quote_rate))  # USD
    return FXForwardModel(market=market, base_curve=_curve(base_rate), spot=spot)  # EUR curve


def _fwd_rate(model, maturity=MATURITY):
    return model.spot * model.base_curve.df(maturity) / model.market.discount_curve.df(maturity)


def _fwd(strike, maturity=MATURITY):
    return fx_forward(Money(NOTIONAL, EUR), quote_ccy=USD, strike=strike, maturity=maturity)


def test_par_fx_forward_prices_to_zero():
    model = _model()
    fwd = _fwd(_fwd_rate(model))               # struck at the forward rate
    result = FXForwardEngine().price(fwd, model, NUM)
    assert isinstance(result, PricingResult)
    assert result.pv.currency is USD           # valued in the quote currency
    assert result.pv.amount == pytest.approx(0.0, abs=1e-6)


def test_pv_matches_covered_interest_parity():
    model = _model()
    strike = 1.15
    fwd = _fwd(strike)
    df_base = model.base_curve.df(MATURITY)
    df_quote = model.market.discount_curve.df(MATURITY)
    expected = NOTIONAL * SPOT * df_base - NOTIONAL * strike * df_quote
    result = FXForwardEngine().price(fwd, model, NUM)
    assert result.pv.amount == pytest.approx(expected, abs=1e-6)


def test_sell_is_negative_buy():
    model = _model()
    buy = _fwd(1.15)
    sell = replace(buy, buy_base=False)
    b = FXForwardEngine().price(buy, model, NUM).pv.amount
    s = FXForwardEngine().price(sell, model, NUM).pv.amount
    assert s == pytest.approx(-b, abs=1e-9)


def test_below_forward_strike_is_valuable_to_buyer():
    model = _model()
    fwd = _fwd(_fwd_rate(model) - 0.05)        # buy base cheaper than fair -> positive
    assert FXForwardEngine().price(fwd, model, NUM).pv.amount > 0.0


def test_matured_forward_is_settled():
    model = _model()
    fwd = _fwd(1.15, maturity=date(2025, 1, 5))  # before valuation
    assert FXForwardEngine().price(fwd, model, NUM).pv.amount == pytest.approx(0.0, abs=1e-9)


def test_is_pure_data_no_pricing():
    fwd = _fwd(1.10)
    assert isinstance(fwd, FXForward)
    assert not hasattr(fwd, "pv")
    assert fwd.base.currency is EUR and fwd.quote.currency is USD
