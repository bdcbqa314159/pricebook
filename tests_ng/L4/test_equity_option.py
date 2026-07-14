"""Equity oracle — European equity option (Black-Scholes with dividends) (L4).

Black-Scholes as Black-76 on the equity forward `F = spot * DF_div / DF_r`
(continuous dividend/repo curve `DF_div`, discount curve `DF_r`), discounted by
`DF_r`. Equity market data (spot, dividend curve, vol) lives in the snapshot,
keyed by ticker. Oracles: put-call parity ties to the forward value; an
independent Black recompute; sigma -> 0 intrinsic; ATM-forward call == put.
"""

import math
from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.distributions import norm_cdf
from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.equity_option import EquityOptionEngine
from pricebook_ng.products.equity_option import EquityOption
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.models.discounting_model import DiscountingModel

USD = Currency.USD
TICKER = "ACME"
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)
SPOT, VOL, QTY = 100.0, 0.20, 1000.0
ACT365 = DC.ACT_365_FIXED
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=ACT365)


def _market(vol=VOL):
    return MarketSnapshot(
        valuation_date=D0, discount_curve=_curve(0.04),
        curves={MarketKey(AssetClass.EQUITY, TICKER): _curve(0.02)}, spots={MarketKey(AssetClass.EQUITY, TICKER): SPOT}, vols={MarketKey(AssetClass.EQUITY, TICKER): vol},
    )


def _fwd(market):
    return SPOT * market.curves[MarketKey(AssetClass.EQUITY, TICKER)].df(MATURITY) / market.discount_curve.df(MATURITY)


def _opt(strike, is_call=True):
    return EquityOption(ticker=TICKER, quantity=QTY, strike=strike,
                        maturity=MATURITY, currency=USD, is_call=is_call)


def _price(opt, market):
    return EquityOptionEngine().price(opt, DiscountingModel(market), NUM)


def test_priced_in_the_option_currency():
    r = _price(_opt(105.0), _market())
    assert isinstance(r, PricingResult)
    assert r.pv.currency is USD


def test_put_call_parity_ties_to_forward_value():
    market = _market()
    k = 105.0
    call = _price(_opt(k, is_call=True), market).pv.amount
    put = _price(_opt(k, is_call=False), market).pv.amount
    df_r = market.discount_curve.df(MATURITY)
    assert (call - put) == pytest.approx(QTY * df_r * (_fwd(market) - k), abs=1e-4)


def test_matches_independent_black():
    market = _market()
    k = 105.0
    f = _fwd(market)
    t = year_fraction(D0, MATURITY, ACT365)
    df_r = market.discount_curve.df(MATURITY)
    std = VOL * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * std * std) / std
    d2 = d1 - std
    expected = QTY * df_r * (f * norm_cdf(d1) - k * norm_cdf(d2))
    assert _price(_opt(k, is_call=True), market).pv.amount == pytest.approx(expected, abs=1e-4)


def test_zero_vol_is_discounted_intrinsic():
    market = _market(vol=0.0)
    f = _fwd(market)
    df_r = market.discount_curve.df(MATURITY)
    call = _price(_opt(90.0, is_call=True), market).pv.amount  # ITM (K < F)
    assert call == pytest.approx(QTY * df_r * max(f - 90.0, 0.0), abs=1e-4)


def test_atm_forward_call_equals_put():
    market = _market()
    k = _fwd(market)
    assert _price(_opt(k, is_call=True), market).pv.amount == pytest.approx(
        _price(_opt(k, is_call=False), market).pv.amount, abs=1e-4
    )


def test_missing_equity_market_is_a_failure():
    bare = MarketSnapshot(valuation_date=D0, discount_curve=_curve(0.04))
    assert isinstance(_price(_opt(105.0), bare), PricingFailure)


def test_is_pure_data_no_pricing():
    assert not hasattr(_opt(100.0), "pv")
