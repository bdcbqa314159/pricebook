"""Equity greeks oracle — delta + vega on the Priceable protocol (L5).

Equity market data (spot, dividend curve, vol) lives in the snapshot keyed by
ticker, so equity risk flows through the same `Priceable`/bump path as FX and
rates: `equity_delta` bumps the spot, `equity_vega` bumps the vol.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.distributions import norm_cdf
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.equity_option import EquityOptionEngine
from pricebook_ng.products.equity_option import EquityOption
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.risk.greeks import bump_spot, bump_vol, spot_delta, vol_vega
from pricebook_ng.risk.priceable import discounting_priceable

USD = Currency.USD
TICKER = "ACME"
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)
SPOT, VOL, QTY = 100.0, 0.20, 1000.0
ACT365 = DC.ACT_365_FIXED
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=ACT365)


MARKET = MarketSnapshot(
    valuation_date=D0, discount_curve=_curve(0.04),
    curves={MarketKey(AssetClass.EQUITY, TICKER): _curve(0.02)}, spots={MarketKey(AssetClass.EQUITY, TICKER): SPOT}, vols={MarketKey(AssetClass.EQUITY, TICKER): VOL},
)


def _opt_priceable(strike=105.0, is_call=True):
    opt = EquityOption(TICKER, QTY, strike, MATURITY, USD, is_call)
    return discounting_priceable(opt, EquityOptionEngine(), NUM)


def _black_inputs(strike=105.0):
    df_r = MARKET.discount_curve.df(MATURITY)
    df_div = MARKET.curves[MarketKey(AssetClass.EQUITY, TICKER)].df(MATURITY)
    f = SPOT * df_div / df_r
    t = year_fraction(D0, MATURITY, ACT365)
    std = VOL * math.sqrt(t)
    d1 = (math.log(f / strike) + 0.5 * std * std) / std
    return df_r, df_div, f, t, std, d1


def test_bump_equity_spot_moves_only_that_spot():
    bumped = bump_spot(MARKET, MarketKey(AssetClass.EQUITY, TICKER), 1.0)
    assert bumped.spots[MarketKey(AssetClass.EQUITY, TICKER)] == pytest.approx(101.0, abs=1e-12)
    assert bumped.vols[MarketKey(AssetClass.EQUITY, TICKER)] == VOL              # vol untouched
    assert MARKET.spots[MarketKey(AssetClass.EQUITY, TICKER)] == SPOT            # original unchanged


def test_bump_equity_vol_moves_only_that_vol():
    bumped = bump_vol(MARKET, MarketKey(AssetClass.EQUITY, TICKER), 0.01)
    assert bumped.vols[MarketKey(AssetClass.EQUITY, TICKER)] == pytest.approx(0.21, abs=1e-12)
    assert bumped.spots[MarketKey(AssetClass.EQUITY, TICKER)] == SPOT


def test_equity_delta_matches_analytic():
    # d(call PV)/d(spot) = quantity * DF_div(T) * N(d1)
    _, df_div, _, _, _, d1 = _black_inputs()
    assert spot_delta(_opt_priceable(), MARKET, MarketKey(AssetClass.EQUITY, TICKER), NUM) == pytest.approx(
        QTY * df_div * norm_cdf(d1), abs=1e-4
    )


def test_put_delta_is_negative():
    assert spot_delta(_opt_priceable(is_call=False), MARKET, MarketKey(AssetClass.EQUITY, TICKER), NUM) < 0.0


def test_equity_vega_matches_analytic():
    # vega = quantity * DF_r(T) * F * phi(d1) * sqrt(T), per unit vol
    df_r, _, f, t, _, d1 = _black_inputs()
    phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    assert vol_vega(_opt_priceable(), MARKET, MarketKey(AssetClass.EQUITY, TICKER), NUM) == pytest.approx(
        QTY * df_r * f * phi * math.sqrt(t), abs=1e-2
    )
