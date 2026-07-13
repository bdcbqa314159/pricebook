"""FX greeks oracle — on the Priceable protocol (L5).

With FX data promoted into the snapshot (§5.1), FX risk flows through the same
`Priceable` as `dv01`/`credit01`: `fx_delta` bumps the snapshot's FX spot and
reprices; the identical pricable also yields the (quote-curve) `dv01`.
"""

from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.engine.fx_forward import FXForwardEngine
from pricebook_ng.products.fx_forward import fx_forward
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.risk.greeks import bump_fx_spot, dv01, fx_delta
from pricebook_ng.risk.priceable import fx_forward_priceable

EUR, USD = Currency.EUR, Currency.USD
D0 = date(2026, 1, 5)
MATURITY = date(2027, 1, 5)
NOTIONAL = 1_000_000.0
NUM = NumericalConfig()


def _curve(rate):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=DC.ACT_365_FIXED)


MARKET = MarketSnapshot(
    valuation_date=D0, discount_curve=_curve(0.03),
    fx_curves={EUR: _curve(0.01)}, fx_spots={EUR: 1.10},
)


def _fwd(strike=1.15, buy=True):
    f = fx_forward(Money(NOTIONAL, EUR), quote_ccy=USD, strike=strike, maturity=MATURITY)
    return f if buy else replace(f, buy_base=False)


def _priceable(f):
    return fx_forward_priceable(f, FXForwardEngine(), NUM)


def test_fx_data_lives_in_snapshot():
    assert MARKET.fx_spots[EUR] == 1.10
    assert MARKET.fx_curves[EUR].df(MATURITY) == _curve(0.01).df(MATURITY)


def test_bump_fx_spot_moves_only_that_spot():
    bumped = bump_fx_spot(MARKET, EUR, 0.01)
    assert bumped.fx_spots[EUR] == pytest.approx(1.11, abs=1e-12)
    assert bumped.discount_curve is MARKET.discount_curve         # rates untouched
    assert MARKET.fx_spots[EUR] == 1.10                            # original unchanged


def test_fx_delta_matches_analytic():
    # d(PV_buy)/d(spot) = base_notional * DF_base(T)
    d = fx_delta(_priceable(_fwd()), MARKET, EUR, NUM)
    assert d == pytest.approx(NOTIONAL * MARKET.fx_curves[EUR].df(MATURITY), abs=1e-3)


def test_sell_fx_delta_is_negative_buy():
    b = fx_delta(_priceable(_fwd(buy=True)), MARKET, EUR, NUM)
    s = fx_delta(_priceable(_fwd(buy=False)), MARKET, EUR, NUM)
    assert s == pytest.approx(-b, abs=1e-6)


def test_same_priceable_gives_quote_rate_dv01():
    # the FX forward Priceable also feeds the generic dv01 (bumps the quote curve)
    assert dv01(_priceable(_fwd()), MARKET, NUM) != 0.0
