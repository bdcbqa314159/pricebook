"""Amendment A1 oracle — the engine depends on the model, not a market arg.

`price(product, model, numerics)`: the model carries the `MarketSnapshot` it was
calibrated to (`model.market`), so a model/market mismatch is structurally
impossible — there is no second market to pass. `DiscountingModel` is the thin
model linear products use.
"""

import inspect
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.instruments.fixed_cashflow import FixedCashflowTrade
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel

D0 = date(2026, 1, 1)
T = date(2028, 1, 1)


def _market(rate=0.03):
    curve = FlatDiscountCurve(rate=rate, anchor=D0, day_count=DC.ACT_365_FIXED)
    return MarketSnapshot(valuation_date=D0, discount_curve=curve)


def _trade():
    return FixedCashflowTrade(Cashflow(date=T, amount=Money(1_000_000.0, Currency.USD)))


def test_engine_price_has_no_market_parameter():
    params = list(inspect.signature(DiscountingEngine.price).parameters)
    assert params == ["self", "instrument", "model", "numerics"]
    assert "market" not in params


def test_model_carries_its_own_market():
    market = _market()
    model = DiscountingModel(market)
    assert model.market is market
    # "today" is reached through the model, not a separate argument
    assert model.market.valuation_date == D0


def test_price_uses_the_models_market():
    engine = DiscountingEngine()
    hi = engine.price(_trade(), DiscountingModel(_market(0.01)), NumericalConfig())
    lo = engine.price(_trade(), DiscountingModel(_market(0.05)), NumericalConfig())
    assert isinstance(hi, PricingResult)
    # a higher-rate market discounts harder -> smaller PV; the model alone decides
    assert hi.pv.amount > lo.pv.amount
