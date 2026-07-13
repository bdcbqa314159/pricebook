"""Credit oracle — CDS priced end-to-end through the engine (L4).

The CDS wraps the S(credit) survival curve: a `CreditModel` carries the market
(discount) + the bootstrapped survival curve + recovery; the `CDSEngine` values
the protection buyer via the L1 CDS leg math. Oracles: a par CDS reprices to
zero; the engine matches the L1 `cds_pv`; seller = -buyer; buyer value falls as
the contract spread rises.
"""

from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.engine.cds import CDSEngine
from pricebook_ng.products.cds import CDS, cds
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote, bootstrap_survival_curve, cds_pv
from pricebook_ng.models.credit_model import CreditModel

CCY = Currency.USD
NOTIONAL = 10_000_000.0
RECOVERY = 0.4
D0 = date(2026, 1, 5)
MATURITY_5Y = date(2031, 1, 5)
PAR_5Y = 0.0200

MARKET = MarketSnapshot(
    valuation_date=D0,
    discount_curve=FlatDiscountCurve(rate=0.02, anchor=D0, day_count=DC.ACT_365_FIXED),
)
QUOTES = [
    CDSQuote(date(2027, 1, 5), 0.0100),
    CDSQuote(date(2029, 1, 5), 0.0150),
    CDSQuote(MATURITY_5Y, PAR_5Y),
]
SURVIVAL = bootstrap_survival_curve(MARKET, QUOTES, RECOVERY)
MODEL = CreditModel(market=MARKET, survival=SURVIVAL, recovery=RECOVERY)
TERMS = ScheduleTerms(Frequency.ANNUAL, DC.ACT_360)


def _cds(spread, maturity=MATURITY_5Y):
    return cds(face=Money(NOTIONAL, CCY), spread=spread, start=D0, maturity=maturity, terms=TERMS)


def test_par_cds_reprices_to_zero():
    result = CDSEngine().price(_cds(PAR_5Y), MODEL, NumericalConfig())
    assert isinstance(result, PricingResult)
    assert result.pv.currency is CCY
    assert result.pv.amount == pytest.approx(0.0, abs=1e-3)


def test_engine_matches_l1_cds_pv():
    cds_obj = _cds(0.025)
    expected = cds_pv(
        MARKET.discount_curve, SURVIVAL, list(cds_obj.premium_schedule), 0.025, RECOVERY
    ) * NOTIONAL
    result = CDSEngine().price(cds_obj, MODEL, NumericalConfig())
    assert result.pv.amount == pytest.approx(expected, abs=1e-6)


def test_seller_is_negative_buyer():
    buyer = _cds(0.025)
    seller = replace(buyer, buy_protection=False)
    b = CDSEngine().price(buyer, MODEL, NumericalConfig()).pv.amount
    s = CDSEngine().price(seller, MODEL, NumericalConfig()).pv.amount
    assert s == pytest.approx(-b, abs=1e-9)


def test_buyer_value_falls_as_spread_rises():
    cheap = CDSEngine().price(_cds(PAR_5Y - 0.005), MODEL, NumericalConfig()).pv.amount
    dear = CDSEngine().price(_cds(PAR_5Y + 0.005), MODEL, NumericalConfig()).pv.amount
    assert cheap > 0.0 > dear  # buyer gains paying below par, loses paying above


def test_is_pure_data_no_pricing():
    c = _cds(PAR_5Y)
    assert isinstance(c, CDS)
    assert not hasattr(c, "pv")
