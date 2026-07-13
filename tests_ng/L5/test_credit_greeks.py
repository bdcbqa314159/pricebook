"""Credit greeks oracle — CDS credit01 / CS01 (L5).

credit01 = CDS PV change per 1bp parallel shift in the credit spread (hazard),
by central finite difference: bump the survival curve's hazard, rebuild the
CreditModel, reprice. The hazard analogue of dv01 (which bumps the discount rate).
"""

import math
from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.cds import CDSEngine
from pricebook_ng.products.cds import cds
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote, SurvivalCurve, bootstrap_survival_curve
from pricebook_ng.models.credit_model import CreditModel
from pricebook_ng.risk.credit_greeks import bump_hazard, credit01

CCY = Currency.USD
NOTIONAL = 10_000_000.0
RECOVERY = 0.4
D0 = date(2026, 1, 5)
MATURITY_5Y = date(2031, 1, 5)
ACT365 = DC.ACT_365_FIXED

MARKET = MarketSnapshot(
    valuation_date=D0, discount_curve=FlatDiscountCurve(0.02, D0, ACT365)
)
SURVIVAL = bootstrap_survival_curve(
    MARKET,
    [CDSQuote(date(2027, 1, 5), 0.0100), CDSQuote(date(2029, 1, 5), 0.0150),
     CDSQuote(MATURITY_5Y, 0.0200)],
    RECOVERY,
)
MODEL = CreditModel(MARKET, SURVIVAL, RECOVERY)
TERMS = ScheduleTerms(Frequency.ANNUAL, DC.ACT_360)
NUM = NumericalConfig()


def _cds(spread=0.025):
    return cds(face=Money(NOTIONAL, CCY), spread=spread, start=D0, maturity=MATURITY_5Y, terms=TERMS)


def test_bump_hazard_shifts_survival_exactly():
    dh = 0.001
    bumped = bump_hazard(SURVIVAL, dh)
    for (d, q), (bd, bq) in zip(SURVIVAL.pillars, bumped.pillars):
        assert bd == d
        t = year_fraction(SURVIVAL.valuation_date, d, ACT365)
        assert bq == pytest.approx(q * math.exp(-dh * t), abs=1e-15)
    assert bumped.survival(D0) == pytest.approx(1.0, abs=1e-15)          # anchor unchanged
    assert bumped.survival(MATURITY_5Y) < SURVIVAL.survival(MATURITY_5Y)  # higher hazard


def test_buyer_credit01_is_positive():
    # protection buyer gains as credit worsens (spreads widen)
    assert credit01(_cds(), MODEL, NUM) > 0.0


def test_seller_credit01_is_negative_buyer():
    buyer = _cds()
    seller = replace(buyer, buy_protection=False)
    assert credit01(seller, MODEL, NUM) == pytest.approx(-credit01(buyer, MODEL, NUM), abs=1e-9)


def test_credit01_matches_independent_central_diff():
    cds_obj = _cds()
    h = NUM.fd_bump
    v = SURVIVAL.valuation_date

    def manual_bump(dh):  # inline hazard shift, independent of bump_hazard
        pillars = tuple(
            (d, q * math.exp(-dh * year_fraction(v, d, ACT365))) for d, q in SURVIVAL.pillars
        )
        return SurvivalCurve(v, pillars)

    def priced(surv):
        return CDSEngine().price(cds_obj, replace(MODEL, survival=surv), NUM).pv.amount

    manual = (priced(manual_bump(h)) - priced(manual_bump(-h))) / (2 * h) * 1e-4
    assert credit01(cds_obj, MODEL, NUM) == pytest.approx(manual, abs=1e-9)
