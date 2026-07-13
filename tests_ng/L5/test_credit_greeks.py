"""Credit greeks oracle — unified on the Pricable protocol (L5).

Ruling §5.1: the survival/hazard curve is market data, so it lives in the
`MarketSnapshot` (like the discount curve + fixings). Credit risk then flows
through the SAME `Pricable` as `dv01`: `credit01` bumps the snapshot's survival
curve, rebuilds the CreditModel via the pricable, and reprices. The identical
pricable also yields the CDS rate `dv01`.
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
from pricebook_ng.market.survival_curve import CDSQuote, bootstrap_survival_curve
from pricebook_ng.risk.greeks import bump_hazard, credit01, dv01
from pricebook_ng.risk.pricable import credit_pricable

CCY = Currency.USD
NOTIONAL = 10_000_000.0
RECOVERY = 0.4
D0 = date(2026, 1, 5)
MATURITY = date(2031, 1, 5)
ACT365 = DC.ACT_365_FIXED
NUM = NumericalConfig()

_BARE = MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(0.02, D0, ACT365))
SURVIVAL = bootstrap_survival_curve(
    _BARE,
    [CDSQuote(date(2027, 1, 5), 0.0100), CDSQuote(date(2029, 1, 5), 0.0150),
     CDSQuote(MATURITY, 0.0200)],
    RECOVERY,
)
MARKET = replace(_BARE, survival_curve=SURVIVAL)   # survival promoted into the snapshot
TERMS = ScheduleTerms(Frequency.ANNUAL, DC.ACT_360)


def _cds(spread=0.025, buy=True):
    c = cds(face=Money(NOTIONAL, CCY), spread=spread, start=D0, maturity=MATURITY, terms=TERMS)
    return c if buy else replace(c, buy_protection=False)


def _pricable(c):
    return credit_pricable(c, RECOVERY, CDSEngine(), NUM)


def test_survival_curve_lives_in_snapshot():
    assert MARKET.survival_curve is SURVIVAL


def test_bump_hazard_shifts_only_the_survival_curve():
    bumped = bump_hazard(MARKET, 0.001)
    assert bumped.survival_curve.survival(MATURITY) < MARKET.survival_curve.survival(MATURITY)
    assert bumped.discount_curve is MARKET.discount_curve  # rates untouched


def test_buyer_credit01_positive_via_pricable():
    assert credit01(_pricable(_cds()), MARKET, NUM) > 0.0


def test_seller_credit01_is_negative_buyer():
    b = credit01(_pricable(_cds(buy=True)), MARKET, NUM)
    s = credit01(_pricable(_cds(buy=False)), MARKET, NUM)
    assert s == pytest.approx(-b, abs=1e-9)


def test_credit01_matches_independent_hazard_fd():
    c = _cds()
    h = NUM.fd_bump
    v = SURVIVAL.valuation_date

    def priced(dh):  # inline hazard shift, independent of bump_hazard
        pillars = tuple(
            (d, q * math.exp(-dh * year_fraction(v, d, ACT365))) for d, q in SURVIVAL.pillars
        )
        m = replace(MARKET, survival_curve=replace(SURVIVAL, pillars=pillars))
        return _pricable(c)(m)

    manual = (priced(h) - priced(-h)) / (2 * h) * 1e-4
    assert credit01(_pricable(c), MARKET, NUM) == pytest.approx(manual, abs=1e-9)


def test_same_pricable_gives_cds_rate_dv01():
    # unification payoff: the CDS Pricable also feeds the generic dv01 (rate risk)
    d = dv01(_pricable(_cds()), MARKET, NUM)
    assert d != 0.0
