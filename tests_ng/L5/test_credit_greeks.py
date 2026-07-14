"""Credit greeks oracle — unified on the Priceable protocol (L5).

Ruling §5.1: the survival/hazard curve is market data, so it lives in the
`MarketSnapshot` (like the discount curve + fixings). Credit risk then flows
through the SAME `Priceable` as `dv01`: `credit01` bumps the snapshot's survival
curve, rebuilds the CreditModel via the priceable, and reprices. The identical
priceable also yields the CDS rate `dv01`.
"""

import math
from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.cds import CDSEngine
from pricebook_ng.products.cds import cds
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.market.survival_curve import CDSQuote, bootstrap_survival_curve
from pricebook_ng.risk.greeks import bump_curve, credit01, dv01
from pricebook_ng.risk.priceable import credit_priceable

CCY = Currency.USD
NOTIONAL = 10_000_000.0
RECOVERY = 0.4
ISSUER = "ACME_CO"
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
CREDIT_KEY = MarketKey(AssetClass.CREDIT, ISSUER)
MARKET = replace(_BARE, curves={CREDIT_KEY: SURVIVAL})   # survival keyed by issuer (A5)


def _cds(spread=0.025, buy=True):
    c = cds(face=Money(NOTIONAL, CCY), issuer=ISSUER, spread=spread, start=D0, maturity=MATURITY)
    return c if buy else replace(c, buy_protection=False)


def _priceable(c):
    return credit_priceable(c, RECOVERY, CDSEngine(), NUM)


def test_survival_curve_lives_in_snapshot():
    assert MARKET.curves[CREDIT_KEY] is SURVIVAL


def test_bump_hazard_shifts_only_the_survival_curve():
    bumped = bump_curve(MARKET, CREDIT_KEY, 0.001)
    assert bumped.curves[CREDIT_KEY].survival(MATURITY) < SURVIVAL.survival(MATURITY)
    assert bumped.discount_curve is MARKET.discount_curve  # rates untouched


def test_buyer_credit01_positive_via_priceable():
    assert credit01(_priceable(_cds()), MARKET, CREDIT_KEY, NUM) > 0.0


def test_seller_credit01_is_negative_buyer():
    b = credit01(_priceable(_cds(buy=True)), MARKET, CREDIT_KEY, NUM)
    s = credit01(_priceable(_cds(buy=False)), MARKET, CREDIT_KEY, NUM)
    assert s == pytest.approx(-b, abs=1e-9)


def test_credit01_matches_independent_hazard_fd():
    c = _cds()
    h = NUM.fd_bump
    v = SURVIVAL.valuation_date

    def priced(dh):  # inline hazard shift, independent of bump_hazard
        pillars = tuple(
            (d, q * math.exp(-dh * year_fraction(v, d, ACT365))) for d, q in SURVIVAL.pillars
        )
        m = replace(MARKET, curves={CREDIT_KEY: replace(SURVIVAL, pillars=pillars)})
        return _priceable(c)(m)

    manual = (priced(h) - priced(-h)) / (2 * h) * 1e-4
    assert credit01(_priceable(c), MARKET, CREDIT_KEY, NUM) == pytest.approx(manual, abs=1e-9)


def test_same_priceable_gives_cds_rate_dv01():
    # unification payoff: the CDS Priceable also feeds the generic dv01 (rate risk)
    d = dv01(_priceable(_cds()), MARKET, NUM)
    assert d != 0.0
