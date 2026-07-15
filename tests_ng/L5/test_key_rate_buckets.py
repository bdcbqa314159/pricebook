"""Key-rate (bucketed) dv01 oracle (L5).

`dv01` is a single parallel shift; `key_rate_dv01` decomposes it into per-pillar
buckets — bump the zero at one pillar only, reprice, and the log-linear
interpolation "tents" the bump (rising to the bumped pillar, back to zero at its
neighbours). Two facts pin it down:

  1. Partition of unity: the buckets sum to the parallel dv01 (bumping every
     pillar at once IS a parallel shift). Σ_j KR01_j = dv01.
  2. Isolation at a node: a cashflow landing exactly on pillar j is touched only
     by bucket j (neighbouring tents are zero at the node), so
     KR01_j = -N·t_j·DF(t_j)·1bp and every other bucket is ~0.

Key-rate is inherently a pillar-curve concept (a flat curve has no pillars), so
it reads the home bootstrapped `DiscountCurve` directly.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.market.discount_curve import DepositQuote, bootstrap_discount_curve
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.products.fixed_cashflow import FixedCashflow
from pricebook_ng.products.fixed_rate_bond import fixed_rate_bond
from pricebook_ng.risk.greeks import dv01, key_rate_dv01
from pricebook_ng.risk.priceable import discounting_priceable

CCY = Currency.USD
ACT365 = DC.ACT_365_FIXED
ACT360 = DC.ACT_360
NUM = NumericalConfig()
D0 = date(2026, 1, 15)
_ONE_BP = 1e-4
_PILLARS = [date(2027, 1, 15), date(2028, 1, 15), date(2029, 1, 15)]


def _boot_curve():
    deposits = [
        DepositQuote(_PILLARS[0], 0.030, ACT360),
        DepositQuote(_PILLARS[1], 0.035, ACT360),
        DepositQuote(_PILLARS[2], 0.040, ACT360),
    ]
    return bootstrap_discount_curve(D0, deposits, [])


def test_buckets_sum_to_parallel_dv01():
    curve = _boot_curve()
    market = MarketSnapshot(valuation_date=D0, discount_curve=curve)
    bond = fixed_rate_bond(
        face=Money(1_000_000.0, CCY), coupon_rate=0.04, start=D0, maturity=_PILLARS[2],
        terms=ScheduleTerms(Frequency.SEMI_ANNUAL, DC.THIRTY_360),
    )
    priceable = discounting_priceable(bond, DiscountingEngine(), NUM)
    buckets = key_rate_dv01(priceable, market, NUM)
    assert [d for d, _ in buckets] == _PILLARS                       # one bucket per non-anchor pillar
    total = sum(kr for _, kr in buckets)
    assert total == pytest.approx(dv01(priceable, market, NUM), abs=1e-6)


def test_bucket_is_isolated_for_a_cashflow_on_a_pillar():
    curve = _boot_curve()
    market = MarketSnapshot(valuation_date=D0, discount_curve=curve)
    pillar = _PILLARS[1]                                             # cashflow exactly on the 2Y node
    notional = 1_000_000.0
    trade = FixedCashflow(Cashflow(date=pillar, amount=Money(notional, CCY)))
    priceable = discounting_priceable(trade, DiscountingEngine(), NUM)
    buckets = dict(key_rate_dv01(priceable, market, NUM))
    t = year_fraction(D0, pillar, ACT365)
    analytic = -notional * t * curve.df(pillar) * _ONE_BP
    assert buckets[pillar] == pytest.approx(analytic, abs=1e-4)      # bucket 2Y carries it
    for other in (_PILLARS[0], _PILLARS[2]):                         # neighbours ~ 0 (tent vanishes at node)
        assert buckets[other] == pytest.approx(0.0, abs=1e-6)
