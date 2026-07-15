"""Netting-set SA-CCR oracle (L5 risk & capital).

Aggregates a netting set of IR swaps the Basel way: each trade's *signed* effective
notional `D = delta * notional * SD * MF` (delta = +1 payer / -1 receiver) is bucketed
by end maturity (<1y, 1-5y, >5y), summed within buckets, and combined across buckets
with the supervisory correlations —

    EffNotional = sqrt(D1^2 + D2^2 + D3^2 + 1.4(D1 D2 + D2 D3) + 0.6 D1 D3)

then AddOn = SF * EffNotional, RC = max(sum of marks, 0), EAD = alpha*(RC + mult*AddOn).

Oracles: (1) a one-trade set equals the single-trade `saccr_ead`; (2) a payer + its
mirror receiver perfectly net — signed notionals cancel and marks cancel, EAD = 0;
(3) a two-bucket set matches the hand-computed correlation aggregation; (4) netting is
sub-additive — a diversified set has EAD below the sum of standalone EADs.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.risk.saccr import netting_set_ead, saccr_ead

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
USD = Currency.USD


def _terms(pay_fixed):
    leg = ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360)
    return SwapTerms(fixed_schedule=leg, float_schedule=leg, pay_fixed=pay_fixed)


def _swap(notional, maturity, pay_fixed=True):
    return vanilla_swap(Money(notional, USD), 0.03, D0, maturity, _terms(pay_fixed))


def _magnitude(notional, end_years):
    sd = (1.0 - math.exp(-0.05 * end_years)) / 0.05  # S=0 (spot)
    return notional * sd * math.sqrt(min(end_years, 1.0))


def test_single_trade_set_equals_saccr_ead():
    swap = _swap(100_000_000.0, date(2036, 1, 15))
    assert netting_set_ead([(swap, 5_000_000.0)], D0) == pytest.approx(
        saccr_ead(swap, 5_000_000.0, D0), abs=1e-6
    )


def test_payer_receiver_mirror_perfectly_nets_to_zero():
    payer = _swap(100_000_000.0, date(2036, 1, 15), pay_fixed=True)
    receiver = _swap(100_000_000.0, date(2036, 1, 15), pay_fixed=False)
    # opposite positions: signed notionals cancel (AddOn 0) and marks cancel (RC 0)
    assert netting_set_ead([(payer, 5_000_000.0), (receiver, -5_000_000.0)], D0) == pytest.approx(
        0.0, abs=1e-6
    )


def test_two_bucket_correlation_aggregation():
    short = _swap(100_000_000.0, date(2026, 10, 15))  # ~0.75y -> bucket 1 (<1y)
    long = _swap(100_000_000.0, date(2036, 1, 15))    # ~10y   -> bucket 3 (>5y)
    e_short = year_fraction(D0, short.float_leg.schedule[-1], ACT365)
    e_long = year_fraction(D0, long.float_leg.schedule[-1], ACT365)
    d1, d3 = _magnitude(100_000_000.0, e_short), _magnitude(100_000_000.0, e_long)
    eff = math.sqrt(d1**2 + d3**2 + 0.6 * d1 * d3)     # buckets 1 & 3: correlation 30%
    expected = 1.4 * 0.005 * eff                        # mark 0 -> RC 0, multiplier 1
    assert netting_set_ead([(short, 0.0), (long, 0.0)], D0) == pytest.approx(expected, abs=1e-3)


def test_netting_is_sub_additive():
    short = _swap(100_000_000.0, date(2026, 10, 15))
    long = _swap(100_000_000.0, date(2036, 1, 15))
    combined = netting_set_ead([(short, 0.0), (long, 0.0)], D0)
    standalone = saccr_ead(short, 0.0, D0) + saccr_ead(long, 0.0, D0)
    assert combined < standalone            # cross-bucket correlation < 1 diversifies
    assert combined > saccr_ead(long, 0.0, D0)  # but more than the larger leg alone
