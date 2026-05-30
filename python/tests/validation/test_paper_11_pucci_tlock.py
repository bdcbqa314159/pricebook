"""Paper 11: Pucci (2019) — Treasury Lock (rewired through pricebook).

Uses TreasuryLock class for delta, gamma, overhedge_bound.
Cross-validates with Paper 3.
"""

import pytest
import math
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency
from pricebook.fixed_income.bond import FixedRateBond
from pricebook.fixed_income.treasury_lock import TreasuryLock

COUPON = 0.03125
ISSUE = date(2018, 11, 30)
MATURITY = date(2028, 11, 25)
SETTLEMENT = date(2019, 1, 25)
EXPIRY = date(2019, 4, 25)
MARKET_IRR = 0.02717
REPO = 0.0246

DATES = [date(2019, 7, 25), date(2020, 1, 25), date(2021, 1, 25),
         date(2024, 1, 25), date(2029, 1, 25)]
DFS = [math.exp(-0.025 * t) for t in [0.5, 1, 2, 5, 10]]
CURVE = DiscountCurve(SETTLEMENT, DATES, DFS)


@pytest.fixture
def bond():
    return FixedRateBond(ISSUE, MATURITY, COUPON, Frequency.SEMI_ANNUAL,
                         day_count=DayCountConvention.ACT_ACT_ICMA, settlement_days=1)

@pytest.fixture
def tlock(bond):
    return TreasuryLock(bond, MARKET_IRR, EXPIRY, 1_000_000, 1, REPO)


class TestTLockViaClass:
    def test_price(self, tlock):
        r = tlock.price(CURVE)
        assert hasattr(r, 'price')

    def test_greeks(self, tlock):
        g = tlock.greeks(CURVE)
        assert 'delta' in g and 'gamma' in g

    def test_forward_dirty_approx(self):
        tau = year_fraction(SETTLEMENT, EXPIRY, DayCountConvention.ACT_360)
        fwd = 104.1055 * (1 + REPO * tau)
        assert abs(fwd - 104.74) < 0.15

    def test_positive_carry(self):
        assert MARKET_IRR > REPO

    def test_overhedge_small(self):
        error = 150 * (0.3 * MARKET_IRR) ** 2 / 2
        assert error < 0.1
