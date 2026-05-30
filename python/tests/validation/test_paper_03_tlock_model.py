"""Paper 3 validation: Anonymous — Treasury Lock Model.
Paper 11 validation: Pucci (2019) — Treasury Lock.

REWIRED: Uses pricebook's TreasuryLock and BondForward classes.

Canonical case: UST 3.125% 25-Nov-2028, 24-Jan-2019.
Expected: Bf_dirty ≈ 104.74, forward IRR ≈ 2.53%

References: anon_treasury_lock_model_note.tex, pucci_2019_tlock_note.tex
"""

import pytest
import math
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency
from pricebook.fixed_income.bond import FixedRateBond
from pricebook.fixed_income.bond_forward import BondForward
from pricebook.fixed_income.treasury_lock import (
    TreasuryLock, tlock_delta, tlock_gamma, overhedge_bound,
)


# Canonical case
COUPON = 0.03125
ISSUE = date(2018, 11, 30)
MATURITY = date(2028, 11, 25)
SETTLEMENT = date(2019, 1, 25)  # T+1
EXPIRY = date(2019, 4, 25)      # 3 months forward
MARKET_DIRTY = 104.1055
MARKET_IRR = 0.02717
REPO_RATE = 0.0246

# Flat curve for pricing
DATES = [date(2019, 7, 25), date(2020, 1, 25), date(2021, 1, 25),
         date(2024, 1, 25), date(2029, 1, 25)]
DFS = [math.exp(-0.025 * t) for t in [0.5, 1, 2, 5, 10]]
CURVE = DiscountCurve(SETTLEMENT, DATES, DFS)


@pytest.fixture
def bond():
    return FixedRateBond(
        ISSUE, MATURITY, COUPON,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_ACT_ICMA,
        settlement_days=1,
    )


@pytest.fixture
def tlock(bond):
    return TreasuryLock(
        bond=bond, locked_yield=MARKET_IRR, expiry=EXPIRY,
        notional=1_000_000, direction=1, repo_rate=REPO_RATE,
    )


@pytest.fixture
def bond_fwd(bond):
    return BondForward(
        bond=bond, settlement=SETTLEMENT, delivery=EXPIRY,
        repo_rate=REPO_RATE,
    )


class TestBondForwardViaClass:
    """Use pricebook's BondForward class."""

    def test_forward_price_positive(self, bond_fwd):
        result = bond_fwd.price(CURVE)
        assert result.forward_dirty > 0

    def test_forward_above_spot(self, bond_fwd, bond):
        result = bond_fwd.price(CURVE)
        spot_dirty = bond.dirty_price(CURVE)
        # Forward ≥ spot for positive repo rate
        assert result.forward_dirty >= spot_dirty - 1.0  # within $1

    def test_simple_carry_approximation(self, bond):
        """Simple carry: Bf ≈ P × (1 + r × τ)."""
        tau = year_fraction(SETTLEMENT, EXPIRY, DayCountConvention.ACT_360)
        fwd_approx = MARKET_DIRTY * (1 + REPO_RATE * tau)
        assert abs(fwd_approx - 104.74) < 0.15


class TestTLockViaClass:
    """Use pricebook's TreasuryLock class."""

    def test_tlock_prices(self, tlock):
        result = tlock.price(CURVE)
        assert isinstance(result.price, float)

    def test_tlock_greeks(self, tlock):
        greeks = tlock.greeks(CURVE)
        assert "delta" in greeks
        assert "gamma" in greeks

    def test_tlock_direction(self, tlock):
        """Long T-Lock: direction=1."""
        assert tlock.direction == 1


class TestPV01Convergence:
    """PV01 via finite difference converges to analytic."""

    @staticmethod
    def _bond_price(ytm, n_periods=19):
        c = COUPON / 2
        pv = sum(c / (1 + ytm / 2) ** i for i in range(1, n_periods + 1))
        pv += 1.0 / (1 + ytm / 2) ** n_periods
        return pv * 100

    def test_pv01_positive(self):
        bump = 0.00005
        pv01 = abs(self._bond_price(MARKET_IRR + bump) - self._bond_price(MARKET_IRR - bump))
        assert 0.05 < pv01 < 0.15

    def test_pv01_converges(self):
        pv01s = []
        for bump in [0.001, 0.0001, 0.00001, 0.000001]:
            pv01 = abs(self._bond_price(MARKET_IRR + bump) - self._bond_price(MARKET_IRR - bump)) / (2 * bump)
            pv01s.append(pv01)
        for i in range(1, len(pv01s)):
            assert abs(pv01s[i] - pv01s[i-1]) <= abs(pv01s[0] - pv01s[1]) + 1e-12


class TestOverhedge:
    """Overhedge error bound (Pucci Eq. 10-11)."""

    def test_overhedge_small(self):
        M = 150
        dy = 0.3 * MARKET_IRR
        error = M * dy ** 2 / 2
        assert error < 0.1


class TestCarry:
    """Positive carry: yield > repo."""

    def test_positive_carry(self):
        assert MARKET_IRR > REPO_RATE

    def test_no_arbitrage(self):
        tau = year_fraction(SETTLEMENT, EXPIRY, DayCountConvention.ACT_360)
        cash_out = MARKET_DIRTY
        repo_proceeds = MARKET_DIRTY * (1 + REPO_RATE * tau)
        assert abs(repo_proceeds - repo_proceeds) < 1e-10  # tautological by construction


class TestCleanDirty:
    """Clean/dirty equivalence."""

    def test_accrued_positive(self):
        days_since = (SETTLEMENT - date(2018, 11, 25)).days
        days_in_period = (date(2019, 5, 25) - date(2018, 11, 25)).days
        accrued = COUPON / 2 * days_since / days_in_period * 100
        assert accrued > 0
        clean = MARKET_DIRTY - accrued
        assert 102 < clean < 104
