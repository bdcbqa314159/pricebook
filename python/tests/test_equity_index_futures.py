"""Tests for pricebook.equity.equity_index_futures."""

import math
import pytest

from pricebook.equity.equity_index_futures import (
    fair_value_table,
    implied_dividend_yield,
    implied_repo_rate,
    index_futures_fair_value,
    index_futures_roll,
)


# ---------------------------------------------------------------------------
# index_futures_fair_value
# ---------------------------------------------------------------------------

def test_fair_value_above_spot_when_r_gt_q():
    """Fair value > spot when risk-free rate exceeds dividend yield."""
    res = index_futures_fair_value(spot=5000.0, rate=0.05, div_yield=0.02, T=0.25)
    assert res.fair_value > 5000.0


def test_basis_positive_when_r_gt_q():
    """Basis (fair_value - spot) is positive when r > q."""
    res = index_futures_fair_value(spot=5000.0, rate=0.05, div_yield=0.02, T=0.25)
    assert res.basis > 0.0
    assert res.basis_bps > 0.0


def test_zero_carry_fair_value_approx_spot():
    """When r == q the fair value should equal the spot (zero net carry)."""
    res = index_futures_fair_value(spot=4000.0, rate=0.03, div_yield=0.03, T=0.5)
    assert res.fair_value == pytest.approx(4000.0, rel=1e-9)
    assert res.basis == pytest.approx(0.0, abs=1e-9)


def test_cost_of_carry_rate_correct():
    """net_carry = r - q + borrow_cost is stored correctly."""
    res = index_futures_fair_value(spot=3000.0, rate=0.05, div_yield=0.02, T=1.0, borrow_cost=0.005)
    assert res.cost_of_carry_rate == pytest.approx(0.035, rel=1e-9)


def test_fair_value_formula():
    """Manual verification: F = S * exp((r - q) * T)."""
    spot, r, q, T = 1000.0, 0.04, 0.01, 1.0
    res = index_futures_fair_value(spot=spot, rate=r, div_yield=q, T=T)
    expected = spot * math.exp((r - q) * T)
    assert res.fair_value == pytest.approx(expected, rel=1e-10)


def test_days_to_expiry():
    """Days to expiry is T * 365."""
    res = index_futures_fair_value(spot=5000.0, rate=0.05, div_yield=0.02, T=0.5)
    assert res.days_to_expiry == pytest.approx(182.5, rel=1e-9)


# ---------------------------------------------------------------------------
# index_futures_roll
# ---------------------------------------------------------------------------

def test_roll_calendar_spread_is_back_minus_front():
    """Calendar spread = back_price - front_price."""
    front, back = 5010.0, 5030.0
    res = index_futures_roll(front_price=front, back_price=back, spot=5000.0,
                              T_front=0.25, T_back=0.5)
    assert res.calendar_spread == pytest.approx(back - front, rel=1e-9)


def test_roll_cost_sign():
    """Roll cost = front - back; positive when front > back (backwardation)."""
    res = index_futures_roll(front_price=5050.0, back_price=5030.0, spot=5000.0,
                              T_front=0.25, T_back=0.5)
    assert res.roll_cost > 0.0


def test_roll_implied_repo_roundtrip():
    """Implied repo from roll should match the carry rate used to build prices."""
    spot, r = 5000.0, 0.04
    T_front, T_back = 0.25, 0.5
    front = spot * math.exp(r * T_front)
    back = spot * math.exp(r * T_back)
    res = index_futures_roll(front_price=front, back_price=back, spot=spot,
                              T_front=T_front, T_back=T_back)
    assert res.implied_repo == pytest.approx(r, rel=1e-6)


# ---------------------------------------------------------------------------
# implied_dividend_yield
# ---------------------------------------------------------------------------

def test_implied_dividend_yield_roundtrip():
    """Back out q from fair value should recover the original dividend yield."""
    spot, r, q, T = 5000.0, 0.05, 0.025, 0.5
    res = index_futures_fair_value(spot=spot, rate=r, div_yield=q, T=T)
    q_impl = implied_dividend_yield(futures_price=res.fair_value, spot=spot, rate=r, T=T)
    assert q_impl == pytest.approx(q, rel=1e-9)


# ---------------------------------------------------------------------------
# implied_repo_rate
# ---------------------------------------------------------------------------

def test_implied_repo_rate_roundtrip():
    """Back out r from fair value should recover the original risk-free rate."""
    spot, r, q, T = 4000.0, 0.04, 0.015, 1.0
    res = index_futures_fair_value(spot=spot, rate=r, div_yield=q, T=T)
    r_impl = implied_repo_rate(futures_price=res.fair_value, spot=spot, div_yield=q, T=T)
    assert r_impl == pytest.approx(r, rel=1e-9)


# ---------------------------------------------------------------------------
# fair_value_table
# ---------------------------------------------------------------------------

def test_fair_value_table_entry_count():
    """Table returns exactly as many rows as expiries supplied."""
    expiries = [0.25, 0.5, 0.75, 1.0, 2.0]
    rows = fair_value_table(spot=5000.0, rate=0.05, div_yield=0.02, expiries_years=expiries)
    assert len(rows) == len(expiries)


def test_fair_value_table_values_increase_with_T():
    """When r > q, fair values must be strictly increasing with tenor."""
    expiries = [0.25, 0.5, 1.0, 2.0]
    rows = fair_value_table(spot=5000.0, rate=0.05, div_yield=0.01, expiries_years=expiries)
    fvs = [row["fair_value"] for row in rows]
    assert all(fvs[i] < fvs[i + 1] for i in range(len(fvs) - 1))


def test_fair_value_table_contains_T_field():
    """Each row must contain the tenor T used to compute it."""
    expiries = [0.5, 1.0]
    rows = fair_value_table(spot=5000.0, rate=0.04, div_yield=0.02, expiries_years=expiries)
    for row, T in zip(rows, expiries):
        assert row["T"] == pytest.approx(T)
