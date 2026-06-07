"""Tests for pricebook.equity.quanto_futures and pricebook.equity.etf."""

import math
import pytest
import numpy as np

from pricebook.equity.quanto_futures import (
    QuantoFuturesResult,
    CompoQuantoResult,
    quanto_futures_price,
    implied_correlation,
    compo_vs_quanto,
)
from pricebook.equity.etf import (
    ETFResult,
    ArbResult,
    etf_nav,
    creation_redemption_arb,
    tracking_error,
    leveraged_etf_decay,
)

# ---------------------------------------------------------------------------
# Common quanto parameters
# ---------------------------------------------------------------------------
QUANTO_BASE = dict(
    spot_index=1000.0,
    rate_domestic=0.03,
    rate_foreign=0.01,
    div_yield=0.02,
    fx_vol=0.10,
    index_vol=0.20,
    T=1.0,
)


# ---------------------------------------------------------------------------
# quanto_futures_price
# ---------------------------------------------------------------------------

def test_quanto_futures_zero_correlation_equals_standard_futures():
    """rho=0 → quanto adjustment = 1 → F_Q = S * exp((r_d - q) * T)."""
    result = quanto_futures_price(**QUANTO_BASE, correlation=0.0)
    expected = QUANTO_BASE["spot_index"] * math.exp(
        (QUANTO_BASE["rate_domestic"] - QUANTO_BASE["div_yield"]) * QUANTO_BASE["T"]
    )
    assert result.fair_value == pytest.approx(expected, rel=1e-10)


def test_quanto_futures_positive_corr_lower_than_zero_corr():
    """Positive correlation → negative quanto drift correction → lower price."""
    r0 = quanto_futures_price(**QUANTO_BASE, correlation=0.0)
    r_pos = quanto_futures_price(**QUANTO_BASE, correlation=0.5)
    assert r_pos.fair_value < r0.fair_value


def test_quanto_futures_returns_result_type():
    result = quanto_futures_price(**QUANTO_BASE, correlation=0.3)
    assert isinstance(result, QuantoFuturesResult)


def test_quanto_futures_quanto_adjustment_positive():
    result = quanto_futures_price(**QUANTO_BASE, correlation=0.3)
    assert result.quanto_adjustment > 0.0


# ---------------------------------------------------------------------------
# implied_correlation — roundtrip
# ---------------------------------------------------------------------------

def test_implied_correlation_roundtrip():
    """Price a quanto then back out correlation — should recover original."""
    rho_in = 0.35
    result = quanto_futures_price(**QUANTO_BASE, correlation=rho_in)
    rho_out = implied_correlation(
        quanto_price=result.fair_value,
        spot_index=QUANTO_BASE["spot_index"],
        rate_domestic=QUANTO_BASE["rate_domestic"],
        rate_foreign=QUANTO_BASE["rate_foreign"],
        div_yield=QUANTO_BASE["div_yield"],
        fx_vol=QUANTO_BASE["fx_vol"],
        index_vol=QUANTO_BASE["index_vol"],
        T=QUANTO_BASE["T"],
    )
    assert rho_out == pytest.approx(rho_in, abs=1e-10)


# ---------------------------------------------------------------------------
# compo_vs_quanto
# ---------------------------------------------------------------------------

def test_compo_vs_quanto_both_positive():
    result = compo_vs_quanto(**QUANTO_BASE, fx_spot=130.0, correlation=0.3)
    assert result.compo_forward > 0.0
    assert result.quanto_forward > 0.0


def test_compo_vs_quanto_returns_result_type():
    result = compo_vs_quanto(**QUANTO_BASE, fx_spot=130.0, correlation=0.3)
    assert isinstance(result, CompoQuantoResult)


def test_compo_vs_quanto_difference_sign_with_positive_corr():
    """Positive corr → quanto_forward < compo_forward (larger drift reduction)."""
    result = compo_vs_quanto(**QUANTO_BASE, fx_spot=130.0, correlation=0.5)
    assert result.difference < result.compo_forward   # quanto < compo


# ---------------------------------------------------------------------------
# etf_nav
# ---------------------------------------------------------------------------

def test_etf_nav_basic_calculation():
    holdings = [100.0, 200.0]
    prices = [50.0, 25.0]
    shares = 1000.0
    nav = etf_nav(holdings=holdings, prices=prices, shares_outstanding=shares)
    expected = (100.0 * 50.0 + 200.0 * 25.0) / 1000.0
    assert nav == pytest.approx(expected, rel=1e-10)


def test_etf_nav_with_cash_and_liabilities():
    holdings = [100.0]
    prices = [10.0]
    nav = etf_nav(holdings=holdings, prices=prices, shares_outstanding=100.0,
                  cash=50.0, liabilities=20.0)
    expected = (1000.0 + 50.0 - 20.0) / 100.0
    assert nav == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# creation_redemption_arb
# ---------------------------------------------------------------------------

def test_creation_redemption_arb_premium_creates():
    """ETF trading at premium → AP should create."""
    result = creation_redemption_arb(
        market_price=101.0, nav=100.0,
        creation_fee=0.10, redemption_fee=0.10, transaction_costs=0.20,
    )
    assert result.direction == "create"
    assert result.is_profitable is True


def test_creation_redemption_arb_discount_redeems():
    """ETF trading at discount → AP should redeem."""
    result = creation_redemption_arb(
        market_price=99.0, nav=100.0,
        creation_fee=0.10, redemption_fee=0.10, transaction_costs=0.20,
    )
    assert result.direction == "redeem"
    assert result.is_profitable is True


def test_creation_redemption_arb_small_spread_no_arb():
    """Spread within costs → no profitable arbitrage."""
    result = creation_redemption_arb(
        market_price=100.05, nav=100.0,
        creation_fee=0.10, redemption_fee=0.10, transaction_costs=0.20,
    )
    assert result.direction == "none"
    assert result.is_profitable is False


# ---------------------------------------------------------------------------
# tracking_error
# ---------------------------------------------------------------------------

def test_tracking_error_zero_for_identical_returns():
    rets = [0.01, -0.005, 0.02, 0.00, -0.01]
    te = tracking_error(rets, rets)
    assert te == pytest.approx(0.0, abs=1e-15)


def test_tracking_error_positive_for_different_returns():
    etf_rets = [0.01, -0.006, 0.019, 0.001, -0.011]
    idx_rets = [0.01, -0.005, 0.020, 0.000, -0.010]
    te = tracking_error(etf_rets, idx_rets)
    assert te > 0.0


# ---------------------------------------------------------------------------
# leveraged_etf_decay
# ---------------------------------------------------------------------------

def test_leveraged_etf_decay_one_x_no_decay():
    """1× leverage → no vol drag, ratio = exp(mu * T)."""
    mu = 0.0
    val = leveraged_etf_decay(daily_leverage=1.0, vol=0.01, T_days=252, mu=mu)
    assert val == pytest.approx(1.0, rel=1e-10)


def test_leveraged_etf_decay_two_x_below_square():
    """2× ETF < (index)^2 due to variance drag; compare ratio with vol=0."""
    no_drag = leveraged_etf_decay(daily_leverage=2.0, vol=0.0, T_days=252, mu=0.001)
    with_drag = leveraged_etf_decay(daily_leverage=2.0, vol=0.01, T_days=252, mu=0.001)
    assert with_drag < no_drag
