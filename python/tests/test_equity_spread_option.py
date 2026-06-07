"""Tests for pricebook.equity.equity_spread_option."""

import math
import pytest

from pricebook.equity.equity_spread_option import (
    bjerksund_stensland_spread,
    kirk_equity_spread,
    mc_spread_option,
    outperformance_option,
    relative_performance_option,
)

# Common market parameters
S1, S2 = 110.0, 100.0
VOL1, VOL2, RHO = 0.25, 0.20, 0.40
T, R = 1.0, 0.05
Q1, Q2 = 0.01, 0.02


# ---------------------------------------------------------------------------
# kirk_equity_spread
# ---------------------------------------------------------------------------

def test_kirk_price_positive():
    """Kirk spread option call price is always non-negative."""
    res = kirk_equity_spread(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R, q1=Q1, q2=Q2)
    assert res.price >= 0.0


def test_kirk_itm_call_positive():
    """Call price is strictly positive when forward(S1) > forward(S2) + K."""
    # S1 >> S2 + K → deep ITM call
    res = kirk_equity_spread(200.0, 100.0, K=10.0, vol1=0.20, vol2=0.20, rho=0.5, T=1.0, r=0.0)
    assert res.price > 0.0


def test_kirk_call_put_price_sign():
    """Put price is non-negative; call is non-negative."""
    call = kirk_equity_spread(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R, q1=Q1, q2=Q2)
    put = kirk_equity_spread(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R, q1=Q1, q2=Q2,
                              option_type="put")
    assert call.price >= 0.0
    assert put.price >= 0.0


def test_higher_correlation_lower_spread_call():
    """Increasing correlation reduces spread option call price (correlation → convergence)."""
    res_low = kirk_equity_spread(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=0.0, T=T, r=R)
    res_high = kirk_equity_spread(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=0.90, T=T, r=R)
    assert res_low.price > res_high.price


# ---------------------------------------------------------------------------
# bjerksund_stensland_spread
# ---------------------------------------------------------------------------

def test_bs_spread_positive():
    """Bjerksund-Stensland spread option price is non-negative."""
    res = bjerksund_stensland_spread(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R)
    assert res.price >= 0.0


def test_bs_spread_close_to_kirk_moderate_k():
    """Bjerksund-Stensland and Kirk should agree closely for moderate strike."""
    kirk = kirk_equity_spread(S1, S2, K=2.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R)
    bs = bjerksund_stensland_spread(S1, S2, K=2.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R)
    assert bs.price == pytest.approx(kirk.price, rel=0.05)


# ---------------------------------------------------------------------------
# mc_spread_option
# ---------------------------------------------------------------------------

def test_mc_spread_close_to_kirk():
    """MC price should be close to Kirk's closed form for large path count."""
    kirk = kirk_equity_spread(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R, q1=Q1, q2=Q2)
    mc = mc_spread_option(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R, q1=Q1, q2=Q2,
                           n_paths=200_000, seed=7)
    assert mc.price == pytest.approx(kirk.price, rel=0.05)


def test_mc_price_non_negative():
    """Monte Carlo price is always non-negative."""
    mc = mc_spread_option(S1, S2, K=5.0, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R, n_paths=50_000)
    assert mc.price >= 0.0


# ---------------------------------------------------------------------------
# outperformance_option
# ---------------------------------------------------------------------------

def test_outperformance_matches_margrabe():
    """outperformance_option (K=0) should match Margrabe formula manually."""
    sigma = math.sqrt(VOL1**2 - 2.0 * RHO * VOL1 * VOL2 + VOL2**2)
    F1 = S1 * math.exp((R - Q1) * T)
    F2 = S2 * math.exp((R - Q2) * T)
    df = math.exp(-R * T)
    sqrt_t = math.sqrt(T)
    from scipy.stats import norm
    d1 = (math.log(F1 / F2) + 0.5 * sigma**2 * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    expected = df * (F1 * norm.cdf(d1) - F2 * norm.cdf(d2))

    res = outperformance_option(S1, S2, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R, q1=Q1, q2=Q2)
    assert res.price == pytest.approx(expected, rel=1e-6)


def test_outperformance_put_call_relationship():
    """call - put = df*(F1 - F2) for outperformance (Margrabe parity)."""
    call_res = outperformance_option(S1, S2, vol1=VOL1, vol2=VOL2, rho=RHO, T=T, r=R, q1=Q1, q2=Q2)
    # Mirror put: swap asset 1 and 2 in a call gives the put on the original
    put_res = outperformance_option(S2, S1, vol1=VOL2, vol2=VOL1, rho=RHO, T=T, r=R, q1=Q2, q2=Q1)
    df = math.exp(-R * T)
    F1 = S1 * math.exp((R - Q1) * T)
    F2 = S2 * math.exp((R - Q2) * T)
    # call - put = df*(F1 - F2)
    assert (call_res.price - put_res.price) == pytest.approx(df * (F1 - F2), rel=1e-4)


# ---------------------------------------------------------------------------
# relative_performance_option
# ---------------------------------------------------------------------------

def test_relative_performance_positive():
    """Relative performance option price is non-negative."""
    res = relative_performance_option(S1, S2, K_pct=0.05, vol1=VOL1, vol2=VOL2,
                                      rho=RHO, T=T, r=R, q1=Q1, q2=Q2)
    assert res.price >= 0.0


def test_relative_performance_call_positive_when_s1_outperforms():
    """Call price is positive when asset 1 has higher expected return than asset 2."""
    # q1 < q2 means forward(S1) > forward(S2); spread call should be positive
    res = relative_performance_option(1.0, 1.0, K_pct=0.0, vol1=0.20, vol2=0.20,
                                      rho=0.5, T=1.0, r=R, q1=0.01, q2=0.04)
    assert res.price > 0.0
