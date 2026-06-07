"""Tests for G2++ product pricers: callable bonds, CMS spread, callable FRN."""

from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.vasicek import G2PlusPlus
from pricebook.fixed_income.callable_bond_g2pp import callable_bond_g2pp, puttable_bond_g2pp
from pricebook.structured.cms_spread_g2pp import cms_spread_g2pp, cms_correlation_g2pp
from pricebook.fixed_income.callable_floater_g2pp import callable_frn_g2pp

REF = date(2024, 1, 15)
curve = DiscountCurve.flat(REF, 0.04)

g2pp = G2PlusPlus(a=0.05, b=0.08, sigma1=0.01, sigma2=0.008, rho=-0.7, curve=curve)

N_STEPS = 30       # small for tree speed
N_PATHS = 5000     # small for MC speed


# ---------------------------------------------------------------------------
# callable_bond_g2pp
# ---------------------------------------------------------------------------


def test_callable_bond_price_positive():
    """Callable bond price must be positive."""
    result = callable_bond_g2pp(
        g2pp,
        coupon_rate=0.05,
        maturity_years=10.0,
        call_dates_years=[5.0, 6.0, 7.0, 8.0, 9.0],
        n_steps=N_STEPS,
    )
    assert result.price > 0.0


def test_callable_bond_price_le_straight():
    """Callable bond price must not exceed the straight bond price (call hurts investor)."""
    result = callable_bond_g2pp(
        g2pp,
        coupon_rate=0.05,
        maturity_years=10.0,
        call_dates_years=[5.0, 6.0, 7.0, 8.0, 9.0],
        n_steps=N_STEPS,
    )
    assert result.price <= result.straight_price + 1e-8


# ---------------------------------------------------------------------------
# puttable_bond_g2pp
# ---------------------------------------------------------------------------


def test_puttable_bond_price_positive():
    """Puttable bond price must be positive."""
    result = puttable_bond_g2pp(
        g2pp,
        coupon_rate=0.05,
        maturity_years=10.0,
        put_dates_years=[5.0, 6.0, 7.0, 8.0, 9.0],
        n_steps=N_STEPS,
    )
    assert result.price > 0


# ---------------------------------------------------------------------------
# cms_spread_g2pp
# ---------------------------------------------------------------------------


def test_cms_spread_correlation_below_one():
    """Implied CMS correlation must be strictly < 1.0 under G2++ (key 2-factor test)."""
    result = cms_spread_g2pp(
        g2pp,
        cms_long_tenor=10.0,
        cms_short_tenor=2.0,
        T=1.0,
        n_paths=N_PATHS,
        seed=42,
    )
    assert result.correlation_implied < 1.0


def test_cms_spread_correlation_positive():
    """Implied CMS correlation should be positive for typical rate tenors."""
    result = cms_spread_g2pp(
        g2pp,
        cms_long_tenor=10.0,
        cms_short_tenor=2.0,
        T=1.0,
        n_paths=N_PATHS,
        seed=42,
    )
    assert result.correlation_implied > 0.0


# ---------------------------------------------------------------------------
# cms_correlation_g2pp
# ---------------------------------------------------------------------------


def test_cms_correlation_below_one():
    """cms_correlation_g2pp must return a value strictly less than 1.0."""
    corr = cms_correlation_g2pp(
        g2pp, tenor1=10.0, tenor2=2.0, T=1.0, n_paths=N_PATHS, seed=42,
    )
    assert corr < 1.0


def test_cms_correlation_positive():
    """cms_correlation_g2pp must return a positive correlation for 10Y vs 2Y."""
    corr = cms_correlation_g2pp(
        g2pp, tenor1=10.0, tenor2=2.0, T=1.0, n_paths=N_PATHS, seed=42,
    )
    assert corr > 0.0


# ---------------------------------------------------------------------------
# callable_frn_g2pp
# ---------------------------------------------------------------------------


def test_callable_frn_price_positive():
    """Callable FRN price must be positive."""
    result = callable_frn_g2pp(
        g2pp,
        maturity_years=5.0,
        spread=0.005,
        call_dates_years=[2.0, 3.0, 4.0],
        n_steps=N_STEPS,
    )
    assert result.price > 0.0


def test_callable_frn_option_value_nonneg():
    """Callable FRN embedded option value must be non-negative."""
    result = callable_frn_g2pp(
        g2pp,
        maturity_years=5.0,
        spread=0.005,
        call_dates_years=[2.0, 3.0, 4.0],
        n_steps=N_STEPS,
    )
    assert result.option_value >= 0.0


def test_callable_frn_price_le_straight():
    """Callable FRN price must not exceed straight FRN (issuer call hurts investor)."""
    result = callable_frn_g2pp(
        g2pp,
        maturity_years=5.0,
        spread=0.005,
        call_dates_years=[2.0, 3.0, 4.0],
        n_steps=N_STEPS,
    )
    assert result.price <= result.straight_price + 1e-8
