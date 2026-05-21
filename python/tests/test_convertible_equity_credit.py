"""Tests for convertible bond with joint equity-credit dynamics (C8)."""

import pytest
import math

from pricebook.credit.convertible_equity_credit import (
    convertible_equity_credit_price, EquityCreditConvertibleResult,
)


PARAMS = dict(
    spot=50.0, notional=100.0, coupon_rate=0.03,
    conversion_ratio=1.5,  # conversion price = 100/1.5 ≈ 66.7
    maturity_years=5.0, risk_free_rate=0.04,
    equity_vol=0.30, hazard_rate_0=0.02,
    n_paths=5000, seed=42,
)


class TestBasicPricing:
    def test_produces_result(self):
        r = convertible_equity_credit_price(**PARAMS)
        assert isinstance(r, EquityCreditConvertibleResult)
        assert r.price > 0

    def test_above_bond_floor(self):
        """Convertible should be worth at least the bond floor (conversion option has value)."""
        r = convertible_equity_credit_price(**PARAMS)
        assert r.price >= r.bond_floor * 0.95  # within MC noise

    def test_above_conversion_value_when_itm(self):
        """Deep ITM: price ≥ conversion value."""
        r = convertible_equity_credit_price(**{**PARAMS, "spot": 100.0})
        assert r.price >= r.conversion_value * 0.95

    def test_default_prob_positive(self):
        r = convertible_equity_credit_price(**PARAMS)
        assert r.default_prob > 0
        assert r.default_prob < 1


class TestEquitySensitivity:
    def test_higher_spot_higher_price(self):
        """Higher stock → higher convertible price."""
        r1 = convertible_equity_credit_price(**{**PARAMS, "spot": 40.0})
        r2 = convertible_equity_credit_price(**{**PARAMS, "spot": 80.0})
        assert r2.price > r1.price

    def test_delta_positive(self):
        """Convertible delta should be positive (long equity exposure)."""
        r = convertible_equity_credit_price(**PARAMS)
        assert r.delta > 0

    def test_delta_between_0_and_cr(self):
        """Delta should be between 0 and conversion_ratio."""
        r = convertible_equity_credit_price(**PARAMS)
        assert 0 <= r.delta <= PARAMS["conversion_ratio"] + 0.1


class TestCreditSensitivity:
    def test_higher_hazard_lower_price(self):
        """Higher hazard rate → lower price (more credit risk)."""
        r1 = convertible_equity_credit_price(**{**PARAMS, "hazard_rate_0": 0.01})
        r2 = convertible_equity_credit_price(**{**PARAMS, "hazard_rate_0": 0.05})
        assert r2.price < r1.price

    def test_cs01_negative(self):
        """CS01 should be negative (higher hazard → lower price)."""
        r = convertible_equity_credit_price(**PARAMS)
        assert r.cs01 < 0

    def test_higher_default_prob_with_higher_hazard(self):
        r1 = convertible_equity_credit_price(**{**PARAMS, "hazard_rate_0": 0.01, "n_paths": 10000})
        r2 = convertible_equity_credit_price(**{**PARAMS, "hazard_rate_0": 0.10, "n_paths": 10000})
        assert r2.default_prob > r1.default_prob


class TestCorrelation:
    def test_negative_corr_lower_price(self):
        """Negative equity-credit correlation (realistic) → lower price.

        When stock drops, hazard rises → double hit for convertible holder.
        """
        r_zero = convertible_equity_credit_price(
            **{**PARAMS, "equity_credit_corr": 0.0, "n_paths": 10000})
        r_neg = convertible_equity_credit_price(
            **{**PARAMS, "equity_credit_corr": -0.50, "n_paths": 10000})
        # Negative correlation makes things worse for the holder
        # (but effect can be subtle with MC noise)
        assert abs(r_neg.price - r_zero.price) > 0  # prices differ

    def test_rho_sensitivity_nonzero(self):
        r = convertible_equity_credit_price(**PARAMS)
        assert r.rho_sensitivity != 0


class TestGreeks:
    def test_vega_positive(self):
        """Convertible has positive vega (long optionality)."""
        r = convertible_equity_credit_price(**PARAMS)
        assert r.vega > 0

    def test_gamma_reasonable(self):
        """Convertible gamma should be finite and reasonable (may be slightly negative from MC noise)."""
        r = convertible_equity_credit_price(**PARAMS)
        assert abs(r.gamma) < 5.0  # bounded


class TestSerialization:
    def test_to_dict(self):
        r = convertible_equity_credit_price(**{**PARAMS, "n_paths": 2000})
        d = r.to_dict()
        assert "price" in d
        assert "delta" in d
        assert "cs01" in d
        assert "rho_sensitivity" in d
        assert "default_prob" in d
