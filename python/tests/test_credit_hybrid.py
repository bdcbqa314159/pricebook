"""Tests for hybrid credit-rates models."""

import math

import pytest

from pricebook.credit_hybrid import (
    CallableRiskyBondResult,
    ConvertibleBondResult,
    FloatingCLNResult,
    callable_risky_bond,
    convertible_bond,
    floating_cln,
)


# ---- Callable risky bond ----

class TestCallableRiskyBond:
    def test_callable_less_than_non_callable(self):
        """Callable risky ≤ non-callable risky (call benefits issuer)."""
        # High coupon + low call price → call is in-the-money
        result = callable_risky_bond(
            100, 0.08, 10, call_price=95, call_start_year=2,
            flat_hazard=0.005, rate_vol=0.015,
        )
        assert result.price <= result.non_callable_price + 0.01

    def test_call_value_non_negative(self):
        result = callable_risky_bond(100, 0.08, 10, 95, 2,
                                      flat_hazard=0.005, rate_vol=0.015)
        assert result.call_value >= -0.01

    def test_oas_non_negative(self):
        result = callable_risky_bond(100, 0.08, 10, 95, 2,
                                      flat_hazard=0.005, rate_vol=0.015)
        assert result.oas >= -0.01

    def test_high_call_price_small_call_value(self):
        """Very high call price → call never exercised → call value ≈ 0."""
        result = callable_risky_bond(100, 0.05, 10, 200, 1)
        assert result.call_value < 1  # essentially zero

    def test_positive_price(self):
        result = callable_risky_bond(100, 0.04, 5, 101, 2)
        assert result.price > 0


# ---- Floating CLN ----

class TestFloatingCLN:
    def test_deterministic_positive_price(self):
        result = floating_cln(100, spread=0.02, maturity_years=5)
        assert result.price > 0

    def test_par_spread_positive(self):
        result = floating_cln(100, 0.02, 5)
        assert result.par_spread > 0

    def test_at_par_spread_price_near_100(self):
        """At par spread, price ≈ 100."""
        result = floating_cln(100, 0.02, 5)
        # Price at par spread
        result2 = floating_cln(100, result.par_spread, 5)
        assert result2.price == pytest.approx(100.0, rel=0.05)

    def test_higher_hazard_lower_price(self):
        low = floating_cln(100, 0.02, 5, flat_hazard=0.01)
        high = floating_cln(100, 0.02, 5, flat_hazard=0.05)
        assert high.price < low.price

    def test_stochastic_hazard_produces_price(self):
        """MC with stochastic hazard should produce a positive price."""
        result = floating_cln(100, 0.02, 3, hazard_vol=0.1,
                               n_paths=10_000, seed=42)
        assert result.price > 0


# ---- Convertible bond ----

class TestConvertibleBond:
    def test_price_exceeds_bond_floor(self):
        """Convertible ≥ bond floor (conversion adds value)."""
        result = convertible_bond(
            100, 0.03, 5, conversion_ratio=1.0, spot=100,
            n_paths=20_000, seed=42,
        )
        assert result.price >= result.bond_floor * 0.95

    def test_price_exceeds_conversion_value(self):
        """Convertible ≥ conversion value (bond floor adds value)."""
        result = convertible_bond(
            100, 0.03, 5, conversion_ratio=1.0, spot=100,
            n_paths=20_000, seed=42,
        )
        assert result.price >= result.conversion_value * 0.95

    def test_high_spot_converts(self):
        """With very high spot, convertible ≈ equity (conversion dominates)."""
        result = convertible_bond(
            100, 0.03, 5, conversion_ratio=1.0, spot=200,
            n_paths=20_000, seed=42,
        )
        assert result.conversion_value > result.bond_floor

    def test_low_spot_bond_floor(self):
        """With very low spot, convertible ≈ risky bond (no conversion)."""
        result = convertible_bond(
            100, 0.03, 5, conversion_ratio=1.0, spot=30,
            n_paths=20_000, seed=42,
        )
        assert result.bond_floor > result.conversion_value

    def test_positive_price(self):
        result = convertible_bond(100, 0.04, 3, 1.0, 100,
                                   n_paths=10_000, seed=42)
        assert result.price > 0

    def test_higher_vol_higher_price(self):
        """More equity vol → conversion option worth more."""
        low = convertible_bond(100, 0.03, 5, 1.0, 100, equity_vol=0.10,
                                n_paths=20_000, seed=42)
        high = convertible_bond(100, 0.03, 5, 1.0, 100, equity_vol=0.50,
                                 n_paths=20_000, seed=42)
        assert high.price >= low.price * 0.95
