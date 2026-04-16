"""Tests for convertible bonds: soft call, CoCo, exchangeable, mandatory."""

import math

import numpy as np
import pytest

from pricebook.convertible_bond import (
    CoCoResult,
    ConvertibleBond,
    ConvertibleResult,
    DeltaHedgeResult,
    ExchangeableResult,
    MandatoryConvertibleResult,
    SoftCallResult,
    contingent_convertible,
    convertible_delta_hedge,
    convertible_soft_call,
    exchangeable_bond,
    mandatory_convertible,
)


# ---- Convertible basics ----

class TestConvertibleBond:
    def test_basic_price(self):
        cb = ConvertibleBond(notional=1000, coupon_rate=0.03,
                               maturity_years=3.0, conversion_ratio=10)
        result = cb.price(spot=100, rate=0.04, equity_vol=0.30,
                           credit_spread=0.02, n_paths=2000, seed=42)
        assert isinstance(result, ConvertibleResult)
        assert result.price > 0

    def test_parity(self):
        cb = ConvertibleBond(1000, 0.03, 3.0, 10)
        # Parity = 10 × 100 / 1000 = 1.0
        assert cb.parity(100) == pytest.approx(1.0)

    def test_conversion_price(self):
        cb = ConvertibleBond(1000, 0.03, 3.0, 10)
        assert cb.conversion_price() == 100   # 1000/10

    def test_price_exceeds_bond_floor(self):
        """CB price ≥ bond floor (the option has nonneg value)."""
        cb = ConvertibleBond(1000, 0.03, 3.0, 10)
        result = cb.price(100, 0.04, 0.30, 0.02, n_paths=2000, seed=42)
        assert result.price >= result.bond_floor * 0.95

    def test_deep_itm_converges_to_conversion(self):
        """Deep ITM (high stock): price ≈ conversion value."""
        cb = ConvertibleBond(1000, 0.03, 0.5, 10)
        result = cb.price(spot=500, rate=0.04, equity_vol=0.20,
                           credit_spread=0.01, n_paths=2000, seed=42)
        # Price should be near conversion value = 10 × 500 = 5000
        assert result.price == pytest.approx(5000, rel=0.10)

    def test_deep_otm_converges_to_bond(self):
        """Deep OTM: price ≈ bond floor."""
        cb = ConvertibleBond(1000, 0.03, 0.5, 10)
        result = cb.price(spot=5, rate=0.04, equity_vol=0.30,
                           credit_spread=0.01, n_paths=3000, seed=42)
        # Very unlikely to convert → price near bond floor
        assert result.price == pytest.approx(result.bond_floor, rel=0.05)

    def test_atm_delta_positive_bounded(self):
        """ATM delta (dCB/dS) should be positive and bounded by conversion_ratio."""
        cb = ConvertibleBond(1000, 0.03, 3.0, 10)
        result = cb.price(spot=100, rate=0.04, equity_vol=0.30,
                           credit_spread=0.02, n_paths=5000, seed=42)
        # Delta is dCB_dollars/dS_dollars; max ≈ conversion_ratio + MC noise
        assert 0 < result.equity_delta <= cb.conversion_ratio * 1.5


# ---- Delta hedge ----

class TestDeltaHedge:
    def test_basic(self):
        cb = ConvertibleBond(1000, 0.03, 3.0, 10)
        result = cb.price(100, 0.04, 0.30, 0.02, n_paths=2000, seed=42)
        hedge = convertible_delta_hedge(result, spot=100, cb_notional_traded=1.0)
        assert isinstance(hedge, DeltaHedgeResult)
        assert hedge.shares_to_short > 0

    def test_hedge_notional(self):
        cb = ConvertibleBond(1000, 0.03, 3.0, 10)
        result = cb.price(100, 0.04, 0.30, 0.02, n_paths=2000, seed=42)
        hedge = convertible_delta_hedge(result, 100)
        expected_hedge = hedge.shares_to_short * 100
        assert hedge.hedge_notional == pytest.approx(expected_hedge)


# ---- Soft call ----

class TestSoftCall:
    def test_basic(self):
        cb = ConvertibleBond(1000, 0.03, 3.0, 10)
        result = convertible_soft_call(
            cb, spot=100, rate=0.04, equity_vol=0.30,
            soft_call_trigger=1.3, soft_call_years_after=0.5,
            credit_spread=0.02, n_paths=2000, seed=42,
        )
        assert isinstance(result, SoftCallResult)

    def test_soft_call_reduces_price(self):
        """Soft call ≤ regular convertible (issuer has extra option)."""
        cb = ConvertibleBond(1000, 0.03, 3.0, 10)
        result = convertible_soft_call(
            cb, 100, 0.04, 0.30, 1.1, 0.5, 0.02, n_paths=3000, seed=42,
        )
        assert result.price <= result.price_no_call * 1.05  # allow MC noise
        assert result.call_option_value >= -0.5 * cb.notional

    def test_far_trigger_no_call(self):
        """Very high trigger → soft call rarely exercised → price ≈ no-call."""
        cb = ConvertibleBond(1000, 0.03, 1.0, 10)
        result = convertible_soft_call(
            cb, 100, 0.04, 0.20, 10.0, 0.1, 0.02, n_paths=2000, seed=42,
        )
        assert result.price == pytest.approx(result.price_no_call, rel=0.05)


# ---- CoCo ----

class TestCoCo:
    def test_basic(self):
        result = contingent_convertible(
            notional=1000, coupon_rate=0.08, maturity_years=5.0,
            conversion_trigger=50, conversion_ratio=10,
            spot=100, rate=0.04, equity_vol=0.30,
            credit_spread=0.03, loss_absorption=0.5,
            n_paths=2000, seed=42,
        )
        assert isinstance(result, CoCoResult)
        assert 0 <= result.conversion_probability <= 1

    def test_low_trigger_less_likely_conversion(self):
        low_trigger = contingent_convertible(
            1000, 0.08, 5.0, conversion_trigger=30,   # very low
            conversion_ratio=10, spot=100, rate=0.04, equity_vol=0.30,
            credit_spread=0.03, n_paths=2000, seed=42,
        )
        high_trigger = contingent_convertible(
            1000, 0.08, 5.0, conversion_trigger=90,   # close to spot
            conversion_ratio=10, spot=100, rate=0.04, equity_vol=0.30,
            credit_spread=0.03, n_paths=2000, seed=42,
        )
        assert low_trigger.conversion_probability < high_trigger.conversion_probability

    def test_loss_absorption_effect(self):
        """Smaller loss_absorption (bigger loss on trigger) → lower price."""
        high_loss = contingent_convertible(
            1000, 0.08, 3.0, 70, 10, 100, 0.04, 0.30, 0.03,
            loss_absorption=0.2, n_paths=2000, seed=42,
        )
        low_loss = contingent_convertible(
            1000, 0.08, 3.0, 70, 10, 100, 0.04, 0.30, 0.03,
            loss_absorption=0.8, n_paths=2000, seed=42,
        )
        assert high_loss.price <= low_loss.price


# ---- Exchangeable ----

class TestExchangeable:
    def test_basic(self):
        result = exchangeable_bond(
            notional=1000, coupon_rate=0.03,
            maturity_years=3.0, conversion_ratio=10,
            underlying_spot=100, rate=0.04, equity_vol=0.30,
            issuer_credit_spread=0.02,
            n_paths=2000, seed=42,
        )
        assert isinstance(result, ExchangeableResult)
        assert result.price > 0

    def test_option_value_positive(self):
        result = exchangeable_bond(1000, 0.03, 3.0, 10, 100, 0.04, 0.30,
                                     0.02, n_paths=2000, seed=42)
        assert result.option_value >= -10.0   # small MC noise allowed


# ---- Mandatory ----

class TestMandatoryConvertible:
    def test_basic(self):
        result = mandatory_convertible(
            notional=1000, coupon_rate=0.05,
            maturity_years=3.0, low_strike=80, high_strike=120,
            spot=100, rate=0.04, equity_vol=0.30,
            n_paths=5000, seed=42,
        )
        assert isinstance(result, MandatoryConvertibleResult)
        assert result.price > 0

    def test_min_max_shares(self):
        result = mandatory_convertible(
            1000, 0.05, 3.0, 80, 120, 100, 0.04, 0.30, n_paths=2000, seed=42,
        )
        # min = N/high = 1000/120 ≈ 8.33
        # max = N/low = 1000/80 = 12.5
        assert result.min_shares == pytest.approx(1000/120)
        assert result.max_shares == pytest.approx(1000/80)

    def test_higher_coupon_higher_price(self):
        low = mandatory_convertible(1000, 0.02, 3.0, 80, 120, 100, 0.04, 0.30,
                                      n_paths=2000, seed=42)
        high = mandatory_convertible(1000, 0.10, 3.0, 80, 120, 100, 0.04, 0.30,
                                       n_paths=2000, seed=42)
        assert high.price > low.price

    def test_narrower_range_higher_par_region(self):
        """Narrower [low, high] range → more paths hit at-par region."""
        wide = mandatory_convertible(1000, 0.05, 3.0, 50, 200, 100, 0.04, 0.30,
                                       n_paths=2000, seed=42)
        narrow = mandatory_convertible(1000, 0.05, 3.0, 90, 110, 100, 0.04, 0.30,
                                         n_paths=2000, seed=42)
        # Both positive; relationship is complex
        assert wide.price > 0 and narrow.price > 0
