"""Tests for `NumericalConfig` and its integration with `PricingContext` (G1 P3 Slice 1)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.numerical_config import (
    DEFAULT_NUMERICAL_CONFIG,
    NumericalConfig,
)
from pricebook.core.pricing_context import PricingContext


# ============================================================
# NumericalConfig dataclass
# ============================================================

class TestNumericalConfig:
    def test_default_construction(self):
        cfg = NumericalConfig()
        # Smoke — fields have sensible defaults.
        assert cfg.mc_paths == 50_000
        assert cfg.mc_seed is None
        assert cfg.mc_antithetic is False
        assert cfg.mc_use_sobol is False
        assert cfg.pde_time_steps == 100
        assert cfg.pde_space_steps == 200
        assert cfg.tree_steps == 100
        assert cfg.cos_n == 1024
        assert cfg.integration_tol == 1.0e-9
        assert cfg.rootfinder_tol == 1.0e-10
        assert cfg.extra == {}

    def test_frozen(self):
        cfg = NumericalConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.mc_paths = 1_000  # type: ignore[misc]

    def test_replace(self):
        a = NumericalConfig()
        b = a.replace(mc_paths=200_000, mc_seed=42)
        assert b.mc_paths == 200_000
        assert b.mc_seed == 42
        # Original unchanged
        assert a.mc_paths == 50_000
        assert a.mc_seed is None
        # Other fields preserved
        assert b.cos_n == a.cos_n

    def test_default_singleton_is_default_config(self):
        assert DEFAULT_NUMERICAL_CONFIG == NumericalConfig()

    def test_extra_is_dict_like(self):
        cfg = NumericalConfig(extra={"my_pricer.knob": 7})
        assert cfg.extra["my_pricer.knob"] == 7

    def test_custom_field(self):
        cfg = NumericalConfig(
            mc_paths=1_000_000, mc_seed=42, mc_antithetic=True,
            pde_time_steps=500, cos_n=2048,
        )
        assert cfg.mc_paths == 1_000_000
        assert cfg.mc_seed == 42
        assert cfg.mc_antithetic is True
        assert cfg.pde_time_steps == 500
        assert cfg.cos_n == 2048


# ============================================================
# PricingContext integration
# ============================================================

class TestPricingContextNumericalConfig:
    @pytest.fixture
    def ctx(self) -> PricingContext:
        return PricingContext(valuation_date=date(2026, 6, 11))

    def test_default_field_is_none(self, ctx):
        assert ctx.numerical_config is None

    def test_get_numerical_config_falls_back_to_default(self, ctx):
        # No config attached → accessor returns the library default.
        got = ctx.get_numerical_config()
        assert got is DEFAULT_NUMERICAL_CONFIG

    def test_get_numerical_config_returns_attached(self):
        cfg = NumericalConfig(mc_paths=10_000)
        ctx = PricingContext(
            valuation_date=date(2026, 6, 11),
            numerical_config=cfg,
        )
        assert ctx.get_numerical_config() is cfg

    def test_replace_preserves_numerical_config(self):
        cfg = NumericalConfig(mc_paths=10_000)
        ctx = PricingContext(
            valuation_date=date(2026, 6, 11),
            numerical_config=cfg,
        )
        ctx2 = ctx.replace(reporting_currency="EUR")
        assert ctx2.numerical_config is cfg

    def test_replace_can_swap_numerical_config(self):
        cfg1 = NumericalConfig(mc_paths=10_000)
        cfg2 = NumericalConfig(mc_paths=200_000)
        ctx = PricingContext(
            valuation_date=date(2026, 6, 11),
            numerical_config=cfg1,
        )
        ctx2 = ctx.replace(numerical_config=cfg2)
        assert ctx2.numerical_config is cfg2
        # Original unchanged
        assert ctx.numerical_config is cfg1

    def test_replace_can_clear_numerical_config(self):
        cfg = NumericalConfig(mc_paths=10_000)
        ctx = PricingContext(
            valuation_date=date(2026, 6, 11),
            numerical_config=cfg,
        )
        ctx2 = ctx.replace(numerical_config=None)
        assert ctx2.numerical_config is None
        # Accessor still works — falls back to default.
        assert ctx2.get_numerical_config() is DEFAULT_NUMERICAL_CONFIG

    def test_no_regression_in_simple_factory(self):
        # PricingContext.simple() must still work — it doesn't pass numerical_config.
        ctx = PricingContext.simple(date(2026, 6, 11), rate=0.04)
        assert ctx.numerical_config is None
        assert ctx.get_numerical_config() is DEFAULT_NUMERICAL_CONFIG

    def test_with_curve_and_config(self):
        # End-to-end: context with a curve and a custom config can be used
        # exactly as before; numerical_config is independent.
        ref = date(2026, 6, 11)
        curve = DiscountCurve.flat(ref, 0.04)
        cfg = NumericalConfig(mc_paths=200_000, mc_seed=42)
        ctx = PricingContext(
            valuation_date=ref,
            discount_curve=curve,
            numerical_config=cfg,
        )
        assert ctx.get_discount_curve() is curve
        assert ctx.get_numerical_config() is cfg
        assert ctx.get_numerical_config().mc_paths == 200_000
