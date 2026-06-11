"""Tests for `PricingContext` round-trip and `replace()` immutability (fixes D.1).

Three bugs from L0 audit D.1, all fixed in one slice:

- B1 — empty-dict fields became `None` on round-trip (then `ctx.foo["bar"]` raised
  `TypeError: 'NoneType' object is not subscriptable` instead of the documented
  `KeyError`).
- B2 — `discount_curves`, `inflation_curves`, `repo_curves`, `reporting_currency`,
  `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, AND
  the freshly-added `numerical_config` (G1 P3 Slice 1) were silently dropped on
  serialisation.
- B3 — `replace()` shared mutable dicts with the parent, breaking the
  "Immutable snapshot" contract.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.numerical_config import NumericalConfig
from pricebook.core.pricing_context import PricingContext


# ============================================================
# Helpers
# ============================================================

REF = date(2024, 1, 1)


def _curve(rate: float = 0.05) -> DiscountCurve:
    return DiscountCurve.flat(REF, rate)


# ============================================================
# D.1 B1 — empty dicts stay empty (not None) on round-trip
# ============================================================

class TestEmptyContainersPreserved:
    def test_default_context_round_trip(self):
        """Default-constructed context has all-empty dict fields. After
        round-trip they must still be (empty) dicts, not None."""
        ctx = PricingContext(valuation_date=REF, discount_curve=_curve())
        d = ctx.to_dict()
        rebuilt = PricingContext.from_dict(d)
        # All container fields are dicts (not None).
        assert isinstance(rebuilt.discount_curves, dict)
        assert isinstance(rebuilt.projection_curves, dict)
        assert isinstance(rebuilt.vol_surfaces, dict)
        assert isinstance(rebuilt.credit_curves, dict)
        assert isinstance(rebuilt.fx_spots, dict)
        assert isinstance(rebuilt.inflation_curves, dict)
        assert isinstance(rebuilt.repo_curves, dict)
        assert isinstance(rebuilt.stochastic_credit_models, dict)
        assert isinstance(rebuilt.credit_vol_surfaces, dict)
        assert isinstance(rebuilt.credit_correlations, dict)
        # All empty.
        assert len(rebuilt.projection_curves) == 0
        assert len(rebuilt.discount_curves) == 0

    def test_empty_field_access_is_keyerror_not_typeerror(self):
        """Pre-fix: an empty container collapsed to None, so accessing
        `ctx.projection_curves["foo"]` raised TypeError. Fixed."""
        ctx = PricingContext(valuation_date=REF, discount_curve=_curve())
        rebuilt = PricingContext.from_dict(ctx.to_dict())
        with pytest.raises(KeyError):
            _ = rebuilt.projection_curves["nonexistent"]


# ============================================================
# D.1 B2 — every dataclass-declared field round-trips
# ============================================================

class TestEveryFieldRoundTrips:
    def test_discount_curves_per_currency(self):
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=_curve(),
            discount_curves={"USD": _curve(0.05), "EUR": _curve(0.04)},
        )
        rebuilt = PricingContext.from_dict(ctx.to_dict())
        assert set(rebuilt.discount_curves.keys()) == {"USD", "EUR"}
        # Curves themselves round-trip too.
        target = date(2025, 1, 1)
        assert rebuilt.discount_curves["USD"].zero_rate(target) == pytest.approx(0.05, abs=1e-6)
        assert rebuilt.discount_curves["EUR"].zero_rate(target) == pytest.approx(0.04, abs=1e-6)

    def test_reporting_currency_preserved(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            reporting_currency="JPY",
        )
        rebuilt = PricingContext.from_dict(ctx.to_dict())
        assert rebuilt.reporting_currency == "JPY"

    def test_repo_curves_preserved(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            repo_curves={"USD": _curve(0.04)},
        )
        rebuilt = PricingContext.from_dict(ctx.to_dict())
        target = date(2025, 1, 1)
        assert rebuilt.repo_curves["USD"].zero_rate(target) == pytest.approx(0.04, abs=1e-6)

    def test_fx_spots_preserved(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            fx_spots={("EUR", "USD"): 1.08, ("GBP", "USD"): 1.27},
        )
        rebuilt = PricingContext.from_dict(ctx.to_dict())
        assert rebuilt.fx_spots == {("EUR", "USD"): 1.08, ("GBP", "USD"): 1.27}

    def test_credit_correlations_preserved(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            credit_correlations={"ACME_VS_ZETA": 0.65},
        )
        rebuilt = PricingContext.from_dict(ctx.to_dict())
        assert rebuilt.credit_correlations == {"ACME_VS_ZETA": 0.65}

    def test_numerical_config_preserved(self):
        cfg = NumericalConfig(mc_paths=200_000, mc_seed=42, cos_n=2048)
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            numerical_config=cfg,
        )
        rebuilt = PricingContext.from_dict(ctx.to_dict())
        # Frozen dataclass equality.
        assert rebuilt.numerical_config == cfg

    def test_numerical_config_none_preserved(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            numerical_config=None,
        )
        rebuilt = PricingContext.from_dict(ctx.to_dict())
        assert rebuilt.numerical_config is None


# ============================================================
# D.1 B3 — replace() defensively copies mutable containers
# ============================================================

class TestReplaceImmutability:
    def test_replace_copies_discount_curves(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            discount_curves={"USD": _curve()},
        )
        ctx2 = ctx.replace(reporting_currency="EUR")
        # Mutating ctx2's container must not affect ctx.
        ctx2.discount_curves["BRL"] = _curve(0.10)
        assert "BRL" not in ctx.discount_curves
        # And the reverse.
        ctx.discount_curves["JPY"] = _curve(0.01)
        assert "JPY" not in ctx2.discount_curves

    def test_replace_copies_fx_spots(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            fx_spots={("EUR", "USD"): 1.08},
        )
        ctx2 = ctx.replace(reporting_currency="EUR")
        ctx2.fx_spots[("GBP", "USD")] = 1.27
        assert ("GBP", "USD") not in ctx.fx_spots

    def test_replace_copies_credit_correlations(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            credit_correlations={"A_B": 0.5},
        )
        ctx2 = ctx.replace(reporting_currency="EUR")
        ctx2.credit_correlations["C_D"] = 0.7
        assert "C_D" not in ctx.credit_correlations

    def test_replace_preserves_field_when_not_replaced(self):
        ctx = PricingContext(
            valuation_date=REF, discount_curve=_curve(),
            reporting_currency="USD",
            fx_spots={("EUR", "USD"): 1.08},
        )
        ctx2 = ctx.replace(reporting_currency="EUR")
        # Values copy through.
        assert ctx2.fx_spots == {("EUR", "USD"): 1.08}
        # But the underlying dict is a different object.
        assert ctx2.fx_spots is not ctx.fx_spots
