"""Regression for L2 Wave-2 audit — `risk.var.stress_test` had two
related bugs:

(a) Silent no-op on unsupported shock keys.  Pre-fix inspected only
    ``rate_shift`` — every other key (``vol_shift``, ``credit_shift``,
    ``fx_shift``, …) was silently dropped.  The function's own
    docstring example included ``vol_shift``, so users following the
    docs got identical-to-base PVs that *looked* like a successful
    stress run.

(b) Lossy context reconstruction.  The bumped context was built by
    `PricingContext(valuation_date=..., discount_curve=..., projection_curves=...,
    vol_surfaces=..., credit_curves=..., fx_spots=...)` — a 6-field
    subset that silently dropped the plural ``discount_curves``,
    ``inflation_curves``, ``repo_curves``, ``reporting_currency``,
    ``stochastic_credit_models``, ``credit_vol_surfaces``,
    ``credit_correlations``, and ``numerical_config``.  Pricers that
    consult any of these silently saw a degraded context.

Fix: use ``dataclasses.replace`` so untouched fields are preserved
by default; raise ``ValueError`` on unknown shock keys.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.core.numerical_config import NumericalConfig
from pricebook.risk.var import stress_test


def _flat_curve(rate: float) -> DiscountCurve:
    """Build a flat 5y curve from a flat continuous rate."""
    val = date(2026, 1, 1)
    pillar_years = [1.0, 2.0, 3.0, 5.0]
    dates = [val + timedelta(days=int(365 * y)) for y in pillar_years]
    dfs = [math.exp(-rate * y) for y in pillar_years]
    return DiscountCurve(reference_date=val, dates=dates, dfs=dfs)


class TestStressTestRaisesOnUnsupportedShocks:
    def test_vol_shift_now_raises(self):
        """Pre-fix: silently dropped.  Now: explicit ValueError."""
        ctx = PricingContext(valuation_date=date(2026, 1, 1),
                             discount_curve=_flat_curve(0.05))
        with pytest.raises(ValueError, match="vol_shift"):
            stress_test(
                pricer=lambda c: 100.0,
                base_ctx=ctx,
                scenarios=[{"vol_shift": 0.05}],
            )

    def test_typo_in_shift_name_raises(self):
        """A typo like ``rate_shifft`` is now caught, not silently dropped."""
        ctx = PricingContext(valuation_date=date(2026, 1, 1),
                             discount_curve=_flat_curve(0.05))
        with pytest.raises(ValueError, match="rate_shifft"):
            stress_test(
                pricer=lambda c: 100.0,
                base_ctx=ctx,
                scenarios=[{"rate_shifft": 0.01}],
            )


class TestStressTestRateShift:
    def test_rate_shift_bumps_singular_discount_curve(self):
        ctx = PricingContext(valuation_date=date(2026, 1, 1),
                             discount_curve=_flat_curve(0.05))

        def pricer(c):
            # 1Y zero-coupon bond PV.
            return c.discount_curve.df(date(2027, 1, 1))

        base = pricer(ctx)
        out = stress_test(pricer, ctx, [{"rate_shift": 0.01}], ["up_100bp"])
        # Shift +100bp → lower DF, lower PV.
        assert out[0]["scenario_pv"] < base
        assert out[0]["pnl"] < 0
        assert out[0]["name"] == "up_100bp"

    def test_rate_shift_bumps_plural_discount_curves(self):
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            discount_curves={"USD": _flat_curve(0.05), "EUR": _flat_curve(0.03)},
        )

        def pricer(c):
            return c.discount_curves["EUR"].df(date(2028, 1, 1))

        base = pricer(ctx)
        out = stress_test(pricer, ctx, [{"rate_shift": -0.005}], ["down_50bp"])
        # Shift -50bp → higher DF.
        assert out[0]["scenario_pv"] > base


class TestStressTestPreservesUntouchedFields:
    def test_numerical_config_survives(self):
        """G1 P3 added numerical_config; pre-fix dropped it on reconstruction."""
        cfg = NumericalConfig(mc_paths=12345)
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            discount_curve=_flat_curve(0.05),
            numerical_config=cfg,
        )
        seen_configs: list[NumericalConfig | None] = []

        def pricer(c):
            seen_configs.append(c.numerical_config)
            return 0.0

        stress_test(pricer, ctx, [{"rate_shift": 0.01}])
        # Both the base and the bumped pricer call must see the config.
        assert all(sc is not None for sc in seen_configs)
        assert all(sc.mc_paths == 12345 for sc in seen_configs)

    def test_reporting_currency_survives(self):
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            discount_curve=_flat_curve(0.05),
            reporting_currency="EUR",
        )
        seen: list[str] = []

        def pricer(c):
            seen.append(c.reporting_currency)
            return 0.0

        stress_test(pricer, ctx, [{"rate_shift": 0.01}])
        assert all(s == "EUR" for s in seen)


class TestStressTestEmpty:
    def test_no_shifts_passes_through(self):
        ctx = PricingContext(valuation_date=date(2026, 1, 1),
                             discount_curve=_flat_curve(0.05))
        out = stress_test(pricer=lambda c: 100.0, base_ctx=ctx, scenarios=[{}],
                          scenario_names=["null"])
        assert out[0]["pnl"] == 0.0
