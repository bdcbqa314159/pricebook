"""Regression for L2 phase-2 audit of `risk.scenario`:

Every scenario constructor (`parallel_shift`, `pillar_bump`, `vol_bump`,
`fx_spot_shock`, `credit_spread_shift`) used ``PricingContext(...)``
with a 6-field subset, silently dropping ``discount_curves`` (plural),
``inflation_curves``, ``repo_curves``, ``reporting_currency``,
``stochastic_credit_models``, ``credit_vol_surfaces``,
``credit_correlations``, and ``numerical_config``.

Same shape as the v0.993 ``var.stress_test`` fix.  Now uses
``dataclasses.replace`` to preserve untouched fields.

Bonus fix: ``parallel_shift`` now also bumps the plural
``discount_curves`` dict, not only the singular ``discount_curve``.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.numerical_config import NumericalConfig
from pricebook.core.pricing_context import PricingContext
from pricebook.risk.scenario import (
    credit_spread_shift, fx_spot_shock, parallel_shift, pillar_bump,
    vol_bump,
)


def _flat_curve(rate: float) -> DiscountCurve:
    val = date(2026, 1, 1)
    pillar_years = [1.0, 2.0, 3.0, 5.0]
    dates = [val + timedelta(days=int(365 * y)) for y in pillar_years]
    dfs = [math.exp(-rate * y) for y in pillar_years]
    return DiscountCurve(reference_date=val, dates=dates, dfs=dfs)


class TestPreservesUntouchedFields:
    def test_parallel_shift_preserves_numerical_config(self):
        cfg = NumericalConfig(mc_paths=12345)
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            discount_curve=_flat_curve(0.05),
            numerical_config=cfg,
        )
        bumped = parallel_shift(0.0001).apply(ctx)
        assert bumped.numerical_config is not None
        assert bumped.numerical_config.mc_paths == 12345

    def test_pillar_bump_preserves_reporting_currency(self):
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            discount_curve=_flat_curve(0.05),
            reporting_currency="EUR",
        )
        bumped = pillar_bump(0, 0.0001).apply(ctx)
        assert bumped.reporting_currency == "EUR"

    def test_vol_bump_preserves_numerical_config(self):
        cfg = NumericalConfig(mc_paths=999)
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            numerical_config=cfg,
        )
        bumped = vol_bump(0.01).apply(ctx)
        assert bumped.numerical_config is not None
        assert bumped.numerical_config.mc_paths == 999

    def test_fx_spot_shock_preserves_reporting_currency(self):
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            fx_spots={("EUR", "USD"): 1.10},
            reporting_currency="EUR",
        )
        bumped = fx_spot_shock("EUR", "USD", 0.05).apply(ctx)
        assert bumped.reporting_currency == "EUR"

    def test_credit_spread_shift_preserves_numerical_config(self):
        cfg = NumericalConfig(mc_paths=7777)
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            numerical_config=cfg,
        )
        bumped = credit_spread_shift(0.01).apply(ctx)
        assert bumped.numerical_config is not None
        assert bumped.numerical_config.mc_paths == 7777


class TestParallelShiftBumpsPluralCurves:
    def test_plural_discount_curves_bumped(self):
        usd = _flat_curve(0.05)
        eur = _flat_curve(0.03)
        ctx = PricingContext(
            valuation_date=date(2026, 1, 1),
            discount_curves={"USD": usd, "EUR": eur},
        )
        bumped = parallel_shift(0.0001).apply(ctx)
        # Both USD and EUR should be bumped.
        # Verify the DF at some pillar is now lower (rate shifted up by 1bp).
        usd_df_before = ctx.discount_curves["USD"].df(date(2027, 1, 1))
        usd_df_after = bumped.discount_curves["USD"].df(date(2027, 1, 1))
        eur_df_before = ctx.discount_curves["EUR"].df(date(2027, 1, 1))
        eur_df_after = bumped.discount_curves["EUR"].df(date(2027, 1, 1))
        assert usd_df_after < usd_df_before
        assert eur_df_after < eur_df_before
