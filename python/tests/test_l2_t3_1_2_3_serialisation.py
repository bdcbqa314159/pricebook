"""Regression for L2 Tier-3 T3.1 / T3.2 / T3.3 — round-trip preservation.

* T3.1 — `SurvivalCurve._sc_to_dict` filtered `if t > 0` to drop the
  synthetic-t=0 prepend, but ALSO dropped any user-supplied pillar AT the
  reference date.  Round-trip then produced a curve missing that pillar
  (length mismatch with the round-tripped `dates`).

* T3.2 / T3.3 — `PricingContext.to_dict`/`from_dict` previously silently
  dropped multi-currency dicts and several container fields; empty
  containers came back as `None`, breaking subsequent accessors.  These
  were addressed in v0.903 (D.1 fix); this file locks in the regression.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import (
    DayCountConvention,
    date_from_year_fraction,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.core.survival_curve import SurvivalCurve


REF = date(2024, 1, 1)


# ============================================================
# T3.1 — SurvivalCurve round-trip preserves reference-date pillar
# ============================================================


class TestSurvivalCurveRoundTrip:
    def test_reference_date_pillar_preserved(self):
        """User supplies a pillar AT reference_date — must survive round-trip."""
        dates = [REF, REF + timedelta(days=365), REF + timedelta(days=5 * 365)]
        survival_probs = [1.0, 0.98, 0.90]
        curve = SurvivalCurve(REF, dates, survival_probs)

        d = curve.to_dict()
        # Round-trip via the serialisable framework.
        from pricebook.core.serialisable import from_dict as _fd
        curve2 = _fd(d)

        # Pre-fix: filter `if t > 0` dropped the REF-date pillar from
        # `survival_probs` but kept it in `dates`, producing length
        # mismatch and a curve missing the user's anchor.
        assert curve2._pillar_dates == dates
        assert math.isclose(curve2.survival(REF), 1.0, abs_tol=1e-12)
        assert math.isclose(
            curve2.survival(REF + timedelta(days=365)), 0.98, abs_tol=1e-6,
        )

    def test_no_reference_pillar_unchanged(self):
        """Sanity: when user does NOT supply a ref-date pillar, the
        synthetic-prepend behaviour must still work."""
        dates = [REF + timedelta(days=365), REF + timedelta(days=5 * 365)]
        survival_probs = [0.98, 0.90]
        curve = SurvivalCurve(REF, dates, survival_probs)
        d = curve.to_dict()
        from pricebook.core.serialisable import from_dict as _fd
        curve2 = _fd(d)
        assert curve2._pillar_dates == dates
        assert math.isclose(curve2.survival(REF), 1.0, abs_tol=1e-12)


# ============================================================
# T3.2 / T3.3 — PricingContext round-trip preserves multi-currency
# ============================================================


def _flat_disc(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    dfs = [math.exp(-rate * t) for t in tenors]
    dates = [date_from_year_fraction(REF, t) for t in tenors]
    return DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


class TestPricingContextMultiCurrencyRoundTrip:
    def test_multi_currency_discount_curves(self):
        """Multiple discount_curves dict must round-trip."""
        ctx = PricingContext(
            valuation_date=REF,
            discount_curves={"USD": _flat_disc(0.04), "EUR": _flat_disc(0.03)},
            reporting_currency="USD",
        )
        d = ctx.to_dict()
        ctx2 = PricingContext.from_dict(d)
        assert set(ctx2.discount_curves.keys()) == {"USD", "EUR"}
        assert ctx2.reporting_currency == "USD"

    def test_fx_spots_round_trip(self):
        """FX spots stored as (base, quote) tuples must round-trip."""
        ctx = PricingContext(
            valuation_date=REF,
            discount_curves={"USD": _flat_disc()},
            fx_spots={("EUR", "USD"): 1.10, ("GBP", "USD"): 1.27},
        )
        d = ctx.to_dict()
        ctx2 = PricingContext.from_dict(d)
        assert ctx2.fx_spots == {("EUR", "USD"): 1.10, ("GBP", "USD"): 1.27}

    def test_empty_containers_stay_dicts(self):
        """Pre-fix: empty containers came back as None on from_dict.
        Post-fix: stay as `{}` so accessors don't break."""
        ctx = PricingContext(valuation_date=REF, discount_curve=_flat_disc())
        d = ctx.to_dict()
        ctx2 = PricingContext.from_dict(d)
        # Each container should be a dict (possibly empty), not None.
        for field_name in [
            "discount_curves", "projection_curves", "credit_curves",
            "vol_surfaces", "fx_spots", "inflation_curves",
            "repo_curves", "stochastic_credit_models",
            "credit_vol_surfaces", "credit_correlations",
        ]:
            val = getattr(ctx2, field_name)
            assert val is not None, f"{field_name} is None (pre-fix bug)"
            assert isinstance(val, dict)

    def test_credit_curves_with_survival_curve_round_trip(self):
        """Combined test: PricingContext with a SurvivalCurve in credit_curves,
        the SurvivalCurve has a reference-date pillar (T3.1 case)."""
        sc = SurvivalCurve(
            REF,
            [REF, REF + timedelta(days=365), REF + timedelta(days=5 * 365)],
            [1.0, 0.98, 0.90],
        )
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=_flat_disc(),
            credit_curves={"default": sc},
        )
        d = ctx.to_dict()
        ctx2 = PricingContext.from_dict(d)
        assert "default" in ctx2.credit_curves
        sc2 = ctx2.credit_curves["default"]
        assert sc2._pillar_dates == [REF, REF + timedelta(days=365), REF + timedelta(days=5 * 365)]
