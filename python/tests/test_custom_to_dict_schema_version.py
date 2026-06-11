"""Tests for `make_payload` / `read_payload` helpers + custom `to_dict`
schema-version coverage (fix B.1 B2).

Confirms:
1. The helper produces an envelope including `schema_version`.
2. `DiscountCurve` now preserves `interpolation` across round-trip.
3. Every class migrated to the helper emits `schema_version` in its `to_dict`.
4. Back-compat: pre-fix payloads (no `schema_version`) still deserialise as v1.
5. Future-version payloads raise the standard `ValueError` from
   `_check_schema_version`.
"""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.pricing_context import PricingContext
from pricebook.core.serialisable import (
    _SCHEMA_VERSION_KEY,
    make_payload,
    read_payload,
)
from pricebook.core.survival_curve import SurvivalCurve


# ============================================================
# make_payload / read_payload — direct tests
# ============================================================

class TestPayloadHelpers:
    def test_make_payload_includes_schema_version(self):
        ref = date(2024, 1, 1)
        curve = DiscountCurve.flat(ref, 0.05)
        d = curve.to_dict()
        assert d[_SCHEMA_VERSION_KEY] == 1
        assert d["type"] == "discount_curve"
        assert "params" in d

    def test_read_payload_strips_version_returns_params(self):
        ref = date(2024, 1, 1)
        curve = DiscountCurve.flat(ref, 0.05)
        d = curve.to_dict()
        # round-trip → still valid
        rebuilt = DiscountCurve.from_dict(d)
        # zero-rate preserved
        target = date(2025, 1, 1)
        assert curve.zero_rate(target) == pytest.approx(rebuilt.zero_rate(target), abs=1e-12)


# ============================================================
# DiscountCurve — interpolation now preserved (B.1 B2 fix)
# ============================================================

class TestDiscountCurveInterpolationRoundTrip:
    def _curve(self, interp):
        ref = date(2024, 1, 1)
        dates = [date(2024, 7, 1), date(2025, 1, 1), date(2026, 1, 1), date(2029, 1, 1)]
        dfs = [0.975, 0.951, 0.905, 0.785]
        return DiscountCurve(
            ref, dates, dfs,
            day_count=DayCountConvention.ACT_365_FIXED,
            interpolation=interp,
        )

    @pytest.mark.parametrize("interp", [
        InterpolationMethod.LINEAR,
        InterpolationMethod.LOG_LINEAR,
        InterpolationMethod.CUBIC_SPLINE,
        InterpolationMethod.MONOTONE_CUBIC,
        InterpolationMethod.AKIMA,
    ])
    def test_interpolation_method_preserved(self, interp):
        curve = self._curve(interp)
        assert curve._interpolation == interp
        rebuilt = DiscountCurve.from_dict(curve.to_dict())
        assert rebuilt._interpolation == interp, (
            f"Pre-fix B.1 B2: {interp} silently became LOG_LINEAR after round-trip"
        )

    def test_pre_fix_payload_without_interpolation_defaults_to_log_linear(self):
        """Back-compat: a payload written before the fix has no 'interpolation' key.
        Reading it must succeed and default to LOG_LINEAR."""
        ref = date(2024, 1, 1)
        d = {
            "type": "discount_curve",
            "params": {
                "reference_date": ref.isoformat(),
                "dates": [date(2024, 7, 1).isoformat(), date(2025, 1, 1).isoformat()],
                "dfs": [0.975, 0.951],
                "day_count": "ACT/365F",
                # No "interpolation" key.
            },
            # No schema_version either — treated as v1.
        }
        rebuilt = DiscountCurve.from_dict(d)
        assert rebuilt._interpolation == InterpolationMethod.LOG_LINEAR


# ============================================================
# Schema-version coverage across migrated classes
# ============================================================

class TestSchemaVersionEverywhere:
    def test_discount_curve_emits_version(self):
        c = DiscountCurve.flat(date(2024, 1, 1), 0.05)
        assert c.to_dict()[_SCHEMA_VERSION_KEY] == 1

    def test_survival_curve_emits_version(self):
        c = SurvivalCurve.flat(date(2024, 1, 1), 0.02)
        assert c.to_dict()[_SCHEMA_VERSION_KEY] == 1

    def test_pricing_context_emits_version(self):
        ctx = PricingContext.simple(date(2024, 1, 1), rate=0.04)
        assert ctx.to_dict()[_SCHEMA_VERSION_KEY] == 1

    def test_future_version_rejected_on_discount_curve(self):
        c = DiscountCurve.flat(date(2024, 1, 1), 0.05)
        d = c.to_dict()
        d[_SCHEMA_VERSION_KEY] = 99
        with pytest.raises(ValueError, match="schema_version=99"):
            DiscountCurve.from_dict(d)

    def test_back_compat_no_version_is_v1(self):
        """A payload without `schema_version` (the format that lives on disk
        before this fix landed) must continue to deserialise as v1."""
        c = DiscountCurve.flat(date(2024, 1, 1), 0.05)
        d = c.to_dict()
        d.pop(_SCHEMA_VERSION_KEY)
        rebuilt = DiscountCurve.from_dict(d)
        assert rebuilt.zero_rate(date(2025, 1, 1)) == pytest.approx(
            c.zero_rate(date(2025, 1, 1)), abs=1e-12
        )
