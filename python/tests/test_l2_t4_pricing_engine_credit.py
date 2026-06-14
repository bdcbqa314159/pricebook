"""Regression for L2 T4 audit of `pricing.pricing_engine._pv_fallback`:

Pre-fix two coupled defects when pricing credit-dependent trades:

1. **First-curve silent pick** (same shape as v0.969 FRA): when multiple
   credit curves were provided in market_data, ``next(iter(...))``
   silently selected the first one regardless of which obligor the
   trade referenced.  A 2-obligor portfolio priced ALL CDS against the
   first obligor's hazard.

2. **Silent 2% hazard default**: when no credit curve was provided at
   all, the engine substituted ``SurvivalCurve.flat(val_date, 0.02)``
   and returned ``status: ok`` — caller never knew the wrong input was
   accepted.

Fix: raise ``ValueError`` with diagnostic in both ambiguous cases.
Single-curve case still works (most common).
"""

from __future__ import annotations

import pytest

from pricebook.pricing.pricing_engine import price_from_dict


class TestMissingSurvivalCurveRaises:
    def test_cds_without_survival_curve_raises(self):
        """Pre-fix: silently used 2% flat hazard → wrong PV with status='ok'.
        Post-fix: per-trade status='error' explaining the missing input."""
        req = {
            "valuation_date": "2026-01-15",
            "market_data": {"flat_rate": 0.04},  # no survival curves
            "trades": [{
                "type": "cds",
                "params": {
                    "start": "2026-01-15", "end": "2031-01-15",
                    "spread": 0.01, "recovery": 0.40,
                    "notional": 1_000_000,
                },
            }],
        }
        result = price_from_dict(req)
        # The trade fails individually (engine catches the ValueError and
        # records it as a per-trade error).
        trade_result = result["results"][0]
        assert trade_result["status"] == "error"
        assert "survival" in trade_result["error"].lower()


class TestAmbiguousSurvivalCurvesFallbackPath:
    """Direct unit test of `_pv_fallback`: when multiple survival curves
    are passed and the instrument requires one (no pv_ctx route), the
    fallback must refuse to silently pick the first one.

    Note: the higher-level ``pv_ctx`` path is opaque to the engine —
    individual instruments choose their own curve.  The fallback is the
    last-resort path; we pin its behaviour here.
    """

    def test_pv_fallback_multiple_curves_raises(self):
        from datetime import date
        from pricebook.core.pricing_context import PricingContext
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.survival_curve import SurvivalCurve
        from pricebook.pricing.pricing_engine import _pv_fallback

        ref = date(2026, 1, 15)
        dc = DiscountCurve.flat(ref, 0.04)
        sc_A = SurvivalCurve.flat(ref, 0.02)
        sc_B = SurvivalCurve.flat(ref, 0.05)
        ctx = PricingContext(
            valuation_date=ref, discount_curve=dc,
            credit_curves={"issuer_A": sc_A, "issuer_B": sc_B},
        )

        # Dummy instrument that only takes (curve, sc) — forces fallback path.
        class DummyCDS:
            def pv(self, curve, sc=None):
                if sc is None:
                    raise TypeError("need sc")
                return 100.0  # placeholder

        with pytest.raises(ValueError, match="Specify which|specify which|multiple|2 are provided"):
            _pv_fallback(DummyCDS(), ctx, ref)


class TestSingleSurvivalCurveWorks:
    def test_single_survival_curve_uses_it(self):
        """One survival curve provided → engine uses it (no ambiguity)."""
        req = {
            "valuation_date": "2026-01-15",
            "market_data": {
                "flat_rate": 0.04,
                "survival_curves": {
                    "issuer": {"type": "survival_curve",
                               "params": {"reference_date": "2026-01-15",
                                          "dates": ["2031-01-15"],
                                          "survival_probs": [0.90],
                                          "day_count": "ACT/365F",
                                          "interpolation": "log_linear"}},
                },
            },
            "trades": [{
                "type": "cds",
                "params": {
                    "start": "2026-01-15", "end": "2031-01-15",
                    "spread": 0.01, "recovery": 0.40,
                    "notional": 1_000_000,
                },
            }],
        }
        result = price_from_dict(req)
        trade_result = result["results"][0]
        # Single-curve path should price successfully.
        assert trade_result["status"] == "ok"
        assert "pv" in trade_result
