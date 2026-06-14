"""Regression for L2 T4 audit of `options.bermudan_capfloor`:

Same defect class as ``bermudan_swaption`` (T4-BERM1) rolled forward
to the Bermudan cap/floor HW trinomial tree (T4-BCF1):

1. **Wrong trinomial probabilities** — drift term used ``/6`` instead
   of textbook ``/2``; net drift 3× too small, mean-reversion damped.

2. **Exercise compared to discounted continuation** — the loop
   computed ``new_values = exp(-r·dt) × continuation`` then took
   ``max(new_values, exercise_at_step+1)``, comparing DISCOUNTED
   continuation to UNDISCOUNTED exercise → exercise systematically
   over-valued by ``exp(+r·dt)`` per step.

This module exposes a flat-curve interface (takes ``r0`` directly, no
DiscountCurve), so α(t) = r0 is implicit and acceptable for the
flat-curve use case.

These tests pin:
- Single-exercise (at near-maturity) Bermudan cap matches the
  European cap to discretization (pre-fix the missing discount
  bias inflated Bermudan above European).
- Drift sensitivity has the right sign (stronger mean-reversion ⇒
  lower option value).
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.options.bermudan_capfloor import (
    bermudan_cap, bermudan_floor,
)


REF = date(2026, 1, 15)


class TestNoOverDiscountOnExercise:
    def test_european_at_first_step_matches_european_benchmark(self):
        """When the single exercise date is very close to t=0 the
        Bermudan must equal the European benchmark closely.  Pre-fix the
        missing discount in the exercise comparison inflated the
        Bermudan price above the European value, often by a few percent
        per exercise period."""
        result = bermudan_cap(
            reference_date=REF, maturity_years=5.0, strike=0.05,
            hw_a=0.1, hw_sigma=0.01, r0=0.04,
            exercise_dates_years=[0.05],  # very early, t≈0
            frequency=4, n_steps=100,
        )
        # The Bermudan with one near-immediate exercise opportunity must
        # not exceed the European benchmark by anything significant.
        assert result.price <= result.european_price * 1.02
        # And early exercise premium should be small (a few % at most).
        assert result.early_exercise_premium <= result.european_price * 0.05


class TestMeanReversionSensitivity:
    def test_stronger_mean_reversion_lowers_price(self):
        """Higher mean-reversion speed ``a`` damps long-end vol → lower
        option value.  Pre-fix the ``/6`` drift defect blunted this
        sensitivity (the drift was 3× too small, so changing ``a`` had a
        damped impact)."""
        kwargs = dict(
            reference_date=REF, maturity_years=5.0, strike=0.05,
            hw_sigma=0.01, r0=0.04,
            exercise_dates_years=[1.0, 2.0, 3.0, 4.0],
            frequency=4, n_steps=100,
        )
        r_low = bermudan_cap(hw_a=0.05, **kwargs)
        r_high = bermudan_cap(hw_a=0.20, **kwargs)
        # Higher mean-reversion → lower price (less vol of long forwards).
        assert r_high.price < r_low.price


class TestBermudanBoundedByEuropean:
    """Note: ``_remaining_value(r0, t=0, …)`` is reported as the
    ``european_price`` but it counts ALL caplets from t=0.  A Bermudan
    with exercise dates starting at t=1 can knock in no earlier than
    t=1, so it CANNOT capture the first-period caplets — its price is
    bounded *above* by the full-strip European (knock in at t=0).
    Pre-fix the missing-exercise-discount bug could push the Bermudan
    artificially above this upper bound for deep-ITM products.
    """

    def test_cap_bounded_above_by_european_strip(self):
        result = bermudan_cap(
            reference_date=REF, maturity_years=5.0, strike=0.04,
            hw_a=0.1, hw_sigma=0.01, r0=0.05,  # caplets ITM
            exercise_dates_years=[1.0, 2.0, 3.0, 4.0],
            frequency=4, n_steps=100,
        )
        # Bermudan ≤ full European strip (knock-in-at-0).  Allow 1%
        # discretization slack.
        assert result.price <= result.european_price * 1.01

    def test_floor_bounded_above_by_european_strip(self):
        result = bermudan_floor(
            reference_date=REF, maturity_years=5.0, strike=0.05,
            hw_a=0.1, hw_sigma=0.01, r0=0.04,  # floorlets ITM
            exercise_dates_years=[1.0, 2.0, 3.0, 4.0],
            frequency=4, n_steps=100,
        )
        assert result.price <= result.european_price * 1.01


class TestPositiveAndFinite:
    def test_cap_finite(self):
        result = bermudan_cap(
            reference_date=REF, maturity_years=5.0, strike=0.05,
            hw_a=0.1, hw_sigma=0.01, r0=0.04,
            exercise_dates_years=[1.0, 2.0, 3.0, 4.0],
            frequency=4, n_steps=100,
        )
        assert math.isfinite(result.price)
        assert result.price >= 0
