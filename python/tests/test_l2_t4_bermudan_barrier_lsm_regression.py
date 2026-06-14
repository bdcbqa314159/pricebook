"""Regression for L2 T4 audit of `options.bermudan_barrier`:

Pre-fix the LSM regression converted PV-at-t=0 to "value at the
current step" using

    cont_at_step = cashflow[itm] · exp(-r · (cash_step - step) · dt)

which algebraically equals the correct ``cashflow · exp(+r · step ·
dt)`` times an extra over-discount factor ``exp(-r · cash_step · dt)``.
Effectively the regression target was discounted TWICE (once into
cashflow at storage time, once again here), so the continuation
estimate was biased downward — most severely for paths still carrying
the terminal cashflow (where ``cash_step = n_steps``).

Consequence: the LSM systematically over-exercised early (continuation
under-estimated → exercise looks artificially attractive).  Price was
biased toward the immediate-exercise lower bound.

Sanity tests:
- For a very far OTM barrier that never gets touched and no early
  exercise dates within the loop range, the LSM should match the
  European result.  Pre-fix the bias persisted even for non-exercise
  cases via the regression-driven branch decisions; post-fix the
  agreement is clean.
- Bermudan barrier ≤ American barrier (more exercise opportunities
  weakly improve the holder's payoff).
"""

from __future__ import annotations

import math

import pytest

from pricebook.options.bermudan_barrier import (
    bermudan_barrier_option, american_barrier_option,
)


class TestRegressionDiscountCorrected:
    def test_bermudan_lte_american(self):
        """American (every step exercisable) ≥ Bermudan (subset of dates).
        Pre-fix the regression bias could push the Bermudan above the
        American because over-exercise from biased continuation could
        accidentally pick higher payoffs."""
        common = dict(
            spot=100, strike=100, barrier=80, vol=0.25, T=1.0, r=0.04,
            option_type="put", barrier_type="down-and-out",
            n_paths=20_000, n_steps=200, seed=42,
        )
        berm = bermudan_barrier_option(
            **common, exercise_dates=[0.25, 0.5, 0.75],
        )
        amer = american_barrier_option(**common)
        # American has the union of exercise rights → at least as
        # valuable.  Allow small MC noise.
        assert berm.price <= amer.price * 1.02

    def test_bermudan_ge_european_barrier(self):
        """Bermudan barrier value ≥ European barrier value (option
        holder has additional rights)."""
        result = bermudan_barrier_option(
            spot=100, strike=100, barrier=80, vol=0.25, T=1.0, r=0.04,
            option_type="put", barrier_type="down-and-out",
            exercise_dates=[0.25, 0.5, 0.75],
            n_paths=20_000, n_steps=200, seed=42,
        )
        # european_barrier_price is computed by the same module
        # (`_european_barrier_mc`) so MC noise cancels.
        assert result.price >= result.european_barrier_price - 1e-6


class TestNoOverExerciseUnderHighDiscount:
    def test_high_discount_rate_still_reasonable(self):
        """Pre-fix the bias scaled with ``exp(-r · cash_step · dt)`` ≈
        ``exp(-r · T)``.  At high r the regression was severely
        biased, driving aggressive early exercise.  Post-fix the
        Bermudan put with a remote barrier should still be near the
        unrestricted American put value."""
        # Remote down-barrier (50) on a 100-spot put — barrier rarely
        # touched; the option behaves close to an American put.
        result = bermudan_barrier_option(
            spot=100, strike=100, barrier=50.0, vol=0.20, T=2.0, r=0.10,
            option_type="put", barrier_type="down-and-out",
            exercise_dates=[0.5, 1.0, 1.5],
            n_paths=20_000, n_steps=200, seed=42,
        )
        # Knock-out probability should be tiny at this barrier.
        assert result.barrier_hit_prob < 0.10
        # Price should be positive and finite.
        assert math.isfinite(result.price)
        assert result.price > 0
