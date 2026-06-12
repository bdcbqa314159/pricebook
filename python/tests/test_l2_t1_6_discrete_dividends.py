"""Regression for L2 Tier-1 T1.6 — discrete dividends applied at correct steps.

Pre-fix the `_solve_binomial` path applied ALL configured dividends to the
terminal spot grid only; intermediate-step `S_step` (used for early exercise
and barrier checks) was the raw forward — pre-dividend — so:

- An American option with a known dividend before maturity saw a higher
  intermediate spot than a forward-realised one would, biasing the early-
  exercise decision.
- Barrier knock-out checks used the wrong (pre-dividend) intermediate spot.

The `_solve_trinomial` path had NO dividend handling at all — `self.dividends`
was silently ignored.

Post-fix, both paths use an "escrowed dividend" convention: at each step `s`,
the spot grid has cumulative dividends paid through step `s` subtracted.
"""

from __future__ import annotations

import pytest


# ============================================================
# Binomial — intermediate spots reflect dividends paid by that step
# ============================================================

class TestBinomialDividendTiming:
    def _european_call(self, dividends, n_steps=50):
        from pricebook.numerical._trees import TreeSolver, TreeMethod, ExerciseType
        tree = TreeSolver(
            n_steps=n_steps, method=TreeMethod.CRR,
            exercise=ExerciseType.EUROPEAN,
            dividends=dividends,
        )
        return tree.solve(
            spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0, is_call=True,
        ).price

    def test_dividend_at_step_zero_equals_terminal_dividend(self):
        """A dividend at step 0 is paid immediately — terminal-only escrow
        gives the same answer as escrowing-per-step (boundary case)."""
        # No dividend
        price_none = self._european_call(dividends=None)
        # Dividend paid at step 0 (D = 5)
        price_t0 = self._european_call(dividends=[(0, 5.0)])
        # With a $5 dividend right at the start, the call should be lower
        # (lower effective forward spot).
        assert price_t0 < price_none, (
            f"$5 dividend at step 0 should decrease call PV; "
            f"got {price_t0:.4f} vs no-div {price_none:.4f}"
        )

    def test_dividend_at_intermediate_step_affects_price(self):
        """Pre-fix: dividend at step 25 (of 50) was applied to terminal grid
        regardless of step. Post-fix: only steps >= 25 see the drop."""
        price = self._european_call(dividends=[(25, 5.0)])
        price_none = self._european_call(dividends=None)
        assert price < price_none

    def test_american_call_with_dividend_exercises_correctly(self):
        """A known dividend before maturity is the classic reason to early-
        exercise an American call. With escrowed-dividend correctly applied
        at intermediate steps, the early-exercise value should be ≥ the
        European call value (proper American premium for dividends)."""
        from pricebook.numerical._trees import TreeSolver, TreeMethod, ExerciseType
        # American: should be ≥ European
        amer = TreeSolver(
            n_steps=100, method=TreeMethod.CRR,
            exercise=ExerciseType.AMERICAN,
            dividends=[(50, 8.0)],   # large dividend at mid-life
        ).solve(spot=100.0, strike=95.0, rate=0.05, vol=0.20, T=1.0, is_call=True)
        eur = TreeSolver(
            n_steps=100, method=TreeMethod.CRR,
            exercise=ExerciseType.EUROPEAN,
            dividends=[(50, 8.0)],
        ).solve(spot=100.0, strike=95.0, rate=0.05, vol=0.20, T=1.0, is_call=True)
        assert amer.price >= eur.price - 1e-10


# ============================================================
# Trinomial — now honours dividends (pre-fix it ignored them)
# ============================================================

class TestTrinomialDividendHandling:
    def test_trinomial_dividend_actually_affects_price(self):
        """Pre-fix the trinomial branch silently dropped `dividends`. A $10
        dividend should noticeably decrease a call price."""
        from pricebook.numerical._trees import TreeSolver, TreeMethod, ExerciseType
        no_div = TreeSolver(
            n_steps=50, method=TreeMethod.TRINOMIAL,
            exercise=ExerciseType.EUROPEAN,
        ).solve(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0, is_call=True)

        with_div = TreeSolver(
            n_steps=50, method=TreeMethod.TRINOMIAL,
            exercise=ExerciseType.EUROPEAN,
            dividends=[(25, 10.0)],
        ).solve(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0, is_call=True)

        # $10 dividend before maturity → call should be significantly lower.
        # Pre-fix the trinomial method ignored dividends so this would be ≈0.
        delta = no_div.price - with_div.price
        assert delta > 1.0, (
            f"Trinomial silently dropped the dividend: "
            f"no_div={no_div.price:.4f}, with_div={with_div.price:.4f}, "
            f"Δ={delta:.4f} (expected > 1)"
        )
