"""Regression for L2 T4 audit of `options.american_dividend.american_with_dividends`:

Pre-fix the escrow-method backward induction used ``t > t_step`` to
collect "future" dividends at each tree step.  At the ex-dividend
step ``k_div`` itself, this excluded the dividend (since
``t_div = t_step``), making the true spot at that step the
POST-DROP value.  The intrinsic-vs-continuation check at that step
therefore compared the post-drop exercise value against continuation,
missing the optimal CUM-DIVIDEND exercise opportunity — the dominant
early-exercise scenario for American calls on dividend-paying stocks
(Roll-Geske-Whaley).

Fix (T4-AMDIV1): treat the dividend as future AT and after its ex-step
(``t >= t_step``), so the true spot at the ex-step is cum-dividend
and the intrinsic captures pre-drop exercise.

Sanity:
- Deep-ITM American call with a near-term large dividend should
  prefer to exercise BEFORE the dividend.  Pre-fix that opportunity
  was lost so the premium over European was understated.
- No-dividend case unchanged.
"""

from __future__ import annotations

import math

import pytest

from pricebook.models.black76 import OptionType
from pricebook.options.american_dividend import american_with_dividends


class TestCumDividendExerciseCaptured:
    def test_large_near_term_dividend_premium_strictly_positive(self):
        """A deep-ITM call (spot ≫ strike) with a large dividend in 1
        month should clearly prefer to exercise cum-dividend.  The
        American premium over European should be substantial.

        Pre-fix the cum-dividend exercise opportunity at the ex-step
        was lost, so the premium was understated.
        """
        # spot = 110, strike = 100, ITM by 10.  Dividend = 10 at 1/12y.
        r = american_with_dividends(
            spot=110, strike=100, rate=0.05, vol=0.20, T=1.0,
            dividends=[(1.0 / 12.0, 10.0)],
            option_type=OptionType.CALL, n_steps=400,
        )
        # American value should be at least the immediate-exercise
        # cum-dividend value: 110 − 100 = 10.
        assert r.price >= 10.0 - 0.5, (
            f"Deep-ITM cum-div American call price ({r.price:.2f}) "
            f"should be ≥ immediate exercise value (10.0)"
        )
        # Early exercise premium should be visibly positive.
        assert r.early_exercise_premium > 0.5, (
            f"early_exercise_premium = {r.early_exercise_premium:.3f} "
            f"should be > 0.5 for this deep-ITM cum-div case"
        )

    def test_no_dividend_unchanged(self):
        """Empty dividends list: American call ≈ European call (no
        early exercise on dividend-free call)."""
        r = american_with_dividends(
            spot=100, strike=100, rate=0.05, vol=0.20, T=1.0,
            dividends=[], option_type=OptionType.CALL, n_steps=200,
        )
        assert r.early_exercise_premium < 0.5


class TestPostDivBehaviourUnchanged:
    def test_dividend_after_expiry_ignored(self):
        """A dividend scheduled after expiry should have no effect."""
        no_div = american_with_dividends(
            spot=100, strike=100, rate=0.05, vol=0.20, T=1.0,
            dividends=[], option_type=OptionType.CALL, n_steps=200,
        )
        post_div = american_with_dividends(
            spot=100, strike=100, rate=0.05, vol=0.20, T=1.0,
            dividends=[(1.5, 5.0)],  # ex-date AFTER expiry
            option_type=OptionType.CALL, n_steps=200,
        )
        assert post_div.price == pytest.approx(no_div.price, rel=2e-2)
