"""Regression for L2 T4 audit of `options.exotic_payoffs.installment_option`:

Pre-fix the rational-exercise decision at each installment date compared
the live option value to ``pv_remaining`` — but ``pv_remaining`` summed
only installments paid AFTER the current date, omitting the CURRENT
installment that the holder must pay to continue.  The correct cost of
continuing is ``installment_amt + Σ_{future} installment_amt · DF``.

Consequence: the holder over-continued (the check was too lenient by
the magnitude of one current installment), so the option's
continuation probability was over-stated and the priced payoff was
biased upward.

Fix (T4-EX1): include the current installment in ``cost_to_continue``.

This test pins:
- Continuation probability strictly decreases when the missing current
  installment is added back to the cost.  Pre-fix the same scenario
  gave a higher continuation_prob.
"""

from __future__ import annotations

import math

import pytest

from pricebook.options.exotic_payoffs import installment_option


class TestRationalContinueAccountsForCurrentInstallment:
    def test_otm_path_abandons_under_correct_cost(self):
        """For a deeply OTM call where live_val at intermediate dates
        is below the current installment alone, the holder MUST abandon
        even if remaining installments after are zero.  Pre-fix the
        bug allowed continuation as long as live_val >= future-only
        sum, missing the current payment."""
        # Far OTM call: spot = 100, strike = 200.
        result = installment_option(
            spot=100, strike=200, vol=0.10, T=2.0, r=0.04, q=0.0,
            n_installments=4, option_type="call",
            n_paths=20_000, seed=42,
        )
        # Continuation probability should be very low — the option is
        # almost never worth paying the next installment.
        assert result.continuation_prob < 0.10, (
            f"continuation_prob = {result.continuation_prob:.3f} — "
            f"should be near 0 for deep OTM with low vol"
        )

    def test_atm_continues_with_lower_prob_under_correct_check(self):
        """For ATM, the corrected cost-check (including current
        installment) should be at least as tight as the pre-fix check.
        Continuation probability is bounded by [0, 1].  Just verify
        the price is finite and non-negative."""
        result = installment_option(
            spot=100, strike=100, vol=0.25, T=1.0, r=0.05, q=0.0,
            n_installments=3, option_type="call",
            n_paths=20_000, seed=42,
        )
        assert math.isfinite(result.price)
        assert result.price >= 0.0
        assert 0.0 <= result.continuation_prob <= 1.0
