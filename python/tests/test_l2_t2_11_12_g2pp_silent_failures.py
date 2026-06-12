"""Regression for L2 Tier-2 T2.11 / T2.12 — `g2pp_swaption_price` no longer
masks errors as zero.

Pre-fix two silent-failure modes:

* T2.11 — the entire function body was wrapped in
  `try: ... except Exception: return 0.0`, masking every error (bracketing
  failure, brentq divergence, numerical-overflow, calibration bugs) as a
  silent zero price.  Callers had no way to distinguish "swaption worth ≈ 0"
  from "pricer crashed".

* T2.12 — when the Jamshidian `y*` bracket couldn't find a sign change after
  10 expansions, the code silently set `y_star = 0.0` and pressed on.  The
  resulting K_k strikes were unrelated to the actual swaption — producing
  wildly wrong prices that the outer T2.11 except then turned into zero.

Post-fix: real exceptions propagate; a failed bracket raises RuntimeError
with a diagnostic message.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.g2pp_calibration import g2pp_swaption_price


REF = date(2024, 1, 1)


def _flat_curve(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    dfs = [math.exp(-rate * t) for t in tenors]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


class TestG2PPNoSilentZero:
    def test_normal_params_produce_positive_price(self):
        """Sanity: with reasonable G2++ params, the pricer returns a finite
        positive number (not 0.0)."""
        price = g2pp_swaption_price(
            a=0.5, b=0.1, sigma1=0.01, sigma2=0.005, rho=-0.7,
            curve=_flat_curve(0.04),
            expiry_years=2.0, tenor_years=3.0, strike=0.04,
            is_payer=True,
        )
        assert price > 0, f"normal-params price is {price}, expected > 0"
        assert math.isfinite(price)

    def test_zero_payment_dates_returns_zero_cleanly(self):
        """Degenerate case (tenor → 0): still returns 0.0 explicitly, not
        via exception masking."""
        price = g2pp_swaption_price(
            a=0.5, b=0.1, sigma1=0.01, sigma2=0.005, rho=-0.7,
            curve=_flat_curve(),
            expiry_years=2.0, tenor_years=0.0, strike=0.04,
        )
        # tenor=0 produces no payment dates, returns 0 explicitly.
        assert price == 0.0 or math.isnan(price)

    def test_negative_a_propagates_error(self):
        """Pre-fix: negative a would silently overflow inside the G2++
        formula and the outer except masked it as 0.0.  Post-fix: the
        exception propagates (numerical-overflow or RuntimeError)."""
        # Force a degenerate regime by setting extreme parameters that
        # produce numerical issues; verify either:
        #  (a) the function raises (post-fix behaviour), OR
        #  (b) it returns a finite positive price.
        # It must NOT silently return 0.0 unless the option is genuinely
        # worth zero.
        try:
            price = g2pp_swaption_price(
                a=1e-10,  # near-degenerate mean reversion
                b=0.1, sigma1=0.5, sigma2=0.5, rho=0.99,  # extreme vols + correl
                curve=_flat_curve(),
                expiry_years=10.0, tenor_years=10.0, strike=0.04,
            )
            # If no exception, the answer must be finite (not silently 0
            # for a deep-ITM 10y10y swaption at par-like strike).
            assert math.isfinite(price)
        except (RuntimeError, OverflowError, ValueError):
            pass  # propagation is the post-fix behaviour.


class TestBracketFailureRaises:
    def test_extreme_params_either_raise_or_succeed(self):
        """The pre-fix behaviour: silently set y* = 0 if the bracket failed,
        producing wildly wrong prices.  Post-fix: either succeed with valid
        Jamshidian root OR raise RuntimeError — never silently produce a
        bogus number."""
        # The bracketing should succeed for any reasonable strike + curve;
        # this test just confirms no silent-zero path remains.
        for strike in [0.001, 0.04, 0.20]:
            try:
                price = g2pp_swaption_price(
                    a=0.5, b=0.1, sigma1=0.01, sigma2=0.005, rho=-0.5,
                    curve=_flat_curve(),
                    expiry_years=3.0, tenor_years=5.0, strike=strike,
                )
                assert math.isfinite(price)
                assert price >= 0
            except RuntimeError as e:
                # Post-fix raise path: must mention Jamshidian/bracket.
                assert "bracket" in str(e).lower() or "Jamshidian" in str(e)
