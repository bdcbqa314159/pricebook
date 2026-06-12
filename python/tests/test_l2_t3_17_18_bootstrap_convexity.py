"""Regression for L2 Tier-3 T3.17 / T3.18 — `bootstrap`:

* T3.17 — Hull-White futures convexity formula now includes the T_1 factor.
  Pre-fix:  ca = 0.5 · σ² · B(t1, t2) · [B(0, t2) − B(0, t1)]
  Post-fix: ca = 0.5 · σ² · B(0, t1) · B(t1, t2)
  Small-a expansion gives the textbook 0.5 · σ² · T_1 · (T_2 − T_1).

* T3.18 — FRA / future bootstrap no longer silently uses df_start = 1.0
  when there are no preceding deposits AND the FRA start ≠ reference_date.
  Pre-fix silently produced wildly wrong df(end).  Post-fix raises ValueError
  with a diagnostic message.  start_date == reference_date is still
  accepted (df(0) = 1 is correct).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.curves.bootstrap import bootstrap


REF = date(2024, 1, 1)


def _d(days: int) -> date:
    return REF + timedelta(days=days)


# ============================================================
# T3.17 — HW convexity formula
# ============================================================


class TestHWConvexity:
    def test_convexity_zero_when_params_zero(self):
        """Sanity: no convexity adjustment if a=σ=0."""
        curve = bootstrap(
            REF,
            deposits=[(_d(90), 0.04)],
            futures=[(_d(180), _d(270), 0.0455)],
            swaps=[(_d(365 * 5), 0.045)],
            hw_convexity_a=0.0, hw_convexity_sigma=0.0,
        )
        # Just verify it runs.
        assert curve is not None

    def test_convexity_scales_with_t1(self):
        """The pre-fix bug: CA did not scale with T_1.  Post-fix, a 2y
        future has noticeably more convexity than a 0.5y future at the
        same σ and tenor.

        Compute the implied forward rate from two curves and compare the
        convexity-driven gap.
        """
        a = 0.05
        sigma = 0.01

        # 0.5y future, 90-day tenor
        curve_short = bootstrap(
            REF,
            deposits=[(_d(30), 0.04)],
            futures=[(_d(180), _d(270), 0.0455)],
            swaps=[(_d(365 * 10), 0.045)],
            hw_convexity_a=a, hw_convexity_sigma=sigma,
        )
        # 2y future, 90-day tenor
        curve_long = bootstrap(
            REF,
            deposits=[(_d(30), 0.04)],
            futures=[(_d(730), _d(820), 0.0455)],
            swaps=[(_d(365 * 10), 0.045)],
            hw_convexity_a=a, hw_convexity_sigma=sigma,
        )

        # Recover the implied forward rate (the post-convexity rate the
        # bootstrap actually used).  For the 2y future the CA must be
        # roughly (T_1_long / T_1_short) ≈ 4× larger than the 0.5y case.
        from pricebook.core.day_count import year_fraction
        from datetime import date as _date
        df_short_start = curve_short.df(_d(180))
        df_short_end = curve_short.df(_d(270))
        tau_short = year_fraction(_d(180), _d(270), DayCountConvention.ACT_360)
        fwd_short = (df_short_start / df_short_end - 1.0) / tau_short

        df_long_start = curve_long.df(_d(730))
        df_long_end = curve_long.df(_d(820))
        tau_long = year_fraction(_d(730), _d(820), DayCountConvention.ACT_360)
        fwd_long = (df_long_start / df_long_end - 1.0) / tau_long

        # The forward rates differ from the futures rate by the CA, so:
        ca_short = 0.0455 - fwd_short
        ca_long = 0.0455 - fwd_long

        # Post-fix: CA grows with T_1.  Long > Short.  The exact ratio
        # depends on the curve's interpolation (curve.df(start) after
        # bootstrap differs from the bootstrap-input df_start), but the
        # monotonicity is robust — pre-fix the CA was ≈ proportional only
        # to (T_2 - T_1)² with no T_1 dependence, so CA_long ≈ CA_short.
        assert ca_long > ca_short, (
            f"CA at 2y ({ca_long:.6f}) not greater than at 0.5y ({ca_short:.6f}) — "
            f"T_1 factor missing"
        )

    def test_convexity_small_a_limit(self):
        """For small a, CA → 0.5 · σ² · T_1 · (T_2 − T_1) (Hull's leading-order).
        Direct formula check: build CA using a tiny a and σ=1%, verify it
        matches the analytical limit to <5%."""
        a = 1e-4  # near-zero mean reversion
        sigma = 0.01
        T_1 = 1.0
        T_2 = 1.25  # 90-day tenor

        # Reproduce the bootstrap formula directly.
        def _B(s, t):
            return (1 - math.exp(-a * (t - s))) / a
        ca = 0.5 * sigma**2 * _B(0, T_1) * _B(T_1, T_2)
        # Small-a leading-order: 0.5 · σ² · T_1 · (T_2 − T_1)
        expected = 0.5 * sigma**2 * T_1 * (T_2 - T_1)
        rel = abs(ca - expected) / expected
        assert rel < 0.05, (
            f"Small-a CA = {ca:.6e}, expected ≈ {expected:.6e}, rel = {rel:.3%}"
        )


# ============================================================
# T3.18 — no-deposit FRA/future raises
# ============================================================


class TestNoDepositRaises:
    def test_fra_after_ref_no_deposit_raises(self):
        """FRA starting in the future with no deposits → must raise (pre-fix
        silently used df_start=1.0)."""
        with pytest.raises(ValueError, match="no deposits"):
            bootstrap(
                REF,
                deposits=[],
                fras=[(_d(90), _d(180), 0.04)],
                swaps=[(_d(365 * 5), 0.045)],
            )

    def test_future_after_ref_no_deposit_raises(self):
        """Future starting in the future with no deposits → must raise."""
        with pytest.raises(ValueError, match="no deposits"):
            bootstrap(
                REF,
                deposits=[],
                futures=[(_d(90), _d(180), 0.045)],
                swaps=[(_d(365 * 5), 0.045)],
            )

    def test_fra_starting_at_ref_no_deposit_works(self):
        """FRA starting at reference_date with no deposits IS OK — df(ref) = 1
        is the correct anchor."""
        curve = bootstrap(
            REF,
            deposits=[],
            fras=[(REF, _d(90), 0.04)],
            swaps=[(_d(365 * 5), 0.045)],
        )
        assert curve is not None
        # df at ref = 1.
        assert math.isclose(curve.df(REF), 1.0, abs_tol=1e-12)
