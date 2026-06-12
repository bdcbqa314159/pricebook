"""Regression for L2 Tier-2 T2.16 / T2.17 / T2.18 — CDS variable-notional handling.

Three coupled bugs in `credit/cds.py`:

* T2.16 — `cds_pnl_attribution` computed convexity P&L as
  `0.5 × spread_convexity × |PV| × Δs²`.  But `spread_convexity` is
  d²PV/ds² / notional, so the second-order P&L correction is
  ½ × (conv × notional) × Δs² — not ×|PV|.  Pre-fix the formula collapsed
  to ~0 near par (where |PV|≈0, e.g. when spread ≈ par spread) and was
  dimensionally inconsistent elsewhere.

* T2.17 — `isda_upfront`, `roll_down`, `theta`, `rec01`, and
  `cds_pnl_attribution` all built an aged/parallel `CDS` with
  `notional=self.notional`.  But `self.notional` is the SCALAR first
  period (`self.notional_schedule[0]`); for variable-notional CDS the
  tail-period notionals were silently dropped.

* T2.18 — `protection_leg_pv` accepted a list `notional` without
  `schedule_dates`, then fell through to `n = notional[0]` and scaled the
  whole leg by the first period's notional.  Now raises.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.cds import (
    CDS,
    cds_pnl_attribution,
    protection_leg_pv,
)


REF = date(2024, 1, 1)


def _flat_disc(rate: float = 0.05) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    dfs = [math.exp(-rate * t) for t in tenors]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


def _flat_surv(hazard: float = 0.02) -> SurvivalCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    survs = [math.exp(-hazard * t) for t in tenors]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return SurvivalCurve(REF, dates, survs, day_count=DayCountConvention.ACT_365_FIXED)


# ============================================================
# T2.18 — protection_leg_pv must not silently drop variable notional
# ============================================================


class TestProtectionLegRequiresScheduleForList:
    def test_raises_on_list_without_schedule(self):
        with pytest.raises(ValueError, match="schedule_dates"):
            protection_leg_pv(
                REF, REF + timedelta(days=365),
                _flat_disc(), _flat_surv(),
                notional=[1_000_000.0, 2_000_000.0, 3_000_000.0],
                schedule_dates=None,
            )

    def test_scalar_path_unchanged(self):
        """Sanity: scalar notional still works (the BC path)."""
        pv = protection_leg_pv(
            REF, REF + timedelta(days=365),
            _flat_disc(), _flat_surv(),
            notional=1_000_000.0,
        )
        assert pv > 0


# ============================================================
# T2.17 — variable notional propagates through aged-CDS methods
# ============================================================


class TestVariableNotionalPropagation:
    """Build a CDS with a non-trivial notional schedule (e.g. amortising) and
    verify that aged-CDS methods see the FULL schedule, not just the first
    period."""

    def _build_variable_cds(self):
        """5y amortising CDS: notional drops from 10M to 5M linearly."""
        start = REF
        end = REF + timedelta(days=5 * 365)
        # Quarterly schedule has ~20 periods.  Build a clearly-non-constant list.
        notionals = [10_000_000.0 * (1 - 0.04 * i) for i in range(20)]  # 10M → 2.4M
        return CDS(start, end, spread=0.01, notional=notionals,
                   recovery=0.4, day_count=DayCountConvention.ACT_360)

    def test_average_notional_not_scalar(self):
        """Sanity: the test instrument really is variable."""
        cds = self._build_variable_cds()
        assert cds.average_notional != cds.notional_schedule[0]
        assert len(set(cds.notional_schedule)) > 1

    def test_isda_upfront_uses_average_notional(self):
        """Pre-fix isda_upfront passed `self.notional` (= first period) to the
        std CDS, so the upfront was computed against an effective constant
        10M notional instead of the variable schedule."""
        cds = self._build_variable_cds()
        disc, surv = _flat_disc(), _flat_surv()
        # Build a constant-10M CDS for comparison.
        const_cds = CDS(cds.start, cds.end, cds.spread,
                        notional=10_000_000.0, recovery=cds.recovery,
                        day_count=DayCountConvention.ACT_360)
        u_var = cds.isda_upfront(disc, surv, standard_coupon=0.005)
        u_const = const_cds.isda_upfront(disc, surv, standard_coupon=0.005)
        # The variable CDS has LOWER total notional exposure → upfront per unit
        # notional may differ.  The two should NOT be identical (pre-fix they
        # would be, since variable just collapsed to const-10M).
        assert abs(u_var - u_const) > 1e-8, (
            "isda_upfront for variable vs constant notional was identical — "
            "likely T2.17 not actually applied."
        )

    def test_roll_down_sees_variable_notional(self):
        """Pre-fix roll_down constructed shorter = CDS(notional=self.notional)
        which was a SCALAR first-period.  Now we pass the sliced list."""
        cds = self._build_variable_cds()
        disc, surv = _flat_disc(), _flat_surv()
        # Roll 200 days forward — that's ~2 quarters into the original schedule.
        rd_var = cds.roll_down(disc, surv, horizon_days=200)
        # Compare with a deliberately constant-first-period CDS.
        const_first = CDS(cds.start, cds.end, cds.spread,
                          notional=cds.notional_schedule[0],
                          recovery=cds.recovery,
                          day_count=DayCountConvention.ACT_360)
        rd_const = const_first.roll_down(disc, surv, horizon_days=200)
        # Variable notional → different roll-down.
        assert abs(rd_var - rd_const) > 1e-6, (
            "roll_down for variable vs first-period-only was identical — "
            "T2.17 fix didn't propagate."
        )

    def test_theta_sees_variable_notional(self):
        cds = self._build_variable_cds()
        disc, surv = _flat_disc(), _flat_surv()
        th_var = cds.theta(disc, surv, days=180)
        const_first = CDS(cds.start, cds.end, cds.spread,
                          notional=cds.notional_schedule[0],
                          recovery=cds.recovery,
                          day_count=DayCountConvention.ACT_360)
        th_const = const_first.theta(disc, surv, days=180)
        assert abs(th_var - th_const) > 1e-6


# ============================================================
# T2.16 — convexity P&L uses notional, not |PV|
# ============================================================


class TestConvexityFormula:
    def test_convexity_pnl_nonzero_at_par(self):
        """At-par CDS has |PV|≈0; pre-fix `pv_for_conv = abs(pv_t0)` collapsed
        the convexity term to ~0.  Post-fix should scale by notional and give
        a non-trivial convexity contribution for non-trivial Δspread."""
        start, end = REF, REF + timedelta(days=5 * 365)
        disc_t0 = _flat_disc(rate=0.05)
        surv_t0 = _flat_surv(hazard=0.02)
        # Build a CDS at PAR — find the par spread first.
        probe = CDS(start, end, 0.01, notional=10_000_000.0,
                    day_count=DayCountConvention.ACT_360)
        par_t0 = probe.par_spread(disc_t0, surv_t0)
        cds_at_par = CDS(start, end, par_t0, notional=10_000_000.0,
                         day_count=DayCountConvention.ACT_360)

        # t1: widen hazard by 100bp.
        disc_t1 = disc_t0
        surv_t1 = _flat_surv(hazard=0.03)
        attr = cds_pnl_attribution(cds_at_par, disc_t0, surv_t0,
                                   disc_t1, surv_t1, horizon_days=1)

        # Convexity P&L should now be NONZERO and scale with notional × Δs².
        # Pre-fix pv_for_conv = abs(pv_t0) ≈ 0, so convexity_pnl was ~0.
        assert abs(attr.convexity) > 0.01, (
            f"Convexity P&L collapsed to {attr.convexity:.6f} at par — "
            "T2.16 fix didn't apply (pre-fix used |PV| ≈ 0 instead of notional)."
        )

    def test_convexity_scales_with_notional(self):
        """Convexity P&L should scale linearly with notional (since both
        spread_convexity normalises by notional, and the formula multiplies
        by notional)."""
        start, end = REF, REF + timedelta(days=5 * 365)
        disc_t0 = _flat_disc()
        surv_t0 = _flat_surv()
        # Two CDS identical except notional.
        cds_1m = CDS(start, end, 0.01, notional=1_000_000.0,
                     day_count=DayCountConvention.ACT_360)
        cds_10m = CDS(start, end, 0.01, notional=10_000_000.0,
                      day_count=DayCountConvention.ACT_360)
        surv_t1 = _flat_surv(hazard=0.03)
        attr_1m = cds_pnl_attribution(cds_1m, disc_t0, surv_t0,
                                      disc_t0, surv_t1, horizon_days=1)
        attr_10m = cds_pnl_attribution(cds_10m, disc_t0, surv_t0,
                                       disc_t0, surv_t1, horizon_days=1)
        # 10× notional → 10× convexity P&L.
        ratio = attr_10m.convexity / attr_1m.convexity
        assert abs(ratio - 10.0) < 0.01, (
            f"Convexity P&L ratio for 10×-notional CDS: {ratio:.4f} "
            f"(expected ~10).  Pre-fix used |PV| which scales differently."
        )
