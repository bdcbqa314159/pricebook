"""Regression for L2 Wave-2 audit — credit_risk survival-curve bumps.

Two related bugs:

1. `_bump_survival_curve_at(curve, pillar_idx, shift)` (key-rate bump):
   pre-fix it shifted only ``survs[pillar_idx]`` without propagating the
   change to later pillars.  As a side-effect, segment (i+1)'s hazard
   (computed lazily as ``-log(Q_{i+1}/Q_i)/dt``) was contaminated by
   ``-shift``, so the bump leaked into the next segment.

2. `_bump_survival_curve(curve, shift)` (parallel bump): pre-fix it
   extracted segment-i hazard using ``prev_q = new_q`` (already bumped
   from the previous iteration) — partially absorbing the upstream shift
   into the extracted hazard.  Net effect on a flat-hazard curve: the 5y
   survival shifted by only HALF of the expected ``-shift·t``.

Both bugs cancelled each other for `test_sum_approx_cs01` pre-fix
(spurious agreement).  Post-fix both routines extract hazards from the
ORIGINAL curve, bump correctly, and propagate forward.  The mathematical
identity ``sum_i d/dh_i PV = d/d(parallel) PV`` now holds.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.cds import CDS
from pricebook.credit.credit_risk import (
    _bump_survival_curve,
    _bump_survival_curve_at,
    cs01,
    spread_dv01,
)


REF = date(2024, 1, 1)
TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
PILLAR_DATES = [REF + timedelta(days=int(t * 365)) for t in TENORS]


def _flat_survival_curve(hazard: float = 0.02) -> SurvivalCurve:
    survs = [math.exp(-hazard * t) for t in TENORS]
    return SurvivalCurve(REF, PILLAR_DATES, survs, day_count=DayCountConvention.ACT_365_FIXED)


def _discount_curve(rate: float = 0.05) -> DiscountCurve:
    dfs = [math.exp(-rate * t) for t in TENORS]
    return DiscountCurve(REF, PILLAR_DATES, dfs, day_count=DayCountConvention.ACT_365_FIXED)


class TestParallelBumpShiftsAllSegments:
    def test_parallel_bump_shifts_log_survival_proportionally(self):
        """A flat 2% hazard curve bumped by 1bp should yield
        log(Q_new/Q_old) ≈ -shift·t for every pillar.  Pre-fix the
        shift was only ~half the expected at long pillars."""
        sc = _flat_survival_curve(0.02)
        shift = 0.0001
        bumped = _bump_survival_curve(sc, shift)
        # Day-count rounding makes actual pillar times slightly off `t`;
        # use SurvivalCurve's own pillar times.
        actual_times = list(sc._times[1:])  # skip the t=0 anchor
        for t_actual, d in zip(actual_times, PILLAR_DATES):
            new_q = bumped.survival(d)
            old_q = sc.survival(d)
            log_diff = math.log(new_q / old_q)
            assert log_diff == \
                __import__("pytest").approx(-shift * t_actual, abs=1e-12), \
                f"at t={t_actual}: log_diff={log_diff}, expected {-shift * t_actual}"


class TestPerPillarBumpLocal:
    def test_pillar_bump_changes_only_target_segment_hazard(self):
        """Bumping pillar i should change segment-i hazard by +shift
        and leave all other segment hazards unchanged."""
        sc = _flat_survival_curve(0.02)
        shift = 0.0001
        i = 3  # the 2y pillar

        bumped = _bump_survival_curve_at(sc, i, shift)

        # Extract per-segment hazards from both curves
        def hazards(curve):
            hs = []
            prev_t, prev_q = 0.0, 1.0
            for t, d in zip(TENORS, PILLAR_DATES):
                q = curve.survival(d)
                dt = t - prev_t
                hs.append(-math.log(q / prev_q) / dt)
                prev_t, prev_q = t, q
            return hs

        h_old = hazards(sc)
        h_new = hazards(bumped)
        for k, (a, b) in enumerate(zip(h_old, h_new)):
            expected_delta = shift if k == i else 0.0
            assert (b - a) == __import__("pytest").approx(expected_delta, abs=1e-9), \
                f"segment {k}: delta={b - a}, expected {expected_delta}"


class TestSumKeyRateEqualsParallel:
    def test_sum_of_key_rate_cs01s_equals_parallel_cs01(self):
        """The sum of per-pillar CS01s should match the parallel CS01
        because dPV/d(parallel) = Σ_i dPV/dh_i.  Pre-fix both bump
        routines had compensating errors that made this hold only
        approximately and unreliably."""
        sc = _flat_survival_curve(0.02)
        dc = _discount_curve(0.05)
        cds = CDS(
            REF,
            REF + timedelta(days=5 * 365),
            spread=0.02,
            notional=10_000_000.0,
            recovery=0.4,
        )

        parallel = cs01(cds, dc, sc)
        per_pillar = spread_dv01(cds, dc, sc)
        s = sum(v for _, v in per_pillar)

        assert s == \
            __import__("pytest").approx(parallel, rel=0.01), \
            f"sum_key_rate={s}, parallel={parallel}, ratio={s / parallel}"
