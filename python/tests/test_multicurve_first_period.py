"""Regression for L1 A.3 B1 — multicurve_newton PV_float must include first period.

Pre-fix, the projection-swap PV_float loop started at j=1, skipping the first
period (reference_date → first pillar) while `_compute_annuity` included it.
Result: a systematic bias model_rate ≈ ((N-1)/N) × true_rate, with the solver
either converging to wrong DFs or emitting `RuntimeWarning: did not converge`.

Post-fix, both sides walk the same period grid starting from reference_date,
the solver converges within tolerance, and projection-swap inputs round-trip.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import pytest

from pricebook.curves.multicurve_solver import multicurve_newton


def _two_pillar_inputs():
    """Minimal 2-pillar OIS + 2-pillar projection setup that previously failed."""
    ref = date(2024, 1, 15)
    one_y = ref + timedelta(days=365)
    two_y = ref + timedelta(days=730)
    ois_insts = [
        {"type": "swap", "maturity": one_y, "rate": 0.030},
        {"type": "swap", "maturity": two_y, "rate": 0.032},
    ]
    proj_insts = [
        {"type": "swap", "maturity": one_y, "rate": 0.034},
        {"type": "swap", "maturity": two_y, "rate": 0.036},
    ]
    return ref, ois_insts, proj_insts


class TestMulticurveFirstPeriodFix:
    def test_solver_converges_within_tolerance(self):
        """Pre-fix: residual ~2.86e-03 (warning emitted). Post-fix: converges."""
        ref, ois_insts, proj_insts = _two_pillar_inputs()
        with warnings.catch_warnings():
            # Promote convergence warning to an error so we can assert it's NOT emitted.
            warnings.simplefilter("error", RuntimeWarning)
            result = multicurve_newton(
                ref, ois_insts, proj_insts,
                [d["maturity"] for d in ois_insts],
                [d["maturity"] for d in proj_insts],
                tol=1e-8, max_iter=50,
            )
        assert result.residual < 1e-8, f"Solver did not converge: residual={result.residual}"

    def test_projection_swaps_round_trip(self):
        """Each input projection swap rate must be recoverable from the calibrated curves."""
        ref, ois_insts, proj_insts = _two_pillar_inputs()
        result = multicurve_newton(
            ref, ois_insts, proj_insts,
            [d["maturity"] for d in ois_insts],
            [d["maturity"] for d in proj_insts],
            tol=1e-10, max_iter=100,
        )
        # Re-derive each projection swap rate from the calibrated curves and
        # verify it matches the market input.
        from pricebook.core.day_count import year_fraction, DayCountConvention
        dc = DayCountConvention.ACT_360
        proj = result.projection_curve
        ois = result.ois_curve
        proj_pillars = [d["maturity"] for d in proj_insts]
        for inst in proj_insts:
            dates_up_to = [d for d in proj_pillars if d <= inst["maturity"]]
            pv_float = 0.0
            prev = ref
            for d_end in dates_up_to:
                tau_j = year_fraction(prev, d_end, dc)
                if tau_j > 0:
                    fwd_j = (proj.df(prev) / proj.df(d_end) - 1.0) / tau_j
                    pv_float += fwd_j * tau_j * ois.df(d_end)
                prev = d_end
            # Annuity on OIS
            ann = 0.0
            prev = ref
            for d in dates_up_to:
                ann += year_fraction(prev, d, dc) * ois.df(d)
                prev = d
            recovered = pv_float / ann
            assert recovered == pytest.approx(inst["rate"], abs=1e-8), (
                f"Projection swap maturity={inst['maturity']} did not round-trip: "
                f"input={inst['rate']} recovered={recovered}"
            )
