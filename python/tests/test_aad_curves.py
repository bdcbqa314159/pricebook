"""Tests for AAD-aware discount and survival curves."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.aad import Number, Tape
from pricebook.aad_curves import AADDiscountCurve, AADSurvivalCurve
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


REF = date(2024, 1, 15)
RATE = 0.05
HAZARD = 0.02


def _pillar_dates():
    return [REF + relativedelta(years=t) for t in [1, 2, 3, 5, 7, 10]]


def _pillar_dfs():
    return [math.exp(-RATE * t) for t in [1, 2, 3, 5, 7, 10]]


def _pillar_survivals():
    return [math.exp(-HAZARD * t) for t in [1, 2, 3, 5, 7, 10]]


class TestAADDiscountCurve:
    def test_df_at_reference(self):
        with Tape():
            dfs = [Number(v) for v in _pillar_dfs()]
            curve = AADDiscountCurve(REF, _pillar_dates(), dfs)
            assert curve.df(REF).value == pytest.approx(1.0)

    def test_df_matches_float_curve(self):
        """AAD curve df matches plain DiscountCurve."""
        float_curve = DiscountCurve(REF, _pillar_dates(), _pillar_dfs())
        query_dates = [REF + relativedelta(months=m) for m in [6, 18, 36, 60, 96]]

        for d in query_dates:
            with Tape():
                dfs = [Number(v) for v in _pillar_dfs()]
                aad_curve = AADDiscountCurve(REF, _pillar_dates(), dfs)
                aad_val = aad_curve.df(d).value
                float_val = float_curve.df(d)
                assert aad_val == pytest.approx(float_val, rel=1e-10)

    def test_pillar_sensitivity(self):
        """d(df(query))/d(df[i]) via AAD for all pillars in one pass."""
        query = REF + relativedelta(months=18)  # between 1Y and 2Y pillars

        with Tape():
            dfs = [Number(v) for v in _pillar_dfs()]
            curve = AADDiscountCurve(REF, _pillar_dates(), dfs)
            result = curve.df(query)
            result.propagate_to_start()

            # Only the two bracketing pillars should have non-zero sensitivity
            adjoints = [df.adjoint for df in dfs]
            # Pillar 0 (1Y) and 1 (2Y) bracket 18M
            assert adjoints[0] != 0.0
            assert adjoints[1] != 0.0
            # Other pillars should be zero
            for a in adjoints[2:]:
                assert a == pytest.approx(0.0, abs=1e-14)

    def test_pillar_sensitivity_fd_check(self):
        """AAD IR01 per pillar matches bump-and-reprice."""
        query = REF + relativedelta(months=30)
        base_dfs = _pillar_dfs()
        eps = 1e-7

        # AAD
        with Tape():
            dfs = [Number(v) for v in base_dfs]
            curve = AADDiscountCurve(REF, _pillar_dates(), dfs)
            result = curve.df(query)
            result.propagate_to_start()
            aad_derivs = [df.adjoint for df in dfs]

        # FD per pillar
        for idx in range(len(base_dfs)):
            up = list(base_dfs)
            up[idx] += eps
            dn = list(base_dfs)
            dn[idx] -= eps

            with Tape():
                v_up = AADDiscountCurve(REF, _pillar_dates(), [Number(v) for v in up]).df(query).value
            with Tape():
                v_dn = AADDiscountCurve(REF, _pillar_dates(), [Number(v) for v in dn]).df(query).value

            fd = (v_up - v_dn) / (2 * eps)
            assert aad_derivs[idx] == pytest.approx(fd, abs=1e-4)

    def test_multiple_queries_one_pass(self):
        """Can compute sensitivities to multiple outputs by summing."""
        queries = [
            REF + relativedelta(months=6),
            REF + relativedelta(years=2),
            REF + relativedelta(years=5),
        ]

        with Tape():
            dfs = [Number(v) for v in _pillar_dfs()]
            curve = AADDiscountCurve(REF, _pillar_dates(), dfs)

            total = Number(0.0)
            for d in queries:
                total = total + curve.df(d)

            total.propagate_to_start()
            # Every pillar that brackets a query should have non-zero adjoint
            adjoints = [df.adjoint for df in dfs]
            assert any(a != 0 for a in adjoints)


class TestAADSurvivalCurve:
    def test_survival_at_reference(self):
        with Tape():
            survs = [Number(v) for v in _pillar_survivals()]
            curve = AADSurvivalCurve(REF, _pillar_dates(), survs)
            assert curve.survival(REF).value == pytest.approx(1.0)

    def test_survival_matches_float(self):
        float_curve = SurvivalCurve(REF, _pillar_dates(), _pillar_survivals())
        query_dates = [REF + relativedelta(months=m) for m in [6, 18, 36, 60]]

        for d in query_dates:
            with Tape():
                survs = [Number(v) for v in _pillar_survivals()]
                aad_curve = AADSurvivalCurve(REF, _pillar_dates(), survs)
                aad_val = aad_curve.survival(d).value
                float_val = float_curve.survival(d)
                assert aad_val == pytest.approx(float_val, rel=1e-10)

    def test_cs01_per_pillar(self):
        """d(survival(query))/d(surv[i]) via AAD matches FD."""
        query = REF + relativedelta(months=30)
        base_survs = _pillar_survivals()
        eps = 1e-7

        with Tape():
            survs = [Number(v) for v in base_survs]
            curve = AADSurvivalCurve(REF, _pillar_dates(), survs)
            result = curve.survival(query)
            result.propagate_to_start()
            aad_derivs = [s.adjoint for s in survs]

        for idx in range(len(base_survs)):
            up = list(base_survs)
            up[idx] += eps
            dn = list(base_survs)
            dn[idx] -= eps

            with Tape():
                v_up = AADSurvivalCurve(REF, _pillar_dates(), [Number(v) for v in up]).survival(query).value
            with Tape():
                v_dn = AADSurvivalCurve(REF, _pillar_dates(), [Number(v) for v in dn]).survival(query).value

            fd = (v_up - v_dn) / (2 * eps)
            assert aad_derivs[idx] == pytest.approx(fd, abs=1e-4)

    def test_bracketing_pillars_only(self):
        """Only bracketing pillars get non-zero sensitivity."""
        query = REF + relativedelta(months=18)  # between 1Y and 2Y

        with Tape():
            survs = [Number(v) for v in _pillar_survivals()]
            curve = AADSurvivalCurve(REF, _pillar_dates(), survs)
            result = curve.survival(query)
            result.propagate_to_start()

            adjoints = [s.adjoint for s in survs]
            assert adjoints[0] != 0.0
            assert adjoints[1] != 0.0
            for a in adjoints[2:]:
                assert a == pytest.approx(0.0, abs=1e-14)
