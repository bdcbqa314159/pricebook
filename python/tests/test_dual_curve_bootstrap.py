"""Tests for dual-curve bootstrap."""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap_forward_curve
from pricebook.ois import bootstrap_ois, OISSwap
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.fra import FRA
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention


REF = date(2024, 1, 15)

# OIS (SOFR) par rates -> discount curve
OIS_RATES = [
    (REF + relativedelta(months=1), 0.0530),
    (REF + relativedelta(months=3), 0.0525),
    (REF + relativedelta(months=6), 0.0510),
    (REF + relativedelta(years=1), 0.0490),
    (REF + relativedelta(years=2), 0.0470),
    (REF + relativedelta(years=3), 0.0455),
    (REF + relativedelta(years=5), 0.0440),
]

# IRS par rates (vs 3M term rate) -> forward curve
# Typically higher than OIS due to credit/liquidity premium
IRS_DEPOSITS = [
    (REF + relativedelta(months=3), 0.0540),
    (REF + relativedelta(months=6), 0.0525),
]

IRS_SWAPS = [
    (REF + relativedelta(years=1), 0.0500),
    (REF + relativedelta(years=2), 0.0485),
    (REF + relativedelta(years=3), 0.0475),
    (REF + relativedelta(years=5), 0.0465),
]


class TestDualCurveBootstrap:

    def _build_curves(self):
        ois_curve = bootstrap_ois(REF, OIS_RATES)
        fwd_curve = bootstrap_forward_curve(
            REF, IRS_SWAPS, ois_curve,
            deposits=IRS_DEPOSITS,
        )
        return ois_curve, fwd_curve

    def test_irs_reprices_at_par(self):
        """IRS priced with dual curves should reprice at par."""
        ois_curve, fwd_curve = self._build_curves()

        for mat, par_rate in IRS_SWAPS:
            swap = InterestRateSwap(
                REF, mat, fixed_rate=par_rate,
                direction=SwapDirection.PAYER,
            )
            pv = swap.pv(ois_curve, projection_curve=fwd_curve)
            assert abs(pv) < 100.0, \
                f"IRS {mat} not at par: PV={pv:.2f}"

    def test_irs_par_rates_recovered(self):
        ois_curve, fwd_curve = self._build_curves()

        for mat, par_rate in IRS_SWAPS:
            swap = InterestRateSwap(REF, mat, fixed_rate=0.0)
            recovered = swap.par_rate(ois_curve, projection_curve=fwd_curve)
            assert recovered == pytest.approx(par_rate, abs=5e-4), \
                f"IRS {mat}: input={par_rate:.4f}, recovered={recovered:.4f}"

    def test_forward_curve_differs_from_ois(self):
        """Forward curve should differ from OIS (credit/liquidity spread)."""
        ois_curve, fwd_curve = self._build_curves()
        mat = REF + relativedelta(years=2)
        assert fwd_curve.zero_rate(mat) != pytest.approx(ois_curve.zero_rate(mat), rel=1e-3)

    def test_both_curves_positive_dfs(self):
        ois_curve, fwd_curve = self._build_curves()
        all_dates = [d for d, _ in IRS_DEPOSITS] + [d for d, _ in IRS_SWAPS]
        for d in all_dates:
            assert ois_curve.df(d) > 0
            assert fwd_curve.df(d) > 0

    def test_fra_consistent_with_forward_curve(self):
        """FRA forward rate from dual-curve should match the forward curve."""
        ois_curve, fwd_curve = self._build_curves()
        fra = FRA(
            REF + relativedelta(months=3),
            REF + relativedelta(months=6),
            strike=0.0,
        )
        # Forward rate from projection curve
        fwd_from_proj = fra.forward_rate(fwd_curve)
        # Should be positive and different from OIS forward
        fwd_from_ois = fra.forward_rate(ois_curve)
        assert fwd_from_proj > 0
        assert fwd_from_proj != pytest.approx(fwd_from_ois, rel=1e-3)

    def test_fra_pv_zero_at_par_dual_curve(self):
        ois_curve, fwd_curve = self._build_curves()
        fra = FRA(
            REF + relativedelta(months=3),
            REF + relativedelta(months=6),
            strike=0.0,
        )
        par = fra.par_rate(ois_curve, projection_curve=fwd_curve)
        fra_at_par = FRA(
            REF + relativedelta(months=3),
            REF + relativedelta(months=6),
            strike=par,
        )
        assert fra_at_par.pv(ois_curve, projection_curve=fwd_curve) == pytest.approx(0.0, abs=0.01)


class TestSingleCurveRecovery:
    """Dual-curve with identical curves should match single-curve results."""

    def test_same_curve_matches_single(self):
        """When discount = projection, dual-curve PV = single-curve PV."""
        ois_curve = bootstrap_ois(REF, OIS_RATES)
        swap = InterestRateSwap(REF, REF + relativedelta(years=3), fixed_rate=0.045)
        pv_single = swap.pv(ois_curve)
        pv_dual = swap.pv(ois_curve, projection_curve=ois_curve)
        assert pv_single == pytest.approx(pv_dual, rel=1e-10)


class TestValidation:

    def test_unsorted_swaps_raises(self):
        ois_curve = bootstrap_ois(REF, OIS_RATES)
        bad = [(REF + relativedelta(years=3), 0.05), (REF + relativedelta(years=1), 0.04)]
        with pytest.raises(ValueError):
            bootstrap_forward_curve(REF, bad, ois_curve)

    def test_unsorted_deposits_raises(self):
        ois_curve = bootstrap_ois(REF, OIS_RATES)
        bad_deps = [(REF + relativedelta(months=6), 0.05), (REF + relativedelta(months=3), 0.04)]
        with pytest.raises(ValueError):
            bootstrap_forward_curve(REF, IRS_SWAPS, ois_curve, deposits=bad_deps)
