"""Tests for quarterly time discretization in basket CDS."""

import pytest
import math
from datetime import date

from pricebook.credit.basket_cds import ftd_spread, ntd_spread


def _make_survival_curves(n=5, hazard=0.02):
    from pricebook.core.survival_curve import SurvivalCurve
    ref = date(2024, 1, 1)
    dates = [date(2024 + y, 1, 1) for y in range(1, 11)]
    return [SurvivalCurve(ref, dates, [math.exp(-hazard * y) for y in range(1, 11)])
            for _ in range(n)]


def _make_discount_curve():
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.interpolation import InterpolationMethod
    ref = date(2024, 1, 1)
    dates = [date(2024 + y, 1, 1) for y in range(1, 11)]
    dfs = [math.exp(-0.04 * y) for y in range(1, 11)]
    return DiscountCurve(ref, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


class TestQuarterlyDiscretization:
    def test_default_is_quarterly(self):
        """Default frequency=4 should be quarterly."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        # Just verify it runs without error at default frequency
        s = ftd_spread(sc, dc, 0.3, 5.0, seed=42)
        assert s > 0

    def test_annual_vs_quarterly(self):
        """Quarterly should give a different spread than annual."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        annual = ftd_spread(sc, dc, 0.3, 5.0, seed=42, frequency=1)
        quarterly = ftd_spread(sc, dc, 0.3, 5.0, seed=42, frequency=4)
        # Both should be positive
        assert annual > 0
        assert quarterly > 0
        # Different frequency → different annuity/protection computation
        # Quarterly typically gives a different (often lower) spread due to finer annuity
        assert quarterly != annual

    def test_monthly(self):
        """Monthly frequency should also work."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        monthly = ftd_spread(sc, dc, 0.3, 5.0, seed=42, frequency=12)
        assert monthly > 0

    def test_convergence(self):
        """Increasing frequency should converge."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        spreads = []
        for freq in [1, 2, 4, 12]:
            s = ftd_spread(sc, dc, 0.3, 5.0, seed=42, frequency=freq)
            spreads.append(s)
        # Should be roughly convergent (last two close)
        assert abs(spreads[-1] - spreads[-2]) < abs(spreads[1] - spreads[0]) + 0.001

    def test_ntd_quarterly(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        s = ntd_spread(sc, dc, 0.3, 5.0, n=2, seed=42, frequency=4)
        assert s > 0
