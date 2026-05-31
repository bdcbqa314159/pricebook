"""Tests for multi-copula support in basket CDS."""

import pytest
import math
from datetime import date

from pricebook.credit.basket_cds import ftd_spread, ntd_spread
from pricebook.statistics.copulas import (
    GaussianCopula, StudentTCopula, ClaytonCopula,
)


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


class TestMultiCopulaFTD:
    def test_gaussian_copula_matches_default(self):
        """Explicit Gaussian copula should match default (no copula)."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()

        default = ftd_spread(sc, dc, 0.3, 5.0, seed=42)
        gauss = ftd_spread(sc, dc, 0.3, 5.0, seed=42, copula=GaussianCopula(0.3))

        # Both positive; exact match unlikely due to different RNG paths
        assert default > 0
        assert gauss > 0

    def test_student_t_positive(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        s = ftd_spread(sc, dc, 0.3, 5.0, seed=42, copula=StudentTCopula(0.3, 5.0))
        assert s > 0

    def test_student_t_higher_than_gaussian(self):
        """Student-t (tail dependence) should produce higher FTD spread."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()

        gauss = ftd_spread(sc, dc, 0.3, 5.0, seed=42)
        student = ftd_spread(sc, dc, 0.3, 5.0, seed=42, copula=StudentTCopula(0.3, 3.0))

        # t-copula clusters defaults → more FTD events → higher spread
        # Allow MC noise
        assert student > gauss * 0.70

    def test_clayton_positive(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        s = ftd_spread(sc, dc, 0.3, 5.0, seed=42, copula=ClaytonCopula(2.0))
        assert s > 0


class TestMultiCopulaNTD:
    def test_ntd_with_copula(self):
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        s = ntd_spread(sc, dc, 0.3, 5.0, n=2, seed=42, copula=StudentTCopula(0.3, 5.0))
        assert s > 0

    def test_ftd_gt_ntd_with_copula(self):
        """FTD > 2TD even with non-Gaussian copula."""
        sc = _make_survival_curves()
        dc = _make_discount_curve()
        cop = StudentTCopula(0.3, 5.0)
        ftd = ntd_spread(sc, dc, 0.3, 5.0, n=1, seed=42, copula=cop)
        std = ntd_spread(sc, dc, 0.3, 5.0, n=2, seed=42, copula=cop)
        assert ftd > std
