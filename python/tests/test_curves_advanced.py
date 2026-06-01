"""Tests for advanced curve infrastructure: caplet stripping, inflation dual curve, NDF."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


def _make_curve(rate=0.04):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 15)]
    dfs = [math.exp(-rate * y) for y in range(1, 15)]
    return DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# Caplet vol stripping (V4)
# ═══════════════════════════════════════════════════════════════

class TestCapletStripping:
    def test_strip_basic(self):
        from pricebook.options.capfloor import strip_caplet_vols_from_quotes as strip_caplet_vols
        curve = _make_curve()
        quotes = [
            {"tenor_years": 1.0, "vol": 0.006},
            {"tenor_years": 2.0, "vol": 0.0058},
            {"tenor_years": 5.0, "vol": 0.0052},
        ]
        stripped = strip_caplet_vols(quotes, curve)
        assert len(stripped) > 0
        assert all(s.vol > 0 for s in stripped)

    def test_strip_increasing_tenors(self):
        from pricebook.options.capfloor import strip_caplet_vols_from_quotes as strip_caplet_vols
        curve = _make_curve()
        quotes = [
            {"tenor_years": 1.0, "vol": 0.007},
            {"tenor_years": 3.0, "vol": 0.006},
            {"tenor_years": 5.0, "vol": 0.005},
            {"tenor_years": 10.0, "vol": 0.004},
        ]
        stripped = strip_caplet_vols(quotes, curve)
        assert len(stripped) > 4
        # Tenor should be increasing
        tenors = [s.tenor_years for s in stripped]
        assert tenors == sorted(tenors)

    def test_strip_with_strike(self):
        from pricebook.options.capfloor import strip_caplet_vols_from_quotes as strip_caplet_vols
        curve = _make_curve()
        quotes = [{"tenor_years": 2.0, "vol": 0.006}]
        stripped = strip_caplet_vols(quotes, curve, strike=0.04)
        assert len(stripped) > 0

    def test_sabr_calibration(self):
        from pricebook.options.capfloor import strip_caplet_vols_from_quotes as strip_caplet_vols, calibrate_capfloor_sabr
        curve = _make_curve()
        quotes = [
            {"tenor_years": 1.0, "vol": 0.007},
            {"tenor_years": 2.0, "vol": 0.006},
            {"tenor_years": 5.0, "vol": 0.005},
        ]
        stripped = strip_caplet_vols(quotes, curve)
        sabr_results = calibrate_capfloor_sabr(stripped)
        assert len(sabr_results) > 0
        assert all("alpha" in r for r in sabr_results)

    def test_to_dict(self):
        from pricebook.options.capfloor import strip_caplet_vols_from_quotes as strip_caplet_vols
        curve = _make_curve()
        quotes = [{"tenor_years": 1.0, "vol": 0.006}]
        stripped = strip_caplet_vols(quotes, curve)
        if stripped:
            d = stripped[0].to_dict()
            assert "vol" in d
            assert "forward" in d


# ═══════════════════════════════════════════════════════════════
# Dual real+nominal inflation curves (U3)
# ═══════════════════════════════════════════════════════════════

class TestInflationCurve:
    def test_basic_construction(self):
        from pricebook.curves.inflation_curve import build_real_nominal_curves
        nom = [(1, 0.04), (2, 0.042), (5, 0.045), (10, 0.048)]
        real = [(1, 0.01), (2, 0.012), (5, 0.015), (10, 0.018)]
        result = build_real_nominal_curves(REF, "USD", nom, real)
        assert result.nominal_curve is not None
        assert result.real_curve is not None

    def test_bei_positive(self):
        from pricebook.curves.inflation_curve import build_real_nominal_curves
        nom = [(1, 0.04), (5, 0.045), (10, 0.048)]
        real = [(1, 0.01), (5, 0.015), (10, 0.018)]
        result = build_real_nominal_curves(REF, "USD", nom, real)
        assert all(row["bei"] > 0 for row in result.bei_term_structure)

    def test_bei_magnitude(self):
        """BEI ≈ nominal - real ≈ 3% for these inputs."""
        from pricebook.curves.inflation_curve import build_real_nominal_curves
        nom = [(5, 0.045), (10, 0.048)]
        real = [(5, 0.015), (10, 0.018)]
        result = build_real_nominal_curves(REF, "USD", nom, real)
        for row in result.bei_term_structure:
            assert 0.02 < row["bei"] < 0.04  # ~3%

    def test_em_inflation(self):
        """BRL: high nominal, moderate real → high BEI."""
        from pricebook.curves.inflation_curve import build_real_nominal_curves
        nom = [(1, 0.11), (5, 0.12), (10, 0.13)]
        real = [(1, 0.05), (5, 0.06), (10, 0.065)]
        result = build_real_nominal_curves(REF, "BRL", nom, real)
        assert all(row["bei"] > 0.05 for row in result.bei_term_structure)

    def test_to_dict(self):
        from pricebook.curves.inflation_curve import build_real_nominal_curves
        result = build_real_nominal_curves(REF, "USD",
                                            [(5, 0.04)], [(5, 0.01)])
        d = result.to_dict()
        assert d["currency"] == "USD"


# ═══════════════════════════════════════════════════════════════
# NDF-implied curves (U4 — verify existing)
# ═══════════════════════════════════════════════════════════════

class TestNDFImplied:
    def test_build_ndf_curve(self):
        from pricebook.curves.ndf_implied import build_ndf_implied_curve, NDFQuote
        domestic = _make_curve(0.05)  # USD
        quotes = [
            NDFQuote(1, 7.20, 7.22, 7.18),    # 1M
            NDFQuote(3, 7.25, 7.28, 7.22),    # 3M
            NDFQuote(6, 7.32, 7.36, 7.28),    # 6M
            NDFQuote(12, 7.45, 7.50, 7.40),   # 1Y
        ]
        result = build_ndf_implied_curve(REF, 7.15, quotes, domestic)
        assert result.em_curve is not None
        # Foreign (CNY) DFs should be positive
        assert result.em_curve.df(REF + relativedelta(months=6)) > 0

    def test_cip_basis(self):
        from pricebook.curves.ndf_implied import cip_basis, NDFQuote
        domestic = _make_curve(0.05)
        foreign = _make_curve(0.02)
        quotes = [NDFQuote(3, 7.25), NDFQuote(12, 7.45)]
        basis = cip_basis(REF, 7.15, quotes, domestic, foreign)
        assert isinstance(basis, list)
        assert len(basis) > 0
