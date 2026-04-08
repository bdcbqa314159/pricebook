"""Tests for credit curve relative value."""

import pytest
from datetime import date

from pricebook.credit_rv import (
    cross_name_rv, term_structure_rv, sector_screen,
)
from pricebook.cds import CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


REF = date(2024, 1, 15)


def _dc(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _sc(hazard=0.02):
    return SurvivalCurve.flat(REF, hazard)


def _cds(end_year=2029, spread=0.01):
    return CDS(REF, date(end_year, 1, 15), spread, notional=1_000_000)


# ---- Cross-name RV ----

class TestCrossNameRV:
    def test_identical_names_zero_z(self):
        """Identical credit → all z-scores = 0."""
        dc = _dc()
        sc = _sc(0.02)
        names = {
            "A": (_cds(), sc),
            "B": (_cds(), sc),
            "C": (_cds(), sc),
        }
        results = cross_name_rv(names, dc)
        for r in results:
            assert r.z_score == pytest.approx(0.0)
            assert r.signal == "fair"

    def test_outlier_detected(self):
        dc = _dc()
        names = {
            "A": (_cds(), _sc(0.02)),
            "B": (_cds(), _sc(0.02)),
            "C": (_cds(), _sc(0.02)),
            "D": (_cds(), _sc(0.10)),  # much wider
        }
        results = cross_name_rv(names, dc, threshold=1.5)
        d_result = [r for r in results if r.name == "D"][0]
        assert d_result.z_score > 1.5
        assert d_result.signal == "cheap"

    def test_sorted_by_zscore(self):
        dc = _dc()
        names = {
            "A": (_cds(), _sc(0.01)),
            "B": (_cds(), _sc(0.05)),
            "C": (_cds(), _sc(0.02)),
        }
        results = cross_name_rv(names, dc)
        z_scores = [r.z_score for r in results]
        assert z_scores == sorted(z_scores)

    def test_percentile(self):
        dc = _dc()
        names = {
            "A": (_cds(), _sc(0.01)),
            "B": (_cds(), _sc(0.03)),
            "C": (_cds(), _sc(0.05)),
        }
        results = cross_name_rv(names, dc)
        # Widest should have highest percentile
        widest = max(results, key=lambda r: r.spread)
        assert widest.percentile == pytest.approx(100.0)

    def test_mean_equals_average(self):
        dc = _dc()
        sc1, sc2 = _sc(0.01), _sc(0.03)
        names = {"A": (_cds(), sc1), "B": (_cds(), sc2)}
        results = cross_name_rv(names, dc)
        s1 = _cds().par_spread(dc, sc1)
        s2 = _cds().par_spread(dc, sc2)
        for r in results:
            assert r.peer_mean == pytest.approx((s1 + s2) / 2)


# ---- Term structure RV ----

class TestTermStructureRV:
    def test_flat_curve_zero_slope(self):
        """Same survival curve at both tenors → zero slope."""
        dc = _dc()
        sc = _sc(0.02)
        short = _cds(end_year=2027)
        long = _cds(end_year=2034)
        result = term_structure_rv("ACME", short, long, sc, sc, dc, 3, 10)
        # With flat hazard, par spread is roughly the same at all tenors
        assert abs(result.slope) < 0.001

    def test_steeper_with_higher_long_hazard(self):
        dc = _dc()
        sc_short = _sc(0.01)
        sc_long = _sc(0.03)
        short = _cds(end_year=2027)
        long = _cds(end_year=2034)
        result = term_structure_rv("ACME", short, long, sc_short, sc_long, dc, 3, 10)
        assert result.slope > 0

    def test_signal_with_history(self):
        dc = _dc()
        sc = _sc(0.02)
        short = _cds(end_year=2027)
        long = _cds(end_year=2034)
        result = term_structure_rv(
            "ACME", short, long, sc, sc, dc, 3, 10,
            history=[0.005, 0.006, 0.004, 0.005],
            threshold=2.0,
        )
        assert result.signal in ("steep", "flat", "fair")

    def test_metadata(self):
        dc = _dc()
        sc = _sc(0.02)
        result = term_structure_rv("ACME", _cds(2027), _cds(2034), sc, sc, dc, 3, 10)
        assert result.name == "ACME"
        assert result.short_tenor == 3
        assert result.long_tenor == 10


# ---- Sector screening ----

class TestSectorScreen:
    def test_single_sector(self):
        dc = _dc()
        names = {
            "A": (_cds(), _sc(0.02), "Tech"),
            "B": (_cds(), _sc(0.03), "Tech"),
        }
        results = sector_screen(names, dc)
        assert len(results) == 1
        assert results[0].sector == "Tech"
        assert results[0].n_names == 2

    def test_mean_is_average(self):
        dc = _dc()
        sc1, sc2 = _sc(0.02), _sc(0.03)
        names = {
            "A": (_cds(), sc1, "Tech"),
            "B": (_cds(), sc2, "Tech"),
        }
        results = sector_screen(names, dc)
        s1 = _cds().par_spread(dc, sc1)
        s2 = _cds().par_spread(dc, sc2)
        assert results[0].mean_spread == pytest.approx((s1 + s2) / 2)

    def test_cheapest_richest(self):
        dc = _dc()
        names = {
            "TIGHT": (_cds(), _sc(0.01), "Fin"),
            "WIDE": (_cds(), _sc(0.05), "Fin"),
        }
        results = sector_screen(names, dc)
        assert results[0].cheapest == "WIDE"
        assert results[0].richest == "TIGHT"

    def test_multiple_sectors(self):
        dc = _dc()
        names = {
            "A": (_cds(), _sc(0.02), "Tech"),
            "B": (_cds(), _sc(0.05), "Energy"),
        }
        results = sector_screen(names, dc)
        assert len(results) == 2
        # Sorted by widest mean first
        assert results[0].mean_spread > results[1].mean_spread

    def test_zero_dispersion_uniform(self):
        dc = _dc()
        sc = _sc(0.02)
        names = {
            "A": (_cds(), sc, "Tech"),
            "B": (_cds(), sc, "Tech"),
        }
        results = sector_screen(names, dc)
        assert results[0].dispersion == pytest.approx(0.0, abs=1e-8)
