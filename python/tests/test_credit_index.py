"""Tests for credit index flow trading."""

import pytest
from datetime import date

from pricebook.credit_index import (
    IndexConstituent, IndexDefinition, IndexSeries,
    index_skew, index_roll_pnl,
)
from pricebook.cds import CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


def _dc(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _sc(hazard=0.02):
    return SurvivalCurve.flat(REF, hazard)


def _cds(spread=0.01):
    return CDS(REF, END, spread, notional=1_000_000)


def _constituents(n=5, hazards=None):
    if hazards is None:
        hazards = [0.01 + 0.005 * i for i in range(n)]
    names = [f"NAME_{i}" for i in range(n)]
    return [
        IndexConstituent(
            name=names[i],
            cds=_cds(spread=0.01),
            survival_curve=_sc(hazards[i]),
        )
        for i in range(n)
    ]


def _uniform_constituents(n=5, hazard=0.02):
    """All names with identical credit."""
    return _constituents(n, hazards=[hazard] * n)


# ---- Index Definition ----

class TestIndexDefinition:
    def test_create(self):
        idx = IndexDefinition("CDX.NA.IG", _constituents(), series=42)
        assert idx.index_name == "CDX.NA.IG"
        assert idx.n_names == 5
        assert idx.series == 42

    def test_intrinsic_spread(self):
        dc = _dc()
        idx = IndexDefinition("CDX", _constituents())
        intrinsic = idx.intrinsic_spread(dc)
        assert intrinsic > 0

    def test_uniform_intrinsic_equals_par(self):
        """Uniform constituents → intrinsic = single-name par spread."""
        dc = _dc()
        sc = _sc(0.02)
        constituents = _uniform_constituents(5, 0.02)
        idx = IndexDefinition("CDX", constituents)
        intrinsic = idx.intrinsic_spread(dc)
        single_par = _cds().par_spread(dc, sc)
        assert intrinsic == pytest.approx(single_par, rel=0.01)

    def test_constituent_spreads(self):
        dc = _dc()
        idx = IndexDefinition("CDX", _constituents(3))
        spreads = idx.constituent_spreads(dc)
        assert len(spreads) == 3
        assert all(s > 0 for s in spreads.values())

    def test_dispersion_zero_for_uniform(self):
        dc = _dc()
        idx = IndexDefinition("CDX", _uniform_constituents(5, 0.02))
        disp = idx.spread_dispersion(dc)
        assert disp == pytest.approx(0.0, abs=1e-8)

    def test_dispersion_positive_for_diverse(self):
        dc = _dc()
        idx = IndexDefinition("CDX", _constituents(5))
        disp = idx.spread_dispersion(dc)
        assert disp > 0

    def test_index_pv(self):
        dc = _dc()
        idx = IndexDefinition("CDX", _constituents())
        pv = idx.index_pv(dc)
        assert isinstance(pv, float)

    def test_weights(self):
        c = _constituents(3)
        c[0] = IndexConstituent("A", _cds(), _sc(0.02), weight=2.0)
        c[1] = IndexConstituent("B", _cds(), _sc(0.02), weight=1.0)
        c[2] = IndexConstituent("C", _cds(), _sc(0.02), weight=1.0)
        idx = IndexDefinition("CDX", c)
        assert idx.weight_normalised(0) == pytest.approx(0.5)
        assert idx.weight_normalised(1) == pytest.approx(0.25)


# ---- Index Series ----

class TestIndexSeries:
    def test_add_and_get(self):
        s = IndexSeries("CDX.NA.IG")
        s.add_series(IndexDefinition("CDX", _constituents(), series=41))
        s.add_series(IndexDefinition("CDX", _constituents(), series=42), is_current=True)
        assert s.current().series == 42

    def test_off_the_run(self):
        s = IndexSeries("CDX.NA.IG")
        s.add_series(IndexDefinition("CDX", _constituents(), series=41))
        s.add_series(IndexDefinition("CDX", _constituents(), series=42), is_current=True)
        otr = s.off_the_run()
        assert len(otr) == 1
        assert otr[0].series == 41

    def test_missing_current_raises(self):
        s = IndexSeries("CDX.NA.IG")
        with pytest.raises(KeyError):
            s.current()


# ---- Index Skew ----

class TestIndexSkew:
    def test_zero_skew_at_intrinsic(self):
        dc = _dc()
        idx = IndexDefinition("CDX", _constituents())
        intrinsic = idx.intrinsic_spread(dc)
        result = index_skew(idx, intrinsic, dc)
        assert result.skew == pytest.approx(0.0, abs=1e-10)

    def test_positive_skew(self):
        dc = _dc()
        idx = IndexDefinition("CDX", _constituents())
        intrinsic = idx.intrinsic_spread(dc)
        result = index_skew(idx, intrinsic + 0.001, dc)
        assert result.skew > 0

    def test_zero_dispersion_uniform(self):
        dc = _dc()
        idx = IndexDefinition("CDX", _uniform_constituents())
        intrinsic = idx.intrinsic_spread(dc)
        result = index_skew(idx, intrinsic, dc)
        assert result.dispersion == pytest.approx(0.0, abs=1e-8)
        assert result.skew == pytest.approx(0.0, abs=1e-10)

    def test_metadata(self):
        dc = _dc()
        idx = IndexDefinition("CDX.NA.IG", _constituents())
        result = index_skew(idx, 0.01, dc)
        assert result.index_name == "CDX.NA.IG"
        assert result.n_names == 5


# ---- Index Roll ----

class TestIndexRoll:
    def test_same_composition_zero_change(self):
        dc = _dc()
        same = _uniform_constituents(5, 0.02)
        old = IndexDefinition("CDX", same, series=41)
        new = IndexDefinition("CDX", same, series=42)
        result = index_roll_pnl(old, new, dc)
        assert result.roll_spread_change == pytest.approx(0.0, abs=1e-10)
        assert result.names_added == []
        assert result.names_removed == []

    def test_composition_change(self):
        dc = _dc()
        old_c = [
            IndexConstituent("A", _cds(), _sc(0.02)),
            IndexConstituent("B", _cds(), _sc(0.02)),
            IndexConstituent("C", _cds(), _sc(0.02)),
        ]
        new_c = [
            IndexConstituent("A", _cds(), _sc(0.02)),
            IndexConstituent("B", _cds(), _sc(0.02)),
            IndexConstituent("D", _cds(), _sc(0.03)),
        ]
        old = IndexDefinition("CDX", old_c, series=41)
        new = IndexDefinition("CDX", new_c, series=42)
        result = index_roll_pnl(old, new, dc)
        assert result.names_added == ["D"]
        assert result.names_removed == ["C"]

    def test_wider_new_series(self):
        """New series with wider credits → positive spread change."""
        dc = _dc()
        old_c = _uniform_constituents(3, 0.01)
        new_c = _uniform_constituents(3, 0.03)
        old = IndexDefinition("CDX", old_c, series=41)
        new = IndexDefinition("CDX", new_c, series=42)
        result = index_roll_pnl(old, new, dc)
        assert result.roll_spread_change > 0

    def test_series_numbers(self):
        dc = _dc()
        old = IndexDefinition("CDX", _constituents(), series=41)
        new = IndexDefinition("CDX", _constituents(), series=42)
        result = index_roll_pnl(old, new, dc)
        assert result.old_series == 41
        assert result.new_series == 42
