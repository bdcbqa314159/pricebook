"""Tests for vol surface."""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.vol_surface import FlatVol, VolTermStructure


REF = date(2024, 1, 15)


class TestFlatVol:

    def test_returns_constant(self):
        v = FlatVol(0.20)
        assert v.vol() == 0.20
        assert v.vol(REF + relativedelta(years=1)) == 0.20
        assert v.vol(REF + relativedelta(years=5), strike=100.0) == 0.20

    def test_zero_vol(self):
        v = FlatVol(0.0)
        assert v.vol() == 0.0

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError):
            FlatVol(-0.10)


class TestVolTermStructure:

    def test_at_pillar(self):
        expiries = [REF + relativedelta(months=3), REF + relativedelta(years=1)]
        vols = [0.18, 0.22]
        vts = VolTermStructure(REF, expiries, vols)
        assert vts.vol(expiries[0]) == pytest.approx(0.18, rel=1e-6)
        assert vts.vol(expiries[1]) == pytest.approx(0.22, rel=1e-6)

    def test_interpolates_between_pillars(self):
        expiries = [REF + relativedelta(months=3), REF + relativedelta(years=1)]
        vols = [0.18, 0.22]
        vts = VolTermStructure(REF, expiries, vols)
        mid = REF + relativedelta(months=6)
        v = vts.vol(mid)
        assert 0.18 < v < 0.22

    def test_flat_extrapolation_short(self):
        expiries = [REF + relativedelta(months=3), REF + relativedelta(years=1)]
        vols = [0.18, 0.22]
        vts = VolTermStructure(REF, expiries, vols)
        v = vts.vol(REF + relativedelta(days=1))
        assert v == pytest.approx(0.18, rel=1e-6)

    def test_flat_extrapolation_long(self):
        expiries = [REF + relativedelta(months=3), REF + relativedelta(years=1)]
        vols = [0.18, 0.22]
        vts = VolTermStructure(REF, expiries, vols)
        v = vts.vol(REF + relativedelta(years=5))
        assert v == pytest.approx(0.22, rel=1e-6)

    def test_strike_ignored(self):
        expiries = [REF + relativedelta(years=1)]
        vols = [0.20]
        vts = VolTermStructure(REF, expiries, vols)
        assert vts.vol(expiries[0], strike=80.0) == vts.vol(expiries[0], strike=120.0)

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError):
            VolTermStructure(REF, [REF + relativedelta(years=1)], [-0.10])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            VolTermStructure(REF, [REF + relativedelta(years=1)], [0.20, 0.25])
