"""Tests for Chilean fixed income derivatives."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.fixed_income.chilean import (
    CamaraSwap, BCPBond, BCUBond,
    build_clp_curve, build_uf_curve,
    synthetic_clp_strip, synthetic_uf_strip,
    breakeven_inflation,
)

REF = date(2024, 11, 4)


class TestCLPCurve:
    def test_build(self):
        strip = synthetic_clp_strip(REF, tpm=0.0575, n_contracts=8)
        curve = build_clp_curve(REF, strip)
        for c in strip:
            assert 0 < curve.df(c["maturity"]) < 1


class TestUFCurve:
    def test_build(self):
        strip = synthetic_uf_strip(REF, real_rate=0.02, n_contracts=6)
        curve = build_uf_curve(REF, strip)
        for c in strip:
            assert 0 < curve.df(c["maturity"]) < 1

    def test_real_below_nominal(self):
        clp = build_clp_curve(REF, synthetic_clp_strip(REF, 0.0575))
        uf = build_uf_curve(REF, synthetic_uf_strip(REF, 0.02))
        # Real rate < nominal rate (inflation positive)
        mat = REF + relativedelta(years=5)
        assert uf.df(mat) > clp.df(mat)  # higher DF = lower rate


class TestCamaraSwap:
    @pytest.fixture
    def curve(self):
        return build_clp_curve(REF, synthetic_clp_strip(REF, 0.0575))

    def test_par_swap(self, curve):
        swap = CamaraSwap(REF, REF + relativedelta(years=2), 0.0575)
        r = swap.price(curve)
        assert abs(r.pv) < r.notional * 0.05

    def test_dv01(self, curve):
        r = CamaraSwap(REF, REF + relativedelta(years=5), 0.06).price(curve)
        assert r.dv01 > 0


class TestBCPBond:
    def test_price_positive(self):
        curve = build_clp_curve(REF, synthetic_clp_strip(REF, 0.0575))
        bcp = BCPBond(REF, REF + relativedelta(years=5), 0.055)
        price = bcp.dirty_price(curve)
        assert 85 < price < 115


class TestBCUBond:
    def test_real_price(self):
        uf_curve = build_uf_curve(REF, synthetic_uf_strip(REF, 0.02))
        bcu = BCUBond(REF, REF + relativedelta(years=10), 0.025, face_uf=1000)
        r = bcu.price(REF, uf_curve, current_uf=37_000)
        assert r.real_price > 0
        assert r.nominal_price > 0
        assert abs(r.nominal_price - r.real_price * 37_000) < 1

    def test_uf_scaling(self):
        uf_curve = build_uf_curve(REF, synthetic_uf_strip(REF, 0.02))
        bcu = BCUBond(REF, REF + relativedelta(years=5), 0.03)
        r1 = bcu.price(REF, uf_curve, current_uf=35_000)
        r2 = bcu.price(REF, uf_curve, current_uf=40_000)
        assert r2.nominal_price > r1.nominal_price


class TestBreakevenInflation:
    def test_bei_positive(self):
        clp = build_clp_curve(REF, synthetic_clp_strip(REF, 0.0575))
        uf = build_uf_curve(REF, synthetic_uf_strip(REF, 0.02))
        bei = breakeven_inflation(clp, uf)
        assert all(b["breakeven_inflation"] > 0 for b in bei)
        # BEI ≈ nominal - real ≈ 5.75% - 2% ≈ 3.75%
        bei_5y = [b for b in bei if b["maturity_years"] == 5][0]
        assert 0.02 < bei_5y["breakeven_inflation"] < 0.06
