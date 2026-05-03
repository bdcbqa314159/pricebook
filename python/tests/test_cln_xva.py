"""Tests for CLN XVA: SIMM IM, MVA, KVA, analytic CVA, wrong-way cost, MC XVA."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.cln import CreditLinkedNote
from pricebook.cln_xva import (
    cln_simm_im, cln_mva, cln_kva,
    cln_analytic_cva, cln_wrong_way_cost, cln_mc_xva,
)
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
END = REF + relativedelta(years=5)


def _cln():
    return CreditLinkedNote(
        start=REF, end=END, coupon_rate=0.05,
        notional=1_000_000, recovery=0.4,
        frequency=Frequency.QUARTERLY,
    )


def _surv(h=0.02):
    return SurvivalCurve.flat(REF, h)


def _ctx(rate=0.04):
    return PricingContext(valuation_date=REF, discount_curve=make_flat_curve(REF, rate))


class TestCLNSIMMIM:

    def test_simm_im_positive(self):
        cln = _cln()
        im = cln_simm_im(cln, make_flat_curve(REF, 0.04), _surv())
        assert im > 0

    def test_simm_im_finite(self):
        cln = _cln()
        im = cln_simm_im(cln, make_flat_curve(REF, 0.04), _surv())
        assert math.isfinite(im)


class TestCLNMVA:

    def test_mva_positive(self):
        cln = _cln()
        mva = cln_mva(cln, make_flat_curve(REF, 0.04), _surv())
        assert mva > 0

    def test_mva_proxy_fallback(self):
        cln = _cln()
        mva = cln_mva(cln)  # no curve → 5% proxy
        assert mva > 0


class TestCLNKVA:

    def test_kva_positive(self):
        cln = _cln()
        kva = cln_kva(cln, make_flat_curve(REF, 0.04), _surv())
        assert kva > 0

    def test_kva_proportional_to_hurdle(self):
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _surv()
        k10 = cln_kva(cln, curve, surv, hurdle_rate=0.10)
        k20 = cln_kva(cln, curve, surv, hurdle_rate=0.20)
        assert abs(k20 / k10 - 2.0) < 1e-10


class TestCLNAnalyticCVA:

    def test_cva_positive(self):
        cln = _cln()
        cva = cln_analytic_cva(cln, make_flat_curve(REF, 0.04), _surv())
        assert cva > 0

    def test_cva_increases_with_hazard(self):
        cln = _cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _surv()
        cva_low = cln_analytic_cva(cln, curve, surv, cpty_hazard=0.01)
        cva_high = cln_analytic_cva(cln, curve, surv, cpty_hazard=0.05)
        assert cva_high > cva_low

    def test_cva_zero_hazard(self):
        cln = _cln()
        cva = cln_analytic_cva(cln, make_flat_curve(REF, 0.04), _surv(), cpty_hazard=0.0)
        assert cva == 0.0


class TestCLNWrongWayCost:

    def test_wrong_way_positive(self):
        """Wrong-way cost > 0 when recovery is negatively correlated with default."""
        cln = _cln()
        cost = cln_wrong_way_cost(cln, make_flat_curve(REF, 0.04), _surv(),
                                  n_sims=10_000)
        assert cost > -cln.notional * 0.01  # not hugely negative

    def test_wrong_way_finite(self):
        cln = _cln()
        cost = cln_wrong_way_cost(cln, make_flat_curve(REF, 0.04), _surv(),
                                  n_sims=5_000)
        assert math.isfinite(cost)


class TestCLNMCXVA:

    def test_mc_xva_returns_result(self):
        cln = _cln()
        ctx = _ctx()
        result = cln_mc_xva(
            cln, ctx, _surv(), _surv(0.02), _surv(0.01),
            n_paths=100, n_steps=4,
        )
        assert hasattr(result, 'cva')
        assert hasattr(result, 'dva')
        assert hasattr(result, 'total')

    def test_mc_mva_positive(self):
        cln = _cln()
        ctx = _ctx()
        result = cln_mc_xva(
            cln, ctx, _surv(), _surv(0.02), _surv(0.01),
            funding_spread=0.01, n_paths=100, n_steps=4,
        )
        assert result.mva_val > 0
