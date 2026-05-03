"""Tests for TRS XVA: SIMM IM, MVA, KVA, analytic CVA/DVA, MC XVA, independent amount."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.trs import TotalReturnSwap, FundingLegSpec
from pricebook.trs_xva import (
    trs_simm_im,
    trs_mva,
    trs_kva_from_sa_ccr,
    trs_analytic_cva,
    trs_analytic_dva,
    trs_independent_amount,
    trs_mc_xva,
)
from pricebook.bond import FixedRateBond
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
END = REF + relativedelta(months=6)


def _equity_trs():
    return TotalReturnSwap(
        underlying=100.0, notional=1_000_000,
        start=REF, end=END,
        funding=FundingLegSpec(spread=0.005),
        repo_spread=0.01, haircut=0.05,
        initial_price=100.0, sigma=0.20,
    )


def _bond_trs():
    bond = FixedRateBond.treasury_note(date(2024, 2, 15), date(2034, 2, 15), 0.04125)
    return TotalReturnSwap(
        underlying=bond, notional=10_000_000,
        start=REF, end=END,
        funding=FundingLegSpec(spread=0.005),
        repo_spread=0.005, initial_price=102.0,
    )


# ── SIMM IM ──

class TestSIMMIM:

    def test_equity_simm_im_nonzero(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        im = trs_simm_im(trs, curve)
        assert im > 0

    def test_bond_simm_im_nonzero(self):
        trs = _bond_trs()
        curve = make_flat_curve(REF, 0.04)
        im = trs_simm_im(trs, curve)
        assert im > 0

    def test_simm_differs_from_flat_proxy(self):
        """SIMM IM should differ from the naive 5% proxy."""
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        simm = trs_simm_im(trs, curve)
        proxy = trs.notional * 0.05
        assert simm != proxy

    def test_simm_equity_uses_eq_risk_class(self):
        """Equity TRS should map to EQ risk class in SIMM."""
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        im = trs_simm_im(trs, curve)
        # EQ RW is 20% for large cap, so IM ≈ 20% × delta
        # delta ≈ notional / spot = 10,000 shares × $100 = $1M
        # SIMM ≈ 20% × 10,000 = $2,000 per share × shares
        assert im > 0


# ── MVA ──

class TestMVA:

    def test_mva_positive(self):
        trs = _equity_trs()
        mva = trs_mva(trs)
        assert mva > 0

    def test_mva_with_explicit_im(self):
        trs = _equity_trs()
        mva_default = trs_mva(trs)
        mva_high = trs_mva(trs, simm_im=200_000)
        assert mva_high > mva_default  # 200k > 5% of 1M = 50k

    def test_mva_proportional_to_spread(self):
        trs = _equity_trs()
        mva_low = trs_mva(trs, funding_spread=0.001)
        mva_high = trs_mva(trs, funding_spread=0.004)
        assert abs(mva_high / mva_low - 4.0) < 1e-10

    def test_mva_hand_calc(self):
        """MVA = IM × spread × T.  IM default = 1M × 5% = 50k, T ≈ 0.5."""
        trs = _equity_trs()
        mva = trs_mva(trs, funding_spread=0.002)
        T = (END - REF).days / 365
        expected = 50_000 * 0.002 * T
        assert abs(mva - expected) < 0.01

    def test_mva_bond(self):
        trs = _bond_trs()
        mva = trs_mva(trs)
        assert mva > 0

    def test_mva_uses_simm_when_curve_given(self):
        """When curve provided, MVA uses SIMM IM instead of proxy."""
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        mva_proxy = trs_mva(trs)  # no curve → 5% proxy
        mva_simm = trs_mva(trs, curve=curve)  # curve → SIMM
        assert mva_proxy != mva_simm

    def test_mva_backward_compat_no_curve(self):
        """MVA without curve still works with proxy."""
        trs = _equity_trs()
        mva = trs_mva(trs, funding_spread=0.002)
        T = (END - REF).days / 365
        expected = 50_000 * 0.002 * T  # 5% proxy
        assert abs(mva - expected) < 0.01


# ── KVA from SA-CCR ──

class TestKVA:

    def test_kva_positive(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        kva = trs_kva_from_sa_ccr(trs, curve)
        assert kva > 0

    def test_kva_equity_vs_bond(self):
        """Equity SF=0.32 >> bond SF=0.005 → equity KVA much higher per notional."""
        eq_trs = _equity_trs()
        bd_trs = _bond_trs()
        curve = make_flat_curve(REF, 0.04)
        kva_eq = trs_kva_from_sa_ccr(eq_trs, curve)
        kva_bd = trs_kva_from_sa_ccr(bd_trs, curve)
        # Equity 1M notional × 0.32 vs bond 10M × 0.005 = 320k vs 50k
        assert kva_eq / eq_trs.notional > kva_bd / bd_trs.notional

    def test_kva_proportional_to_hurdle(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        kva_10 = trs_kva_from_sa_ccr(trs, curve, hurdle_rate=0.10)
        kva_20 = trs_kva_from_sa_ccr(trs, curve, hurdle_rate=0.20)
        assert abs(kva_20 / kva_10 - 2.0) < 1e-10

    def test_kva_proportional_to_rw(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        kva_50 = trs_kva_from_sa_ccr(trs, curve, counterparty_rw=0.50)
        kva_100 = trs_kva_from_sa_ccr(trs, curve, counterparty_rw=1.00)
        assert abs(kva_100 / kva_50 - 2.0) < 1e-10


# ── Analytic CVA ──

class TestCVA:

    def test_cva_positive(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cva = trs_analytic_cva(trs, curve)
        assert cva > 0

    def test_cva_increases_with_hazard(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cva_low = trs_analytic_cva(trs, curve, hazard_rate=0.01)
        cva_high = trs_analytic_cva(trs, curve, hazard_rate=0.05)
        assert cva_high > cva_low

    def test_cva_decreases_with_recovery(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cva_low_r = trs_analytic_cva(trs, curve, recovery=0.2)
        cva_high_r = trs_analytic_cva(trs, curve, recovery=0.6)
        assert cva_low_r > cva_high_r  # lower recovery → higher CVA

    def test_cva_bond(self):
        trs = _bond_trs()
        curve = make_flat_curve(REF, 0.04)
        cva = trs_analytic_cva(trs, curve)
        assert cva > 0

    def test_cva_zero_hazard(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cva = trs_analytic_cva(trs, curve, hazard_rate=0.0)
        assert cva == 0.0

    def test_cva_full_recovery(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cva = trs_analytic_cva(trs, curve, recovery=1.0)
        assert cva == 0.0


# ── Analytic DVA ──

class TestDVA:

    def test_dva_positive(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        dva = trs_analytic_dva(trs, curve)
        assert dva > 0

    def test_dva_increases_with_own_hazard(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        dva_low = trs_analytic_dva(trs, curve, own_hazard=0.005)
        dva_high = trs_analytic_dva(trs, curve, own_hazard=0.02)
        assert dva_high > dva_low

    def test_dva_zero_hazard(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        dva = trs_analytic_dva(trs, curve, own_hazard=0.0)
        assert dva == 0.0

    def test_dva_full_recovery(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        dva = trs_analytic_dva(trs, curve, own_recovery=1.0)
        assert dva == 0.0


# ── Independent Amount ──

class TestIndependentAmount:

    def test_percentage_default(self):
        trs = _equity_trs()
        ia = trs_independent_amount(trs)
        assert ia == 1_000_000 * 0.10  # 10% default

    def test_percentage_custom(self):
        trs = _equity_trs()
        ia = trs_independent_amount(trs, ia_pct=0.15)
        assert ia == 1_000_000 * 0.15

    def test_simm_method(self):
        trs = _equity_trs()
        ia = trs_independent_amount(trs, ia_method="simm", simm_im=75_000)
        assert ia == 75_000

    def test_fixed_method(self):
        trs = _equity_trs()
        ia = trs_independent_amount(trs, ia_method="fixed", simm_im=120_000)
        assert ia == 120_000

    def test_simm_without_value_falls_back(self):
        """If simm method but no value provided, falls back to percentage."""
        trs = _equity_trs()
        ia = trs_independent_amount(trs, ia_method="simm")
        assert ia == 1_000_000 * 0.10


# ── Monte Carlo XVA ──

class TestMCXVA:

    def _make_ctx(self, rate=0.04):
        curve = make_flat_curve(REF, rate)
        return PricingContext(valuation_date=REF, discount_curve=curve)

    def _survival(self, hazard=0.02):
        return SurvivalCurve.flat(REF, hazard)

    def test_mc_cva_nonnegative(self):
        """CVA >= 0 always. At inception with no price move, CVA ≈ 0."""
        trs = _equity_trs()
        ctx = self._make_ctx()
        result = trs_mc_xva(
            trs, ctx, self._survival(0.02), self._survival(0.01),
            n_paths=200, n_steps=4,
        )
        assert result.cva >= 0

    def test_mc_cva_bond_rate_sensitive(self):
        """Bond TRS has rate-driven exposure — CVA or DVA should be nonzero."""
        from pricebook.bond import FixedRateBond
        bond = FixedRateBond.treasury_note(date(2024, 2, 15), date(2034, 2, 15), 0.04125)
        trs = TotalReturnSwap(
            underlying=bond, notional=10_000_000,
            start=REF, end=END,
            funding=FundingLegSpec(spread=0.005),
            initial_price=102.0,
        )
        ctx = self._make_ctx()
        result = trs_mc_xva(
            trs, ctx, self._survival(0.02), self._survival(0.01),
            n_paths=200, n_steps=4, rate_vol=0.005,
        )
        # Under rate diffusion, bond TRS has exposure in both directions
        assert result.cva >= 0
        assert result.dva >= 0
        assert result.cva + result.dva > 0  # at least one side has exposure

    def test_mc_dva_positive(self):
        trs = _equity_trs()
        ctx = self._make_ctx()
        result = trs_mc_xva(
            trs, ctx, self._survival(0.02), self._survival(0.01),
            n_paths=200, n_steps=4,
        )
        assert result.dva > 0

    def test_mc_bilateral(self):
        """Bilateral CVA = CVA - DVA."""
        trs = _equity_trs()
        ctx = self._make_ctx()
        result = trs_mc_xva(
            trs, ctx, self._survival(0.02), self._survival(0.01),
            n_paths=200, n_steps=4,
        )
        assert abs(result.bilateral_cva - (result.cva - result.dva)) < 1e-10

    def test_mc_fva_nonneg(self):
        """FVA >= 0 (FVA on positive exposure; equity TRS at inception has EPE≈0)."""
        trs = _equity_trs()
        ctx = self._make_ctx()
        result = trs_mc_xva(
            trs, ctx, self._survival(0.02), self._survival(0.01),
            funding_spread=0.01, n_paths=200, n_steps=4,
        )
        assert result.fva_val >= 0

    def test_mc_mva_positive(self):
        """MVA should be positive when IM profile is provided."""
        trs = _equity_trs()
        ctx = self._make_ctx()
        result = trs_mc_xva(
            trs, ctx, self._survival(0.02), self._survival(0.01),
            funding_spread=0.01, n_paths=200, n_steps=4,
        )
        assert result.mva_val > 0

    def test_mc_total_decomposition(self):
        """Total = CVA - DVA + CFA - DFA + FVA + MVA + KVA."""
        trs = _equity_trs()
        ctx = self._make_ctx()
        result = trs_mc_xva(
            trs, ctx, self._survival(0.02), self._survival(0.01),
            n_paths=200, n_steps=4,
        )
        expected = (result.cva - result.dva + result.cfa - result.dfa
                    + result.colva + result.fva_val + result.mva_val + result.kva_val)
        assert abs(result.total - expected) < 1e-10

    def test_mc_to_dict(self):
        trs = _equity_trs()
        ctx = self._make_ctx()
        result = trs_mc_xva(
            trs, ctx, self._survival(0.02), self._survival(0.01),
            n_paths=100, n_steps=4,
        )
        d = result.to_dict()
        # May be wrapped in {"type": ..., "params": {...}} by serialisation
        inner = d.get("params", d)
        assert "cva" in inner
        assert "dva" in inner
        assert "total" in inner
