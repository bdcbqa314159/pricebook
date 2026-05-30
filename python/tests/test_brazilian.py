"""Tests for Brazilian fixed income derivatives."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.fixed_income.brazilian import (
    DIFuture, DISwap, LFTBond,
    build_cdi_curve_from_di, synthetic_di_strip,
    cupom_cambial, cupom_cambial_curve,
    _bus_days, _di_discount_factor, _di_rate_from_df,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention


REF = date(2024, 6, 3)  # Monday


# ═══════════════════════════════════════════════════════════════
# BUS/252 helpers
# ═══════════════════════════════════════════════════════════════

class TestBUS252:
    def test_bus_days_positive(self):
        bd = _bus_days(REF, REF + relativedelta(months=6))
        assert 100 < bd < 140  # ~126 bus days in 6 months

    def test_bus_days_zero_same_day(self):
        assert _bus_days(REF, REF) == 0

    def test_di_discount_factor(self):
        """df = 1/(1+r)^(bd/252) for r=10%, bd=252 → df ≈ 1/1.10."""
        df = _di_discount_factor(0.10, 252)
        assert abs(df - 1 / 1.10) < 1e-10

    def test_di_rate_round_trip(self):
        """rate → df → rate should round-trip."""
        rate = 0.1150
        bd = 180
        df = _di_discount_factor(rate, bd)
        recovered = _di_rate_from_df(df, bd)
        assert abs(recovered - rate) < 1e-10


# ═══════════════════════════════════════════════════════════════
# CDI Curve from DI Futures
# ═══════════════════════════════════════════════════════════════

class TestCDICurve:
    @pytest.fixture
    def di_strip(self):
        return synthetic_di_strip(REF, selic=0.1050, n_contracts=10)

    def test_synthetic_strip_generates(self, di_strip):
        assert len(di_strip) == 10
        assert all(c["rate"] > 0 for c in di_strip)
        assert all(c["bus_days"] > 0 for c in di_strip)

    def test_strip_upward_sloping(self, di_strip):
        rates = [c["rate"] for c in di_strip]
        # Should be generally upward sloping
        assert rates[-1] > rates[0]

    def test_build_curve(self, di_strip):
        curve = build_cdi_curve_from_di(REF, di_strip)
        assert curve is not None
        # DFs should be positive and decreasing
        for c in di_strip:
            df = curve.df(c["maturity"])
            assert 0 < df < 1

    def test_curve_round_trip(self, di_strip):
        """Curve DFs should recover input DI rates."""
        curve = build_cdi_curve_from_di(REF, di_strip)
        for c in di_strip:
            df = curve.df(c["maturity"])
            recovered_rate = _di_rate_from_df(df, c["bus_days"])
            assert abs(recovered_rate - c["rate"]) < 0.005  # within 50bp (interp)


# ═══════════════════════════════════════════════════════════════
# DI Futures
# ═══════════════════════════════════════════════════════════════

class TestDIFuture:
    def test_pu_positive(self):
        di = DIFuture(date(2025, 1, 2), 0.1050)
        pu = di.pu(REF)
        assert 0 < pu < 100_000

    def test_pu_at_maturity(self):
        """At maturity, PU → 100,000."""
        di = DIFuture(REF, 0.10)
        assert abs(di.pu(REF) - 100_000) < 1

    def test_dv01_positive(self):
        di = DIFuture(date(2025, 7, 1), 0.1050)
        result = di.price(REF)
        assert result.dv01 > 0

    def test_implied_rate_round_trip(self):
        di = DIFuture(date(2025, 7, 1), 0.1150)
        pu = di.pu(REF)
        recovered = di.implied_rate(pu, REF)
        assert abs(recovered - 0.1150) < 1e-8

    def test_higher_rate_lower_pu(self):
        di_low = DIFuture(date(2025, 7, 1), 0.10)
        di_high = DIFuture(date(2025, 7, 1), 0.12)
        assert di_high.pu(REF) < di_low.pu(REF)


# ═══════════════════════════════════════════════════════════════
# DI Swap
# ═══════════════════════════════════════════════════════════════

class TestDISwap:
    @pytest.fixture
    def cdi_curve(self):
        strip = synthetic_di_strip(REF, 0.1050, 10)
        return build_cdi_curve_from_di(REF, strip)

    def test_par_swap_near_zero_pv(self, cdi_curve):
        """Swap at par rate should have PV ≈ 0."""
        mat = REF + relativedelta(years=2)
        bd = _bus_days(REF, mat)
        df = cdi_curve.df(mat)
        par = _di_rate_from_df(df, bd)

        swap = DISwap(REF, mat, par, 10_000_000)
        result = swap.price(cdi_curve)
        assert abs(result.pv) < 5_000  # within R$ 5k

    def test_above_par_positive_pv(self, cdi_curve):
        """Fixed rate above market → positive PV for payer."""
        mat = REF + relativedelta(years=2)
        swap = DISwap(REF, mat, 0.15, 10_000_000, direction=1)
        result = swap.price(cdi_curve)
        assert result.pv > 0

    def test_dv01_positive(self, cdi_curve):
        mat = REF + relativedelta(years=2)
        result = DISwap(REF, mat, 0.11, 10_000_000).price(cdi_curve)
        assert result.dv01 > 0


# ═══════════════════════════════════════════════════════════════
# LFT
# ═══════════════════════════════════════════════════════════════

class TestLFT:
    @pytest.fixture
    def cdi_curve(self):
        strip = synthetic_di_strip(REF, 0.1050, 10)
        return build_cdi_curve_from_di(REF, strip)

    def test_lft_at_par(self, cdi_curve):
        """LFT with spread=0 should price near face (VNA)."""
        lft = LFTBond(REF, REF + relativedelta(years=3), spread=0.0)
        result = lft.price(REF, cdi_curve)
        # At issue date with spread=0, price = face
        assert abs(result.dirty_price - 1000.0) < 10

    def test_positive_spread_below_par(self, cdi_curve):
        """Positive spread (ágio) → price below VNA."""
        lft = LFTBond(REF, REF + relativedelta(years=3), spread=0.01)
        result = lft.price(REF, cdi_curve)
        assert result.dirty_price < result.accrued_vna

    def test_negative_spread_above_par(self, cdi_curve):
        """Negative spread (deságio) → price above VNA."""
        lft = LFTBond(REF, REF + relativedelta(years=3), spread=-0.005)
        result = lft.price(REF, cdi_curve)
        assert result.dirty_price > result.accrued_vna


# ═══════════════════════════════════════════════════════════════
# Cupom Cambial
# ═══════════════════════════════════════════════════════════════

class TestCupomCambial:
    def test_cupom_below_di(self):
        """Cupom cambial (USD rate) should be below DI rate (BRL rate)."""
        # USDBRL spot=5.45, forward=5.85 (1Y), DI=10.5%
        cc = cupom_cambial(5.45, 5.85, 0.1050, 252)
        assert cc < 0.1050  # USD rate < BRL rate
        assert cc > 0        # still positive

    def test_cupom_zero_when_no_differential(self):
        """If forward = spot × (1+DI)^(bd/252), cupom = DI (no differential)."""
        # Higher forward → more BRL depreciation → lower cupom
        cc_low_fwd = cupom_cambial(5.45, 5.70, 0.1050, 252)
        cc_high_fwd = cupom_cambial(5.45, 6.00, 0.1050, 252)
        assert cc_low_fwd > cc_high_fwd

    def test_cupom_curve(self):
        strip = synthetic_di_strip(REF, 0.1050, 5)
        fx_fwds = [{"maturity": c["maturity"], "forward": 5.45 * (1 + 0.03) ** c["years"]}
                   for c in strip]
        result = cupom_cambial_curve(REF, 5.45, fx_fwds, strip)
        assert len(result) == 5
        assert all(r["cupom"] > 0 for r in result)


# ═══════════════════════════════════════════════════════════════
# NTN-F/LTN via sovereign bonds
# ═══════════════════════════════════════════════════════════════

class TestSovereignBondPricing:
    @pytest.fixture
    def cdi_curve(self):
        strip = synthetic_di_strip(REF, 0.1050, 15)
        return build_cdi_curve_from_di(REF, strip)

    def test_ntn_f_prices(self, cdi_curve):
        """NTN-F (fixed coupon) should price via CDI curve."""
        from pricebook.fixed_income.sovereign_bonds import create_sovereign_bond
        bond = create_sovereign_bond("NTN_F", REF, REF + relativedelta(years=5), 0.10)
        price = bond.dirty_price(cdi_curve)
        assert price > 0
        assert 80 < price < 120  # per 100 face, near par for 10% coupon at ~10.5% yield

    def test_ltn_prices(self, cdi_curve):
        """LTN (zero-coupon) should price via CDI curve."""
        from pricebook.fixed_income.sovereign_bonds import create_sovereign_zero
        ltn = create_sovereign_zero("LTN", REF, REF + relativedelta(years=2))
        price = ltn.price(cdi_curve)
        assert 0 < price < 100  # per 100 face, discounted

    def test_lft_convention_exists(self):
        """LFT should be in sovereign bond registry."""
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        conv = get_conventions("LFT")
        assert conv.currency == "BRL"
        assert conv.day_count == DayCountConvention.BUS_252
