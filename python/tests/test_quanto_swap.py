"""Tests for quanto (differential) interest rate swaps."""

import pytest
from datetime import date

from pricebook.fixed_income.quanto_swap import (
    quanto_swap_price,
    differential_swap_price,
    quanto_adjustment_term_structure,
    quanto_fra,
    QuantoSwapResult,
    DifferentialSwapResult,
    QuantoFRAResult,
)
from pricebook.core.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


def _dom_curve():
    return make_flat_curve(REF, 0.05)


def _for_curve():
    return make_flat_curve(REF, 0.03)


class TestQuantoSwapBasic:
    def test_zero_correlation_no_adjustment(self):
        """With ρ=0, quanto adjustment vanishes."""
        res = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.0,
        )
        assert isinstance(res, QuantoSwapResult)
        assert res.quanto_adjustment_total == pytest.approx(0.0, abs=1e-10)
        assert res.foreign_rate_avg == pytest.approx(res.adjusted_rate_avg, abs=1e-10)

    def test_positive_correlation_negative_adjustment(self):
        """ρ > 0 → negative adjustment (foreign rate up ↔ FX appreciates)."""
        res = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.5,
        )
        assert res.quanto_adjustment_total < 0
        assert res.adjusted_rate_avg < res.foreign_rate_avg

    def test_negative_correlation_positive_adjustment(self):
        """ρ < 0 → positive adjustment."""
        res = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=-0.5,
        )
        assert res.quanto_adjustment_total > 0
        assert res.adjusted_rate_avg > res.foreign_rate_avg

    def test_par_spread_makes_pv_near_zero(self):
        """Applying par_spread on floating leg should yield PV ≈ 0."""
        res = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.3,
        )
        res2 = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.3, spread=res.par_spread,
        )
        assert res2.pv == pytest.approx(0.0, abs=1000)

    def test_pay_vs_receive_sign(self):
        """pay_foreign_float True vs False should flip PV sign."""
        kwargs = dict(
            reference_date=REF, maturity_years=5.0,
            domestic_curve=_dom_curve(), foreign_curve=_for_curve(),
            fixed_rate=0.04, rate_vol=0.15, fx_vol=0.10,
            correlation=0.3,
        )
        res_pay = quanto_swap_price(**kwargs, pay_foreign_float=True)
        res_recv = quanto_swap_price(**kwargs, pay_foreign_float=False)
        assert res_pay.pv == pytest.approx(-res_recv.pv, rel=1e-10)

    def test_n_periods_quarterly(self):
        """5Y quarterly = 20 periods."""
        res = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.0, frequency=Frequency.QUARTERLY,
        )
        assert res.n_periods == 20

    def test_n_periods_semi_annual(self):
        """5Y semi-annual = 10 periods."""
        res = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.0, frequency=Frequency.SEMI_ANNUAL,
        )
        assert res.n_periods == 10

    def test_adjustment_grows_with_maturity(self):
        """Longer maturity → larger total adjustment (in absolute terms)."""
        res_2y = quanto_swap_price(
            REF, 2.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.5,
        )
        res_10y = quanto_swap_price(
            REF, 10.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.5,
        )
        assert abs(res_10y.quanto_adjustment_total) > abs(res_2y.quanto_adjustment_total)

    def test_higher_vol_larger_adjustment(self):
        """Higher rate vol → larger adjustment."""
        res_low = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.05, fx_vol=0.10,
            correlation=0.5,
        )
        res_high = quanto_swap_price(
            REF, 5.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.30, fx_vol=0.10,
            correlation=0.5,
        )
        assert abs(res_high.quanto_adjustment_total) > abs(res_low.quanto_adjustment_total)

    def test_to_dict(self):
        """Result should serialize to dict."""
        res = quanto_swap_price(
            REF, 2.0, _dom_curve(), _for_curve(),
            fixed_rate=0.03, rate_vol=0.15, fx_vol=0.10,
            correlation=0.3,
        )
        d = res.to_dict()
        assert "pv" in d
        assert "par_spread" in d
        assert "quanto_adjustment_total" in d


class TestDifferentialSwap:
    def test_equal_curves_zero_pv(self):
        """Same rates → differential = 0 → PV = spread-driven."""
        curve = make_flat_curve(REF, 0.04)
        res = differential_swap_price(
            REF, 5.0, curve, curve, curve,
            fixed_spread=0.0,
            vol_1=0.15, vol_2=0.15, fx_vol=0.10,
            corr_1_fx=0.0, corr_2_fx=0.0,
        )
        # rate1 = rate2 but rate2 gets quanto adj from corr_2_fx=0 → no adj
        # so differential ≈ 0, pv ≈ 0
        assert res.pv == pytest.approx(0.0, abs=100)

    def test_rate_differential_sign(self):
        """Higher rate_1 → positive differential."""
        high = make_flat_curve(REF, 0.05)
        low = make_flat_curve(REF, 0.02)
        pay = make_flat_curve(REF, 0.04)
        res = differential_swap_price(
            REF, 5.0, high, low, pay,
            fixed_spread=0.0,
            vol_1=0.15, vol_2=0.15, fx_vol=0.10,
            corr_1_fx=0.0, corr_2_fx=0.0,
        )
        assert res.pv > 0
        assert res.rate_differential > 0

    def test_quanto_adj_on_foreign_only(self):
        """Rate 1 (payment ccy) gets no adjustment; rate 2 does."""
        curve = make_flat_curve(REF, 0.04)
        res = differential_swap_price(
            REF, 5.0, curve, curve, curve,
            fixed_spread=0.0,
            vol_1=0.15, vol_2=0.15, fx_vol=0.10,
            corr_1_fx=0.5, corr_2_fx=0.5,
        )
        assert res.quanto_adj_1 == 0.0  # rate_1 in payment ccy
        assert res.quanto_adj_2 != 0.0  # rate_2 foreign → adjusted

    def test_to_dict(self):
        curve = make_flat_curve(REF, 0.04)
        res = differential_swap_price(
            REF, 2.0, curve, curve, curve,
            fixed_spread=0.0,
            vol_1=0.15, vol_2=0.15, fx_vol=0.10,
            corr_1_fx=0.0, corr_2_fx=0.0,
        )
        d = res.to_dict()
        assert "pv" in d
        assert "rate_differential" in d


class TestQuantoAdjustmentTermStructure:
    def test_adjustment_increases_with_tenor(self):
        """Adjustment (bps) grows with tenor for positive correlation."""
        ts = quanto_adjustment_term_structure(
            _for_curve(), REF, rate_vol=0.15, fx_vol=0.10, correlation=0.5,
        )
        assert len(ts) == 8  # default tenors
        # Absolute adjustment should increase
        for i in range(1, len(ts)):
            assert abs(ts[i]["adjustment_bps"]) >= abs(ts[i - 1]["adjustment_bps"]) - 0.01

    def test_zero_correlation_zero_adjustment(self):
        """ρ = 0 → all adjustments zero."""
        ts = quanto_adjustment_term_structure(
            _for_curve(), REF, rate_vol=0.15, fx_vol=0.10, correlation=0.0,
        )
        for entry in ts:
            assert entry["adjustment_bps"] == pytest.approx(0.0, abs=1e-10)

    def test_custom_tenors(self):
        ts = quanto_adjustment_term_structure(
            _for_curve(), REF, rate_vol=0.15, fx_vol=0.10, correlation=0.3,
            tenors_years=[1.0, 5.0, 10.0],
        )
        assert len(ts) == 3
        assert ts[0]["tenor_years"] == 1.0
        assert ts[2]["tenor_years"] == 10.0

    def test_forward_rate_positive(self):
        """Forward rates should be positive for positive-rate curve."""
        ts = quanto_adjustment_term_structure(
            _for_curve(), REF, rate_vol=0.15, fx_vol=0.10, correlation=0.3,
        )
        for entry in ts:
            assert entry["forward_rate"] > 0


class TestQuantoFRA:
    def test_at_the_money_near_zero(self):
        """FRA struck at adjusted forward → PV ≈ 0."""
        fix_date = date(2024, 7, 15)
        mat_date = date(2024, 10, 15)
        res = quanto_fra(
            REF, fix_date, mat_date, _dom_curve(), _for_curve(),
            strike=0.0, rate_vol=0.15, fx_vol=0.10, correlation=0.3,
        )
        # Strike at adjusted forward
        res2 = quanto_fra(
            REF, fix_date, mat_date, _dom_curve(), _for_curve(),
            strike=res.quanto_adjusted_forward,
            rate_vol=0.15, fx_vol=0.10, correlation=0.3,
        )
        assert res2.pv == pytest.approx(0.0, abs=1.0)

    def test_adjustment_sign(self):
        """Positive correlation → adjusted < unadjusted."""
        fix_date = date(2025, 1, 15)
        mat_date = date(2025, 4, 15)
        res = quanto_fra(
            REF, fix_date, mat_date, _dom_curve(), _for_curve(),
            strike=0.03, rate_vol=0.15, fx_vol=0.10, correlation=0.5,
        )
        assert res.quanto_adjusted_forward < res.foreign_forward
        assert res.adjustment_bps < 0

    def test_zero_correlation(self):
        """ρ = 0 → no adjustment."""
        fix_date = date(2025, 1, 15)
        mat_date = date(2025, 4, 15)
        res = quanto_fra(
            REF, fix_date, mat_date, _dom_curve(), _for_curve(),
            strike=0.03, rate_vol=0.15, fx_vol=0.10, correlation=0.0,
        )
        assert res.quanto_adjusted_forward == pytest.approx(res.foreign_forward, abs=1e-12)
        assert res.adjustment_bps == pytest.approx(0.0, abs=1e-8)

    def test_to_dict(self):
        fix_date = date(2025, 1, 15)
        mat_date = date(2025, 4, 15)
        res = quanto_fra(
            REF, fix_date, mat_date, _dom_curve(), _for_curve(),
            strike=0.03, rate_vol=0.15, fx_vol=0.10, correlation=0.3,
        )
        d = res.to_dict()
        assert "pv" in d
        assert "adjustment_bps" in d
