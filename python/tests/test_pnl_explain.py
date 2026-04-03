"""Tests for P&L explain."""

import pytest

from pricebook.pnl_explain import (
    PnLResult,
    compute_carry,
    greek_pnl,
    pnl_decompose,
)


class TestGreekPnL:
    def test_delta_only(self):
        pnl = greek_pnl(sensitivity=100.0, risk_factor_change=0.01)
        assert pnl == pytest.approx(1.0)

    def test_delta_gamma(self):
        pnl = greek_pnl(sensitivity=100.0, risk_factor_change=0.01, gamma=-500.0)
        # 100*0.01 + 0.5*(-500)*0.01^2 = 1.0 - 0.025 = 0.975
        assert pnl == pytest.approx(0.975)

    def test_zero_change(self):
        assert greek_pnl(100.0, 0.0) == 0.0

    def test_negative_delta(self):
        pnl = greek_pnl(sensitivity=-50.0, risk_factor_change=0.02)
        assert pnl == pytest.approx(-1.0)


class TestCarry:
    def test_positive_carry(self):
        c = compute_carry(coupon_income=5000, funding_cost=3000, dt=1/252)
        assert c > 0

    def test_negative_carry(self):
        c = compute_carry(coupon_income=2000, funding_cost=4000, dt=1/252)
        assert c < 0

    def test_zero_funding(self):
        c = compute_carry(coupon_income=5000, funding_cost=0, dt=1/252)
        assert c == pytest.approx(5000 / 252)


class TestPnLDecompose:
    def test_total_pnl(self):
        r = pnl_decompose(base_pv=100_000, current_pv=101_500)
        assert r.total == pytest.approx(1500)

    def test_components_sum(self):
        r = pnl_decompose(
            base_pv=100_000, current_pv=101_500,
            carry=200, rolldown=100,
            rate_delta=-5000, rate_change=0.001,
            vol_vega=3000, vol_change=0.005,
            credit_cs01=-200, credit_change=0.002,
        )
        assert r.explained == pytest.approx(
            r.carry + r.rolldown + r.rate_pnl + r.vol_pnl +
            r.credit_pnl + r.fx_pnl + r.theta_pnl
        )

    def test_unexplained(self):
        r = pnl_decompose(
            base_pv=100_000, current_pv=101_500,
            carry=500, rolldown=200,
            rate_delta=-5000, rate_change=-0.001,  # +5 from rates
        )
        expected_explained = 500 + 200 + 5.0  # carry + rolldown + rate
        assert r.unexplained == pytest.approx(1500 - expected_explained)

    def test_all_zero_fully_unexplained(self):
        r = pnl_decompose(base_pv=100, current_pv=110)
        assert r.total == 10
        assert r.unexplained == 10
        assert r.explained == 0

    def test_perfectly_explained(self):
        """If Greeks perfectly explain the move, unexplained ≈ 0."""
        r = pnl_decompose(
            base_pv=100_000, current_pv=100_100,
            rate_delta=10_000, rate_change=0.01,
        )
        # rate_pnl = 10000 * 0.01 = 100
        assert r.rate_pnl == pytest.approx(100)
        assert r.total == pytest.approx(100)
        assert r.unexplained == pytest.approx(0, abs=0.01)

    def test_result_dataclass(self):
        r = pnl_decompose(base_pv=100, current_pv=105)
        assert isinstance(r, PnLResult)
        assert r.base_pv == 100
        assert r.current_pv == 105

    def test_other_field(self):
        r = PnLResult(
            base_pv=100, current_pv=110, total=10,
            carry=3, other={"convexity": 2},
        )
        assert r.explained == 5  # carry + convexity
        assert r.unexplained == 5
