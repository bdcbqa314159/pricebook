"""Tests for bond futures delivery options."""
import math
import pytest
from pricebook.bond_futures_options import (
    end_of_month_option, quality_option, timing_option,
    net_basis_decomposition, joint_delivery_option_value,
)

class TestEOM:
    def test_positive(self):
        r = end_of_month_option(futures_dv01=80, daily_vol_bps=5.0, n_business_days=5)
        assert r.value > 0
    def test_more_days_higher_value(self):
        short = end_of_month_option(80, 5.0, 3)
        long = end_of_month_option(80, 5.0, 7)
        assert long.value > short.value
    def test_daily_option(self):
        r = end_of_month_option(100, 4.0, 1)
        assert r.daily_option_value == pytest.approx(0.8 * 100 * 4.0 / 100)

class TestQualityOption:
    def test_basic(self):
        bases = {"bond_a": 0.02, "bond_b": 0.05, "bond_c": 0.10}
        r = quality_option(bases, yield_vol_bps=80, futures_dv01=80)
        assert r.current_ctd == "bond_a"
        assert r.value >= 0
    def test_single_bond_zero(self):
        r = quality_option({"only": 0.05}, 80, 80)
        assert r.value == 0.0
    def test_switch_probability(self):
        """Tighter gap → higher switch probability."""
        tight = quality_option({"a": 0.01, "b": 0.015}, 80, 80)
        wide = quality_option({"a": 0.01, "b": 0.50}, 80, 80)
        assert tight.ctd_switch_probability >= wide.ctd_switch_probability

class TestTimingOption:
    def test_positive_carry_deliver_late(self):
        r = timing_option(100, coupon_rate=0.06, repo_rate=0.02, conversion_factor=0.9)
        assert r.optimal_delivery_day == "last_day"
        assert r.value > 0
    def test_negative_carry_deliver_early(self):
        r = timing_option(100, coupon_rate=0.01, repo_rate=0.06, conversion_factor=0.9)
        assert r.optimal_delivery_day == "first_day"

class TestNetBasis:
    def test_decomposition(self):
        r = net_basis_decomposition(bond_price=105, conversion_factor=0.95,
                                      futures_price=110, coupon_accrued=1.5, repo_cost=0.8)
        assert r.gross_basis == pytest.approx(105 - 110 * 0.95)
        assert r.carry == pytest.approx(1.5 - 0.8)
        assert r.net_basis == pytest.approx(r.gross_basis - r.carry)

class TestJointDelivery:
    def test_basic(self):
        eom = end_of_month_option(80, 5.0, 5)
        qual = quality_option({"a": 0.01, "b": 0.05}, 80, 80)
        timing = timing_option(100, 0.05, 0.03, 0.9)
        result = joint_delivery_option_value(eom, qual, timing)
        assert result.total_value > 0
        assert result.total_value < eom.value + qual.value + timing.value + 1e-6
