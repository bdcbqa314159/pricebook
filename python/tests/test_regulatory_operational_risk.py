"""Tests for operational risk SMA."""

import math
import pytest

from pricebook.regulatory.operational_risk import (
    SMAInputs, SMAResult,
    calculate_sma_full, calculate_bic, calculate_ilm,
    sma_bucket, sma_sensitivity, calculate_business_indicator,
)


@pytest.fixture
def small_bank():
    """Bucket 1 bank: BI ≈ 500M."""
    return SMAInputs(
        interest_income=[200e6, 210e6, 220e6],
        interest_expense=[100e6, 105e6, 110e6],
        fee_income=[50e6, 52e6, 55e6],
        fee_expense=[10e6, 10e6, 10e6],
        net_trading_income=[20e6, 25e6, 30e6],
        other_operating_income=[10e6, 10e6, 10e6],
        other_operating_expense=[5e6, 5e6, 5e6],
    )


@pytest.fixture
def large_bank():
    """Bucket 2 bank: BI ≈ 5bn."""
    return SMAInputs(
        interest_income=[2000e6, 2100e6, 2200e6],
        interest_expense=[1000e6, 1050e6, 1100e6],
        fee_income=[800e6, 850e6, 900e6],
        fee_expense=[200e6, 200e6, 200e6],
        net_trading_income=[500e6, 600e6, 400e6],
        other_operating_income=[100e6, 100e6, 100e6],
        other_operating_expense=[50e6, 50e6, 50e6],
        annual_op_losses=[50e6, 60e6, 40e6, 55e6, 45e6, 70e6, 35e6, 50e6, 65e6, 55e6],
    )


class TestBucket:
    def test_bucket1(self):
        assert sma_bucket(500e6) == 1

    def test_bucket2(self):
        assert sma_bucket(5e9) == 2

    def test_bucket3(self):
        assert sma_bucket(50e9) == 3

    def test_boundary(self):
        assert sma_bucket(1e9) == 1
        assert sma_bucket(1e9 + 1) == 2


class TestBIC:
    def test_bucket1_only(self):
        # BI = 500M → BIC = 500M × 12% = 60M
        assert abs(calculate_bic(500e6) - 60e6) < 1.0

    def test_bucket2_marginal(self):
        # BI = 2bn → BIC = 1bn × 12% + 1bn × 15% = 120M + 150M = 270M
        assert abs(calculate_bic(2e9) - 270e6) < 1.0

    def test_bucket3_marginal(self):
        # BI = 35bn → BIC = 1bn×12% + 29bn×15% + 5bn×18%
        # = 120M + 4350M + 900M = 5370M
        assert abs(calculate_bic(35e9) - 5370e6) < 1.0


class TestILM:
    def test_ilm_at_one(self):
        # When LC = BIC: ILM = ln(e-1 + 1^0.8) = ln(e) = 1.0
        bic = 100e6
        assert abs(calculate_ilm(bic, bic) - 1.0) < 1e-10

    def test_ilm_above_one(self):
        # LC > BIC → ILM > 1
        assert calculate_ilm(100e6, 200e6) > 1.0

    def test_ilm_below_one(self):
        # LC < BIC → ILM < 1
        assert calculate_ilm(100e6, 50e6) < 1.0

    def test_zero_bic(self):
        assert calculate_ilm(0, 100) == 1.0


class TestSMAFull:
    def test_small_bank(self, small_bank):
        r = calculate_sma_full(small_bank)
        assert r.bucket == 1
        assert not r.use_ilm  # bucket 1: ILM = 1
        assert r.ilm == 1.0
        assert r.capital_requirement == r.bic
        assert r.rwa == r.capital_requirement / 0.08

    def test_large_bank_with_losses(self, large_bank):
        r = calculate_sma_full(large_bank)
        assert r.bucket == 2
        assert r.use_ilm  # bucket 2 with ≥5 years of loss data
        assert r.capital_requirement == r.bic * r.ilm

    def test_bi_averaging(self, small_bank):
        r = calculate_sma_full(small_bank)
        assert len(r.bi_yearly) == 3
        assert abs(r.bi_average - sum(r.bi_yearly) / 3) < 1.0

    def test_legacy_comparison(self, small_bank):
        r = calculate_sma_full(small_bank)
        assert r.legacy_bia > 0  # BIA should be computable

    def test_to_dict(self, small_bank):
        d = calculate_sma_full(small_bank).to_dict()
        assert "bic" in d
        assert "bucket" in d
        assert "rwa" in d


class TestSensitivity:
    def test_sensitivity_shape(self):
        results = sma_sensitivity(5e9, (0.5, 2.0), n_points=10)
        assert len(results) == 10
        assert "ilm" in results[0]

    def test_sensitivity_monotonic(self):
        results = sma_sensitivity(5e9, (0.5, 2.0), n_points=10)
        # Higher LC/BIC → higher ILM → higher capital
        caps = [r["capital"] for r in results]
        for i in range(len(caps) - 1):
            assert caps[i + 1] >= caps[i]
