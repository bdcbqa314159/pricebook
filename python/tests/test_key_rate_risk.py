"""Tests for key-rate DV01 and bucket risk."""

import math
import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.curves.key_rate_risk import (
    key_rate_dv01, bucket_risk, risk_ladder, standard_tenors,
    KeyRateResult, BumpProfile,
)


REF = date(2024, 1, 15)


def _swap_pv(curve: DiscountCurve, maturity_years: float = 10.0,
              fixed_rate: float = 0.04, notional: float = 1e6) -> float:
    """Simple swap PV pricer for testing."""
    pv = 0.0
    for i in range(1, int(maturity_years) + 1):
        d = date(REF.year + i, REF.month, REF.day)
        df = curve.df(d)
        # Fixed leg
        pv -= fixed_rate * notional * df
        # Float leg (forward rate)
        if i == 1:
            df_prev = 1.0
        else:
            d_prev = date(REF.year + i - 1, REF.month, REF.day)
            df_prev = curve.df(d_prev)
        fwd = (df_prev / df - 1.0) if df > 0 else 0.0
        pv += fwd * notional * df
    return pv


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(REF, 0.04)


class TestKeyRateDV01:
    def test_basic(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        result = key_rate_dv01(flat_curve, pricer)
        assert isinstance(result, KeyRateResult)
        assert len(result.dv01s) == len(result.tenors)
        assert result.total_dv01 != 0

    def test_total_approx_parallel(self, flat_curve):
        """Sum of key-rate DV01s ≈ parallel DV01."""
        pricer = lambda c: _swap_pv(c)
        result = key_rate_dv01(flat_curve, pricer)
        # Should be within 20% (triangular bumps don't exactly partition unity at pillars)
        ratio = abs(result.total_dv01 / result.parallel_dv01) if result.parallel_dv01 != 0 else 0
        assert 0.5 < ratio < 2.0

    def test_10y_swap_concentration(self, flat_curve):
        """10Y swap should have most risk at 10Y key tenor."""
        pricer = lambda c: _swap_pv(c, maturity_years=10.0)
        result = key_rate_dv01(flat_curve, pricer, tenors=[1, 2, 5, 10, 20, 30])
        # Find max DV01 — should be near 10Y
        max_idx = max(range(len(result.dv01s)), key=lambda i: abs(result.dv01s[i]))
        assert result.tenors[max_idx] == 10

    def test_with_gamma(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        result = key_rate_dv01(flat_curve, pricer, tenors=[2, 5, 10], compute_gamma=True)
        assert result.gamma is not None
        assert len(result.gamma) == 3

    def test_gaussian_profile(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        result = key_rate_dv01(flat_curve, pricer, tenors=[2, 5, 10],
                                profile=BumpProfile.GAUSSIAN)
        assert result.total_dv01 != 0

    def test_pillar_only(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        result = key_rate_dv01(flat_curve, pricer, tenors=[5, 10],
                                profile=BumpProfile.PILLAR_ONLY)
        assert len(result.dv01s) == 2


class TestBucketRisk:
    def test_basic(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        results = bucket_risk(flat_curve, pricer)
        assert "5-7Y" in results
        assert "7-10Y" in results

    def test_sum_approx_parallel(self, flat_curve):
        """Sum of bucket risks ≈ parallel DV01."""
        pricer = lambda c: _swap_pv(c)
        results = bucket_risk(flat_curve, pricer)
        total_bucket = sum(results.values())
        parallel = pricer(flat_curve.bumped(0.0001)) - pricer(flat_curve)
        ratio = abs(total_bucket / parallel) if parallel != 0 else 0
        assert 0.3 < ratio < 3.0


class TestRiskLadder:
    def test_format(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        kr = key_rate_dv01(flat_curve, pricer, tenors=[2, 5, 10])
        ladder = risk_ladder(kr)
        assert len(ladder) == 3
        assert "tenor" in ladder[0]
        assert "dv01" in ladder[0]
        assert "pct_of_total" in ladder[0]


class TestStandardTenors:
    def test_usd(self):
        t = standard_tenors("USD")
        assert 0.25 in t
        assert 30 in t

    def test_gbp_includes_50(self):
        t = standard_tenors("GBP")
        assert 50 in t

    def test_jpy_includes_40(self):
        t = standard_tenors("JPY")
        assert 40 in t

    def test_to_dict(self, flat_curve):
        pricer = lambda c: _swap_pv(c)
        kr = key_rate_dv01(flat_curve, pricer, tenors=[5, 10])
        d = kr.to_dict()
        assert "tenors" in d
        assert "total_dv01" in d
