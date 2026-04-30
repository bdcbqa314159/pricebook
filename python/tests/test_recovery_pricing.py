"""Tests for recovery pricing: RecoverySpec, default-recovery correlation, wrong-way premium."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

from pricebook.recovery_pricing import (
    RecoverySpec, SENIORITY_RECOVERY,
    correlated_default_recovery, DefaultRecoveryResult,
    wrong_way_premium, lgd_term_structure,
)
from tests.conftest import make_flat_survival


REF = date(2024, 1, 15)


# ---- RecoverySpec ----

class TestRecoverySpec:

    def test_fixed(self):
        spec = RecoverySpec.fixed(0.4)
        assert spec.is_deterministic
        samples = spec.sample(1000)
        assert np.all(samples == 0.4)

    def test_beta_samples_in_range(self):
        spec = RecoverySpec(mean=0.4, std=0.15, distribution="beta")
        samples = spec.sample(10_000)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0
        assert samples.mean() == pytest.approx(0.4, abs=0.02)

    def test_from_seniority(self):
        spec = RecoverySpec.from_seniority("1L")
        assert spec.mean == 0.77
        assert spec.std == 0.20

    def test_from_seniority_unknown(self):
        with pytest.raises(ValueError, match="Unknown seniority"):
            RecoverySpec.from_seniority("ZZZ")

    def test_expected_lgd(self):
        spec = RecoverySpec(mean=0.4)
        assert spec.expected_lgd == pytest.approx(0.6)

    def test_correlated_samples(self):
        """Negative ρ_DR: when Z_D is low (default), R should be low."""
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.5)
        Z_D = np.full(10_000, -2.0)  # deep default region
        R_low = spec.sample(10_000, systematic_factor=Z_D).mean()

        Z_D_high = np.full(10_000, 2.0)  # no-default region
        R_high = spec.sample(10_000, systematic_factor=Z_D_high).mean()

        # With negative correlation: low Z_D → low R
        assert R_low < R_high

    def test_zero_correlation_independent(self):
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=0.0)
        Z_low = np.full(10_000, -2.0)
        Z_high = np.full(10_000, 2.0)
        R_low = spec.sample(10_000, systematic_factor=Z_low, seed=42).mean()
        R_high = spec.sample(10_000, systematic_factor=Z_high, seed=42).mean()
        # Independent: should be similar
        assert abs(R_low - R_high) < 0.05

    def test_to_dict_from_dict(self):
        spec = RecoverySpec(mean=0.45, std=0.18, distribution="beta",
                            correlation_to_default=-0.3)
        d = spec.to_dict()
        spec2 = RecoverySpec.from_dict(d)
        assert spec2.mean == spec.mean
        assert spec2.std == spec.std
        assert spec2.correlation_to_default == spec.correlation_to_default

    def test_validation(self):
        with pytest.raises(ValueError):
            RecoverySpec(mean=1.5)
        with pytest.raises(ValueError):
            RecoverySpec(std=-0.1)
        with pytest.raises(ValueError):
            RecoverySpec(correlation_to_default=2.0)


# ---- Default-recovery correlation ----

class TestCorrelatedDefaultRecovery:

    def test_negative_correlation_increases_loss(self):
        """Negative ρ_DR → E[(1-R)×1_D] > (1-E[R])×PD."""
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.4)
        result = correlated_default_recovery(0.05, spec, n_sims=50_000)
        assert result.wrong_way_premium > 0

    def test_zero_correlation_near_zero_premium(self):
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=0.0)
        result = correlated_default_recovery(0.05, spec, n_sims=50_000)
        assert abs(result.wrong_way_premium) < 0.005

    def test_positive_correlation_negative_premium(self):
        """Positive ρ_DR → recovery is high when defaults occur."""
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=0.4)
        result = correlated_default_recovery(0.05, spec, n_sims=50_000)
        assert result.wrong_way_premium < 0

    def test_lgd_given_default_lower_with_negative_rho(self):
        """E[R|default] < E[R] when ρ_DR < 0."""
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.5)
        result = correlated_default_recovery(0.05, spec, n_sims=100_000)
        assert result.expected_lgd_given_default > spec.expected_lgd

    def test_fixed_recovery_near_zero_premium(self):
        spec = RecoverySpec.fixed(0.4)
        result = correlated_default_recovery(0.05, spec, n_sims=100_000)
        assert abs(result.wrong_way_premium) < 0.005

    def test_result_fields(self):
        spec = RecoverySpec(mean=0.4, std=0.15)
        result = correlated_default_recovery(0.05, spec, n_sims=10_000)
        assert result.default_indicators.shape == (10_000,)
        assert result.recovery_rates.shape == (10_000,)
        assert 0 <= result.expected_loss
        assert math.isfinite(result.wrong_way_premium)


# ---- Wrong-way premium ----

class TestWrongWayPremium:

    def test_positive_for_negative_rho(self):
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.3)
        prem = wrong_way_premium(0.05, spec)
        assert prem > 0

    def test_near_zero_for_fixed(self):
        spec = RecoverySpec.fixed(0.4)
        prem = wrong_way_premium(0.05, spec)
        assert abs(prem) < 0.005

    def test_increases_with_correlation_magnitude(self):
        """Stronger negative correlation → larger premium."""
        spec_low = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.2)
        spec_high = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.6)
        p_low = wrong_way_premium(0.05, spec_low)
        p_high = wrong_way_premium(0.05, spec_high)
        assert p_high > p_low

    def test_increases_with_recovery_vol(self):
        """Higher recovery vol → larger wrong-way premium."""
        spec_low = RecoverySpec(mean=0.4, std=0.10, correlation_to_default=-0.3)
        spec_high = RecoverySpec(mean=0.4, std=0.20, correlation_to_default=-0.3)
        p_low = wrong_way_premium(0.05, spec_low)
        p_high = wrong_way_premium(0.05, spec_high)
        assert p_high > p_low


# ---- LGD term structure ----

class TestLGDTermStructure:

    def test_increasing_pd(self):
        """Cumulative PD should increase with tenor."""
        sc = make_flat_survival(REF, 0.02)
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.3)
        ts = lgd_term_structure(sc, spec, tenors=[1.0, 5.0, 10.0], n_sims=20_000)
        pds = [t["pd"] for t in ts]
        for i in range(1, len(pds)):
            assert pds[i] > pds[i - 1]

    def test_correlated_lgd_higher(self):
        """With negative ρ_DR, correlated LGD > independent LGD."""
        sc = make_flat_survival(REF, 0.03)
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.4)
        ts = lgd_term_structure(sc, spec, tenors=[5.0], n_sims=50_000)
        assert ts[0]["lgd_correlated"] > ts[0]["lgd_independent"]

    def test_premium_positive(self):
        sc = make_flat_survival(REF, 0.03)
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.3)
        ts = lgd_term_structure(sc, spec, tenors=[5.0], n_sims=50_000)
        assert ts[0]["premium"] > 0

    def test_zero_correlation_equal(self):
        """Zero correlation → correlated LGD ≈ independent LGD."""
        sc = make_flat_survival(REF, 0.03)
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=0.0)
        ts = lgd_term_structure(sc, spec, tenors=[5.0], n_sims=50_000)
        assert ts[0]["lgd_correlated"] == pytest.approx(
            ts[0]["lgd_independent"], abs=0.03
        )
