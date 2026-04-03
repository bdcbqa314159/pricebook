"""Tests for stochastic credit intensity models."""

import pytest
import math
import numpy as np

from pricebook.stochastic_credit import (
    CIRIntensity,
    cox_default_mc,
    JointRateHazard,
    calibrate_cir_intensity,
)


class TestCIRIntensity:
    def test_survival_positive(self):
        m = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
        s = m.survival_analytical(0.02, 5.0)
        assert 0 < s < 1

    def test_survival_at_zero(self):
        m = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
        assert m.survival_analytical(0.02, 0.0) == pytest.approx(1.0)

    def test_survival_decreasing(self):
        m = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
        assert m.survival_analytical(0.02, 10.0) < m.survival_analytical(0.02, 5.0)

    def test_higher_intensity_lower_survival(self):
        m = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
        assert m.survival_analytical(0.05, 5.0) < m.survival_analytical(0.01, 5.0)

    def test_constant_lambda_matches_exponential(self):
        """With very low xi, CIR → deterministic → survival = exp(-λ*T)."""
        m = CIRIntensity(kappa=10.0, theta=0.03, xi=0.001)
        # Fast mean reversion + low vol → λ ≈ theta = 0.03
        s = m.survival_analytical(0.03, 5.0)
        expected = math.exp(-0.03 * 5)
        assert s == pytest.approx(expected, rel=0.05)

    def test_mc_matches_analytical(self):
        m = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
        analytical = m.survival_analytical(0.02, 3.0)
        mc = m.survival_mc(0.02, T=3.0, n_steps=100, n_paths=100_000)
        assert mc == pytest.approx(analytical, rel=0.03)


class TestCoxProcess:
    def test_default_rate_reasonable(self):
        m = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
        paths = m.simulate_intensity(0.02, T=5.0, n_steps=100, n_paths=10_000)
        defaults = cox_default_mc(paths, dt=0.05)
        default_rate = defaults.mean()
        # With hazard ≈ 0.02, P(default in 5Y) ≈ 1 - exp(-0.1) ≈ 0.095
        assert 0.03 < default_rate < 0.20

    def test_higher_intensity_more_defaults(self):
        m_low = CIRIntensity(kappa=1.0, theta=0.01, xi=0.05)
        m_high = CIRIntensity(kappa=1.0, theta=0.05, xi=0.05)
        p_low = m_low.simulate_intensity(0.01, 5.0, 100, 10_000, seed=42)
        p_high = m_high.simulate_intensity(0.05, 5.0, 100, 10_000, seed=42)
        d_low = cox_default_mc(p_low, 0.05).mean()
        d_high = cox_default_mc(p_high, 0.05).mean()
        assert d_high > d_low


class TestJointRateHazard:
    def test_zero_corr_independent(self):
        """Zero correlation: survival under joint ≈ survival under independent."""
        m = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
        indep_surv = m.survival_mc(0.02, T=5.0, n_paths=50_000, seed=42)

        joint = JointRateHazard(
            a_r=0.5, mu_r=0.05, sigma_r=0.01,
            kappa=1.0, theta=0.02, xi=0.1,
            rho=0.0,
        )
        joint_surv = joint.survival_mc(0.05, 0.02, T=5.0, n_paths=50_000, seed=42)
        assert joint_surv == pytest.approx(indep_surv, rel=0.05)

    def test_shape(self):
        joint = JointRateHazard(0.5, 0.05, 0.01, 1.0, 0.02, 0.1, rho=-0.3)
        r, lam = joint.simulate(0.05, 0.02, T=2.0, n_steps=20, n_paths=100)
        assert r.shape == (100, 21)
        assert lam.shape == (100, 21)

    def test_positive_intensity(self):
        joint = JointRateHazard(0.5, 0.05, 0.01, 1.0, 0.02, 0.1, rho=-0.5)
        _, lam = joint.simulate(0.02, 0.02, T=5.0, n_steps=100, n_paths=1000)
        assert np.all(lam >= 0)

    def test_wrong_way_risk(self):
        """Wrong-way risk: corr(r, λ) affects discounted default probability.

        Negative corr: rates low when default likely → higher discounted loss.
        Pure survival doesn't depend on r, but discounted quantities do.
        """
        joint = JointRateHazard(0.5, 0.05, 0.03, 0.5, 0.03, 0.3, rho=-0.5)
        r, lam = joint.simulate(0.05, 0.03, 5.0, n_steps=100, n_paths=50_000)
        dt = 0.05

        # Compute discounted default indicator: exp(-∫r ds) * 1_{default}
        r_integral = r[:, :-1].sum(axis=1) * dt
        lam_integral = lam[:, :-1].sum(axis=1) * dt

        df = np.exp(-r_integral)
        default_prob = 1 - np.exp(-lam_integral)

        # The correlation between df and default_prob should be positive
        # (negative rho: low r → high λ → high df AND high default)
        corr = np.corrcoef(df, default_prob)[0, 1]
        assert corr > 0.01  # positive correlation = wrong-way risk


class TestCalibration:
    def test_calibrate_recovers(self):
        """Generate spreads from known params, calibrate back."""
        true_model = CIRIntensity(kappa=1.0, theta=0.02, xi=0.1)
        lam0 = 0.02
        recovery = 0.4

        # Generate "market" spreads: spread ≈ (1-R) * hazard
        par_spreads = []
        for T in [1, 3, 5, 7, 10]:
            surv = true_model.survival_analytical(lam0, T)
            spread = -(1 - recovery) * math.log(surv) / T
            par_spreads.append((float(T), spread))

        result = calibrate_cir_intensity(par_spreads, recovery, lam0_guess=lam0)

        assert result["theta"] == pytest.approx(0.02, rel=0.3)
        assert result["kappa"] > 0
        assert result["xi"] > 0

    def test_returns_all_keys(self):
        result = calibrate_cir_intensity([(5.0, 0.012)], recovery=0.4)
        assert "kappa" in result
        assert "theta" in result
        assert "xi" in result
        assert "lam0" in result
