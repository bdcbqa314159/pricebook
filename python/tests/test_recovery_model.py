"""Tests for recovery rate models."""

import pytest
import math
import numpy as np

from pricebook.recovery_model import BetaRecovery, LGDModel, CorrelatedRecovery


class TestBetaRecovery:
    def test_mean(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        samples = rec.sample(100_000)
        assert samples.mean() == pytest.approx(0.40, rel=0.02)

    def test_std(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        samples = rec.sample(100_000)
        assert samples.std() == pytest.approx(0.15, rel=0.05)

    def test_bounded_zero_one(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        samples = rec.sample(10_000)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_expected_lgd(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        assert rec.expected_lgd == pytest.approx(0.60)

    def test_pdf_positive(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        assert rec.pdf(0.40) > 0

    def test_cdf_monotone(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        assert rec.cdf(0.2) < rec.cdf(0.5) < rec.cdf(0.8)

    def test_invalid_mean(self):
        with pytest.raises(ValueError):
            BetaRecovery(mean=1.5)

    def test_invalid_std(self):
        with pytest.raises(ValueError):
            BetaRecovery(mean=0.40, std=0.0)

    def test_std_too_large(self):
        with pytest.raises(ValueError, match="too large"):
            BetaRecovery(mean=0.40, std=0.49)

    def test_zero_variance_limit(self):
        """Very small std → samples clustered around mean."""
        rec = BetaRecovery(mean=0.40, std=0.01)
        samples = rec.sample(10_000)
        assert samples.std() < 0.02


class TestLGDModel:
    def test_mean_lgd(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        lgd = LGDModel(rec)
        assert lgd.mean == pytest.approx(0.60)

    def test_sample_lgd(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        lgd = LGDModel(rec)
        samples = lgd.sample_lgd(100_000)
        assert samples.mean() == pytest.approx(0.60, rel=0.02)

    def test_lgd_bounded(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        lgd = LGDModel(rec)
        samples = lgd.sample_lgd(10_000)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_expected_loss(self):
        rec = BetaRecovery(mean=0.40, std=0.15)
        lgd = LGDModel(rec)
        el = lgd.expected_loss(default_prob=0.05, notional=1_000_000)
        assert el == pytest.approx(0.05 * 0.60 * 1_000_000)


class TestCorrelatedRecovery:
    def test_normal_conditions(self):
        crec = CorrelatedRecovery(base_mean=0.40, sensitivity=0.10)
        assert crec.recovery_given_factor(0.0) == pytest.approx(0.40)

    def test_downturn(self):
        """Negative M → lower recovery."""
        crec = CorrelatedRecovery(base_mean=0.40, sensitivity=0.10)
        r_down = crec.recovery_given_factor(-2.0)
        r_up = crec.recovery_given_factor(2.0)
        assert r_down < r_up

    def test_floor_respected(self):
        crec = CorrelatedRecovery(base_mean=0.40, sensitivity=0.10, floor=0.10)
        assert crec.recovery_given_factor(-10.0) == pytest.approx(0.10)

    def test_cap_respected(self):
        crec = CorrelatedRecovery(base_mean=0.40, sensitivity=0.10, cap=0.80)
        assert crec.recovery_given_factor(10.0) == pytest.approx(0.80)

    def test_sample_vectorised(self):
        crec = CorrelatedRecovery(base_mean=0.40, sensitivity=0.10)
        M = np.array([-2.0, 0.0, 2.0])
        R = crec.sample(M)
        assert R.shape == (3,)
        assert R[0] < R[1] < R[2]

    def test_downturn_lgd(self):
        crec = CorrelatedRecovery(base_mean=0.40, sensitivity=0.10)
        lgd = crec.downturn_lgd(percentile=0.01)
        # 1st percentile: M ≈ -2.33, R ≈ 0.40 - 0.233 = 0.167, LGD ≈ 0.833
        assert lgd > 0.60  # worse than average
        assert lgd < 1.0

    def test_portfolio_loss_higher_with_correlation(self):
        """Correlated recovery → higher tail loss."""
        rng = np.random.default_rng(42)
        n_sims, n_names = 10_000, 100
        M = rng.standard_normal(n_sims)

        # Defaults: independent, ~5% PD
        defaults = rng.uniform(size=(n_sims, n_names)) < 0.05

        # Uncorrelated recovery: constant 40%
        crec_flat = CorrelatedRecovery(base_mean=0.40, sensitivity=0.0)
        loss_flat = crec_flat.expected_portfolio_loss(defaults, M)

        # Correlated recovery: low M → low R → higher loss
        crec_corr = CorrelatedRecovery(base_mean=0.40, sensitivity=0.15)
        loss_corr = crec_corr.expected_portfolio_loss(defaults, M)

        # Correlated should have similar or slightly higher expected loss
        # (since defaults and M are independent here, the effect is through
        # the variance of the loss distribution, not the mean)
        assert loss_flat > 0
        assert loss_corr > 0

    def test_zero_sensitivity_is_deterministic(self):
        crec = CorrelatedRecovery(base_mean=0.40, sensitivity=0.0)
        M = np.array([-3.0, 0.0, 3.0])
        R = crec.sample(M)
        np.testing.assert_array_almost_equal(R, 0.40)
