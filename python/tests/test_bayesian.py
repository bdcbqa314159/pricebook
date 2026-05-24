"""Tests for Bayesian statistics module."""

import math
import pytest
import numpy as np

from pricebook.statistics.bayesian import (
    MetropolisHastings, GibbsSampler, MCMCResult,
    BayesianLinearRegression, BayesianLinearRegressionResult,
    beta_binomial_update, BetaBinomialResult,
    bayes_factor, credible_interval, hpd_interval, posterior_predictive,
    bayesian_changepoint, ChangepointResult,
)


# ═══════════════════════════════════════════════════════════════
# Metropolis-Hastings
# ═══════════════════════════════════════════════════════════════

class TestMH:
    def test_normal_posterior(self):
        """Sample from N(3, 1): log_posterior = -0.5(θ-3)²."""
        def log_post(theta):
            return -0.5 * (theta[0] - 3.0)**2

        mh = MetropolisHastings(log_post, proposal_std=0.5, param_names=["mu"])
        result = mh.sample(np.array([0.0]), n_samples=5000, burn_in=1000)
        assert isinstance(result, MCMCResult)
        assert abs(result.mean()[0] - 3.0) < 0.3
        assert 0.1 < result.acceptance_rate < 0.9

    def test_2d_normal(self):
        """Sample from bivariate normal."""
        mu = np.array([2.0, -1.0])
        def log_post(theta):
            return -0.5 * np.sum((theta - mu)**2)

        mh = MetropolisHastings(log_post, proposal_std=np.array([0.5, 0.5]))
        result = mh.sample(np.array([0.0, 0.0]), n_samples=10000, burn_in=2000)
        np.testing.assert_allclose(result.mean(), mu, atol=0.3)

    def test_credible_interval(self):
        def log_post(theta):
            return -0.5 * theta[0]**2

        mh = MetropolisHastings(log_post, 0.5)
        result = mh.sample(np.array([0.0]), n_samples=5000, burn_in=1000)
        ci = result.credible_interval(0.05)
        assert ci[0][0] < 0 < ci[0][1]  # 0 should be inside CI

    def test_ess(self):
        def log_post(theta):
            return -0.5 * theta[0]**2

        mh = MetropolisHastings(log_post, 1.0)
        result = mh.sample(np.array([0.0]), n_samples=5000, burn_in=500)
        ess = result.effective_sample_size()
        assert ess[0] > 100  # should have decent ESS

    def test_to_dict(self):
        def log_post(theta):
            return -0.5 * theta[0]**2

        mh = MetropolisHastings(log_post, 1.0)
        d = mh.sample(np.array([0.0]), n_samples=1000, burn_in=100).to_dict()
        assert "mean" in d
        assert "acceptance_rate" in d


# ═══════════════════════════════════════════════════════════════
# Gibbs Sampler
# ═══════════════════════════════════════════════════════════════

class TestGibbs:
    def test_independent_normals(self):
        """Sample θ₁ ~ N(2, 1), θ₂ ~ N(-1, 0.5²) independently."""
        conditionals = [
            lambda theta, rng: rng.normal(2.0, 1.0),
            lambda theta, rng: rng.normal(-1.0, 0.5),
        ]
        gibbs = GibbsSampler(conditionals, ["theta1", "theta2"])
        result = gibbs.sample(np.array([0.0, 0.0]), n_samples=5000, burn_in=500)
        assert abs(result.mean()[0] - 2.0) < 0.2
        assert abs(result.mean()[1] - (-1.0)) < 0.2
        assert result.acceptance_rate == 1.0


# ═══════════════════════════════════════════════════════════════
# Bayesian Linear Regression
# ═══════════════════════════════════════════════════════════════

class TestBayesianRegression:
    def test_simple(self):
        """y = 2x + 1 + noise → recover β ≈ [1, 2]."""
        rng = np.random.default_rng(42)
        n = 100
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + rng.normal(0, 0.5, n)

        blr = BayesianLinearRegression()
        result = blr.fit(X, y)
        assert isinstance(result, BayesianLinearRegressionResult)
        assert abs(result.beta_mean[0] - 1.0) < 0.3
        assert abs(result.beta_mean[1] - 2.0) < 0.3

    def test_credible_intervals(self):
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([3.0, -1.0]) + rng.normal(0, 1, n)

        result = BayesianLinearRegression().fit(X, y)
        ci = result.credible_intervals()
        assert len(ci) == 2
        assert ci[0]["ci_lower"] < 3.0 < ci[0]["ci_upper"]

    def test_predict(self):
        rng = np.random.default_rng(42)
        n = 100
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)

        result = BayesianLinearRegression().fit(X, y)
        pred = result.predict(np.array([[1, 3.0]]))
        # E[y|x=3] ≈ 1 + 2×3 = 7
        assert abs(pred["mean"][0] - 7.0) < 1.0
        assert pred["std"][0] > 0

    def test_log_marginal_likelihood(self):
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, n)

        result = BayesianLinearRegression().fit(X, y)
        assert math.isfinite(result.log_marginal_likelihood)

    def test_to_dict(self):
        X = np.column_stack([np.ones(20), np.arange(20)])
        y = np.arange(20) * 2.0 + 1
        d = BayesianLinearRegression().fit(X, y).to_dict()
        assert "beta_mean" in d
        assert "log_marginal_likelihood" in d


# ═══════════════════════════════════════════════════════════════
# Beta-Binomial (PD estimation)
# ═══════════════════════════════════════════════════════════════

class TestBetaBinomial:
    def test_uniform_prior(self):
        """With uniform prior and 5/100 defaults: PD ≈ 5/100."""
        result = beta_binomial_update(5, 100)
        assert abs(result.posterior_mean - 6/102) < 0.01

    def test_credible_interval(self):
        result = beta_binomial_update(10, 200)
        lo, hi = result.credible_interval_95
        assert lo < result.posterior_mean < hi
        assert lo > 0 and hi < 1

    def test_informative_prior(self):
        """Strong prior (α=50, β=950) → posterior pulled toward prior."""
        result = beta_binomial_update(5, 100, prior_alpha=50, prior_beta=950)
        # Prior mean = 50/1000 = 5%, data mean = 5%
        assert abs(result.posterior_mean - 0.05) < 0.02

    def test_zero_defaults(self):
        result = beta_binomial_update(0, 100)
        assert result.posterior_mean > 0  # not exactly 0 due to prior
        assert result.posterior_mean < 0.02

    def test_to_dict(self):
        d = beta_binomial_update(3, 50).to_dict()
        assert "posterior_mean" in d
        assert "credible_interval_95" in d


# ═══════════════════════════════════════════════════════════════
# Model Selection
# ═══════════════════════════════════════════════════════════════

class TestModelSelection:
    def test_bayes_factor(self):
        bf = bayes_factor(-100, -110)
        assert bf["bayes_factor"] > 1
        assert bf["preferred_model"] == "M1"

    def test_equal_models(self):
        bf = bayes_factor(-100, -100)
        assert abs(bf["bayes_factor"] - 1.0) < 0.01

    def test_credible_interval_fn(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(5, 2, 10000)
        lo, hi = credible_interval(samples, 0.05)
        assert lo < 5 < hi
        assert abs(lo - 1) < 1  # ~5-2×2 = 1
        assert abs(hi - 9) < 1  # ~5+2×2 = 9

    def test_hpd_interval(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)
        lo, hi = hpd_interval(samples, 0.05)
        assert lo < 0 < hi
        assert (hi - lo) < 4.5  # should be ~3.92 for N(0,1)


# ═══════════════════════════════════════════════════════════════
# Changepoint Detection
# ═══════════════════════════════════════════════════════════════

class TestChangepoint:
    def test_clear_changepoint(self):
        """Two segments with different means should detect changepoint."""
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
        result = bayesian_changepoint(data)
        assert isinstance(result, ChangepointResult)
        # The most likely changepoint should have high probability
        assert result.changepoint_probs[result.most_likely_changepoint] > 0.5

    def test_no_changepoint(self):
        """Stationary data: max changepoint probability should be modest."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        result = bayesian_changepoint(data)
        # No single point should have overwhelming evidence
        assert result.changepoint_probs.max() < 0.99

    def test_to_dict(self):
        data = np.random.default_rng(42).normal(0, 1, 50)
        d = bayesian_changepoint(data).to_dict()
        assert "most_likely_changepoint" in d


# ═══════════════════════════════════════════════════════════════
# Posterior Predictive
# ═══════════════════════════════════════════════════════════════

class TestPosteriorPredictive:
    def test_basic(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(3.0, 0.5, (1000, 1))  # posterior samples of mean
        pred = posterior_predictive(
            samples,
            predict_fn=lambda theta, x: theta[0] * x,
            x_new=2.0,
        )
        assert abs(pred["mean"] - 6.0) < 0.5
        assert pred["std"] > 0
