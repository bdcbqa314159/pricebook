"""Bayesian statistics for quantitative finance.

MCMC sampling, conjugate priors, posterior inference, credible intervals,
model selection, and Bayesian calibration.

    from pricebook.statistics.bayesian import (
        MetropolisHastings, GibbsSampler, MCMCResult,
        BayesianLinearRegression, BetaBinomial,
        bayes_factor, credible_interval, posterior_predictive,
    )

References:
    Gelman et al. (2013). Bayesian Data Analysis, 3rd ed.
    Robert & Casella (2004). Monte Carlo Statistical Methods.
    Geweke (2005). Contemporary Bayesian Econometrics and Statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm, t as t_dist, invgamma, beta as beta_dist


# ═══════════════════════════════════════════════════════════════
# MCMC Results
# ═══════════════════════════════════════════════════════════════


@dataclass
class MCMCResult:
    """Result of MCMC sampling."""
    samples: np.ndarray          # (n_samples, n_params)
    log_posteriors: np.ndarray   # (n_samples,)
    acceptance_rate: float
    n_samples: int
    burn_in: int
    param_names: list[str]

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def std(self) -> np.ndarray:
        return self.samples.std(axis=0)

    def credible_interval(self, alpha: float = 0.05) -> list[tuple[float, float]]:
        """Credible intervals for each parameter."""
        lo = np.percentile(self.samples, 100 * alpha / 2, axis=0)
        hi = np.percentile(self.samples, 100 * (1 - alpha / 2), axis=0)
        return [(float(l), float(h)) for l, h in zip(lo, hi)]

    def effective_sample_size(self) -> np.ndarray:
        """ESS per parameter (accounting for autocorrelation)."""
        n = self.n_samples
        ess = np.zeros(self.samples.shape[1])
        for j in range(self.samples.shape[1]):
            x = self.samples[:, j]
            x = x - x.mean()
            acf = np.correlate(x, x, mode='full')[n-1:] / (np.var(x) * n)
            # Sum autocorrelations until they go negative
            tau = 1.0
            for k in range(1, min(n // 2, 500)):
                if acf[k] < 0:
                    break
                tau += 2 * acf[k]
            ess[j] = n / tau
        return ess

    def to_dict(self) -> dict:
        return {
            "mean": self.mean().tolist(),
            "std": self.std().tolist(),
            "credible_95": self.credible_interval(0.05),
            "acceptance_rate": self.acceptance_rate,
            "n_samples": self.n_samples,
            "param_names": self.param_names,
        }


# ═══════════════════════════════════════════════════════════════
# Metropolis-Hastings
# ═══════════════════════════════════════════════════════════════


class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler.

    Args:
        log_posterior: callable(theta) → float (unnormalised log-posterior).
        proposal_std: (D,) proposal standard deviations (Gaussian random walk).
        param_names: optional parameter names.
    """

    def __init__(
        self,
        log_posterior: callable,
        proposal_std: np.ndarray | float,
        param_names: list[str] | None = None,
    ):
        self.log_posterior = log_posterior
        self.proposal_std = np.atleast_1d(np.asarray(proposal_std, dtype=float))
        self.param_names = param_names

    def sample(
        self,
        theta0: np.ndarray,
        n_samples: int = 10_000,
        burn_in: int = 1_000,
        seed: int = 42,
    ) -> MCMCResult:
        """Run the MH sampler.

        Args:
            theta0: (D,) initial parameter values.
            n_samples: number of post-burn-in samples.
            burn_in: number of initial samples to discard.
        """
        rng = np.random.default_rng(seed)
        theta = np.asarray(theta0, dtype=float)
        D = len(theta)
        total = n_samples + burn_in

        if self.param_names is None:
            self.param_names = [f"theta_{i}" for i in range(D)]

        samples = np.zeros((total, D))
        log_posts = np.zeros(total)
        accepted = 0
        current_lp = self.log_posterior(theta)

        for i in range(total):
            # Propose
            proposal = theta + rng.normal(0, self.proposal_std, D)
            proposed_lp = self.log_posterior(proposal)

            # Accept/reject
            log_alpha = proposed_lp - current_lp
            if math.log(max(rng.uniform(), 1e-300)) < log_alpha:
                theta = proposal
                current_lp = proposed_lp
                accepted += 1

            samples[i] = theta
            log_posts[i] = current_lp

        return MCMCResult(
            samples=samples[burn_in:],
            log_posteriors=log_posts[burn_in:],
            acceptance_rate=accepted / total,
            n_samples=n_samples,
            burn_in=burn_in,
            param_names=self.param_names,
        )


# ═══════════════════════════════════════════════════════════════
# Gibbs Sampler
# ═══════════════════════════════════════════════════════════════


class GibbsSampler:
    """Gibbs sampler for models with known conditional distributions.

    Each conditional must sample one parameter block given all others.

    Args:
        conditionals: list of callables. conditionals[i](theta, rng) → new theta[i].
            Each receives the full parameter vector and returns the updated value
            for component i, sampled from its full conditional.
        param_names: optional names.
    """

    def __init__(
        self,
        conditionals: list[callable],
        param_names: list[str] | None = None,
    ):
        self.conditionals = conditionals
        self.n_params = len(conditionals)
        self.param_names = param_names or [f"theta_{i}" for i in range(self.n_params)]

    def sample(
        self,
        theta0: np.ndarray,
        n_samples: int = 10_000,
        burn_in: int = 1_000,
        seed: int = 42,
    ) -> MCMCResult:
        rng = np.random.default_rng(seed)
        theta = np.asarray(theta0, dtype=float).copy()
        total = n_samples + burn_in
        samples = np.zeros((total, self.n_params))

        for i in range(total):
            for j in range(self.n_params):
                theta[j] = self.conditionals[j](theta, rng)
            samples[i] = theta.copy()

        return MCMCResult(
            samples=samples[burn_in:],
            log_posteriors=np.zeros(n_samples),  # Gibbs doesn't track log-post
            acceptance_rate=1.0,  # Gibbs always accepts
            n_samples=n_samples,
            burn_in=burn_in,
            param_names=self.param_names,
        )


# ═══════════════════════════════════════════════════════════════
# Conjugate Priors
# ═══════════════════════════════════════════════════════════════


@dataclass
class BayesianLinearRegressionResult:
    """Posterior of Bayesian linear regression."""
    beta_mean: np.ndarray        # posterior mean of coefficients
    beta_cov: np.ndarray         # posterior covariance
    sigma2_mean: float           # posterior mean of error variance
    sigma2_shape: float          # inverse-gamma shape (a)
    sigma2_scale: float          # inverse-gamma scale (b)
    n_obs: int
    n_params: int
    log_marginal_likelihood: float

    def credible_intervals(self, alpha: float = 0.05) -> list[dict]:
        """Credible intervals for each coefficient."""
        z = norm.ppf(1 - alpha / 2)
        results = []
        for i in range(self.n_params):
            se = math.sqrt(self.beta_cov[i, i] * self.sigma2_mean)
            results.append({
                "param_idx": i,
                "mean": float(self.beta_mean[i]),
                "ci_lower": float(self.beta_mean[i] - z * se),
                "ci_upper": float(self.beta_mean[i] + z * se),
            })
        return results

    def predict(self, X_new: np.ndarray) -> dict:
        """Posterior predictive distribution at new points."""
        X = np.asarray(X_new)
        y_mean = X @ self.beta_mean
        y_var = self.sigma2_mean * (1 + np.sum((X @ self.beta_cov) * X, axis=1))
        return {"mean": y_mean, "std": np.sqrt(y_var)}

    def to_dict(self) -> dict:
        return {
            "beta_mean": self.beta_mean.tolist(),
            "sigma2_mean": self.sigma2_mean,
            "n_obs": self.n_obs,
            "log_marginal_likelihood": self.log_marginal_likelihood,
        }


class BayesianLinearRegression:
    """Bayesian linear regression with Normal-Inverse-Gamma conjugate prior.

    Prior: β|σ² ~ N(β₀, σ²Λ₀⁻¹), σ² ~ IG(a₀, b₀)
    Posterior: β|σ²,y ~ N(β_n, σ²Λ_n⁻¹), σ²|y ~ IG(a_n, b_n)

    Closed-form posterior — no MCMC needed.
    """

    def __init__(
        self,
        prior_beta_mean: np.ndarray | None = None,
        prior_beta_precision: np.ndarray | None = None,
        prior_sigma2_shape: float = 1.0,
        prior_sigma2_scale: float = 1.0,
    ):
        self.beta0 = prior_beta_mean
        self.Lambda0 = prior_beta_precision
        self.a0 = prior_sigma2_shape
        self.b0 = prior_sigma2_scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesianLinearRegressionResult:
        """Compute posterior given data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        # Default uninformative prior
        if self.beta0 is None:
            self.beta0 = np.zeros(p)
        if self.Lambda0 is None:
            self.Lambda0 = np.eye(p) * 0.01  # weak prior

        # Posterior precision and mean
        Lambda_n = self.Lambda0 + X.T @ X
        Lambda_n_inv = np.linalg.inv(Lambda_n)
        beta_n = Lambda_n_inv @ (self.Lambda0 @ self.beta0 + X.T @ y)

        # Posterior for sigma²
        a_n = self.a0 + n / 2
        residual = y - X @ beta_n
        b_n = self.b0 + 0.5 * (residual @ residual
                                 + (beta_n - self.beta0) @ self.Lambda0 @ (beta_n - self.beta0))

        sigma2_mean = b_n / (a_n - 1) if a_n > 1 else b_n

        # Log marginal likelihood (evidence)
        sign0, logdet0 = np.linalg.slogdet(self.Lambda0)
        sign_n, logdet_n = np.linalg.slogdet(Lambda_n)
        log_ml = (0.5 * logdet0 - 0.5 * logdet_n
                   + self.a0 * math.log(max(self.b0, 1e-15))
                   - a_n * math.log(max(b_n, 1e-15))
                   + math.lgamma(a_n) - math.lgamma(self.a0)
                   - n / 2 * math.log(2 * math.pi))

        return BayesianLinearRegressionResult(
            beta_mean=beta_n,
            beta_cov=Lambda_n_inv,
            sigma2_mean=sigma2_mean,
            sigma2_shape=a_n,
            sigma2_scale=b_n,
            n_obs=n,
            n_params=p,
            log_marginal_likelihood=log_ml,
        )


# ═══════════════════════════════════════════════════════════════
# Beta-Binomial (for PD estimation)
# ═══════════════════════════════════════════════════════════════


@dataclass
class BetaBinomialResult:
    """Posterior of Beta-Binomial model."""
    alpha_posterior: float       # Beta shape α
    beta_posterior: float        # Beta shape β
    posterior_mean: float        # E[p] = α/(α+β)
    posterior_mode: float        # mode = (α-1)/(α+β-2) if α,β > 1
    posterior_std: float
    credible_interval_95: tuple[float, float]
    n_defaults: int
    n_observations: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def beta_binomial_update(
    n_defaults: int,
    n_observations: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> BetaBinomialResult:
    """Bayesian PD estimation with Beta-Binomial conjugate model.

    Prior: p ~ Beta(α₀, β₀)
    Likelihood: k|p ~ Binomial(n, p)
    Posterior: p|k ~ Beta(α₀ + k, β₀ + n - k)

    With uninformative prior (α=β=1), posterior mean = (k+1)/(n+2).

    Args:
        n_defaults: observed defaults.
        n_observations: total observations.
        prior_alpha: prior Beta shape α (default 1 = uniform).
        prior_beta: prior Beta shape β.
    """
    alpha_n = prior_alpha + n_defaults
    beta_n = prior_beta + n_observations - n_defaults

    mean = alpha_n / (alpha_n + beta_n)
    mode = (alpha_n - 1) / (alpha_n + beta_n - 2) if alpha_n > 1 and beta_n > 1 else mean
    var = (alpha_n * beta_n) / ((alpha_n + beta_n)**2 * (alpha_n + beta_n + 1))
    std = math.sqrt(var)

    ci = (float(beta_dist.ppf(0.025, alpha_n, beta_n)),
          float(beta_dist.ppf(0.975, alpha_n, beta_n)))

    return BetaBinomialResult(
        alpha_posterior=alpha_n,
        beta_posterior=beta_n,
        posterior_mean=mean,
        posterior_mode=mode,
        posterior_std=std,
        credible_interval_95=ci,
        n_defaults=n_defaults,
        n_observations=n_observations,
    )


# ═══════════════════════════════════════════════════════════════
# Model Selection
# ═══════════════════════════════════════════════════════════════


def bayes_factor(log_ml_1: float, log_ml_2: float) -> dict:
    """Bayes factor for model comparison.

    BF₁₂ = p(data|M₁) / p(data|M₂) = exp(log_ml_1 - log_ml_2)

    Interpretation (Kass & Raftery 1995):
    BF > 100: decisive evidence for M₁
    BF > 10: strong evidence
    BF > 3: moderate evidence
    BF > 1: weak evidence
    """
    log_bf = log_ml_1 - log_ml_2
    bf = math.exp(min(log_bf, 700))  # prevent overflow

    if bf > 100:
        strength = "decisive"
    elif bf > 10:
        strength = "strong"
    elif bf > 3:
        strength = "moderate"
    elif bf > 1:
        strength = "weak_for_M1"
    elif bf > 1/3:
        strength = "inconclusive"
    else:
        strength = "evidence_for_M2"

    return {
        "bayes_factor": bf,
        "log_bayes_factor": log_bf,
        "preferred_model": "M1" if bf > 1 else "M2",
        "strength": strength,
    }


def credible_interval(
    samples: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Equal-tailed credible interval from posterior samples."""
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return (lo, hi)


def hpd_interval(
    samples: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Highest Posterior Density (HPD) interval — shortest interval containing (1-α)%.

    More informative than equal-tailed for skewed posteriors.
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    n_included = int(math.ceil((1 - alpha) * n))
    widths = sorted_samples[n_included:] - sorted_samples[:n - n_included]
    best = int(np.argmin(widths))
    return (float(sorted_samples[best]), float(sorted_samples[best + n_included]))


def posterior_predictive(
    samples: np.ndarray,
    predict_fn: callable,
    x_new: np.ndarray | float,
) -> dict:
    """Posterior predictive distribution via Monte Carlo.

    For each posterior sample θ, compute predict_fn(θ, x_new).
    Returns mean, std, credible interval of predictions.

    Args:
        samples: (N, D) posterior parameter samples.
        predict_fn: callable(theta, x) → float prediction.
        x_new: new input point(s).
    """
    predictions = np.array([predict_fn(theta, x_new) for theta in samples])
    return {
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions)),
        "ci_95": credible_interval(predictions, 0.05),
        "median": float(np.median(predictions)),
    }


# ═══════════════════════════════════════════════════════════════
# Bayesian Changepoint Detection
# ═══════════════════════════════════════════════════════════════


@dataclass
class ChangepointResult:
    """Result of Bayesian changepoint detection."""
    changepoint_probs: np.ndarray  # (T,) posterior probability of changepoint at each t
    most_likely_changepoint: int
    n_changepoints_expected: float
    segments: list[dict]           # [{start, end, mean, std}, ...]

    def to_dict(self) -> dict:
        return {
            "most_likely_changepoint": self.most_likely_changepoint,
            "n_changepoints_expected": self.n_changepoints_expected,
            "n_segments": len(self.segments),
        }


def bayesian_changepoint(
    data: np.ndarray,
    prior_prob: float = 0.01,
) -> ChangepointResult:
    """Bayesian online changepoint detection (Adams & MacKay 2007).

    At each time t, compute the posterior probability that a changepoint
    occurred at t. Uses a run-length model with geometric prior on
    run length.

    Args:
        data: (T,) time series.
        prior_prob: prior probability of changepoint at each step (1/expected_run_length).
    """
    data = np.asarray(data, dtype=float)
    T = len(data)

    # Run-length posterior: R(t, r) = P(run_length = r at time t)
    # Simplified: compute P(changepoint at t) marginally
    cp_probs = np.zeros(T)

    # Use cumulative sufficient statistics
    for tau in range(1, T):
        # Log-likelihood ratio: data[tau:] vs data[:tau] as separate segments
        seg1 = data[:tau]
        seg2 = data[tau:]
        if len(seg1) < 2 or len(seg2) < 2:
            continue

        # Evidence for changepoint at tau
        ll_joint = _normal_log_likelihood(data)
        ll_split = _normal_log_likelihood(seg1) + _normal_log_likelihood(seg2)
        log_bf = ll_split - ll_joint + math.log(prior_prob) - math.log(1 - prior_prob)
        cp_probs[tau] = 1.0 / (1.0 + math.exp(-min(log_bf, 500)))

    # Most likely changepoint
    best = int(np.argmax(cp_probs))
    expected = float(cp_probs.sum())

    # Segments
    segments = []
    if cp_probs[best] > 0.5 and best > 0 and best < T:
        segments = [
            {"start": 0, "end": best, "mean": float(data[:best].mean()),
             "std": float(data[:best].std())},
            {"start": best, "end": T, "mean": float(data[best:].mean()),
             "std": float(data[best:].std())},
        ]
    else:
        segments = [{"start": 0, "end": T, "mean": float(data.mean()),
                      "std": float(data.std())}]

    return ChangepointResult(
        changepoint_probs=cp_probs,
        most_likely_changepoint=best,
        n_changepoints_expected=expected,
        segments=segments,
    )


def _normal_log_likelihood(x):
    n = len(x)
    if n < 2:
        return 0.0
    mu = x.mean()
    var = max(x.var(), 1e-10)
    return -0.5 * n * math.log(2 * math.pi * var) - 0.5 * np.sum((x - mu)**2) / var
