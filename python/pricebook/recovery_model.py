"""
Recovery rate models: stochastic, LGD distribution, correlated recovery.

BetaRecovery: recovery rate drawn from Beta distribution at default.
LGDModel: Loss Given Default = 1 - R, with distributional properties.
CorrelatedRecovery: recovery depends on a systematic factor M (downturn LGD).

    from pricebook.recovery_model import BetaRecovery, CorrelatedRecovery

    rec = BetaRecovery(mean=0.40, std=0.20)
    samples = rec.sample(n=10000)

    crec = CorrelatedRecovery(base_mean=0.40, sensitivity=0.3)
    R = crec.recovery_given_factor(M=-2.0)  # downturn: lower recovery
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta as beta_dist


class BetaRecovery:
    """Recovery rate from a Beta distribution.

    Beta is natural for [0, 1]-valued random variables.
    Parameterised by mean and std (converted to alpha, beta internally).

    Args:
        mean: expected recovery rate (e.g. 0.40).
        std: standard deviation of recovery (e.g. 0.20).
    """

    def __init__(self, mean: float = 0.40, std: float = 0.20):
        if not 0 < mean < 1:
            raise ValueError(f"mean must be in (0,1), got {mean}")
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")

        self.mean = mean
        self.std = std

        # Beta distribution params from mean and variance
        var = std**2
        if var >= mean * (1 - mean):
            raise ValueError(
                f"std={std} too large for mean={mean} (variance must be < mean*(1-mean))"
            )
        self.alpha = mean * (mean * (1 - mean) / var - 1)
        self.beta = (1 - mean) * (mean * (1 - mean) / var - 1)

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        """Draw n recovery rate samples. Shape: (n,)."""
        rng = np.random.default_rng(seed)
        return rng.beta(self.alpha, self.beta, size=n)

    def pdf(self, r: float) -> float:
        """Probability density at recovery rate r."""
        return float(beta_dist.pdf(r, self.alpha, self.beta))

    def cdf(self, r: float) -> float:
        """Cumulative distribution at r."""
        return float(beta_dist.cdf(r, self.alpha, self.beta))

    @property
    def expected_lgd(self) -> float:
        """E[LGD] = 1 - E[R] = 1 - mean."""
        return 1 - self.mean


class LGDModel:
    """Loss Given Default model.

    LGD = 1 - R. Provides distributional properties and sampling.

    Args:
        recovery_model: a BetaRecovery (or similar with .sample method).
    """

    def __init__(self, recovery_model: BetaRecovery):
        self.recovery = recovery_model

    def sample_lgd(self, n: int, seed: int = 42) -> np.ndarray:
        """Sample LGD values. Shape: (n,)."""
        return 1.0 - self.recovery.sample(n, seed)

    @property
    def mean(self) -> float:
        return self.recovery.expected_lgd

    @property
    def std(self) -> float:
        return self.recovery.std

    def expected_loss(self, default_prob: float, notional: float = 1.0) -> float:
        """Expected loss = PD * LGD * notional."""
        return default_prob * self.mean * notional


class CorrelatedRecovery:
    """Recovery correlated with the systematic factor.

    In downturns (low M), recovery is lower (higher LGD).
    Model: R(M) = mean + sensitivity * M, clipped to [floor, cap].

    This creates wrong-way risk for portfolio credit products:
    when more defaults occur (low M), each default has higher loss.

    Args:
        base_mean: average recovery in normal conditions.
        sensitivity: dR/dM (positive: recovery increases with M).
        floor: minimum recovery.
        cap: maximum recovery.
    """

    def __init__(
        self,
        base_mean: float = 0.40,
        sensitivity: float = 0.10,
        floor: float = 0.05,
        cap: float = 0.95,
    ):
        self.base_mean = base_mean
        self.sensitivity = sensitivity
        self.floor = floor
        self.cap = cap

    def recovery_given_factor(self, M: float) -> float:
        """Recovery rate given systematic factor M (standard normal)."""
        R = self.base_mean + self.sensitivity * M
        return max(self.floor, min(self.cap, R))

    def sample(self, M_values: np.ndarray) -> np.ndarray:
        """Recovery for each systematic factor value. Shape matches M_values."""
        R = self.base_mean + self.sensitivity * M_values
        return np.clip(R, self.floor, self.cap)

    def downturn_lgd(self, percentile: float = 0.01) -> float:
        """LGD in a downturn scenario (low M percentile).

        E.g. percentile=0.01 → M at 1st percentile of N(0,1) ≈ -2.33.
        """
        from scipy.stats import norm
        M = norm.ppf(percentile)
        R = self.recovery_given_factor(M)
        return 1 - R

    def expected_portfolio_loss(
        self,
        default_indicators: np.ndarray,
        M_values: np.ndarray,
        notional_per_name: float = 1.0,
    ) -> float:
        """Expected portfolio loss with correlated recovery.

        Args:
            default_indicators: boolean (n_sims, n_names).
            M_values: systematic factor per simulation (n_sims,).
        """
        R = self.sample(M_values)  # (n_sims,)
        LGD = 1 - R  # (n_sims,)
        n_defaults = default_indicators.sum(axis=1)  # (n_sims,)
        loss_per_sim = n_defaults * LGD * notional_per_name
        return float(loss_per_sim.mean())
