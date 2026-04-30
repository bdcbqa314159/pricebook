"""Recovery pricing: stochastic recovery, default-recovery correlation, wrong-way risk.

The core insight: E[(1-R) × 1_default] ≠ (1-E[R]) × P[default] when
recovery R and default are correlated. This module provides the tools
to price with stochastic recovery and quantify the wrong-way premium.

    from pricebook.recovery_pricing import (
        RecoverySpec, correlated_default_recovery,
        wrong_way_premium, lgd_term_structure,
    )

References:
    Altman, Resti, Sironi (2004). Default Recovery Rates in Credit Risk
    Modelling. J. Banking & Finance.
    Frye (2000). Depressing Recoveries. Risk.
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives,
    Ch. 15-16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
from scipy.stats import norm, beta as beta_dist

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.survival_curve import SurvivalCurve


# ---------------------------------------------------------------------------
# Seniority recovery table (Moody's 2022)
# ---------------------------------------------------------------------------

SENIORITY_RECOVERY = {
    "1L": (0.77, 0.20),     # mean, std
    "2L": (0.43, 0.25),
    "senior": (0.45, 0.22),
    "senior_secured": (0.65, 0.22),
    "senior_unsecured": (0.45, 0.22),
    "sub": (0.28, 0.20),
    "mezzanine": (0.35, 0.22),
}


# ---------------------------------------------------------------------------
# RecoverySpec
# ---------------------------------------------------------------------------

@dataclass
class RecoverySpec:
    """Unified recovery specification for any credit product.

    Describes recovery behaviour: mean, volatility, distribution shape,
    and correlation with the default driver.

    Args:
        mean: expected recovery rate (e.g. 0.40).
        std: recovery volatility (0.0 = deterministic).
        distribution: "fixed", "beta", or "lognormal".
        correlation_to_default: ρ_DR, typically negative (-0.3 to -0.5).
            Negative = wrong-way risk (recovery falls when defaults rise).
    """
    mean: float = 0.40
    std: float = 0.20
    distribution: str = "beta"
    correlation_to_default: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.mean <= 1.0:
            raise ValueError(f"mean must be in [0, 1], got {self.mean}")
        if self.std < 0:
            raise ValueError(f"std must be >= 0, got {self.std}")
        if not -1.0 <= self.correlation_to_default <= 1.0:
            raise ValueError(f"correlation must be in [-1, 1], got {self.correlation_to_default}")

    @classmethod
    def from_seniority(cls, seniority: str, correlation_to_default: float = -0.3) -> RecoverySpec:
        """Build from seniority level using Moody's historical data."""
        if seniority not in SENIORITY_RECOVERY:
            raise ValueError(f"Unknown seniority '{seniority}'. Options: {list(SENIORITY_RECOVERY)}")
        mean, std = SENIORITY_RECOVERY[seniority]
        return cls(mean=mean, std=std, correlation_to_default=correlation_to_default)

    @classmethod
    def fixed(cls, recovery: float = 0.40) -> RecoverySpec:
        """Deterministic (fixed) recovery."""
        return cls(mean=recovery, std=0.0, distribution="fixed")

    @property
    def is_deterministic(self) -> bool:
        return self.std == 0.0 or self.distribution == "fixed"

    @property
    def expected_lgd(self) -> float:
        return 1.0 - self.mean

    def _beta_params(self) -> tuple[float, float]:
        """Convert mean/std to beta distribution alpha, beta."""
        var = self.std ** 2
        if var >= self.mean * (1 - self.mean):
            # Clamp variance
            var = self.mean * (1 - self.mean) * 0.99
        alpha = self.mean * (self.mean * (1 - self.mean) / var - 1)
        beta_p = (1 - self.mean) * (self.mean * (1 - self.mean) / var - 1)
        return max(alpha, 0.01), max(beta_p, 0.01)

    def sample(
        self,
        n: int,
        systematic_factor: np.ndarray | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Sample recovery rates.

        If systematic_factor is provided and correlation_to_default != 0,
        samples are correlated with the factor via Gaussian copula.

        Args:
            n: number of samples.
            systematic_factor: Z values driving default (shape (n,)).
            seed: random seed.

        Returns:
            Recovery samples, shape (n,).
        """
        if self.is_deterministic:
            return np.full(n, self.mean)

        rng = np.random.default_rng(seed)

        if systematic_factor is not None and abs(self.correlation_to_default) > 1e-10:
            # Sign convention: correlation_to_default < 0 means wrong-way risk
            # (recovery falls when defaults rise). Since low Z_D = default,
            # we negate ρ so that negative correlation → Z_R co-moves with Z_D.
            rho_copula = -self.correlation_to_default
            eps = rng.standard_normal(n)
            Z_R = rho_copula * systematic_factor + math.sqrt(1 - rho_copula ** 2) * eps
            # Transform to uniform via Φ
            U = norm.cdf(Z_R)
        else:
            U = rng.uniform(size=n)

        # Map uniform to recovery distribution
        if self.distribution == "beta":
            a, b = self._beta_params()
            return np.clip(beta_dist.ppf(U, a, b), 0.0, 1.0)
        elif self.distribution == "lognormal":
            # Lognormal mapped to [0,1]: R = min(exp(μ + σ × Φ⁻¹(U)), 1)
            mu = math.log(self.mean) - 0.5 * self.std ** 2
            sigma = self.std
            return np.clip(np.exp(mu + sigma * norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))), 0.0, 1.0)
        else:
            return np.full(n, self.mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean, "std": self.std,
            "distribution": self.distribution,
            "correlation_to_default": self.correlation_to_default,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RecoverySpec:
        return cls(
            mean=d["mean"], std=d.get("std", 0.2),
            distribution=d.get("distribution", "beta"),
            correlation_to_default=d.get("correlation_to_default", 0.0),
        )


# ---------------------------------------------------------------------------
# Default-recovery correlation
# ---------------------------------------------------------------------------

@dataclass
class DefaultRecoveryResult:
    """Joint default and recovery simulation result."""
    default_indicators: np.ndarray  # (n_sims,) bool
    recovery_rates: np.ndarray      # (n_sims,) float
    expected_lgd_given_default: float
    expected_loss: float            # E[(1-R) × 1_D]
    naive_expected_loss: float      # (1-E[R]) × PD
    wrong_way_premium: float        # expected_loss - naive_expected_loss


def correlated_default_recovery(
    pd: float,
    recovery_spec: RecoverySpec,
    n_sims: int = 100_000,
    seed: int = 42,
) -> DefaultRecoveryResult:
    """Joint simulation of default and recovery via Gaussian copula.

    Z_D ~ N(0,1), default if Z_D < Φ⁻¹(PD)
    Z_R = ρ_DR × Z_D + √(1-ρ_DR²) × ε_R
    R = F_beta⁻¹(Φ(Z_R))

    When ρ_DR < 0: default (low Z_D) → low Z_R → low R (wrong-way).
    """
    rng = np.random.default_rng(seed)
    Z_D = rng.standard_normal(n_sims)

    threshold = norm.ppf(max(pd, 1e-15))
    defaults = Z_D < threshold

    # Recovery correlated with default driver
    R = recovery_spec.sample(n_sims, systematic_factor=Z_D, seed=seed + 1)

    # Conditional recovery given default
    n_defaults = defaults.sum()
    if n_defaults > 0:
        lgd_given_default = float((1.0 - R[defaults]).mean())
    else:
        lgd_given_default = recovery_spec.expected_lgd

    # Expected loss: E[(1-R) × 1_D]
    expected_loss = float(((1.0 - R) * defaults).mean())
    naive_loss = recovery_spec.expected_lgd * pd
    premium = expected_loss - naive_loss

    return DefaultRecoveryResult(
        default_indicators=defaults,
        recovery_rates=R,
        expected_lgd_given_default=lgd_given_default,
        expected_loss=expected_loss,
        naive_expected_loss=naive_loss,
        wrong_way_premium=premium,
    )


# ---------------------------------------------------------------------------
# Wrong-way premium
# ---------------------------------------------------------------------------

def wrong_way_premium(
    pd: float,
    recovery_spec: RecoverySpec,
    n_sims: int = 100_000,
    seed: int = 42,
) -> float:
    """Wrong-way recovery premium: E[(1-R)×1_D] - (1-E[R])×PD.

    The amount by which naive (independent R) pricing under-estimates
    expected loss. Positive when ρ_DR < 0 (the common case).
    """
    result = correlated_default_recovery(pd, recovery_spec, n_sims, seed)
    return result.wrong_way_premium


# ---------------------------------------------------------------------------
# LGD term structure
# ---------------------------------------------------------------------------

def lgd_term_structure(
    survival_curve: SurvivalCurve,
    recovery_spec: RecoverySpec,
    tenors: list[float] | None = None,
    n_sims: int = 100_000,
    seed: int = 42,
) -> list[dict[str, float]]:
    """LGD term structure with default-recovery correlation.

    At each tenor: compute cumulative PD, then E[LGD|default] considering
    the correlation between default probability and recovery.

    Returns list of {tenor, pd, lgd_independent, lgd_correlated, premium}.
    """
    if tenors is None:
        tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    ref = survival_curve.reference_date
    results = []

    for T in tenors:
        d = ref + timedelta(days=round(T * 365))
        pd = 1.0 - survival_curve.survival(d)

        if pd < 1e-10:
            results.append({
                "tenor": T, "pd": pd,
                "lgd_independent": recovery_spec.expected_lgd,
                "lgd_correlated": recovery_spec.expected_lgd,
                "premium": 0.0,
            })
            continue

        dr = correlated_default_recovery(pd, recovery_spec, n_sims, seed + int(T * 100))

        results.append({
            "tenor": T,
            "pd": pd,
            "lgd_independent": recovery_spec.expected_lgd,
            "lgd_correlated": dr.expected_lgd_given_default,
            "premium": dr.wrong_way_premium,
        })

    return results
