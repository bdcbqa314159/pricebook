"""Correlated recovery model.

Recovery rates are not independent of default — they decline when many
firms default simultaneously (systematic risk). This module implements
a factor-based correlated recovery model.

    from pricebook.credit.correlated_recovery import (
        CorrelatedRecoveryModel, systematic_recovery,
    )

    model = CorrelatedRecoveryModel(
        base_recovery=0.40, recovery_vol=0.25, systematic_factor_loading=0.30,
    )
    # In a stress scenario where the systematic factor is -2σ:
    r = model.conditional_recovery(systematic_factor=-2.0)

References:
    Altman, Brady, Resti & Sironi (2005). The Link between Default and
    Recovery Rates. Journal of Business.
    Frye (2000). Depressing Recoveries. Risk Magazine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, beta as beta_dist


@dataclass
class CorrelatedRecoveryResult:
    """Result of correlated recovery computation."""
    base_recovery: float
    conditional_recovery: float
    systematic_factor: float
    stress_multiplier: float
    recovery_samples: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "base_recovery": self.base_recovery,
            "conditional_recovery": self.conditional_recovery,
            "systematic_factor": self.systematic_factor,
            "stress_multiplier": self.stress_multiplier,
        }


class CorrelatedRecoveryModel:
    """Factor-based correlated recovery model (Frye 2000).

    Recovery depends on a systematic factor M:
        R(M) = E[R] + β_R × M × σ_R

    where:
    - E[R] = unconditional mean recovery
    - β_R = systematic factor loading (how much recovery co-moves with the economy)
    - M = systematic factor (standard normal: M < 0 = recession)
    - σ_R = recovery volatility

    In bad economic states (M << 0), both default rates increase AND
    recovery rates decline — the double hit of systematic risk.
    """

    def __init__(
        self,
        base_recovery: float = 0.40,
        recovery_vol: float = 0.25,
        systematic_factor_loading: float = 0.30,
    ):
        self.base_recovery = base_recovery
        self.recovery_vol = recovery_vol
        self.beta = systematic_factor_loading

    def conditional_recovery(self, systematic_factor: float) -> float:
        """Recovery rate conditional on the systematic factor.

        R(M) = base_R + β × M × σ_R, clipped to [0.01, 0.95].
        """
        r = self.base_recovery + self.beta * systematic_factor * self.recovery_vol
        return max(0.01, min(0.95, r))

    def stress_recovery(self, n_sigma: float = 2.0) -> float:
        """Recovery in a stress scenario (systematic factor = -n_sigma)."""
        return self.conditional_recovery(-n_sigma)

    def expansion_recovery(self, n_sigma: float = 1.0) -> float:
        """Recovery in an expansion (systematic factor = +n_sigma)."""
        return self.conditional_recovery(n_sigma)

    def sample_recoveries(
        self, n_defaults: int, systematic_factor: float,
        idio_vol: float = 0.15, rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample recovery rates for n defaults given systematic factor.

        Each default's recovery = conditional_mean + idiosyncratic noise.
        Uses beta distribution to ensure [0, 1].
        """
        if rng is None:
            rng = np.random.default_rng()

        mu = self.conditional_recovery(systematic_factor)
        mu = max(0.02, min(0.98, mu))
        var = idio_vol ** 2
        var = min(var, mu * (1 - mu) * 0.99)

        if var <= 0:
            return np.full(n_defaults, mu)

        factor = mu * (1 - mu) / var - 1.0
        if factor <= 0:
            return np.full(n_defaults, mu)

        a = mu * factor
        b = (1 - mu) * factor
        return rng.beta(a, b, n_defaults)

    def recovery_distribution(
        self, systematic_scenarios: list[float],
    ) -> list[dict]:
        """Recovery at multiple systematic factor levels."""
        return [
            {
                "systematic_factor": m,
                "recovery": self.conditional_recovery(m),
                "scenario": "stress" if m < -1 else "expansion" if m > 1 else "normal",
            }
            for m in systematic_scenarios
        ]

    def to_dict(self) -> dict:
        return {
            "base_recovery": self.base_recovery,
            "recovery_vol": self.recovery_vol,
            "systematic_factor_loading": self.beta,
            "stress_2sigma": self.stress_recovery(2.0),
            "expansion_1sigma": self.expansion_recovery(1.0),
        }


def systematic_recovery(
    base_recovery: float,
    default_rate: float,
    rho: float = 0.20,
) -> float:
    """Frye (2000) recovery conditional on portfolio default rate.

    When the default rate is high (more defaults than expected),
    recovery is low (systematic risk). Uses the Vasicek single-factor
    model to link default rates to the systematic factor.

    Args:
        base_recovery: unconditional mean recovery.
        default_rate: observed/assumed portfolio default rate.
        rho: asset correlation (Vasicek model).

    Returns:
        Conditional recovery rate.
    """
    if default_rate <= 0 or default_rate >= 1:
        return base_recovery

    # Invert Vasicek to get systematic factor
    # default_rate = Φ((Φ⁻¹(PD) - √ρ × M) / √(1-ρ))
    # Assuming PD ≈ default_rate for the marginal:
    # M ≈ (Φ⁻¹(PD) - √(1-ρ) × Φ⁻¹(default_rate)) / √ρ
    # Simplified: M ≈ -Φ⁻¹(default_rate) (higher default → negative M)
    M = -norm.ppf(default_rate)

    # Recovery response: empirically ~-0.10 per unit of M
    recovery_loading = 0.10
    r = base_recovery + recovery_loading * M
    return max(0.01, min(0.95, r))
