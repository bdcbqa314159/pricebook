"""Loan credit modelling: recovery by seniority, stochastic LGD,
spread-dependent prepayment, regulatory LGD.

    from pricebook.loan_credit import (
        RecoveryModel, StochasticRecovery, SpreadPrepayModel,
        lgd_regulatory, capital_requirement,
    )

    model = RecoveryModel(seniority="1L")
    print(model.expected_recovery())  # 0.77

    prepay = SpreadPrepayModel(loan_spread=0.04)
    cpr = prepay.conditional_cpr(market_spread=0.03)  # high — refinancing

References:
    Moody's (2022). Annual Default Study. Recovery rates by seniority.
    Altman & Kishore (1996). Almost Everything You Wanted to Know About
    Recoveries on Defaulted Bonds. Financial Analysts Journal.
    Basel Committee (2006). International Convergence of Capital Measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm, beta as beta_dist


# ---------------------------------------------------------------------------
# P2.1: Recovery by seniority
# ---------------------------------------------------------------------------

# Historical recovery rates (Moody's 2022 Annual Default Study)
# Issuer-weighted, post-default trading prices
RECOVERY_BY_SENIORITY = {
    "1L":         {"mean": 0.77, "std": 0.22, "label": "First Lien Secured"},
    "2L":         {"mean": 0.43, "std": 0.28, "label": "Second Lien Secured"},
    "senior":     {"mean": 0.45, "std": 0.26, "label": "Senior Unsecured"},
    "sub":        {"mean": 0.28, "std": 0.21, "label": "Subordinated"},
    "mezzanine":  {"mean": 0.35, "std": 0.25, "label": "Mezzanine"},
}

# Industry recovery adjustments (relative to average)
INDUSTRY_ADJUSTMENT = {
    "utilities":    +0.10,
    "telecom":      +0.05,
    "healthcare":   +0.03,
    "manufacturing": 0.00,
    "retail":       -0.05,
    "energy":       -0.03,
    "technology":   -0.02,
}


class RecoveryModel:
    """Seniority and industry-dependent recovery model.

    Uses historical Moody's data calibrated by seniority tier.
    Optionally adjusts for industry and collateral type.

    Args:
        seniority: "1L", "2L", "senior", "sub", "mezzanine".
        collateral_type: "assets", "real_estate", "ip", "none".
        industry: industry name for adjustment.
    """

    def __init__(
        self,
        seniority: str = "1L",
        collateral_type: str = "assets",
        industry: str = "manufacturing",
    ):
        if seniority not in RECOVERY_BY_SENIORITY:
            raise ValueError(
                f"Unknown seniority '{seniority}'. "
                f"Available: {list(RECOVERY_BY_SENIORITY.keys())}"
            )
        self.seniority = seniority
        self.collateral_type = collateral_type
        self.industry = industry

    def expected_recovery(self) -> float:
        """Expected recovery rate adjusted for industry."""
        base = RECOVERY_BY_SENIORITY[self.seniority]["mean"]
        adj = INDUSTRY_ADJUSTMENT.get(self.industry, 0.0)
        return max(0.0, min(1.0, base + adj))

    def recovery_std(self) -> float:
        """Standard deviation of recovery (historical)."""
        return RECOVERY_BY_SENIORITY[self.seniority]["std"]

    def recovery_distribution(self) -> tuple[float, float]:
        """Beta distribution parameters (alpha, beta) for MC sampling.

        Calibrated to match mean and std from historical data.
        Beta distribution is natural for recovery ∈ [0, 1].
        """
        mu = self.expected_recovery()
        sigma = self.recovery_std()
        # Method of moments for beta distribution
        if sigma <= 0 or mu <= 0 or mu >= 1:
            return (2.0, 2.0)  # symmetric fallback
        var = sigma ** 2
        common = mu * (1 - mu) / var - 1
        alpha = mu * common
        beta_param = (1 - mu) * common
        return (max(alpha, 0.1), max(beta_param, 0.1))

    def sample(self, n: int = 1, seed: int = 42) -> np.ndarray:
        """Sample recovery rates from the calibrated beta distribution."""
        a, b = self.recovery_distribution()
        rng = np.random.default_rng(seed)
        return rng.beta(a, b, size=n)

    def to_dict(self) -> dict:
        return {"seniority": self.seniority, "collateral_type": self.collateral_type,
                "industry": self.industry}

    @classmethod
    def from_dict(cls, d: dict) -> RecoveryModel:
        return cls(d.get("seniority", "1L"), d.get("collateral_type", "assets"),
                   d.get("industry", "manufacturing"))


# ---------------------------------------------------------------------------
# P2.2: Stochastic recovery (LGD cyclicality)
# ---------------------------------------------------------------------------

class StochasticRecovery:
    """Recovery rate correlated with default intensity.

    In recessions: default rates rise AND recovery drops.
    Correlation ρ ≈ -0.3 to -0.5 historically.

    For MC: sample (hazard, recovery) jointly:
        Z_default = standard normal (drives default timing)
        Z_recovery = ρ × Z_default + √(1-ρ²) × Z_independent
        recovery = beta_ppf(Φ(Z_recovery), alpha, beta)

    Args:
        recovery_model: base RecoveryModel for mean/std.
        default_correlation: ρ between default indicator and recovery.
    """

    def __init__(
        self,
        recovery_model: RecoveryModel,
        default_correlation: float = -0.3,
    ):
        if not -1.0 <= default_correlation <= 1.0:
            raise ValueError(f"correlation must be in [-1,1], got {default_correlation}")
        self.recovery_model = recovery_model
        self.default_correlation = default_correlation

    def sample_correlated(
        self,
        default_normals: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Sample recovery rates correlated with default drivers.

        Args:
            default_normals: standard normal variates driving default
                (one per simulation path).

        Returns:
            Recovery rates in [0, 1], correlated with default_normals.
        """
        n = len(default_normals)
        rng = np.random.default_rng(seed)
        z_indep = rng.standard_normal(n)

        rho = self.default_correlation
        z_recovery = rho * default_normals + math.sqrt(1 - rho**2) * z_indep

        # Map to uniform via Φ, then to beta
        u = norm.cdf(z_recovery)
        a, b = self.recovery_model.recovery_distribution()
        return beta_dist.ppf(np.clip(u, 1e-10, 1 - 1e-10), a, b)

    @staticmethod
    def lgd_downturn(mean_recovery: float, stress_factor: float = 0.75) -> float:
        """Basel downturn LGD: stressed recovery for regulatory capital.

        downturn_recovery = mean_recovery × stress_factor
        downturn_lgd = 1 - downturn_recovery

        Typical stress_factor: 0.75 (25% haircut on recovery).
        """
        return 1.0 - mean_recovery * stress_factor


# ---------------------------------------------------------------------------
# P2.3: Spread-dependent prepayment
# ---------------------------------------------------------------------------

class SpreadPrepayModel:
    """Prepayment model driven by credit spread refinancing incentive.

    Unlike mortgage CPR/PSA (rate-driven), leveraged loan prepayment is
    spread-driven. When market spreads tighten below the loan's spread,
    the borrower refinances.

    S-curve: CPR = base + (max_cpr - base) × Φ((loan_spread - market - premium) / vol)

    Args:
        loan_spread: the loan's contractual spread.
        call_premium: prepayment penalty in first year (e.g. 0.01 = 1%).
        base_cpr: base turnover rate when no incentive (annual).
        max_cpr: maximum CPR when deep in-the-money.
        vol: spread volatility for S-curve width.
    """

    def __init__(
        self,
        loan_spread: float,
        call_premium: float = 0.01,
        base_cpr: float = 0.10,
        max_cpr: float = 0.70,
        vol: float = 0.02,
    ):
        self.loan_spread = loan_spread
        self.call_premium = call_premium
        self.base_cpr = base_cpr
        self.max_cpr = max_cpr
        self.vol = vol

    def conditional_cpr(self, market_spread: float) -> float:
        """Conditional CPR given current market spread.

        S-curve: high CPR when market < loan (refinancing incentive),
        low CPR when market > loan (no incentive).
        """
        if self.vol <= 0:
            # Step function
            if market_spread < self.loan_spread - self.call_premium:
                return self.max_cpr
            return self.base_cpr

        incentive = (self.loan_spread - market_spread - self.call_premium) / self.vol
        phi = float(norm.cdf(incentive))
        return self.base_cpr + (self.max_cpr - self.base_cpr) * phi

    def s_curve(self, market_spreads: list[float]) -> list[float]:
        """Compute CPR across a range of market spreads."""
        return [self.conditional_cpr(s) for s in market_spreads]

    def to_dict(self) -> dict:
        return {"loan_spread": self.loan_spread, "call_premium": self.call_premium,
                "base_cpr": self.base_cpr, "max_cpr": self.max_cpr, "vol": self.vol}

    @classmethod
    def from_dict(cls, d: dict) -> SpreadPrepayModel:
        return cls(**d)


# ---------------------------------------------------------------------------
# P2.4: Regulatory LGD
# ---------------------------------------------------------------------------

# Basel II Foundation IRB LGD values
REGULATORY_LGD = {
    "senior_secured": 0.45,
    "senior_unsecured": 0.45,
    "subordinated": 0.75,
    "1L": 0.35,   # with eligible collateral
    "2L": 0.65,
}


def lgd_regulatory(
    seniority: str = "senior_unsecured",
    downturn: bool = True,
    stress_factor: float = 0.80,
) -> float:
    """Regulatory LGD for Basel capital calculation.

    Foundation IRB: uses supervisory LGD values.
    Downturn: applies stress factor (LGD increases in downturn).
    """
    base = REGULATORY_LGD.get(seniority, 0.45)
    if downturn:
        # Downturn LGD = min(base / stress_factor, 1.0)
        # stress_factor < 1 means recovery drops → LGD rises
        return min(base / stress_factor, 1.0)
    return base


def capital_requirement(
    pd: float,
    lgd: float,
    maturity: float = 2.5,
    correlation: float | None = None,
) -> float:
    """Basel II IRB capital requirement (K).

    K = LGD × [Φ(√ρ × Φ⁻¹(0.999) + √(1-ρ) × Φ⁻¹(PD)) / (1 + (M-2.5) × b)]
        - PD × LGD

    where ρ is the asset correlation (derived from PD if not given).
    """
    if pd <= 0:
        return 0.0
    if pd >= 1:
        return lgd

    # Asset correlation (Basel formula)
    if correlation is None:
        rho = 0.12 * (1 - math.exp(-50 * pd)) / (1 - math.exp(-50)) + \
              0.24 * (1 - (1 - math.exp(-50 * pd)) / (1 - math.exp(-50)))
    else:
        rho = correlation

    # Maturity adjustment
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    mat_adj = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

    # Conditional PD at 99.9% confidence
    sqrt_rho = math.sqrt(max(rho, 1e-10))
    sqrt_1_rho = math.sqrt(max(1 - rho, 1e-10))
    cond_pd = norm.cdf(
        (norm.ppf(pd) + sqrt_rho * norm.ppf(0.999)) / sqrt_1_rho
    )

    # Capital
    k = lgd * cond_pd * mat_adj - pd * lgd
    return max(k, 0.0)
