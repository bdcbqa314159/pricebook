"""CreditGrades model (JP Morgan / RiskMetrics, 2002).

Extended Merton model with a stochastic default barrier. Unlike the
classical Merton model where default only occurs at maturity, CreditGrades
allows first-passage default at any time.

    from pricebook.credit.credit_grades import (
        CreditGrades, credit_grades_survival, credit_grades_spread,
    )

    model = CreditGrades(
        asset_vol=0.30, leverage=0.50, recovery_mean=0.50,
        recovery_vol=0.25, risk_free_rate=0.05,
    )
    q = model.survival(t=5.0)
    spread = model.cds_spread(t=5.0)

The default barrier L is lognormal:
    L = D × exp(λ × Z - λ²/2)

where D = debt per share, λ = recovery uncertainty, Z ~ N(0,1).

Survival probability:
    Q(t) = Φ(-d₁(t) + λ) × exp(A) + Φ(-d₂(t) - λ) × exp(B)

where d₁, d₂ are functions of leverage, vol, and time.

References:
    Finger et al. (2002). CreditGrades Technical Document. RiskMetrics.
    Stamicar & Finger (2005). Incorporating Equity Options into CreditGrades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class CreditGradesResult:
    """Result from CreditGrades model evaluation."""
    survival_prob: float
    default_prob: float
    cds_spread: float           # par CDS spread (approximate)
    distance_to_default: float
    implied_hazard: float
    asset_vol: float
    leverage: float

    def to_dict(self) -> dict:
        return dict(vars(self))


class CreditGrades:
    """CreditGrades model: first-passage Merton with stochastic barrier.

    Args:
        asset_vol: asset volatility (annualised, e.g. 0.30 = 30%).
        leverage: debt / (debt + equity) = D / V.
        recovery_mean: expected recovery rate (mean of lognormal barrier).
        recovery_vol: volatility of recovery (λ in the model).
        risk_free_rate: risk-free rate (continuous compounding).
    """

    def __init__(
        self,
        asset_vol: float,
        leverage: float,
        recovery_mean: float = 0.50,
        recovery_vol: float = 0.25,
        risk_free_rate: float = 0.05,
    ):
        if not 0 < leverage < 1:
            raise ValueError(f"leverage must be in (0, 1), got {leverage}")
        self.asset_vol = asset_vol
        self.leverage = leverage
        self.recovery_mean = recovery_mean
        self.recovery_vol = recovery_vol  # λ
        self.risk_free_rate = risk_free_rate

        # Pre-compute model parameters
        self._sigma = asset_vol
        self._lambda = recovery_vol
        self._d = leverage  # D/V ratio (barrier-to-value)
        self._ld = self._lambda ** 2  # λ²

    def survival(self, t: float) -> float:
        """Survival probability to time t.

        From CreditGrades Technical Document (Finger et al. 2002, Eq 11):

            Q(t) = Φ(−a + σ̄√t/2) − d̄ × Φ(−a − σ̄√t/2)

        where:
            d̄ = (D/S) × exp(λ²/2)        (adjusted debt-equity ratio)
            σ̄² = σ² + λ²                   (total variance including barrier noise)
            a = log(d̄) / (σ̄√t) + σ̄√t/2   (NOT the formula — see below)

        More precisely, the first-passage survival for a GBM hitting
        a barrier at d̄ with vol σ̄ is:

            Q(t) = Φ(z₊) − d̄^(2μ/σ̄²) × Φ(z₋)

        where μ = σ̄²/2 (risk-neutral drift), giving exponent = 1, so:

            Q(t) = Φ(z₊) − d̄ × Φ(z₋)

            z₊ = (−ln(d̄) + σ̄²t/2) / (σ̄√t)
            z₋ = (−ln(d̄) − σ̄²t/2) / (σ̄√t)

        But this is the barrier-crossing probability for V/L hitting 1 from
        above when V₀/L₀ = S/D_bar > 1 (i.e. firm value > barrier).
        For a firm with d̄ < 1 (healthy), Q > 0.5 for short t.
        For d̄ > 1 (distressed), default is near-certain.
        """
        if t <= 0:
            return 1.0

        sigma = self._sigma
        lam = self._lambda
        ld = self._ld

        # σ̄² = σ² + λ²
        sigma_bar_sq = sigma ** 2 + ld
        sigma_bar = math.sqrt(sigma_bar_sq)

        # d̄ = (D/S) × exp(λ²/2)
        d_bar = self._d * math.exp(ld / 2.0)

        if d_bar <= 0:
            return 1.0

        sqrt_t = math.sqrt(t)
        log_d = math.log(d_bar)

        # α = (-ln(d̄) + σ̄²t/2) / (σ̄√t)  — "up" term
        # β = ( ln(d̄) + σ̄²t/2) / (σ̄√t)  — "down" term (note: +ln, not -ln)
        alpha = (-log_d + sigma_bar_sq * t / 2.0) / (sigma_bar * sqrt_t)
        beta = (log_d + sigma_bar_sq * t / 2.0) / (sigma_bar * sqrt_t)

        q = norm.cdf(alpha) - d_bar * norm.cdf(beta)

        # Ensure in [0, 1]
        return float(max(0.0, min(1.0, q)))

    def default_probability(self, t: float) -> float:
        """Cumulative default probability to time t."""
        return 1.0 - self.survival(t)

    def cds_spread(self, t: float, recovery: float | None = None) -> float:
        """Approximate par CDS spread for maturity t.

        spread ≈ -ln(Q(t)) / t × (1 - R)
        """
        r = recovery if recovery is not None else self.recovery_mean
        q = self.survival(t)
        if q <= 0 or t <= 0:
            return 0.0
        h = -math.log(q) / t
        return h * (1.0 - r)

    def distance_to_default(self) -> float:
        """Merton distance to default: DD = -log(D/V) / σ.

        Higher DD = further from default.
        """
        d_bar = self._d * math.exp(self._ld / 2.0)
        if d_bar <= 0:
            return float("inf")
        return -math.log(d_bar) / self._sigma if self._sigma > 0 else float("inf")

    def implied_hazard(self, t: float) -> float:
        """Implied flat hazard rate from CreditGrades survival."""
        q = self.survival(t)
        if q <= 0 or t <= 0:
            return 0.0
        return -math.log(q) / t

    def spread_term_structure(
        self, tenors: list[float], recovery: float | None = None,
    ) -> list[float]:
        """CDS spread at multiple tenors."""
        return [self.cds_spread(t, recovery) for t in tenors]

    def evaluate(self, t: float = 5.0, recovery: float | None = None) -> CreditGradesResult:
        """Full evaluation at a given horizon."""
        r = recovery if recovery is not None else self.recovery_mean
        q = self.survival(t)
        return CreditGradesResult(
            survival_prob=q,
            default_prob=1.0 - q,
            cds_spread=self.cds_spread(t, r),
            distance_to_default=self.distance_to_default(),
            implied_hazard=self.implied_hazard(t),
            asset_vol=self.asset_vol,
            leverage=self.leverage,
        )

    def to_dict(self) -> dict:
        return {
            "asset_vol": self.asset_vol,
            "leverage": self.leverage,
            "recovery_mean": self.recovery_mean,
            "recovery_vol": self.recovery_vol,
            "risk_free_rate": self.risk_free_rate,
            "distance_to_default": self.distance_to_default(),
        }


def credit_grades_survival(
    asset_vol: float,
    leverage: float,
    t: float,
    recovery_mean: float = 0.50,
    recovery_vol: float = 0.25,
) -> float:
    """Convenience: compute CreditGrades survival probability."""
    model = CreditGrades(asset_vol, leverage, recovery_mean, recovery_vol)
    return model.survival(t)


def credit_grades_spread(
    asset_vol: float,
    leverage: float,
    t: float = 5.0,
    recovery_mean: float = 0.50,
    recovery_vol: float = 0.25,
) -> float:
    """Convenience: compute CreditGrades CDS spread."""
    model = CreditGrades(asset_vol, leverage, recovery_mean, recovery_vol)
    return model.cds_spread(t)
