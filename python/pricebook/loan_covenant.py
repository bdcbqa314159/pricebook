"""Loan covenant analytics: specification, cushion, breach probability,
waiver/amendment pricing, covenant-adjusted PV.

    from pricebook.loan_covenant import (
        Covenant, CovenantSchedule, covenant_cushion,
        breach_probability, waiver_cost, covenant_adjusted_pv,
    )

    cov = Covenant(type="maintenance", metric="leverage", threshold=5.0)
    cushion = covenant_cushion(actual_leverage=4.2, threshold=5.0)
    prob = breach_probability(cushion=0.16, ebitda_vol=0.20, horizon=1.0)

References:
    S&P LCD (2023). Leveraged Loan Primer — Covenants.
    Becker & Ivashina (2016). Covenant-Light Contracts and Creditor
    Coordination. Working Paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
from scipy.stats import norm

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


# ---------------------------------------------------------------------------
# P3.1: Covenant specification
# ---------------------------------------------------------------------------

COVENANT_TYPES = {"maintenance", "incurrence"}
COVENANT_METRICS = {"leverage", "coverage", "fixed_charge", "capex", "debt_to_equity"}


@dataclass
class Covenant:
    """Single financial covenant.

    Args:
        type: "maintenance" (tested quarterly) or "incurrence" (on action).
        metric: what ratio is tested.
        threshold: covenant level (e.g. 5.0x for leverage ≤ 5.0x).
        direction: "max" (must stay below, e.g. leverage) or
                   "min" (must stay above, e.g. coverage).
    """
    type: str = "maintenance"
    metric: str = "leverage"
    threshold: float = 5.0
    direction: str = "max"  # "max" = must be ≤ threshold, "min" = must be ≥ threshold

    def __post_init__(self):
        if self.type not in COVENANT_TYPES:
            raise ValueError(f"type must be one of {COVENANT_TYPES}")
        if self.metric not in COVENANT_METRICS:
            raise ValueError(f"metric must be one of {COVENANT_METRICS}")
        if self.direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min'")

    def is_breached(self, actual: float) -> bool:
        """Check if the covenant is breached at the actual ratio."""
        if self.direction == "max":
            return actual > self.threshold
        return actual < self.threshold

    def to_dict(self) -> dict:
        return {"type": self.type, "metric": self.metric,
                "threshold": self.threshold, "direction": self.direction}

    @classmethod
    def from_dict(cls, d: dict) -> Covenant:
        return cls(**d)


@dataclass
class CovenantSchedule:
    """Collection of covenants with test dates.

    Args:
        covenants: list of Covenant objects.
        test_dates: quarterly (or other) test dates.
    """
    covenants: list[Covenant] = field(default_factory=list)
    test_dates: list[date] = field(default_factory=list)

    @property
    def is_cov_lite(self) -> bool:
        """True if no maintenance covenants (only incurrence)."""
        return all(c.type == "incurrence" for c in self.covenants)

    @property
    def n_maintenance(self) -> int:
        return sum(1 for c in self.covenants if c.type == "maintenance")

    def to_dict(self) -> dict:
        return {
            "covenants": [c.to_dict() for c in self.covenants],
            "test_dates": [d.isoformat() for d in self.test_dates],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CovenantSchedule:
        return cls(
            covenants=[Covenant.from_dict(c) for c in d.get("covenants", [])],
            test_dates=[date.fromisoformat(s) for s in d.get("test_dates", [])],
        )


# ---------------------------------------------------------------------------
# P3.2: Covenant cushion
# ---------------------------------------------------------------------------

def covenant_cushion(actual: float, threshold: float, direction: str = "max") -> float:
    """Covenant cushion: how far the ratio is from the threshold.

    For leverage (direction="max"):
        cushion = (threshold - actual) / threshold

    For coverage (direction="min"):
        cushion = (actual - threshold) / threshold

    Returns:
        Cushion as fraction (0.16 = 16% headroom). Negative = breached.
    """
    if direction == "max":
        return (threshold - actual) / threshold if threshold != 0 else 0.0
    return (actual - threshold) / threshold if threshold != 0 else 0.0


def cushion_trajectory(
    historical_ratios: list[float],
    threshold: float,
    direction: str = "max",
) -> str:
    """Assess trend of covenant cushion over time.

    Returns "improving", "stable", or "deteriorating".
    """
    if len(historical_ratios) < 2:
        return "stable"

    cushions = [covenant_cushion(r, threshold, direction) for r in historical_ratios]
    # Linear trend
    n = len(cushions)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(cushions) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, cushions))
    den = sum((xi - x_mean) ** 2 for xi in x)
    slope = num / den if den > 0 else 0.0

    if slope > 0.01:
        return "improving"
    elif slope < -0.01:
        return "deteriorating"
    return "stable"


def periods_to_breach(
    current_ratio: float,
    threshold: float,
    trend_per_period: float,
    direction: str = "max",
) -> int | None:
    """Estimate periods until breach at current trend.

    Returns None if ratio is moving away from threshold.
    """
    if direction == "max":
        if trend_per_period <= 0:
            return None  # improving, won't breach
        gap = threshold - current_ratio
        if gap <= 0:
            return 0  # already breached
    else:
        if trend_per_period >= 0:
            return None
        gap = current_ratio - threshold
        if gap <= 0:
            return 0

    return max(1, int(math.ceil(gap / abs(trend_per_period))))


# ---------------------------------------------------------------------------
# P3.3: Breach probability
# ---------------------------------------------------------------------------

def breach_probability(
    cushion: float,
    ebitda_vol: float,
    horizon: float = 1.0,
) -> float:
    """Probability of covenant breach over horizon.

    Model: EBITDA ~ GBM with volatility ebitda_vol.
    For leverage covenant (Debt/EBITDA): breach when EBITDA drops enough
    that leverage exceeds threshold.

    P(breach) = Φ(-ln(1 + cushion) / (ebitda_vol × √T))

    Args:
        cushion: current covenant cushion (fraction, e.g. 0.16).
        ebitda_vol: annual EBITDA volatility (e.g. 0.20).
        horizon: time horizon in years.
    """
    if cushion <= 0:
        return 1.0  # already breached
    if ebitda_vol <= 0:
        return 0.0  # no uncertainty

    d = math.log(1 + cushion) / (ebitda_vol * math.sqrt(horizon))
    return float(norm.cdf(-d))


def breach_probability_mc(
    current_ebitda: float,
    debt: float,
    threshold: float,
    ebitda_vol: float,
    ebitda_drift: float = 0.0,
    horizon: float = 1.0,
    n_steps: int = 4,
    n_paths: int = 50_000,
    seed: int = 42,
) -> float:
    """MC breach probability: simulate EBITDA paths, count breaches.

    Leverage = debt / EBITDA. Breach if leverage > threshold at any test date.
    """
    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    sqrt_dt = math.sqrt(dt)

    ebitda = np.full(n_paths, current_ebitda)
    breached = np.zeros(n_paths, dtype=bool)

    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        ebitda = ebitda * np.exp((ebitda_drift - 0.5 * ebitda_vol**2) * dt
                                 + ebitda_vol * sqrt_dt * z)
        leverage = debt / np.maximum(ebitda, 1e-10)
        breached |= leverage > threshold

    return float(breached.mean())


# ---------------------------------------------------------------------------
# P3.4: Waiver / amendment / cure costs
# ---------------------------------------------------------------------------

def waiver_cost(
    notional: float,
    waiver_fee_bps: float = 25.0,
    spread_stepup_bps: float = 0.0,
    remaining_life: float = 3.0,
) -> float:
    """Cost of a covenant waiver.

    Total cost = upfront fee + PV of spread step-up.
    """
    if notional <= 0:
        raise ValueError(f"notional must be positive, got {notional}")
    if remaining_life <= 0:
        raise ValueError(f"remaining_life must be positive, got {remaining_life}")
    upfront = notional * waiver_fee_bps / 10_000
    stepup_annual = notional * spread_stepup_bps / 10_000
    stepup_pv = stepup_annual * remaining_life  # undiscounted for simplicity
    return upfront + stepup_pv


def amendment_cost(
    notional: float,
    amendment_fee_bps: float = 50.0,
    spread_change_bps: float = 25.0,
    remaining_life: float = 3.0,
) -> float:
    """Cost of a covenant amendment (reset threshold).

    Higher than waiver: includes legal fees + spread change.
    """
    upfront = notional * amendment_fee_bps / 10_000
    spread_pv = notional * spread_change_bps / 10_000 * remaining_life
    return upfront + spread_pv


def equity_cure_cost(
    cure_amount: float,
    sponsor_required_return: float = 0.15,
    cure_periods: int = 4,
) -> float:
    """Cost of equity cure: sponsor injects cash to fix ratios.

    The sponsor's required return on injected equity is the true cost.
    """
    return cure_amount * sponsor_required_return * cure_periods / 4


# ---------------------------------------------------------------------------
# P3.5: Covenant-adjusted PV
# ---------------------------------------------------------------------------

@dataclass
class CovenantAdjustedResult:
    """Result of covenant-adjusted valuation."""
    base_pv: float
    expected_waiver_cost: float
    expected_amendment_cost: float
    expected_acceleration_loss: float
    adjusted_pv: float
    breach_prob: float

    def to_dict(self) -> dict:
        return {
            "base_pv": self.base_pv,
            "expected_waiver_cost": self.expected_waiver_cost,
            "expected_amendment_cost": self.expected_amendment_cost,
            "expected_acceleration_loss": self.expected_acceleration_loss,
            "adjusted_pv": self.adjusted_pv, "breach_prob": self.breach_prob,
        }


def covenant_adjusted_pv(
    base_pv: float,
    notional: float,
    cushion: float,
    ebitda_vol: float,
    remaining_life: float = 3.0,
    recovery: float = 0.6,
    waiver_prob: float = 0.70,
    amendment_prob: float = 0.20,
    acceleration_prob: float = 0.10,
    waiver_fee_bps: float = 25.0,
    amendment_fee_bps: float = 50.0,
    amendment_spread_bps: float = 25.0,
) -> CovenantAdjustedResult:
    """PV adjusted for expected covenant breach costs.

    PV_adjusted = base_PV - P(breach) × E[cost | breach]

    Where cost is weighted average of waiver, amendment, and acceleration:
    E[cost] = p_waiver × waiver_cost + p_amend × amend_cost + p_accel × accel_loss

    Args:
        waiver_prob: probability that breach leads to waiver (most common).
        amendment_prob: probability of amendment.
        acceleration_prob: probability of acceleration (rare).
    """
    bp = breach_probability(cushion, ebitda_vol, remaining_life)

    wc = waiver_cost(notional, waiver_fee_bps, 0.0, remaining_life)
    ac = amendment_cost(notional, amendment_fee_bps, amendment_spread_bps, remaining_life)
    accel_loss = notional * (1 - recovery)  # worst case: acceleration + default

    expected_waiver = bp * waiver_prob * wc
    expected_amend = bp * amendment_prob * ac
    expected_accel = bp * acceleration_prob * accel_loss

    adjusted = base_pv - expected_waiver - expected_amend - expected_accel

    return CovenantAdjustedResult(
        base_pv=base_pv,
        expected_waiver_cost=expected_waiver,
        expected_amendment_cost=expected_amend,
        expected_acceleration_loss=expected_accel,
        adjusted_pv=adjusted, breach_prob=bp,
    )
