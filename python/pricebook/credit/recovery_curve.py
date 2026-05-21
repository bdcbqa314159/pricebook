"""Term structure of recovery rates.

Recovery rates are not constant — they vary by seniority, maturity,
economic conditions, and instrument type. This module provides recovery
curves that can be used in credit pricing instead of flat scalars.

    from pricebook.credit.recovery_curve import (
        RecoveryCurve, recovery_by_seniority, RecoverySeniority,
    )

    # Maturity-dependent recovery
    curve = RecoveryCurve.linear(short_rate=0.50, long_rate=0.35, pivot_years=5)
    r_3y = curve.recovery(3.0)   # higher for short maturities
    r_10y = curve.recovery(10.0) # lower for long maturities

References:
    Altman & Kishore (1996). Almost Everything You Wanted to Know
    About Recoveries on Defaulted Bonds.
    Moody's (2023). Annual Default Study.
    Duffie & Singleton (1999). Sec 4.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RecoverySeniority(Enum):
    """Bond seniority for recovery rate estimation."""
    SENIOR_SECURED = "senior_secured"
    SENIOR_UNSECURED = "senior_unsecured"
    SENIOR_SUBORDINATED = "senior_subordinated"
    SUBORDINATED = "subordinated"
    JUNIOR_SUBORDINATED = "junior_subordinated"


# Historical average recovery rates by seniority (Moody's 2023 study)
_SENIORITY_RECOVERY: dict[RecoverySeniority, float] = {
    RecoverySeniority.SENIOR_SECURED: 0.53,
    RecoverySeniority.SENIOR_UNSECURED: 0.40,
    RecoverySeniority.SENIOR_SUBORDINATED: 0.32,
    RecoverySeniority.SUBORDINATED: 0.28,
    RecoverySeniority.JUNIOR_SUBORDINATED: 0.18,
}

# Historical recovery volatility by seniority
_SENIORITY_VOL: dict[RecoverySeniority, float] = {
    RecoverySeniority.SENIOR_SECURED: 0.22,
    RecoverySeniority.SENIOR_UNSECURED: 0.25,
    RecoverySeniority.SENIOR_SUBORDINATED: 0.27,
    RecoverySeniority.SUBORDINATED: 0.28,
    RecoverySeniority.JUNIOR_SUBORDINATED: 0.30,
}


def recovery_by_seniority(seniority: RecoverySeniority) -> float:
    """Historical average recovery rate for a given seniority."""
    return _SENIORITY_RECOVERY[seniority]


def recovery_vol_by_seniority(seniority: RecoverySeniority) -> float:
    """Historical recovery rate volatility for a given seniority."""
    return _SENIORITY_VOL[seniority]


class RecoveryCurve:
    """Term structure of recovery rates.

    Recovery may decline with maturity (longer bonds have lower recovery
    due to accrued coupon loss and price erosion pre-default).

    Supports: flat, linear interpolation, and piecewise.
    """

    def __init__(
        self,
        tenors: list[float],
        recovery_rates: list[float],
    ):
        """Create from explicit pillar points.

        Args:
            tenors: year fractions [t₁, t₂, ...] (ascending).
            recovery_rates: [R₁, R₂, ...] at each tenor.
        """
        if len(tenors) != len(recovery_rates):
            raise ValueError("tenors and recovery_rates must have same length")
        if len(tenors) == 0:
            raise ValueError("At least one pillar required")
        self._tenors = np.array(tenors)
        self._rates = np.array(recovery_rates)

    def recovery(self, t: float) -> float:
        """Interpolated recovery rate at time t (years).

        Flat extrapolation outside range.
        """
        if t <= self._tenors[0]:
            return float(self._rates[0])
        if t >= self._tenors[-1]:
            return float(self._rates[-1])
        return float(np.interp(t, self._tenors, self._rates))

    @classmethod
    def flat(cls, rate: float) -> RecoveryCurve:
        """Constant recovery at all maturities."""
        return cls([1.0], [rate])

    @classmethod
    def linear(
        cls, short_rate: float, long_rate: float, pivot_years: float = 5.0,
    ) -> RecoveryCurve:
        """Linear interpolation from short to long rate.

        Args:
            short_rate: recovery at t=0.
            long_rate: recovery at t=pivot_years (and beyond).
            pivot_years: maturity where long_rate kicks in.
        """
        return cls([0.0, pivot_years], [short_rate, long_rate])

    @classmethod
    def from_seniority(
        cls,
        seniority: RecoverySeniority,
        maturity_adjustment: float = -0.02,
    ) -> RecoveryCurve:
        """Create from seniority with a maturity adjustment.

        Recovery declines by `maturity_adjustment` per year beyond 5Y.
        """
        base = recovery_by_seniority(seniority)
        r_short = base
        r_long = max(base + maturity_adjustment * 5, 0.05)
        return cls.linear(r_short, r_long, pivot_years=5.0)

    def average(self, t_start: float, t_end: float, n_points: int = 20) -> float:
        """Average recovery rate over a period [t_start, t_end]."""
        if t_end <= t_start:
            return self.recovery(t_start)
        ts = np.linspace(t_start, t_end, n_points)
        return float(np.mean([self.recovery(t) for t in ts]))

    def to_dict(self) -> dict:
        return {
            "tenors": self._tenors.tolist(),
            "recovery_rates": self._rates.tolist(),
        }


@dataclass
class StochasticRecovery:
    """Recovery rate with uncertainty (for simulation / stress testing).

    Models recovery as beta-distributed with given mean and vol.
    """
    mean: float
    vol: float
    seniority: RecoverySeniority | None = None

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw n recovery samples from beta distribution.

        Beta parameterised by mean μ and vol σ:
            α = μ × ((μ(1-μ)/σ² - 1)
            β = (1-μ) × ((μ(1-μ)/σ² - 1)
        """
        if rng is None:
            rng = np.random.default_rng()

        mu = min(max(self.mean, 0.01), 0.99)
        var = min(self.vol ** 2, mu * (1 - mu) * 0.99)

        if var <= 0:
            return np.full(n, mu)

        factor = mu * (1 - mu) / var - 1.0
        if factor <= 0:
            return np.full(n, mu)

        a = mu * factor
        b = (1 - mu) * factor
        return rng.beta(a, b, n)

    def percentile(self, p: float) -> float:
        """p-th percentile of the recovery distribution."""
        from scipy.stats import beta as beta_dist
        mu = min(max(self.mean, 0.01), 0.99)
        var = min(self.vol ** 2, mu * (1 - mu) * 0.99)
        if var <= 0:
            return mu
        factor = mu * (1 - mu) / var - 1.0
        if factor <= 0:
            return mu
        a = mu * factor
        b = (1 - mu) * factor
        return float(beta_dist.ppf(p, a, b))

    @classmethod
    def from_seniority(cls, seniority: RecoverySeniority) -> StochasticRecovery:
        """Create from historical seniority statistics."""
        return cls(
            mean=recovery_by_seniority(seniority),
            vol=recovery_vol_by_seniority(seniority),
            seniority=seniority,
        )

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "vol": self.vol,
            "seniority": self.seniority.value if self.seniority else None,
        }
