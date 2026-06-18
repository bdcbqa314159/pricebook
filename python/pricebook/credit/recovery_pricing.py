"""Recovery pricing: stochastic recovery, default-recovery correlation, wrong-way risk.

The core insight: E[(1-R) × 1_default] ≠ (1-E[R]) × P[default] when
recovery R and default are correlated. This module provides the tools
to price with stochastic recovery and quantify the wrong-way premium.

    from pricebook.credit.recovery_pricing import (
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

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.survival_curve import SurvivalCurve


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



    def to_dict(self) -> dict:
        return dict(vars(self))
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


# ---------------------------------------------------------------------------
# Unified MC Engine migration
# ---------------------------------------------------------------------------

def correlated_default_recovery_via_engine(
    pd: float,
    recovery_spec: RecoverySpec,
    n_sims: int = 100_000,
    seed: int = 42,
) -> DefaultRecoveryResult:
    """Correlated default-recovery via unified MC engine (copula sampling).

    Delegates to original — copula sampling of (default, recovery) is
    inherently single-step and doesn't benefit from SDE path generation.
    """
    return correlated_default_recovery(pd, recovery_spec, n_sims, seed)


# ═══════════════════════════════════════════════════════════════
# Per-name heterogeneous RecoverySpec utilities
# ═══════════════════════════════════════════════════════════════


def build_recovery_specs(
    seniorities: list[str],
    correlation: float = -0.3,
) -> list[RecoverySpec]:
    """Build a list of RecoverySpec from seniority levels.

    Args:
        seniorities: per-name seniority (e.g. ["1L", "senior", "sub"]).
        correlation: default-recovery correlation (same for all names).

    Returns:
        List of RecoverySpec, one per name.
    """
    return [RecoverySpec.from_seniority(s, correlation) for s in seniorities]


def validate_recovery_specs(
    specs: list[RecoverySpec],
    n_names: int,
) -> None:
    """Validate that recovery specs match portfolio size.

    Raises ValueError if lengths don't match.
    """
    if len(specs) != n_names:
        raise ValueError(f"Expected {n_names} RecoverySpec, got {len(specs)}")


def recovery_spec_summary(specs: list[RecoverySpec]) -> dict:
    """Portfolio-level summary of recovery specifications.

    Returns dict with weighted average recovery, LGD, and per-seniority counts.
    """
    if not specs:
        return {"n_names": 0, "avg_recovery": 0.0, "avg_lgd": 0.0}

    avg_r = sum(s.mean for s in specs) / len(specs)
    avg_lgd = 1 - avg_r
    min_r = min(s.mean for s in specs)
    max_r = max(s.mean for s in specs)
    avg_corr = sum(s.correlation_to_default for s in specs) / len(specs)

    return {
        "n_names": len(specs),
        "avg_recovery": avg_r,
        "avg_lgd": avg_lgd,
        "min_recovery": min_r,
        "max_recovery": max_r,
        "avg_correlation": avg_corr,
    }


# ═══════════════════════════════════════════════════════════════
# Seniority waterfall
# ═══════════════════════════════════════════════════════════════


@dataclass
class SeniorityWaterfall:
    """Capital structure waterfall for recovery distribution.

    Distributes total recovery amount across tranches in priority order:
    senior_secured → senior → mezzanine → subordinated → equity.

    Args:
        tranches: list of (seniority_name, notional) in priority order.
    """
    tranches: list[tuple[str, float]]

    def distribute(self, total_recovery: float) -> dict[str, float]:
        """Distribute total recovery amount across tranches.

        Higher-priority tranches absorb recovery first (waterfall).

        Args:
            total_recovery: total recovery amount (in currency).

        Returns:
            {seniority: recovery_amount} per tranche.
        """
        remaining = total_recovery
        result = {}
        for name, notional in self.tranches:
            absorbed = min(remaining, notional)
            result[name] = absorbed
            remaining -= absorbed
        return result

    def recovery_rates(self, total_recovery_pct: float) -> dict[str, float]:
        """Per-tranche recovery rates given total asset recovery %.

        Args:
            total_recovery_pct: total recovery as fraction of total notional (0 to 1).

        Returns:
            {seniority: recovery_rate} per tranche (0 to 1).
        """
        total_notional = sum(n for _, n in self.tranches)
        total_amount = total_recovery_pct * total_notional
        dist = self.distribute(total_amount)
        return {name: dist[name] / notional if notional > 0 else 0.0
                for name, notional in self.tranches}

    def to_recovery_specs(
        self,
        total_recovery_mean: float = 0.45,
        total_recovery_std: float = 0.20,
        correlation: float = -0.3,
    ) -> list[RecoverySpec]:
        """Generate per-tranche RecoverySpec consistent with waterfall.

        Uses the mean recovery rate per tranche from the waterfall
        with the given total recovery distribution.
        """
        rates = self.recovery_rates(total_recovery_mean)
        return [RecoverySpec(mean=max(0.01, min(0.99, rates[name])),
                             std=total_recovery_std * 0.5,  # tranche vol < total
                             correlation_to_default=correlation)
                for name, _ in self.tranches]

    def to_dict(self) -> dict:
        return {"tranches": [(n, v) for n, v in self.tranches]}


# ═══════════════════════════════════════════════════════════════
# Recovery bid-ask surface (CDS-implied)
# ═══════════════════════════════════════════════════════════════


@dataclass
class RecoveryBidAsk:
    """Recovery bid-ask for a single tenor."""
    tenor: float
    bid: float      # lower recovery estimate
    ask: float      # upper recovery estimate
    mid: float      # mid-market recovery

    def to_dict(self) -> dict:
        return dict(vars(self))


def implied_recovery(
    cds_spread: float,
    hazard_rate: float,
) -> float:
    """Back out implied recovery from CDS spread and hazard rate.

    From: spread ≈ (1-R) × h
    Solve: R = 1 - spread/h

    Args:
        cds_spread: CDS par spread (decimal, e.g. 0.01 = 100bp).
        hazard_rate: default intensity (continuous).

    Returns:
        Implied recovery rate.
    """
    if hazard_rate <= 1e-10:
        return 0.4  # default assumption
    r = 1 - cds_spread / hazard_rate
    return max(0.0, min(1.0, r))


def recovery_bid_ask_surface(
    spreads_by_tenor: dict[float, float],
    hazards_by_tenor: dict[float, float],
    spread_width_bp: float = 5.0,
) -> list[RecoveryBidAsk]:
    """Build term structure of implied recovery with bid-ask.

    Args:
        spreads_by_tenor: {tenor_years: cds_spread}.
        hazards_by_tenor: {tenor_years: hazard_rate}.
        spread_width_bp: bid-ask spread width in bp (default 5bp).

    Returns:
        List of RecoveryBidAsk, one per tenor.
    """
    half_width = spread_width_bp / 10_000 / 2

    results = []
    for tenor in sorted(spreads_by_tenor.keys()):
        s = spreads_by_tenor[tenor]
        h = hazards_by_tenor.get(tenor, 0.02)

        mid = implied_recovery(s, h)
        # Wider spread → lower recovery (bid), tighter → higher (ask)
        bid = implied_recovery(s + half_width, h)
        ask = implied_recovery(s - half_width, h)

        results.append(RecoveryBidAsk(tenor, bid, ask, mid))

    return results
