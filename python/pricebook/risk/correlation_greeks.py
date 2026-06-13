"""Correlation Greeks: ∂V/∂ρ, cross-gamma, correlation P&L attribution.

* :func:`correlation_delta` — ∂V/∂ρ via bump-and-reprice.
* :func:`correlation_gamma` — ∂²V/∂ρ².
* :func:`cross_gamma` — ∂²V/∂S₁∂S₂.
* :func:`correlation_pnl_attribution` — decompose P&L into ρ contributions.
* :func:`correlation_sensitivity_ladder` — ladder across full ρ matrix.

References:
    Bossu, *Advanced Equity Derivatives: Volatility and Correlation*, Wiley, 2014.
    Alexander, *Market Risk Analysis*, Vol. IV, Wiley, 2008.
    De Weert, *Exotic Options Trading*, Wiley, 2008.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Correlation delta ----

@dataclass
class CorrelationDeltaResult:
    """Correlation delta result."""
    rho_delta: float            # ∂V/∂ρ
    base_price: float
    bumped_up_price: float
    bumped_down_price: float
    bump_size: float



    def to_dict(self) -> dict:
        return vars(self)
def correlation_delta(
    price_fn,
    rho: float,
    bump: float = 0.01,
) -> CorrelationDeltaResult:
    """Correlation delta: ∂V/∂ρ via central difference.

    Args:
        price_fn: callable(rho) → option price.
        rho: current correlation.
        bump: ρ bump size (default 1%).
    """
    rho_up = min(rho + bump, 0.999)
    rho_dn = max(rho - bump, -0.999)
    actual_bump = rho_up - rho_dn

    p_up = price_fn(rho_up)
    p_dn = price_fn(rho_dn)
    p_base = price_fn(rho)

    delta = (p_up - p_dn) / actual_bump

    return CorrelationDeltaResult(
        rho_delta=float(delta),
        base_price=float(p_base),
        bumped_up_price=float(p_up),
        bumped_down_price=float(p_dn),
        bump_size=float(actual_bump / 2),
    )


# ---- Correlation gamma ----

@dataclass
class CorrelationGammaResult:
    """Correlation gamma result."""
    rho_gamma: float            # ∂²V/∂ρ²
    rho_delta: float
    convexity_pnl_per_unit: float   # 0.5 × Γ_ρ × (Δρ)²



    def to_dict(self) -> dict:
        return vars(self)
def correlation_gamma(
    price_fn,
    rho: float,
    bump: float = 0.01,
) -> CorrelationGammaResult:
    """Correlation gamma: ∂²V/∂ρ².

    Measures convexity of option value to correlation changes.
    Positive γ_ρ: option benefits from ρ moves in either direction.

    Args:
        price_fn: callable(rho) → price.
    """
    rho_up = min(rho + bump, 0.999)
    rho_dn = max(rho - bump, -0.999)
    actual_bump = (rho_up - rho_dn) / 2

    p_up = price_fn(rho_up)
    p_mid = price_fn(rho)
    p_dn = price_fn(rho_dn)

    gamma = (p_up - 2 * p_mid + p_dn) / (actual_bump ** 2)
    delta = (p_up - p_dn) / (2 * actual_bump)
    conv_pnl = 0.5 * gamma * actual_bump**2

    return CorrelationGammaResult(
        rho_gamma=float(gamma),
        rho_delta=float(delta),
        convexity_pnl_per_unit=float(conv_pnl),
    )


# ---- Cross-gamma ----

@dataclass
class CrossGammaResult:
    """Cross-gamma: ∂²V/∂S₁∂S₂."""
    cross_gamma: float
    delta1: float               # ∂V/∂S₁
    delta2: float               # ∂V/∂S₂
    assets: tuple[str, str]



    def to_dict(self) -> dict:
        return vars(self)
def cross_gamma(
    price_fn,
    spot1: float,
    spot2: float,
    asset1_name: str = "S1",
    asset2_name: str = "S2",
    bump_pct: float = 0.01,
) -> CrossGammaResult:
    """Cross-gamma: ∂²V/∂S₁∂S₂ via finite difference.

    Measures how delta of asset 1 changes when asset 2 moves.
    Important for hedging multi-asset exotics.

    Uses the formula:
        ∂²V/∂S₁∂S₂ ≈ [V(S₁+, S₂+) - V(S₁+, S₂-) - V(S₁-, S₂+) + V(S₁-, S₂-)] / (4 h₁ h₂)
    """
    h1 = spot1 * bump_pct
    h2 = spot2 * bump_pct

    p_pp = price_fn(spot1 + h1, spot2 + h2)
    p_pm = price_fn(spot1 + h1, spot2 - h2)
    p_mp = price_fn(spot1 - h1, spot2 + h2)
    p_mm = price_fn(spot1 - h1, spot2 - h2)

    cg = (p_pp - p_pm - p_mp + p_mm) / (4 * h1 * h2)

    # Individual deltas
    p_base = price_fn(spot1, spot2)
    p_1up = price_fn(spot1 + h1, spot2)
    p_1dn = price_fn(spot1 - h1, spot2)
    p_2up = price_fn(spot1, spot2 + h2)
    p_2dn = price_fn(spot1, spot2 - h2)

    d1 = (p_1up - p_1dn) / (2 * h1)
    d2 = (p_2up - p_2dn) / (2 * h2)

    return CrossGammaResult(
        cross_gamma=float(cg),
        delta1=float(d1),
        delta2=float(d2),
        assets=(asset1_name, asset2_name),
    )


# ---- Correlation P&L attribution ----

@dataclass
class CorrelationPnLAttribution:
    """P&L attribution from correlation changes."""
    total_pnl: float
    delta_pnl: float            # Δ_ρ × Δρ
    gamma_pnl: float            # 0.5 × Γ_ρ × Δρ²
    explained: float            # delta + gamma
    unexplained: float          # total − explained
    rho_change: float



    def to_dict(self) -> dict:
        return vars(self)
def correlation_pnl_attribution(
    price_fn,
    rho_old: float,
    rho_new: float,
    bump: float = 0.01,
) -> CorrelationPnLAttribution:
    """Attribute P&L to correlation changes via Taylor expansion.

    P&L ≈ Δ_ρ × Δρ + 0.5 × Γ_ρ × Δρ² + higher order.

    Args:
        price_fn: callable(rho) → price.
        rho_old, rho_new: correlation before and after.

    Fix T4-RISK9: pre-fix called ``correlation_delta`` (3 pricer calls)
    AND ``correlation_gamma`` (3 more) AND ``price_fn(rho_old)`` /
    ``price_fn(rho_new)`` (2 more) = 8 calls.  Calls overlapped: the
    sub-functions both compute ``price_fn(rho_old)``,
    ``price_fn(rho_old+bump)``, ``price_fn(rho_old-bump)``.  Only 4
    unique evaluations are needed (rho_old, rho_old±bump, rho_new).
    Now computes inline with 4 calls — halves the cost.
    """
    drho = rho_new - rho_old

    # Greeks at old ρ — compute inline, share the rho_old eval with the
    # P&L denominator.
    rho_up = min(rho_old + bump, 0.999)
    rho_dn = max(rho_old - bump, -0.999)
    h = (rho_up - rho_dn) / 2  # symmetric half-bump (may be smaller near ±1).

    p_base = price_fn(rho_old)
    p_up = price_fn(rho_up)
    p_dn = price_fn(rho_dn)
    p_new = price_fn(rho_new)

    if h > 1e-10:
        rho_delta = (p_up - p_dn) / (2 * h)
        rho_gamma = (p_up - 2 * p_base + p_dn) / (h ** 2)
    else:
        rho_delta = 0.0
        rho_gamma = 0.0

    delta_pnl = rho_delta * drho
    gamma_pnl = 0.5 * rho_gamma * drho**2
    explained = delta_pnl + gamma_pnl

    total = p_new - p_base
    unexplained = total - explained

    return CorrelationPnLAttribution(
        total_pnl=float(total),
        delta_pnl=float(delta_pnl),
        gamma_pnl=float(gamma_pnl),
        explained=float(explained),
        unexplained=float(unexplained),
        rho_change=float(drho),
    )


# ---- Correlation sensitivity ladder ----

@dataclass
class CorrelationLadderEntry:
    """One entry in the correlation sensitivity ladder."""
    asset_pair: tuple[str, str]
    rho: float
    rho_delta: float
    rho_gamma: float



    def to_dict(self) -> dict:
        return vars(self)
@dataclass
class CorrelationLadder:
    """Full correlation sensitivity ladder.

    Fix T4-RISK10: pre-fix exposed only ``total_rho_delta`` /
    ``total_rho_gamma`` which both sum ``abs(...)`` over pairs (a gross
    magnitude — useful for sizing hedges but not for portfolio P&L).
    The signed sums are what move portfolio value when correlations
    drift together.  Added ``net_rho_delta`` / ``net_rho_gamma`` as
    the signed totals; the original gross fields are preserved for
    backwards compatibility but renamed in docs as "gross" totals.
    """
    entries: list[CorrelationLadderEntry]
    total_rho_delta: float       # gross — sum of |delta_i|
    total_rho_gamma: float       # gross — sum of |gamma_i|
    net_rho_delta: float         # signed — sum of delta_i (portfolio rho sensitivity)
    net_rho_gamma: float         # signed — sum of gamma_i
    n_pairs: int



    def to_dict(self) -> dict:
        return vars(self)
def correlation_sensitivity_ladder(
    asset_names: list[str],
    correlations: np.ndarray,
    price_fn_matrix,
    bump: float = 0.01,
) -> CorrelationLadder:
    """Compute correlation sensitivity for each pair in the ρ matrix.

    Args:
        asset_names: list of asset names.
        correlations: (n×n) correlation matrix.
        price_fn_matrix: callable(corr_matrix) → total portfolio price.
        bump: ρ bump for each pair.
    """
    n = len(asset_names)
    entries = []
    total_delta = 0.0
    total_gamma = 0.0
    net_delta = 0.0
    net_gamma = 0.0

    base_price = price_fn_matrix(correlations)

    for i in range(n):
        for j in range(i + 1, n):
            rho = correlations[i, j]

            # Bump ρ_{ij} up
            corr_up = correlations.copy()
            corr_up[i, j] = min(rho + bump, 0.999)
            corr_up[j, i] = corr_up[i, j]

            # Bump ρ_{ij} down
            corr_dn = correlations.copy()
            corr_dn[i, j] = max(rho - bump, -0.999)
            corr_dn[j, i] = corr_dn[i, j]

            p_up = price_fn_matrix(corr_up)
            p_dn = price_fn_matrix(corr_dn)

            actual_bump = (corr_up[i, j] - corr_dn[i, j]) / 2
            delta = (p_up - p_dn) / (2 * actual_bump) if actual_bump > 1e-10 else 0.0
            gamma = (p_up - 2 * base_price + p_dn) / (actual_bump**2) if actual_bump > 1e-10 else 0.0

            entries.append(CorrelationLadderEntry(
                asset_pair=(asset_names[i], asset_names[j]),
                rho=float(rho),
                rho_delta=float(delta),
                rho_gamma=float(gamma),
            ))

            total_delta += abs(delta)
            total_gamma += abs(gamma)
            net_delta += delta
            net_gamma += gamma

    return CorrelationLadder(
        entries=entries,
        total_rho_delta=float(total_delta),
        total_rho_gamma=float(total_gamma),
        net_rho_delta=float(net_delta),
        net_rho_gamma=float(net_gamma),
        n_pairs=len(entries),
    )
