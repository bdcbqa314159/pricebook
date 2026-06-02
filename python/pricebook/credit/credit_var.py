"""Credit portfolio VaR: historical, parametric, and copula-based.

Value-at-Risk and Expected Shortfall for credit portfolios,
using spread-based (CS01) and default-based approaches.

* :class:`CreditVaRResult` — VaR/ES result with diagnostics.
* :func:`historical_credit_var` — historical simulation VaR.
* :func:`parametric_credit_var` — delta-normal (parametric) VaR.
* :func:`copula_credit_var` — Gaussian copula joint-default VaR.

References:
    McNeil, Frey & Embrechts, *Quantitative Risk Management*, Princeton, 2005.
    Crouhy, Galai & Mark, *Risk Management*, McGraw-Hill, 2001.
    Li, *On Default Correlation: A Copula Function Approach*, J. Fixed Income, 2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from pricebook.statistics.copulas import GaussianCopula


@dataclass
class CreditVaRResult:
    """Credit VaR / ES result."""
    var_amount: float
    es_amount: float
    confidence: float
    method: str
    n_positions: int
    worst_name: str
    worst_contribution: float

    def to_dict(self) -> dict:
        return {
            "var_amount": self.var_amount,
            "es_amount": self.es_amount,
            "confidence": self.confidence,
            "method": self.method,
            "n_positions": self.n_positions,
            "worst_name": self.worst_name,
            "worst_contribution": self.worst_contribution,
        }


# ---- Historical simulation ----

def historical_credit_var(
    positions: list[dict[str, Any]],
    spread_changes: dict[str, list[float]],
    cs01s: list[float] | None = None,
    confidence: float = 0.99,
) -> CreditVaRResult:
    """Historical simulation credit VaR.

    PnL per day = sum_i cs01_i * delta_spread_i.
    VaR is the (1 - confidence) percentile of the PnL distribution
    (negative = loss).

    Args:
        positions: list of {"name": str, "cs01": float}.
        spread_changes: dict {name: [daily spread changes in bp]}.
        cs01s: optional override for CS01 vector (uses positions if None).
        confidence: VaR confidence level (e.g. 0.99).
    """
    n = len(positions)
    names = [p["name"] for p in positions]
    cs01_vec = np.array(cs01s if cs01s is not None else [p["cs01"] for p in positions])

    # Determine number of historical observations
    n_obs = min(len(spread_changes.get(name, [])) for name in names)
    if n_obs == 0:
        return CreditVaRResult(0.0, 0.0, confidence, "historical", n, "", 0.0)

    # Build spread change matrix (n_obs, n_positions)
    delta_matrix = np.zeros((n_obs, n))
    for j, name in enumerate(names):
        changes = spread_changes.get(name, [])
        delta_matrix[:, j] = changes[:n_obs]

    # PnL per day: negative cs01 * positive spread change = loss
    # Convention: cs01 > 0 means long protection, spread widening = profit
    pnl = delta_matrix @ cs01_vec  # (n_obs,)

    # VaR: loss at (1-confidence) percentile
    var_pctl = (1.0 - confidence) * 100.0
    var_amount = float(np.percentile(pnl, var_pctl))

    # ES: mean of losses worse than VaR
    tail = pnl[pnl <= var_amount]
    es_amount = float(np.mean(tail)) if len(tail) > 0 else var_amount

    # Worst name: largest marginal contribution
    # Marginal contribution = cs01_i * std(spread_changes_i)
    contributions: list[tuple[float, str]] = []
    for j, name in enumerate(names):
        std_j = float(np.std(delta_matrix[:, j])) if n_obs > 1 else 0.0
        contributions.append((abs(cs01_vec[j]) * std_j, name))
    contributions.sort(reverse=True)

    worst_name = contributions[0][1] if contributions else ""
    worst_contrib = contributions[0][0] if contributions else 0.0

    return CreditVaRResult(
        var_amount=var_amount,
        es_amount=es_amount,
        confidence=confidence,
        method="historical",
        n_positions=n,
        worst_name=worst_name,
        worst_contribution=worst_contrib,
    )


# ---- Parametric (delta-normal) ----

def parametric_credit_var(
    positions: list[dict[str, Any]],
    vols: list[float],
    correlation_matrix: list[list[float]],
    confidence: float = 0.99,
) -> CreditVaRResult:
    """Delta-normal (parametric) credit VaR.

    VaR = z_alpha * sqrt(CS01^T * Sigma * CS01)
    where Sigma = diag(vols) * correlation * diag(vols).

    Args:
        positions: list of {"name": str, "cs01": float}.
        vols: annualised spread vol per position (bp).
        correlation_matrix: n×n correlation matrix.
        confidence: VaR confidence level.
    """
    n = len(positions)
    names = [p["name"] for p in positions]
    cs01_vec = np.array([p["cs01"] for p in positions])
    vol_vec = np.array(vols)
    corr = np.array(correlation_matrix)

    # Covariance matrix: Sigma = diag(vols) @ corr @ diag(vols)
    D = np.diag(vol_vec)
    cov = D @ corr @ D

    # Portfolio variance: CS01^T * Sigma * CS01
    port_var = float(cs01_vec @ cov @ cs01_vec)
    port_std = math.sqrt(max(port_var, 0.0))

    # z_alpha for the confidence level (left tail)
    z_alpha = norm.ppf(1.0 - confidence)  # negative

    var_amount = z_alpha * port_std  # negative (loss)

    # ES for normal distribution: ES = -sigma * phi(z) / (1 - alpha)
    phi_z = norm.pdf(norm.ppf(1.0 - confidence))
    es_amount = -port_std * float(phi_z) / (1.0 - confidence)

    # Marginal contributions: component VaR
    marginal = cov @ cs01_vec  # (n,)
    contributions: list[tuple[float, str]] = []
    for j, name in enumerate(names):
        mc = abs(float(cs01_vec[j] * marginal[j]))
        contributions.append((mc, name))
    contributions.sort(reverse=True)

    worst_name = contributions[0][1] if contributions else ""
    worst_contrib = contributions[0][0] if contributions else 0.0

    return CreditVaRResult(
        var_amount=var_amount,
        es_amount=es_amount,
        confidence=confidence,
        method="parametric",
        n_positions=n,
        worst_name=worst_name,
        worst_contribution=worst_contrib,
    )


# ---- Copula (joint default simulation) ----

def copula_credit_var(
    positions: list[dict[str, Any]],
    pds: list[float],
    lgds: list[float],
    correlation: float,
    confidence: float = 0.99,
    n_sims: int = 100_000,
    seed: int = 42,
) -> CreditVaRResult:
    """Gaussian copula joint-default credit VaR.

    Simulates correlated defaults using :class:`GaussianCopula`,
    computes portfolio loss distribution, and extracts VaR / ES.

    Loss per simulation = sum_i lgd_i * 1{default_i}.

    Args:
        positions: list of {"name": str, "notional": float}.
        pds: per-name default probability.
        lgds: per-name loss-given-default (fraction).
        correlation: equi-correlation for Gaussian copula.
        confidence: VaR confidence level.
        n_sims: number of Monte Carlo simulations.
        seed: random seed for reproducibility.
    """
    n = len(positions)
    names = [p["name"] for p in positions]
    notionals = np.array([p.get("notional", 1.0) for p in positions])
    lgd_vec = np.array(lgds)

    # Simulate correlated defaults via Gaussian copula
    copula = GaussianCopula(correlation)
    defaults = copula.default_indicators(pds, n_sims, seed=seed)  # (n_sims, n)

    # Loss per simulation (positive = loss)
    unit_loss = notionals * lgd_vec  # (n,)
    losses = defaults.astype(float) @ unit_loss  # (n_sims,)

    # Convert to P&L convention: negative = loss (consistent with historical/parametric)
    pnl = -losses

    # VaR: left tail of P&L distribution
    var_pctl = (1.0 - confidence) * 100.0
    var_amount = float(np.percentile(pnl, var_pctl))

    # ES: mean of P&L worse than VaR
    tail = pnl[pnl <= var_amount]
    es_amount = float(np.mean(tail)) if len(tail) > 0 else var_amount

    # Worst name: highest expected loss contribution
    expected_loss = np.array(pds) * lgd_vec * notionals
    contributions: list[tuple[float, str]] = []
    for j, name in enumerate(names):
        contributions.append((float(expected_loss[j]), name))
    contributions.sort(reverse=True)

    worst_name = contributions[0][1] if contributions else ""
    worst_contrib = contributions[0][0] if contributions else 0.0

    return CreditVaRResult(
        var_amount=var_amount,
        es_amount=es_amount,
        confidence=confidence,
        method="copula",
        n_positions=n,
        worst_name=worst_name,
        worst_contribution=worst_contrib,
    )
