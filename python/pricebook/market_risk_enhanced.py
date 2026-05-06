"""Enhanced market risk: Incremental VaR, Stressed VaR, copula tail ES.

Extends the existing VaR/ES framework (var.py, regulatory/var_es.py)
with position-level risk attribution and stressed market measures.

    from pricebook.market_risk_enhanced import (
        incremental_var, IncrementalVaRResult,
        stressed_var, StressedVaRResult,
        copula_es, CopulaESResult,
    )

References:
    Jorion (2007). Value at Risk: The New Benchmark. McGraw-Hill.
    Basel Committee (2019). MAR31 — Internal Models Approach.
    McNeil, Frey, Embrechts (2015). Quantitative Risk Management.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as t_dist


# ---------------------------------------------------------------------------
# Incremental VaR
# ---------------------------------------------------------------------------

@dataclass
class IncrementalVaRResult:
    """Incremental VaR decomposition per position."""
    portfolio_var: float           # total portfolio VaR
    positions: list[str]           # position identifiers
    incremental_vars: list[float]  # marginal contribution per position
    component_vars: list[float]    # percentage contribution per position
    diversification_benefit: float  # sum(individual) - portfolio

    def to_dict(self) -> dict:
        return {
            "portfolio_var": self.portfolio_var,
            "positions": dict(zip(self.positions, self.incremental_vars)),
            "component_pct": dict(zip(self.positions,
                [cv / max(self.portfolio_var, 1e-10) for cv in self.component_vars])),
            "diversification": self.diversification_benefit,
        }


def incremental_var(
    position_pnls: dict[str, list[float]],
    confidence: float = 0.99,
    method: str = "parametric",
) -> IncrementalVaRResult:
    """Compute incremental VaR for each position in a portfolio.

    Incremental VaR (IVaR) measures each position's marginal contribution
    to portfolio VaR. Uses the Euler decomposition: sum(IVaR_i) = portfolio VaR.

    For parametric (delta-normal):
        IVaR_i = (Σ × w_i) × z_α / (w' × Σ × w)^0.5

    For historical:
        IVaR_i = VaR(portfolio) - VaR(portfolio without i)

    Args:
        position_pnls: dict of {position_id: [daily_pnl_series]}.
        confidence: VaR confidence level (e.g. 0.99 for 99%).
        method: "parametric" (Euler) or "historical" (leave-one-out).
    """
    names = list(position_pnls.keys())
    n = len(names)
    if n == 0:
        return IncrementalVaRResult(0.0, [], [], [], 0.0)

    # Build P&L matrix: rows = observations, columns = positions
    pnl_lists = [position_pnls[name] for name in names]
    min_len = min(len(p) for p in pnl_lists)
    pnl_matrix = np.array([p[:min_len] for p in pnl_lists]).T  # (T, n)

    # Portfolio P&L
    portfolio_pnl = pnl_matrix.sum(axis=1)

    if method == "historical":
        # Historical VaR: percentile
        alpha = 1 - confidence
        portfolio_var = -float(np.percentile(portfolio_pnl, alpha * 100))

        # Leave-one-out: VaR without each position
        incremental = []
        for i in range(n):
            pnl_without = portfolio_pnl - pnl_matrix[:, i]
            var_without = -float(np.percentile(pnl_without, alpha * 100))
            incremental.append(portfolio_var - var_without)

        # Component VaR (Euler approx for historical)
        component = incremental  # simplified

    else:  # parametric
        # Covariance matrix
        cov = np.cov(pnl_matrix, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Equal-weighted portfolio (sum of positions)
        w = np.ones(n)
        port_vol = float(np.sqrt(w @ cov @ w))
        z = norm.ppf(confidence)
        portfolio_var = port_vol * z

        # Euler decomposition: IVaR_i = (Σ × w)_i × z / port_vol
        marginal = cov @ w  # (n,) vector
        incremental = [float(marginal[i] * z / max(port_vol, 1e-15)) for i in range(n)]
        component = incremental

    # Individual VaRs (undiversified)
    individual_vars = []
    for i in range(n):
        if method == "historical":
            alpha = 1 - confidence
            iv = -float(np.percentile(pnl_matrix[:, i], alpha * 100))
        else:
            iv = float(np.std(pnl_matrix[:, i]) * norm.ppf(confidence))
        individual_vars.append(iv)

    diversification = sum(individual_vars) - portfolio_var

    return IncrementalVaRResult(
        portfolio_var=portfolio_var,
        positions=names,
        incremental_vars=incremental,
        component_vars=component,
        diversification_benefit=diversification,
    )


# ---------------------------------------------------------------------------
# Stressed VaR
# ---------------------------------------------------------------------------

@dataclass
class StressedVaRResult:
    """Stressed VaR result (Basel 2.5 / MAR33)."""
    current_var: float
    stressed_var: float
    stress_multiplier: float
    stressed_period: str      # description of the stress window
    capital_charge: float     # max(SVaR_t-1, mc × SVaR_avg) + max(VaR_t-1, mc × VaR_avg)

    def to_dict(self) -> dict:
        return {"current_var": self.current_var, "stressed_var": self.stressed_var,
                "multiplier": self.stress_multiplier,
                "period": self.stressed_period, "capital": self.capital_charge}


def stressed_var(
    current_pnls: list[float],
    stressed_pnls: list[float],
    confidence: float = 0.99,
    multiplier: float = 3.0,
    stressed_period: str = "2008-Q4 to 2009-Q1",
) -> StressedVaRResult:
    """Compute stressed VaR (Basel 2.5).

    SVaR uses the same VaR methodology but applied to a stressed market
    period (worst 12 months in recent history).

    Capital = max(VaR_t-1, mc × VaR_60d_avg) + max(SVaR_t-1, mc × SVaR_60d_avg)

    Args:
        current_pnls: recent P&L series (current market conditions).
        stressed_pnls: P&L series from the stressed period.
        confidence: VaR confidence level.
        multiplier: regulatory multiplier (Basel: 3.0 base).
        stressed_period: description of the stress window used.
    """
    alpha = 1 - confidence
    current_var = -float(np.percentile(current_pnls, alpha * 100))
    svar = -float(np.percentile(stressed_pnls, alpha * 100))

    # Capital: simplified (single-day, no 60d average)
    capital = multiplier * current_var + multiplier * svar

    return StressedVaRResult(
        current_var=current_var,
        stressed_var=svar,
        stress_multiplier=svar / max(current_var, 1e-10),
        stressed_period=stressed_period,
        capital_charge=capital,
    )


# ---------------------------------------------------------------------------
# Copula-based ES
# ---------------------------------------------------------------------------

@dataclass
class CopulaESResult:
    """Expected Shortfall with copula-based tail estimation."""
    es_normal: float          # ES assuming normal tail
    es_t: float               # ES assuming t-distribution tail
    es_empirical: float       # ES from empirical tail
    tail_index: float         # estimated tail heaviness (t-df)
    selected_es: float        # conservative (max of estimates)

    def to_dict(self) -> dict:
        return {"es_normal": self.es_normal, "es_t": self.es_t,
                "es_empirical": self.es_empirical,
                "tail_index": self.tail_index, "selected": self.selected_es}


def copula_es(
    pnls: list[float],
    confidence: float = 0.975,
    t_df: float | None = None,
) -> CopulaESResult:
    """Expected Shortfall with multiple tail models.

    Computes ES under three distributional assumptions:
    1. Normal (Gaussian tail)
    2. Student-t (heavier tails, estimated df)
    3. Empirical (direct tail average)

    Selects the most conservative (highest) estimate.

    Args:
        pnls: historical P&L series.
        confidence: ES confidence level (e.g. 0.975 for 97.5%).
        t_df: degrees of freedom for t-distribution (None = estimate from data).
    """
    pnl_arr = np.array(pnls)
    mu = float(np.mean(pnl_arr))
    sigma = float(np.std(pnl_arr, ddof=1))
    alpha = 1 - confidence

    # Normal ES
    z = norm.ppf(alpha)
    es_normal = -(mu + sigma * norm.pdf(z) / alpha)

    # t-distribution ES
    if t_df is None:
        # Estimate df from excess kurtosis: kurt = 6/(df-4) → df = 6/kurt + 4
        kurt = float(np.mean((pnl_arr - mu) ** 4) / sigma ** 4 - 3)
        t_df = max(6.0 / max(kurt, 0.1) + 4, 3.0)  # floor at 3
    t_z = t_dist.ppf(alpha, t_df)
    es_t = -(mu + sigma * t_dist.pdf(t_z, t_df) / alpha * (t_df + t_z ** 2) / (t_df - 1))

    # Empirical ES
    sorted_pnls = np.sort(pnl_arr)
    n_tail = max(int(len(sorted_pnls) * alpha), 1)
    es_empirical = -float(np.mean(sorted_pnls[:n_tail]))

    selected = max(es_normal, es_t, es_empirical)

    return CopulaESResult(
        es_normal=es_normal, es_t=es_t, es_empirical=es_empirical,
        tail_index=t_df, selected_es=selected,
    )
