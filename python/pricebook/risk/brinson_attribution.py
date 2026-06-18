"""Brinson-Fachler performance attribution.

Decomposes portfolio excess return into allocation, selection,
and interaction effects relative to a benchmark.

* :func:`brinson_attribution` — full Brinson-Fachler decomposition.
* :func:`brinson_multi_period` — multi-period linking.
* :func:`factor_based_attribution` — factor-model attribution.

References:
    Brinson, Hood & Beebower, *Determinants of Portfolio Performance*,
    FAJ, 1986.
    Brinson & Fachler, *Measuring Non-US Equity Portfolio Performance*,
    JPM, 1985.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BrinsonSectorResult:
    """Attribution for a single sector."""
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class BrinsonResult:
    """Complete Brinson-Fachler attribution."""
    sectors: list[BrinsonSectorResult]
    total_allocation: float
    total_selection: float
    total_interaction: float
    total_active_return: float
    portfolio_return: float
    benchmark_return: float

    def to_dict(self) -> dict:
        return {
            "allocation": self.total_allocation,
            "selection": self.total_selection,
            "interaction": self.total_interaction,
            "active_return": self.total_active_return,
            "portfolio_return": self.portfolio_return,
            "benchmark_return": self.benchmark_return,
            "n_sectors": len(self.sectors),
        }


def brinson_attribution(
    portfolio_weights: list[float],
    benchmark_weights: list[float],
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    sector_names: list[str] | None = None,
) -> BrinsonResult:
    """Brinson-Fachler performance attribution.

    Decomposes active return (portfolio − benchmark) into:

    Allocation: (w_p − w_b) × (r_b − R_b)
        Overweighting sectors that outperform the benchmark.

    Selection: w_b × (r_p − r_b)
        Picking better securities within sectors.

    Interaction: (w_p − w_b) × (r_p − r_b)
        Combined effect of overweighting AND outperforming.

    Args:
        portfolio_weights: sector weights in portfolio.
        benchmark_weights: sector weights in benchmark.
        portfolio_returns: sector returns in portfolio.
        benchmark_returns: sector returns in benchmark.
        sector_names: sector labels.
    """
    n = len(portfolio_weights)
    names = sector_names or [f"Sector_{i}" for i in range(n)]

    R_b = sum(w * r for w, r in zip(benchmark_weights, benchmark_returns))
    R_p = sum(w * r for w, r in zip(portfolio_weights, portfolio_returns))

    sectors = []
    total_alloc = 0.0
    total_sel = 0.0
    total_inter = 0.0

    for i in range(n):
        w_p = portfolio_weights[i]
        w_b = benchmark_weights[i]
        r_p = portfolio_returns[i]
        r_b = benchmark_returns[i]

        # Brinson-Fachler
        alloc = (w_p - w_b) * (r_b - R_b)
        sel = w_b * (r_p - r_b)
        inter = (w_p - w_b) * (r_p - r_b)

        total_alloc += alloc
        total_sel += sel
        total_inter += inter

        sectors.append(BrinsonSectorResult(
            sector=names[i],
            portfolio_weight=w_p,
            benchmark_weight=w_b,
            portfolio_return=r_p,
            benchmark_return=r_b,
            allocation_effect=alloc,
            selection_effect=sel,
            interaction_effect=inter,
            total_effect=alloc + sel + inter,
        ))

    return BrinsonResult(
        sectors=sectors,
        total_allocation=total_alloc,
        total_selection=total_sel,
        total_interaction=total_inter,
        total_active_return=R_p - R_b,
        portfolio_return=R_p,
        benchmark_return=R_b,
    )


def brinson_multi_period(
    period_results: list[BrinsonResult],
) -> dict:
    """Multi-period Brinson linking via Frongello (2002).

    Compounds single-period attributions while preserving the
    geometric active-return identity:
        Σ_t F_t·(alloc_t + sel_t + inter_t) = Π(1+r_p_t) − Π(1+r_b_t)

    where the Frongello linking coefficient F_t is the recursive
    update:
        cum_t = cum_{t-1} · (1+r_b_t) + effect_t · cum_port_{t-1}

    Equivalent closed-form: F_t = (Π_{s<t}(1+r_p_s)) · (Π_{s>t}(1+r_b_s)).

    Fix T4-RISK19: pre-fix used the ad-hoc scaling
    ``cum_alloc += effect_t × cum_bench_before_t`` which is neither
    Frongello nor Carino, and does NOT preserve the identity.  For
    two periods of equal P&L (port = bench = 1%/period), pre-fix
    gives Σ effects = 2.00% while geometric active is 2.01% — small
    in this case but quadratic-in-T as horizon grows.

    Args:
        period_results: list of single-period Brinson results.
    """
    cum_alloc = 0.0
    cum_sel = 0.0
    cum_inter = 0.0
    cum_port = 1.0
    cum_bench = 1.0

    for r in period_results:
        # Frongello: prior cumulative effects compound by this period's
        # benchmark return; the new period's effects scale by the prior
        # cumulative portfolio growth.
        cum_alloc = cum_alloc * (1 + r.benchmark_return) + r.total_allocation * cum_port
        cum_sel = cum_sel * (1 + r.benchmark_return) + r.total_selection * cum_port
        cum_inter = cum_inter * (1 + r.benchmark_return) + r.total_interaction * cum_port
        cum_port *= (1 + r.portfolio_return)
        cum_bench *= (1 + r.benchmark_return)

    return {
        "cumulative_allocation": cum_alloc,
        "cumulative_selection": cum_sel,
        "cumulative_interaction": cum_inter,
        "cumulative_active_return": cum_port - cum_bench,
        "portfolio_total_return": cum_port - 1,
        "benchmark_total_return": cum_bench - 1,
        "n_periods": len(period_results),
    }


def factor_based_attribution(
    portfolio_returns: np.ndarray,
    factor_returns: np.ndarray,
    factor_names: list[str] | None = None,
) -> dict:
    """Factor-based return attribution via OLS.

    r_p = α + Σ β_i × f_i + ε

    Args:
        portfolio_returns: (T,) portfolio return series.
        factor_returns: (T, K) factor return matrix.
        factor_names: factor labels.
    """
    T, K = factor_returns.shape
    names = factor_names or [f"Factor_{i}" for i in range(K)]

    # OLS: add intercept
    X = np.column_stack([np.ones(T), factor_returns])
    try:
        beta = np.linalg.lstsq(X, portfolio_returns, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(K + 1)

    alpha = beta[0]
    factor_betas = beta[1:]

    # Attribution
    factor_contributions = {}
    for i, name in enumerate(names):
        contrib = float(factor_betas[i] * np.mean(factor_returns[:, i]))
        factor_contributions[name] = {
            "beta": float(factor_betas[i]),
            "factor_return": float(np.mean(factor_returns[:, i])),
            "contribution": contrib,
        }

    residual = portfolio_returns - X @ beta
    r_squared = 1 - np.var(residual) / np.var(portfolio_returns) if np.var(portfolio_returns) > 0 else 0

    return {
        "alpha": float(alpha),
        "factor_contributions": factor_contributions,
        "r_squared": float(r_squared),
        "residual_vol": float(np.std(residual)),
        "total_return": float(np.mean(portfolio_returns)),
    }
