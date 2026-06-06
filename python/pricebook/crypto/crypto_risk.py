"""Crypto risk metrics: 24/7 VaR, tail risk, exchange risk.

* :func:`crypto_var` — VaR adapted for 24/7 markets.
* :func:`tail_risk` — power-law tail analysis.
* :func:`exchange_risk` — concentration risk across exchanges.

References:
    Bouri et al., *Return Connectedness Across Asset Classes Around
    the COVID-19 Outbreak*, IJF, 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CryptoVaRResult:
    """Crypto VaR result (24/7 adapted)."""
    var_1d: float               # 1-day VaR
    var_1h: float               # 1-hour VaR (useful for 24/7)
    es_1d: float                # 1-day Expected Shortfall
    confidence: float
    method: str
    annualised_vol: float

    def to_dict(self) -> dict:
        return vars(self)


def crypto_var(
    returns: list[float],
    confidence: float = 0.99,
    interval_hours: float = 1.0,
    method: str = "historical",
) -> CryptoVaRResult:
    """VaR for 24/7 crypto markets.

    Key difference from TradFi: annualisation uses 365×24 hours,
    not 252 trading days.

    Args:
        returns: return series (log or simple).
        confidence: VaR confidence level.
        interval_hours: observation interval.
        method: "historical" or "parametric".
    """
    arr = np.array(returns)
    n = len(arr)

    if method == "parametric":
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        var_interval = -(mu + z * sigma)
    else:
        var_interval = float(-np.percentile(arr, (1 - confidence) * 100))

    # Scale to different horizons
    # 1-hour VaR from interval VaR
    scale_1h = math.sqrt(1.0 / interval_hours) if interval_hours > 0 else 1
    var_1h = var_interval * scale_1h

    # 1-day VaR (24 hours)
    scale_1d = math.sqrt(24.0 / interval_hours) if interval_hours > 0 else 1
    var_1d = var_interval * scale_1d

    # ES (tail average beyond VaR)
    threshold = np.percentile(arr, (1 - confidence) * 100)
    tail = arr[arr <= threshold]
    es_1d = float(-np.mean(tail)) * scale_1d if len(tail) > 0 else var_1d

    # Annualised vol (365×24)
    periods_per_year = 365 * 24 / interval_hours
    ann_vol = float(np.std(arr, ddof=1)) * math.sqrt(periods_per_year)

    return CryptoVaRResult(
        var_1d=var_1d,
        var_1h=var_1h,
        es_1d=es_1d,
        confidence=confidence,
        method=method,
        annualised_vol=ann_vol,
    )


@dataclass
class TailRiskResult:
    """Power-law tail analysis."""
    tail_index: float           # Hill estimator (α)
    expected_max_loss_pct: float
    kurtosis: float
    is_heavy_tailed: bool       # α < 4 suggests heavy tails

    def to_dict(self) -> dict:
        return vars(self)


def tail_risk(
    returns: list[float],
    tail_pct: float = 0.05,
) -> TailRiskResult:
    """Power-law tail analysis for crypto returns.

    Crypto returns exhibit heavier tails than equities (α ≈ 2–3
    vs α ≈ 3–5 for equities).

    Uses Hill estimator for tail index:
    α = n / Σ log(x_i / x_min)

    Args:
        returns: return series.
        tail_pct: fraction of observations in the tail.
    """
    arr = np.abs(np.array(returns))
    arr_sorted = np.sort(arr)[::-1]  # descending
    n_tail = max(int(len(arr) * tail_pct), 2)
    tail_obs = arr_sorted[:n_tail]

    x_min = tail_obs[-1]
    if x_min > 0:
        log_ratios = np.log(tail_obs / x_min)
        alpha = n_tail / float(np.sum(log_ratios)) if np.sum(log_ratios) > 0 else 10
    else:
        alpha = 10  # thin tails

    kurtosis = float(np.mean((np.array(returns) - np.mean(returns))**4) /
                     np.std(returns)**4 - 3) if np.std(returns) > 0 else 0

    # Expected max loss (simplified)
    max_loss = float(np.percentile(-np.array(returns), 99.9))

    return TailRiskResult(
        tail_index=alpha,
        expected_max_loss_pct=max_loss * 100,
        kurtosis=kurtosis,
        is_heavy_tailed=alpha < 4,
    )


def exchange_risk(
    balances: dict[str, float],
    total_portfolio: float | None = None,
) -> dict:
    """Exchange concentration risk.

    Measures how concentrated your crypto is across exchanges.
    Herfindahl index: H = Σ(share_i²). H > 0.25 = concentrated.

    Args:
        balances: {exchange_name: USD_value}.
        total_portfolio: optional total (if not sum of balances).
    """
    total = total_portfolio or sum(balances.values())
    if total <= 0:
        return {"herfindahl": 0, "concentrated": False, "n_exchanges": 0}

    shares = {k: v / total for k, v in balances.items()}
    herfindahl = sum(s**2 for s in shares.values())
    largest = max(shares.items(), key=lambda x: x[1])

    return {
        "shares": {k: round(v * 100, 1) for k, v in shares.items()},
        "herfindahl": round(herfindahl, 4),
        "concentrated": herfindahl > 0.25,
        "largest_exchange": largest[0],
        "largest_share_pct": round(largest[1] * 100, 1),
        "n_exchanges": len(balances),
    }
