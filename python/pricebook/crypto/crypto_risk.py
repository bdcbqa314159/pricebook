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
        return dict(vars(self))


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
        return dict(vars(self))


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


# ═══════════════════════════════════════════════════════════════
# CD13: Liquidity Depth, Correlation Breakdown, Gas Cost
# ═══════════════════════════════════════════════════════════════

@dataclass
class LiquidityDepthResult:
    """Orderbook liquidity depth analysis."""
    bid_depth_usd: float        # total bid liquidity within X%
    ask_depth_usd: float
    bid_ask_spread_bps: float
    slippage_1pct: float        # expected slippage for 1% of depth
    time_to_exit_hours: float   # estimated time to exit position
    liquidity_score: float      # 0–100

    def to_dict(self) -> dict:
        return dict(vars(self))


def liquidity_depth(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    position_size_usd: float = 100_000.0,
    daily_volume_usd: float = 10_000_000.0,
    pct_range: float = 0.02,
) -> LiquidityDepthResult:
    """Analyse orderbook depth and execution quality.

    Args:
        bids: [(price, qty_usd), ...] sorted descending by price.
        asks: [(price, qty_usd), ...] sorted ascending by price.
        position_size_usd: position to exit.
        daily_volume_usd: average daily volume.
        pct_range: depth range (±2% from mid).
    """
    if not bids or not asks:
        return LiquidityDepthResult(0, 0, 0, 0, 0, 0)

    mid = (bids[0][0] + asks[0][0]) / 2
    spread_bps = (asks[0][0] - bids[0][0]) / mid * 10_000

    # Depth within range
    bid_depth = sum(qty for p, qty in bids if p >= mid * (1 - pct_range))
    ask_depth = sum(qty for p, qty in asks if p <= mid * (1 + pct_range))

    # Slippage estimate: walk the book
    remaining = position_size_usd
    total_cost = 0.0
    for price, qty in bids:
        fill = min(remaining, qty)
        total_cost += fill
        remaining -= fill
        if remaining <= 0:
            break
    slippage = (mid - total_cost / position_size_usd * mid) / mid * 100 if position_size_usd > 0 and total_cost > 0 else 0

    # Time to exit: position / (participation_rate × daily_volume)
    participation = 0.10  # 10% of volume
    tte = position_size_usd / (participation * daily_volume_usd) * 24 if daily_volume_usd > 0 else float('inf')

    # Score: 0–100 based on depth relative to position
    depth_ratio = (bid_depth + ask_depth) / max(position_size_usd, 1)
    score = min(depth_ratio * 50, 100)

    return LiquidityDepthResult(bid_depth, ask_depth, spread_bps, abs(slippage), tte, score)


@dataclass
class StressCorrelationResult:
    """Correlation breakdown analysis under stress."""
    normal_correlation: float
    stress_correlation: float
    correlation_jump: float
    diversification_loss_pct: float
    n_assets: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def correlation_stress_test(
    returns: dict[str, list[float]],
    stress_threshold: float = -0.05,
) -> StressCorrelationResult:
    """Test correlation breakdown during market stress.

    Crypto correlations spike to ~1.0 in crashes.
    Diversification benefit evaporates exactly when needed most.

    Computes correlation in normal times vs stress times
    (days when any major asset drops > threshold).

    Args:
        returns: {asset: [daily_returns]}.
        stress_threshold: return below which = stress day.
    """
    assets = list(returns.keys())
    n = len(assets)
    min_len = min(len(v) for v in returns.values())

    data = np.column_stack([np.array(returns[a][:min_len]) for a in assets])

    # Identify stress days: any asset drops below threshold
    stress_mask = np.any(data < stress_threshold, axis=1)
    normal_mask = ~stress_mask

    if np.sum(stress_mask) < 3 or np.sum(normal_mask) < 3:
        # Not enough data
        corr_all = np.corrcoef(data, rowvar=False)
        avg = float(np.mean(corr_all[np.triu_indices(n, k=1)]))
        return StressCorrelationResult(avg, avg, 0, 0, n)

    # Normal correlation
    normal_data = data[normal_mask]
    corr_normal = np.corrcoef(normal_data, rowvar=False)
    avg_normal = float(np.mean(corr_normal[np.triu_indices(n, k=1)]))

    # Stress correlation
    stress_data = data[stress_mask]
    corr_stress = np.corrcoef(stress_data, rowvar=False)
    avg_stress = float(np.mean(corr_stress[np.triu_indices(n, k=1)]))

    jump = avg_stress - avg_normal
    div_loss = jump / (1 - avg_normal) * 100 if avg_normal < 1 else 0

    return StressCorrelationResult(avg_normal, avg_stress, jump, div_loss, n)


@dataclass
class GasCostResult:
    """Gas cost analysis for DeFi operations."""
    base_fee_gwei: float
    priority_fee_gwei: float
    total_gas_usd: float
    cost_as_pct_of_trade: float
    profitable_above_usd: float  # min trade size to be profitable

    def to_dict(self) -> dict:
        return dict(vars(self))


def gas_cost_analysis(
    base_fee_gwei: float = 30.0,
    priority_fee_gwei: float = 2.0,
    gas_units: int = 150_000,
    eth_price_usd: float = 3000.0,
    trade_size_usd: float = 10_000.0,
    expected_profit_pct: float = 0.005,
) -> GasCostResult:
    """Analyse gas costs impact on DeFi profitability.

    EIP-1559: total_fee = (base_fee + priority_fee) × gas_units.
    Cost in USD: total_fee_gwei × 1e-9 × eth_price.

    Args:
        base_fee_gwei: current base fee.
        priority_fee_gwei: tip for faster inclusion.
        gas_units: gas needed for operation (swap ~150k, complex ~500k).
        eth_price_usd: ETH price for USD conversion.
        trade_size_usd: trade notional.
        expected_profit_pct: expected profit before gas.
    """
    total_gwei = (base_fee_gwei + priority_fee_gwei) * gas_units
    total_eth = total_gwei * 1e-9
    total_usd = total_eth * eth_price_usd

    cost_pct = total_usd / trade_size_usd * 100 if trade_size_usd > 0 else 0

    # Min trade size to be profitable
    min_trade = total_usd / expected_profit_pct if expected_profit_pct > 0 else float('inf')

    return GasCostResult(base_fee_gwei, priority_fee_gwei, total_usd, cost_pct, min_trade)
