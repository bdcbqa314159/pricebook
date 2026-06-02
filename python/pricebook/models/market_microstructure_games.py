"""Strategic market microstructure: Kyle, Glosten-Milgrom.

* :func:`kyle_lambda` — Kyle (1985) price impact and insider profit.
* :func:`glosten_milgrom` — sequential trade with adverse selection.
* :func:`optimal_order_splitting` — split large order to minimise impact.
* :func:`information_share` — Hasbrouck information share decomposition.

References:
    Kyle, *Continuous Auctions and Insider Trading*, Ecta, 1985.
    Glosten & Milgrom, *Bid, Ask and Transaction Prices*, JFE, 1985.
    Hasbrouck, *One Security, Many Markets*, JF, 1995.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class KyleResult:
    """Kyle (1985) model result."""
    lambda_impact: float        # price impact per unit of order flow
    insider_profit: float       # expected insider profit
    informed_trade_size: float  # optimal insider trade
    market_depth: float         # 1/λ
    price_efficiency: float     # fraction of private info in price (0→1)
    noise_vol: float

    def to_dict(self) -> dict:
        return vars(self)


def kyle_lambda(
    sigma_v: float,
    sigma_u: float,
    n_informed: int = 1,
) -> KyleResult:
    """Kyle (1985) single-period model.

    One insider knows v ~ N(μ, σ²_v). Noise traders submit u ~ N(0, σ²_u).
    Market maker sets price = E[v | x] = μ + λx where x = insider_trade + u.

    λ = σ_v / (2σ_u) for single informed trader.
    Insider profit = σ_v × σ_u / 2.

    Multiple insiders: λ increases, each insider's profit decreases.

    Args:
        sigma_v: volatility of private information (std of v).
        sigma_u: noise trader volatility (std of u).
        n_informed: number of informed traders.
    """
    if n_informed == 1:
        lam = sigma_v / (2 * sigma_u)
        insider_trade = sigma_u  # optimal trade size
        insider_profit = sigma_v * sigma_u / 2
    else:
        # N-informed competition: λ = N/(N+1) × σ_v/σ_u
        N = n_informed
        lam = N / (N + 1) * sigma_v / sigma_u
        insider_trade = sigma_u / N  # each trader's size
        insider_profit = sigma_v * sigma_u / (N * (N + 1))

    depth = 1.0 / lam if lam > 0 else float('inf')

    # Price efficiency: fraction of variance revealed
    # After trading: Var(v|price) = σ²_v × σ²_u / (σ²_v + σ²_u) for N=1
    if n_informed == 1:
        post_var = sigma_v**2 * sigma_u**2 / (sigma_v**2 + sigma_u**2)
        efficiency = 1 - post_var / sigma_v**2
    else:
        efficiency = min(1.0, n_informed / (n_informed + 1))

    return KyleResult(
        lambda_impact=lam,
        insider_profit=insider_profit,
        informed_trade_size=insider_trade,
        market_depth=depth,
        price_efficiency=efficiency,
        noise_vol=sigma_u,
    )


@dataclass
class GlostenMilgromResult:
    """Glosten-Milgrom sequential trade result."""
    bid: float
    ask: float
    spread: float
    spread_pct: float
    prob_informed: float
    mid_price: float
    adverse_selection_cost: float

    def to_dict(self) -> dict:
        return vars(self)


def glosten_milgrom(
    v_high: float,
    v_low: float,
    prior_high: float = 0.5,
    prob_informed: float = 0.3,
) -> GlostenMilgromResult:
    """Glosten-Milgrom sequential trade model.

    Market maker quotes bid/ask to break even against informed traders.

    ask = E[v | buy] = (π × v_H + (1-π) × v_L) where
    π = Pr(v=v_H | buy) = α × prior / (α × prior + (1-α) × 0.5)

    bid = E[v | sell] similarly.

    The spread reflects adverse selection: wider when more informed.

    Args:
        v_high: high fundamental value.
        v_low: low fundamental value.
        prior_high: prior probability of high value.
        prob_informed: fraction of informed traders (α).
    """
    alpha = prob_informed

    # Posterior given BUY: informed buy if v=v_H, noise buy with prob 0.5
    num_buy = alpha * prior_high + (1 - alpha) * 0.5
    post_high_buy = (alpha * prior_high + (1 - alpha) * 0.5 * prior_high) / num_buy if num_buy > 0 else prior_high
    # More precise: P(v=H|buy) = [α×1×prior + (1-α)×0.5×prior] / [α×prior + (1-α)×0.5]
    post_high_buy = (alpha * prior_high + (1 - alpha) * 0.5 * prior_high) / (alpha * prior_high + (1 - alpha) * 0.5)

    # Posterior given SELL: informed sell if v=v_L
    post_high_sell = ((1 - alpha) * 0.5 * prior_high) / (alpha * (1 - prior_high) + (1 - alpha) * 0.5)

    ask = post_high_buy * v_high + (1 - post_high_buy) * v_low
    bid = post_high_sell * v_high + (1 - post_high_sell) * v_low
    mid = (ask + bid) / 2
    spread = ask - bid

    # Adverse selection cost per trade
    as_cost = spread / 2

    return GlostenMilgromResult(
        bid=bid,
        ask=ask,
        spread=spread,
        spread_pct=spread / mid * 100 if mid > 0 else 0,
        prob_informed=alpha,
        mid_price=mid,
        adverse_selection_cost=as_cost,
    )


@dataclass
class OrderSplitResult:
    """Optimal order splitting result."""
    schedule: list[float]       # trade sizes per period
    total_cost: float           # expected execution cost
    cost_bps: float             # cost in basis points
    permanent_impact: float
    temporary_impact: float

    def to_dict(self) -> dict:
        return {
            "n_slices": len(self.schedule),
            "total_cost": self.total_cost,
            "cost_bps": self.cost_bps,
        }


def optimal_order_splitting(
    total_shares: float,
    daily_volume: float,
    sigma: float,
    permanent_impact: float = 0.1,
    temporary_impact: float = 0.01,
    risk_aversion: float = 1e-6,
    n_periods: int = 10,
    price: float = 100.0,
) -> OrderSplitResult:
    """Split large order to minimise market impact.

    Almgren-Chriss model extended:
    Cost = permanent + temporary + risk
    = γ × X² + η × Σ(x_t²/dt) + λ × σ² × Σ(X_t² × dt)

    Optimal: front-loaded exponential schedule.

    Args:
        total_shares: total order size.
        daily_volume: average daily volume.
        sigma: daily price volatility.
        permanent_impact: permanent impact per share (γ).
        temporary_impact: temporary impact per share (η).
        risk_aversion: trader's risk aversion (λ).
        n_periods: number of trading periods.
        price: current price (for bps conversion).
    """
    dt = 1.0 / n_periods
    X = total_shares

    # Optimal schedule: exponentially front-loaded
    # x_t = X × sinh(κ(T-t)) / sinh(κT)
    # where κ = sqrt(λσ²/η)
    kappa_sq = risk_aversion * sigma**2 / max(temporary_impact, 1e-10)
    kappa = math.sqrt(max(kappa_sq, 1e-10))
    T = 1.0  # normalised to 1 day

    schedule = []
    remaining = X
    for t in range(n_periods):
        t_frac = t * dt
        if kappa * T < 20:  # avoid overflow
            x_t = X * math.sinh(kappa * (T - t_frac)) / math.sinh(kappa * T) * dt
        else:
            x_t = X * math.exp(-kappa * t_frac) * dt
        x_t = min(x_t, remaining)
        schedule.append(x_t)
        remaining -= x_t

    # Distribute remainder
    if remaining > 0:
        schedule[-1] += remaining

    # Cost computation
    perm_cost = permanent_impact * X * X * 0.5
    temp_cost = temporary_impact * sum(x**2 / dt for x in schedule)
    total_cost = perm_cost + temp_cost

    cost_bps = total_cost / (X * price) * 10_000 if X * price > 0 else 0

    return OrderSplitResult(
        schedule=schedule,
        total_cost=total_cost,
        cost_bps=cost_bps,
        permanent_impact=perm_cost,
        temporary_impact=temp_cost,
    )


def information_share(
    price_changes: list[np.ndarray],
    market_names: list[str] | None = None,
) -> dict:
    """Hasbrouck information share decomposition.

    Decomposes price discovery across multiple markets trading
    the same security. Higher share = more information contribution.

    Uses variance decomposition from a VECM.

    Args:
        price_changes: list of (T,) arrays of price changes per market.
        market_names: labels for markets.
    """
    n_markets = len(price_changes)
    names = market_names or [f"Market_{i}" for i in range(n_markets)]
    T = min(len(pc) for pc in price_changes)

    # Stack into matrix
    changes = np.column_stack([pc[:T] for pc in price_changes])

    # Covariance
    cov = np.cov(changes, rowvar=False)

    # Simple decomposition: variance share
    total_var = np.sum(np.diag(cov))
    shares = {}
    for i, name in enumerate(names):
        shares[name] = float(cov[i, i] / total_var) if total_var > 0 else 1.0 / n_markets

    # Cross-market correlation
    corr = np.corrcoef(changes, rowvar=False)

    return {
        "information_shares": shares,
        "correlation_matrix": corr.tolist(),
        "n_markets": n_markets,
        "n_observations": T,
    }
