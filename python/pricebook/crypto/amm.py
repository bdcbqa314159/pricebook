"""Automated Market Maker (AMM) pricing.

Uniswap v2 (constant product), v3 (concentrated liquidity),
Curve (stableswap). Slippage, price impact, LP returns.

* :func:`uniswap_v2_price` — constant product x×y=k pricing.
* :func:`uniswap_v3_price` — concentrated liquidity with tick math.
* :func:`curve_stableswap` — Curve's StableSwap invariant.
* :func:`price_impact` — slippage for a given trade size.
* :func:`lp_return` — LP fee income vs holding.

References:
    Adams et al., *Uniswap v2 Core*, 2020.
    Adams et al., *Uniswap v3 Core*, 2021.
    Egorov, *StableSwap — Efficient Mechanism for Stablecoin Liquidity*, 2019.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AMMQuote:
    """AMM trade quote."""
    amount_in: float
    amount_out: float
    price: float                # effective execution price
    price_impact_pct: float     # vs mid price
    fee: float                  # fee charged
    pool_type: str              # "v2", "v3", "curve"

    def to_dict(self) -> dict:
        return dict(vars(self))


# ═══════════════════════════════════════════════════════════════
# Uniswap v2: x × y = k
# ═══════════════════════════════════════════════════════════════

def uniswap_v2_price(
    reserve_x: float,
    reserve_y: float,
    amount_in: float,
    fee: float = 0.003,
    buy_y: bool = True,
) -> AMMQuote:
    """Uniswap v2 constant product pricing.

    Invariant: x × y = k (after fees).
    amount_out = y − k / (x + amount_in × (1 − fee))

    Args:
        reserve_x: pool reserve of token X.
        reserve_y: pool reserve of token Y.
        amount_in: amount of X being swapped.
        fee: swap fee (0.3% default).
        buy_y: True = swap X for Y.
    """
    if not buy_y:
        reserve_x, reserve_y = reserve_y, reserve_x

    mid_price = reserve_y / reserve_x if reserve_x > 0 else 0
    k = reserve_x * reserve_y

    amount_in_after_fee = amount_in * (1 - fee)
    new_reserve_x = reserve_x + amount_in_after_fee
    new_reserve_y = k / new_reserve_x if new_reserve_x > 0 else 0
    amount_out = reserve_y - new_reserve_y

    exec_price = amount_out / amount_in if amount_in > 0 else 0
    impact = (mid_price - exec_price) / mid_price * 100 if mid_price > 0 else 0
    fee_amount = amount_in * fee

    return AMMQuote(
        amount_in=amount_in,
        amount_out=max(amount_out, 0),
        price=exec_price,
        price_impact_pct=abs(impact),
        fee=fee_amount,
        pool_type="v2",
    )


# ═══════════════════════════════════════════════════════════════
# Uniswap v3: concentrated liquidity
# ═══════════════════════════════════════════════════════════════

def uniswap_v3_price(
    liquidity: float,
    sqrt_price: float,
    amount_in: float,
    sqrt_price_lower: float,
    sqrt_price_upper: float,
    fee: float = 0.003,
) -> AMMQuote:
    """Uniswap v3 concentrated liquidity pricing.

    Liquidity is concentrated in [sqrt_price_lower, sqrt_price_upper].
    Within range: x × y = L² (virtual reserves).

    amount_out = L × (1/√P_new − 1/√P_old) for buying token Y.

    Args:
        liquidity: L, concentrated liquidity.
        sqrt_price: current √P (√(Y/X)).
        amount_in: amount of X being swapped.
        sqrt_price_lower: lower tick √price.
        sqrt_price_upper: upper tick √price.
        fee: swap fee tier (0.05%, 0.3%, 1%).
    """
    mid_price = sqrt_price ** 2

    amount_in_after_fee = amount_in * (1 - fee)

    # New sqrt price after swap
    # Δ√P = amount_in / L
    delta_sqrt_p = amount_in_after_fee / liquidity if liquidity > 0 else 0
    new_sqrt_price = sqrt_price + delta_sqrt_p

    # Clamp to range
    new_sqrt_price = min(new_sqrt_price, sqrt_price_upper)

    # Amount out (in token Y)
    amount_out = liquidity * (1 / sqrt_price - 1 / new_sqrt_price) if new_sqrt_price > 0 and sqrt_price > 0 else 0
    amount_out = max(amount_out, 0)

    exec_price = amount_out / amount_in if amount_in > 0 else 0
    impact = abs(mid_price - exec_price) / mid_price * 100 if mid_price > 0 else 0

    return AMMQuote(
        amount_in=amount_in,
        amount_out=amount_out,
        price=exec_price,
        price_impact_pct=impact,
        fee=amount_in * fee,
        pool_type="v3",
    )


# ═══════════════════════════════════════════════════════════════
# Curve: StableSwap
# ═══════════════════════════════════════════════════════════════

def curve_stableswap(
    reserves: list[float],
    amount_in: float,
    token_in: int = 0,
    token_out: int = 1,
    A: float = 100.0,
    fee: float = 0.0004,
) -> AMMQuote:
    """Curve StableSwap invariant pricing.

    Combines constant-product and constant-sum:
    A×n^n×ΣD_i + D = A×n^n×D + D^{n+1}/(n^n × Π D_i)

    For stablecoins near peg: very low slippage.
    As imbalance grows: approaches constant product.

    Args:
        reserves: pool reserves per token.
        amount_in: amount of token_in to swap.
        token_in: index of input token.
        token_out: index of output token.
        A: amplification coefficient (higher = more stable).
        fee: swap fee.
    """
    n = len(reserves)
    D = _compute_D(reserves, A, n)

    # New reserve of token_in after swap
    new_reserves = list(reserves)
    new_reserves[token_in] += amount_in * (1 - fee)

    # Solve for new reserve of token_out
    y = _get_y(new_reserves, token_out, D, A, n)
    amount_out = reserves[token_out] - y

    mid_price = 1.0  # stablecoins should be 1:1
    exec_price = amount_out / amount_in if amount_in > 0 else 0
    impact = abs(1 - exec_price) * 100

    return AMMQuote(
        amount_in=amount_in,
        amount_out=max(amount_out, 0),
        price=exec_price,
        price_impact_pct=impact,
        fee=amount_in * fee,
        pool_type="curve",
    )


def _compute_D(reserves: list[float], A: float, n: int) -> float:
    """Newton's method for StableSwap invariant D."""
    S = sum(reserves)
    D = S
    Ann = A * n**n

    for _ in range(256):
        D_P = D
        for r in reserves:
            D_P = D_P * D / (n * r) if r > 0 else D_P
        D_prev = D
        D = (Ann * S + D_P * n) * D / ((Ann - 1) * D + (n + 1) * D_P)
        if abs(D - D_prev) < 1e-10:
            break
    return D


def _get_y(reserves: list[float], j: int, D: float, A: float, n: int) -> float:
    """Solve for y_j given other reserves and D."""
    Ann = A * n**n
    c = D
    S = 0.0
    for i, r in enumerate(reserves):
        if i == j:
            continue
        S += r
        c = c * D / (n * r) if r > 0 else c
    c = c * D / (Ann * n)
    b = S + D / Ann

    y = D
    for _ in range(256):
        y_prev = y
        y = (y * y + c) / (2 * y + b - D)
        if abs(y - y_prev) < 1e-10:
            break
    return y


# ═══════════════════════════════════════════════════════════════
# Analytics
# ═══════════════════════════════════════════════════════════════

def price_impact(
    reserve_x: float,
    reserve_y: float,
    trade_size: float,
    fee: float = 0.003,
) -> float:
    """Price impact as percentage for a v2 trade."""
    quote = uniswap_v2_price(reserve_x, reserve_y, trade_size, fee)
    return quote.price_impact_pct


@dataclass
class LPReturnResult:
    """LP return analysis."""
    fee_income: float           # total fees earned
    impermanent_loss: float     # IL (negative = loss)
    net_return: float           # fees − IL
    hodl_value: float           # value if just held
    lp_value: float             # value as LP

    def to_dict(self) -> dict:
        return dict(vars(self))


def lp_return_v2(
    initial_x: float,
    initial_y: float,
    final_price_ratio: float,
    fee_income: float = 0.0,
) -> LPReturnResult:
    """LP return for Uniswap v2 (constant product).

    IL = 2√r / (1 + r) − 1  where r = final_price / initial_price.

    Args:
        initial_x: initial deposit of token X.
        initial_y: initial deposit of token Y.
        final_price_ratio: P_final / P_initial.
        fee_income: cumulative fee income.
    """
    r = final_price_ratio
    initial_value = initial_x + initial_y  # at initial prices

    # HODL value
    hodl_value = initial_x + initial_y * r

    # LP value (constant product)
    il_factor = 2 * math.sqrt(r) / (1 + r)
    lp_value_no_fees = initial_value * il_factor
    il = lp_value_no_fees - hodl_value

    lp_value = lp_value_no_fees + fee_income
    net = lp_value - hodl_value

    return LPReturnResult(
        fee_income=fee_income,
        impermanent_loss=il,
        net_return=net,
        hodl_value=hodl_value,
        lp_value=lp_value,
    )


# ═══════════════════════════════════════════════════════════════
# CD5: Balancer Weighted Pools, MEV Sandwich Cost
# ═══════════════════════════════════════════════════════════════

def balancer_weighted_price(
    reserves: list[float],
    weights: list[float],
    amount_in: float,
    token_in: int = 0,
    token_out: int = 1,
    fee: float = 0.003,
) -> AMMQuote:
    """Balancer weighted pool: Π(R_i^{w_i}) = k.

    Amount out = R_out × (1 − (R_in/(R_in+in))^{w_in/w_out}).

    Args:
        reserves: reserves per token.
        weights: normalised weights (sum to 1).
    """
    r_in, r_out = reserves[token_in], reserves[token_out]
    w_in, w_out = weights[token_in], weights[token_out]
    mid_price = (r_out / w_out) / (r_in / w_in)
    in_after_fee = amount_in * (1 - fee)
    ratio = r_in / (r_in + in_after_fee)
    amount_out = r_out * (1 - ratio ** (w_in / w_out))
    exec_price = amount_out / amount_in if amount_in > 0 else 0
    impact = abs(mid_price - exec_price) / mid_price * 100 if mid_price > 0 else 0
    return AMMQuote(amount_in, max(amount_out, 0), exec_price, impact,
                     amount_in * fee, "balancer")


def mev_sandwich_cost(
    trade_size: float,
    pool_reserves: float,
    gas_cost_usd: float = 5.0,
) -> dict:
    """Estimate MEV sandwich attack cost.

    Your loss ≈ (trade_size / pool_reserves)² × pool_reserves × 0.5.
    Sandwich profitable when loss > 2 × gas_cost.
    Safe trade size: where MEV profit < gas cost.
    """
    impact_ratio = trade_size / pool_reserves if pool_reserves > 0 else 0
    expected_loss = impact_ratio * trade_size * 0.5
    safe_size = math.sqrt(2 * gas_cost_usd * pool_reserves) if pool_reserves > 0 else 0
    return {
        "expected_loss": expected_loss,
        "safe_trade_size": safe_size,
        "profitable_for_attacker": expected_loss > 2 * gas_cost_usd,
    }
