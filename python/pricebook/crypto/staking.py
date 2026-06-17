"""Staking economics: validator yield, slashing, liquid staking.

* :func:`eth_staking_yield` — ETH PoS validator yield model.
* :func:`liquid_staking_premium` — stETH/rETH discount/premium.
* :func:`slashing_risk` — expected loss from slashing events.

References:
    Ethereum Foundation, *Proof of Stake FAQ*.
    Lido, *stETH Mechanics*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StakingYieldResult:
    """Staking yield analysis."""
    base_yield: float           # protocol issuance yield
    mev_yield: float            # MEV/tips component
    total_yield: float          # base + MEV
    net_yield: float            # after operator fee
    operator_fee: float
    total_staked: float
    participation_rate: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def eth_staking_yield(
    total_eth_staked: float = 30_000_000,
    total_eth_supply: float = 120_000_000,
    base_reward_factor: float = 64,
    mev_per_block_eth: float = 0.05,
    blocks_per_year: int = 2_628_000,
    operator_fee: float = 0.10,
) -> StakingYieldResult:
    """ETH Proof of Stake validator yield model.

    Base yield decreases with more validators (√N scaling):
    yield ≈ base_reward_factor / √(total_staked / 32)

    MEV/tips are additive on top.

    Args:
        total_eth_staked: total ETH staked across all validators.
        total_eth_supply: total ETH supply.
        base_reward_factor: protocol constant (64 in Ethereum).
        mev_per_block_eth: average MEV + tips per block.
        blocks_per_year: ~2.63M for 12s blocks.
        operator_fee: staking provider fee (e.g. 10% for Lido).
    """
    n_validators = total_eth_staked / 32
    participation = total_eth_staked / total_eth_supply

    # Base yield: inversely proportional to √validators
    if n_validators > 0:
        base_yield = base_reward_factor / math.sqrt(n_validators) / 32
    else:
        base_yield = 0

    # MEV yield
    mev_total = mev_per_block_eth * blocks_per_year
    mev_yield = mev_total / total_eth_staked if total_eth_staked > 0 else 0

    total = base_yield + mev_yield
    net = total * (1 - operator_fee)

    return StakingYieldResult(
        base_yield=base_yield,
        mev_yield=mev_yield,
        total_yield=total,
        net_yield=net,
        operator_fee=operator_fee,
        total_staked=total_eth_staked,
        participation_rate=participation,
    )


@dataclass
class LiquidStakingResult:
    """Liquid staking derivative analysis."""
    derivative_price: float     # e.g. stETH price in ETH
    nav: float                  # underlying ETH value per token
    premium_pct: float          # premium (positive) or discount (negative)
    implied_yield_pickup: float # extra yield from buying at discount

    def to_dict(self) -> dict:
        return dict(vars(self))


def liquid_staking_premium(
    derivative_price: float,
    nav: float,
    staking_yield: float,
    time_to_exit_days: float = 7.0,
) -> LiquidStakingResult:
    """Liquid staking derivative premium/discount analysis.

    stETH, rETH, etc. can trade at premium or discount to NAV.
    Discount = opportunity: buy cheap, earn staking yield on full NAV.

    premium = (price / nav − 1) × 100
    If trading at 0.99 ETH (1% discount):
    implied_yield_pickup = discount × (365 / exit_days) extra yield.

    Args:
        derivative_price: market price of derivative (e.g. stETH in ETH).
        nav: net asset value (underlying ETH per token).
        staking_yield: current annualised staking yield.
        time_to_exit_days: days to unstake (exit queue).
    """
    premium = (derivative_price / nav - 1) * 100 if nav > 0 else 0
    discount = -premium / 100  # positive when trading cheap

    # Yield pickup from buying at discount
    if time_to_exit_days > 0 and discount > 0:
        implied_pickup = discount * 365 / time_to_exit_days
    else:
        implied_pickup = 0

    return LiquidStakingResult(
        derivative_price=derivative_price,
        nav=nav,
        premium_pct=premium,
        implied_yield_pickup=implied_pickup,
    )


@dataclass
class SlashingRiskResult:
    """Slashing risk analysis."""
    expected_loss_pct: float    # expected annual loss from slashing
    max_loss_pct: float         # maximum single slashing penalty
    prob_slashing: float        # annual probability
    correlation_penalty: float  # additional penalty if many slash simultaneously

    def to_dict(self) -> dict:
        return dict(vars(self))


def slashing_risk(
    prob_slashing_annual: float = 0.001,
    base_penalty_eth: float = 1.0,
    stake_per_validator: float = 32.0,
    correlation_factor: float = 0.0,
    total_validators_slashed_pct: float = 0.0,
) -> SlashingRiskResult:
    """Expected loss from slashing events.

    ETH slashing: minimum 1/32 of stake. If many validators slash
    simultaneously, penalty scales with % of network slashed.

    correlated_penalty = base + 3 × (total_slashed / total_validators)

    Args:
        prob_slashing_annual: annual probability of being slashed.
        base_penalty_eth: minimum penalty in ETH.
        stake_per_validator: ETH per validator (32).
        correlation_factor: fraction of validators slashed simultaneously.
        total_validators_slashed_pct: % of all validators involved.
    """
    base_loss = base_penalty_eth / stake_per_validator
    corr_penalty = 3 * total_validators_slashed_pct  # proportional penalty
    max_loss = min(base_loss + corr_penalty, 1.0)  # capped at 100%

    expected_loss = prob_slashing_annual * (base_loss + corr_penalty)

    return SlashingRiskResult(
        expected_loss_pct=expected_loss * 100,
        max_loss_pct=max_loss * 100,
        prob_slashing=prob_slashing_annual,
        correlation_penalty=corr_penalty * 100,
    )
