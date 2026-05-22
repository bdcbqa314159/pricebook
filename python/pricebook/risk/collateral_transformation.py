"""Collateral transformation — upgrade/downgrade optimization.

    from pricebook.risk.collateral_transformation import (
        optimise_transformation, transformation_cost, TransformationResult,
    )

References:
    Singh (2011). Velocity of Pledged Collateral. IMF Working Paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class TransformationTrade:
    """A single collateral transformation trade."""
    give_asset: str              # asset pledged
    receive_asset: str           # asset received
    give_haircut: float          # haircut on pledged
    receive_haircut: float       # haircut on received
    repo_spread_bp: float        # repo rate spread to pay
    xccy_basis_bp: float = 0.0   # cross-currency basis if different currencies
    capital_charge_bp: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class TransformationResult:
    """Result of collateral transformation optimization."""
    trades: list[TransformationTrade]
    total_cost_bp: float
    haircut_improvement: float   # reduction in effective haircut
    capital_cost_bp: float
    is_profitable: bool

    def to_dict(self) -> dict:
        return {
            "n_trades": len(self.trades),
            "total_cost_bp": self.total_cost_bp,
            "haircut_improvement": self.haircut_improvement,
            "capital_cost_bp": self.capital_cost_bp,
            "is_profitable": self.is_profitable,
        }


def transformation_cost(
    give_haircut: float,
    receive_haircut: float,
    repo_spread_bp: float,
    xccy_basis_bp: float = 0.0,
    capital_charge_bp: float = 0.0,
    holding_days: int = 90,
) -> dict:
    """All-in cost of a single collateral transformation.

    Cost = repo_spread + xccy_basis + capital_charge - haircut_benefit

    The haircut benefit comes from receiving lower-haircut collateral
    that can be pledged more efficiently elsewhere.
    """
    denom = 360.0
    t = holding_days / denom

    repo_cost = repo_spread_bp * t
    basis_cost = xccy_basis_bp * t
    cap_cost = capital_charge_bp * t
    haircut_benefit = (give_haircut - receive_haircut) * 10_000 * t

    total = repo_cost + basis_cost + cap_cost
    net = total - max(haircut_benefit, 0)

    return {
        "repo_cost_bp": repo_cost,
        "xccy_basis_cost_bp": basis_cost,
        "capital_cost_bp": cap_cost,
        "haircut_benefit_bp": haircut_benefit,
        "total_cost_bp": total,
        "net_cost_bp": net,
        "is_profitable": net < 0,
    }


def optimise_transformation(
    available_collateral: list[dict],
    target_collateral_type: str,
    target_haircut: float,
    max_notional: float,
) -> TransformationResult:
    """Optimize collateral upgrade via repo chain.

    Finds the cheapest path to convert available collateral into
    target-quality collateral.

    Args:
        available_collateral: list of {asset, haircut, notional, repo_spread_bp}.
        target_collateral_type: e.g. "HQLA_L1".
        target_haircut: haircut of target collateral.
        max_notional: maximum transformation notional.
    """
    if not available_collateral:
        return TransformationResult([], 0.0, 0.0, 0.0, False)

    # Sort by cost-effectiveness: cheapest spread first
    sorted_coll = sorted(available_collateral, key=lambda c: c.get("repo_spread_bp", 0))

    trades = []
    remaining = max_notional
    total_cost = 0.0
    total_haircut_improvement = 0.0

    for coll in sorted_coll:
        if remaining <= 0:
            break

        use_notional = min(coll.get("notional", remaining), remaining)
        spread = coll.get("repo_spread_bp", 0)
        give_haircut = coll["haircut"]

        cost = transformation_cost(give_haircut, target_haircut, spread)
        trade = TransformationTrade(
            give_asset=coll["asset"],
            receive_asset=target_collateral_type,
            give_haircut=give_haircut,
            receive_haircut=target_haircut,
            repo_spread_bp=spread,
        )
        trades.append(trade)

        total_cost += cost["total_cost_bp"] * use_notional / 10_000
        total_haircut_improvement += (give_haircut - target_haircut) * use_notional
        remaining -= use_notional

    return TransformationResult(
        trades=trades,
        total_cost_bp=total_cost,
        haircut_improvement=total_haircut_improvement,
        capital_cost_bp=0.0,
        is_profitable=total_cost < total_haircut_improvement * 100,
    )


def funding_arbitrage(
    secured_rates: dict[str, float],
    unsecured_rates: dict[str, float],
    haircuts: dict[str, float],
) -> list[dict]:
    """Identify mispriced collateral vs funding value across asset classes.

    If secured_rate + haircut_cost < unsecured_rate, there's an arbitrage
    from pledging that collateral.
    """
    results = []
    for asset, sec_rate in secured_rates.items():
        unsec = unsecured_rates.get(asset, list(unsecured_rates.values())[0])
        h = haircuts.get(asset, 0.05)

        # Haircut cost: need to fund the haircut portion unsecured
        blended = (1 - h) * sec_rate + h * unsec
        spread_to_unsecured = (unsec - blended) * 10_000

        results.append({
            "asset": asset,
            "secured_rate": sec_rate,
            "unsecured_rate": unsec,
            "haircut": h,
            "blended_rate": blended,
            "arbitrage_bp": spread_to_unsecured,
            "is_arbitrage": spread_to_unsecured > 5,
        })

    return sorted(results, key=lambda r: -r["arbitrage_bp"])
