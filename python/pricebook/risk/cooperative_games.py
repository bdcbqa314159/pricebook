"""Cooperative game theory for netting sets and collateral pools.

    from pricebook.risk.cooperative_games import (
        CooperativeGame, NettingSetGame, CollateralPoolGame,
    )

References:
    Shapley (1953). A Value for N-Person Games.
    Schmeidler (1969). The Nucleolus of a Characteristic Function Game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pricebook.risk.shapley import shapley_value, shapley_sampling, ShapleyResult


@dataclass
class CoreCheckResult:
    """Result of core stability check."""
    is_in_core: bool
    max_violation: float         # max excess over any coalition
    violating_coalition: frozenset | None

    def to_dict(self) -> dict:
        return {
            "is_in_core": self.is_in_core,
            "max_violation": self.max_violation,
        }


class CooperativeGame:
    """N-player cooperative game with characteristic function.

    Args:
        players: list of player names.
        characteristic_fn: v(frozenset) → float (value of coalition).
    """

    def __init__(self, players: list[str], characteristic_fn: callable):
        self.players = players
        self.v = characteristic_fn
        self.n = len(players)

    def shapley(self) -> ShapleyResult:
        """Compute Shapley value allocation."""
        if self.n <= 15:
            return shapley_value(self.v, self.players)
        return shapley_sampling(self.v, self.players)

    def core_check(self, allocation: dict[str, float]) -> CoreCheckResult:
        """Check if an allocation is in the core.

        An allocation x is in the core if:
        1. Σ xᵢ = v(N) (efficiency)
        2. Σ_{i∈S} xᵢ ≥ v(S) for all coalitions S (stability)
        """
        from itertools import combinations

        total_alloc = sum(allocation.values())
        grand = self.v(frozenset(self.players))
        if abs(total_alloc - grand) > 1e-6:
            return CoreCheckResult(False, abs(total_alloc - grand), None)

        max_violation = 0.0
        worst_coalition = None

        for size in range(1, self.n):
            for subset in combinations(self.players, size):
                S = frozenset(subset)
                v_S = self.v(S)
                alloc_S = sum(allocation.get(p, 0) for p in S)
                violation = v_S - alloc_S
                if violation > max_violation:
                    max_violation = violation
                    worst_coalition = S

        return CoreCheckResult(
            is_in_core=max_violation <= 1e-6,
            max_violation=max_violation,
            violating_coalition=worst_coalition,
        )

    def grand_coalition_value(self) -> float:
        return self.v(frozenset(self.players))

    def to_dict(self) -> dict:
        return {
            "players": self.players,
            "n_players": self.n,
            "grand_coalition_value": self.grand_coalition_value(),
        }


class NettingSetGame(CooperativeGame):
    """Cooperative game where v(S) = netting benefit of coalition S.

    v(S) = Σ_{i∈S} standalone_exposure_i - netted_exposure(S)

    The netting benefit is the reduction in total exposure from netting.
    """

    def __init__(
        self,
        counterparties: list[str],
        standalone_exposures: dict[str, float],
        netted_exposure_fn: callable,
    ):
        """
        Args:
            counterparties: list of counterparty names.
            standalone_exposures: {cp: bilateral exposure without netting}.
            netted_exposure_fn: callable(frozenset) → netted exposure of the set.
        """
        self.standalone = standalone_exposures

        def v(S):
            if not S:
                return 0.0
            gross = sum(standalone_exposures.get(p, 0) for p in S)
            net = netted_exposure_fn(S)
            return gross - net  # benefit = reduction

        super().__init__(counterparties, v)


class CollateralPoolGame(CooperativeGame):
    """Cooperative game where v(S) = funding cost reduction from shared pool.

    v(S) = Σ_{i∈S} individual_cost_i - pooled_cost(S)
    """

    def __init__(
        self,
        participants: list[str],
        individual_costs: dict[str, float],
        pooled_cost_fn: callable,
    ):
        self.individual_costs = individual_costs

        def v(S):
            if not S:
                return 0.0
            individual = sum(individual_costs.get(p, 0) for p in S)
            pooled = pooled_cost_fn(S)
            return individual - pooled  # benefit = saving

        super().__init__(participants, v)
