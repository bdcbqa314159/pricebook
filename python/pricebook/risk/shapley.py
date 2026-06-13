"""Shapley value for fair capital allocation and risk attribution.

The Shapley value is the unique allocation satisfying efficiency, symmetry,
dummy, and additivity — stronger than Euler allocation (which only
satisfies efficiency + linearity).

    from pricebook.risk.shapley import (
        shapley_value, shapley_sampling, ShapleyResult,
    )

References:
    Shapley (1953). A Value for N-Person Games. Annals of Mathematics Studies.
    Denault (2001). Coherent Allocation of Risk Capital. Journal of Risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class ShapleyResult:
    """Result of Shapley value computation.

    Fix T4-RISK20: added optional ``diversification`` field carrying
    the per-player {shapley_allocation, standalone_risk,
    diversification_benefit} dict computed by
    ``shapley_capital_allocation``.  Pre-fix that function built the
    dict but discarded it, contradicting its docstring promise of
    "diversification benefit" reporting.
    """
    players: list[str]
    values: dict[str, float]     # {player: Shapley value}
    total_value: float           # v(grand coalition)
    method: str                  # "exact" or "sampling"
    n_coalitions_evaluated: int
    diversification: dict[str, dict] | None = None  # per-player enrichment

    @property
    def is_efficient(self) -> bool:
        """Check Σ φᵢ = v(N) (efficiency axiom)."""
        return abs(sum(self.values.values()) - self.total_value) < 1e-6

    def to_dict(self) -> dict:
        return {
            "players": self.players,
            "values": self.values,
            "total_value": self.total_value,
            "method": self.method,
            "is_efficient": self.is_efficient,
            "diversification": self.diversification,
        }


def shapley_value(
    v: callable,
    players: list[str],
) -> ShapleyResult:
    """Exact Shapley value (enumerates all 2^N coalitions).

    φᵢ = Σ_{S⊆N\\{i}} |S|!(|N|-|S|-1)!/|N|! × [v(S∪{i}) - v(S)]

    Feasible for N ≤ 15 (~32K coalitions).

    Args:
        v: characteristic function. v(frozenset of player names) → float.
           v(frozenset()) should return 0 (empty coalition has no value).
        players: list of player names.

    Returns:
        ShapleyResult with per-player values.
    """
    n = len(players)
    if n == 0:
        return ShapleyResult([], {}, 0.0, "exact", 0)
    if n > 20:
        raise ValueError(f"Exact Shapley infeasible for N={n} (>20). Use shapley_sampling().")

    player_set = frozenset(players)
    total = v(player_set)
    phi = {p: 0.0 for p in players}
    n_evals = 0

    for i, player in enumerate(players):
        others = [p for p in players if p != player]
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                S = frozenset(subset)
                S_with_i = S | {player}
                marginal = v(S_with_i) - v(S)
                weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)
                          / math.factorial(n))
                phi[player] += weight * marginal
                n_evals += 2  # two v() calls per marginal

    return ShapleyResult(
        players=players,
        values=phi,
        total_value=total,
        method="exact",
        n_coalitions_evaluated=n_evals,
    )


def shapley_sampling(
    v: callable,
    players: list[str],
    n_samples: int = 10_000,
    seed: int = 42,
) -> ShapleyResult:
    """Monte Carlo approximation of Shapley value for large N.

    Randomly permutes players, computes marginal contribution of each
    player in their position. Average over many permutations.

    φᵢ ≈ (1/M) Σ_{π} [v(S^π_i ∪ {i}) - v(S^π_i)]

    where S^π_i = set of players before i in permutation π.

    Args:
        v: characteristic function.
        players: list of player names.
        n_samples: number of random permutations.
        seed: random seed.
    """
    n = len(players)
    if n == 0:
        return ShapleyResult([], {}, 0.0, "sampling", 0)

    rng = np.random.default_rng(seed)
    phi = {p: 0.0 for p in players}
    n_evals = 0

    player_arr = np.array(players)
    total = v(frozenset(players))

    for _ in range(n_samples):
        perm = rng.permutation(n)
        S = frozenset()
        for idx in perm:
            player = player_arr[idx]
            v_with = v(S | {player})
            v_without = v(S)
            phi[player] += (v_with - v_without) / n_samples
            S = S | {player}
            n_evals += 2

    return ShapleyResult(
        players=players,
        values=phi,
        total_value=total,
        method="sampling",
        n_coalitions_evaluated=n_evals,
    )


def shapley_capital_allocation(
    desk_names: list[str],
    standalone_risks: dict[str, float],
    portfolio_risk_fn: callable,
) -> ShapleyResult:
    """Shapley-based capital allocation across desks.

    The characteristic function v(S) = portfolio risk of subset S.

    Args:
        desk_names: list of desk/sub-portfolio names.
        standalone_risks: {desk: standalone risk} (for reporting only).
        portfolio_risk_fn: callable(frozenset of desks) → portfolio risk.
    """
    result = shapley_value(portfolio_risk_fn, desk_names)

    # Enrich with standalone comparison — Fix T4-RISK20: pre-fix built
    # this dict then immediately threw it away by returning ``result``.
    # The function docstring promised diversification info that the
    # caller never received.  Now attached as ``result.diversification``.
    enriched_values: dict[str, dict] = {}
    for desk in desk_names:
        enriched_values[desk] = {
            "shapley_allocation": result.values[desk],
            "standalone_risk": standalone_risks.get(desk, 0.0),
            "diversification_benefit": standalone_risks.get(desk, 0.0) - result.values[desk],
        }
    result.diversification = enriched_values

    return result
