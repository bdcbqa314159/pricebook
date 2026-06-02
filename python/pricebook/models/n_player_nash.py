"""N-player Nash equilibrium: fictitious play, Lemke-Howson support.

* :func:`fictitious_play` — iterative best-response for N players.
* :func:`lemke_howson_2p` — Lemke-Howson for bimatrix games.
* :func:`correlated_equilibrium` — LP for correlated equilibrium.

References:
    Nash, *Equilibrium Points in N-Person Games*, PNAS, 1950.
    Lemke & Howson, *Equilibrium Points of Bimatrix Games*, SIAM J, 1964.
    Brown, *Iterative Solution of Games by Fictitious Play*, 1951.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class NashEquilibrium:
    """N-player Nash equilibrium result."""
    strategies: list[np.ndarray]    # mixed strategy per player
    payoffs: list[float]            # expected payoff per player
    n_players: int
    converged: bool
    iterations: int
    method: str

    def to_dict(self) -> dict:
        return {
            "n_players": self.n_players,
            "payoffs": self.payoffs,
            "converged": self.converged,
            "iterations": self.iterations,
            "method": self.method,
        }


def fictitious_play(
    payoff_matrices: list[np.ndarray],
    max_iter: int = 10_000,
    tol: float = 1e-6,
) -> NashEquilibrium:
    """Fictitious play for N-player games.

    Each player plays best response to the empirical frequency of
    opponents' past actions. Converges to Nash for 2-player
    zero-sum and potential games.

    Args:
        payoff_matrices: list of N matrices, each of shape
            (A₁ × A₂ × ... × Aₙ) giving player i's payoff.
        max_iter: maximum iterations.
        tol: convergence tolerance on strategy change.
    """
    n_players = len(payoff_matrices)
    actions = [m.shape[i] for i, m in enumerate(payoff_matrices)]

    # Initialise: uniform play counts
    counts = [np.ones(a) for a in actions]

    prev_strategies = [c / c.sum() for c in counts]

    for iteration in range(max_iter):
        # Each player best-responds to current empirical frequencies
        strategies = [c / c.sum() for c in counts]

        for p in range(n_players):
            # Expected payoff for each action of player p
            # given opponents' empirical strategies
            expected = np.zeros(actions[p])
            opp_strategies = [strategies[j] for j in range(n_players) if j != p]

            for a in range(actions[p]):
                # Compute expected payoff for action a
                # by marginalising over all opponent actions
                payoff = _marginalise(payoff_matrices[p], p, a, opp_strategies, n_players)
                expected[a] = payoff

            # Best response
            best = np.argmax(expected)
            counts[p][best] += 1

        # Convergence check
        new_strategies = [c / c.sum() for c in counts]
        max_change = max(
            float(np.max(np.abs(new_strategies[p] - prev_strategies[p])))
            for p in range(n_players)
        )
        prev_strategies = new_strategies

        if max_change < tol:
            strategies = new_strategies
            payoffs = _compute_payoffs(payoff_matrices, strategies)
            return NashEquilibrium(strategies, payoffs, n_players, True, iteration + 1, "fictitious_play")

    strategies = [c / c.sum() for c in counts]
    payoffs = _compute_payoffs(payoff_matrices, strategies)
    return NashEquilibrium(strategies, payoffs, n_players, False, max_iter, "fictitious_play")


def lemke_howson_2p(
    A: np.ndarray,
    B: np.ndarray,
) -> NashEquilibrium:
    """Lemke-Howson for 2-player bimatrix games.

    Simplified vertex enumeration: finds one Nash equilibrium
    via complementary pivoting.

    Args:
        A: (m × n) payoff matrix for player 1.
        B: (m × n) payoff matrix for player 2.
    """
    m, n = A.shape

    # Support enumeration: try all support pairs
    best_result = None
    best_gap = float('inf')

    for support_size_1 in range(1, m + 1):
        for support_size_2 in range(1, n + 1):
            if support_size_1 != support_size_2:
                continue  # supports must match in size for non-degenerate

            from itertools import combinations
            for s1 in combinations(range(m), support_size_1):
                for s2 in combinations(range(n), support_size_2):
                    result = _try_support(A, B, list(s1), list(s2))
                    if result is not None:
                        gap = result[2]
                        if gap < best_gap:
                            best_gap = gap
                            best_result = result

            if best_result is not None and best_gap < 1e-8:
                break
        if best_result is not None and best_gap < 1e-8:
            break

    if best_result is not None:
        p, q = best_result[0], best_result[1]
        payoff_1 = float(p @ A @ q)
        payoff_2 = float(p @ B @ q)
        return NashEquilibrium([p, q], [payoff_1, payoff_2], 2, True, 0, "support_enum")

    # Fallback: uniform
    p = np.ones(m) / m
    q = np.ones(n) / n
    return NashEquilibrium([p, q], [float(p @ A @ q), float(p @ B @ q)], 2, False, 0, "fallback")


def correlated_equilibrium(
    payoff_matrices: list[np.ndarray],
) -> NashEquilibrium:
    """Correlated equilibrium via LP (2-player).

    Maximises social welfare subject to incentive compatibility.

    Args:
        payoff_matrices: [A, B] where A, B are (m × n).
    """
    A, B = payoff_matrices[0], payoff_matrices[1]
    m, n = A.shape

    # Variables: π(i,j) for joint distribution
    n_vars = m * n

    # Objective: maximise social welfare Σ π(i,j) × (A(i,j) + B(i,j))
    c = -(A + B).flatten()  # negative for minimisation

    # IC constraints for player 1: for each i, i':
    # Σ_j π(i,j) × A(i,j) ≥ Σ_j π(i,j) × A(i',j)
    A_ub_rows = []
    b_ub_rows = []

    for i in range(m):
        for i_prime in range(m):
            if i == i_prime:
                continue
            row = np.zeros(n_vars)
            for j in range(n):
                row[i * n + j] = -(A[i, j] - A[i_prime, j])
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

    # IC for player 2
    for j in range(n):
        for j_prime in range(n):
            if j == j_prime:
                continue
            row = np.zeros(n_vars)
            for i in range(m):
                row[i * n + j] = -(B[i, j] - B[i, j_prime])
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    # Σ π = 1
    A_eq = np.ones((1, n_vars))
    b_eq = np.array([1.0])

    bounds = [(0, 1)] * n_vars

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        pi = result.x.reshape(m, n)
        p = pi.sum(axis=1)
        q = pi.sum(axis=0)
        payoff_1 = float(np.sum(pi * A))
        payoff_2 = float(np.sum(pi * B))
    else:
        p = np.ones(m) / m
        q = np.ones(n) / n
        payoff_1 = float(p @ A @ q)
        payoff_2 = float(p @ B @ q)

    return NashEquilibrium([p, q], [payoff_1, payoff_2], 2, result.success, 0, "correlated_eq")


# ---- Internal helpers ----

def _marginalise(payoff_matrix, player, action, opp_strategies, n_players):
    """Expected payoff for player taking action, marginalising opponents."""
    # For 2-player: simple matrix multiply
    if n_players == 2:
        if player == 0:
            return float(payoff_matrix[action, :] @ opp_strategies[0])
        else:
            return float(opp_strategies[0] @ payoff_matrix[:, action])

    # General N-player: tensor contraction (simplified for 3-player)
    shape = payoff_matrix.shape
    total = 0.0
    opp_idx = 0
    # Iterate over all opponent action combinations
    from itertools import product as iterproduct
    opp_actions = [range(shape[j]) for j in range(n_players) if j != player]
    for combo in iterproduct(*opp_actions):
        idx = list(combo)
        idx.insert(player, action)
        prob = 1.0
        oi = 0
        for j in range(n_players):
            if j != player:
                prob *= opp_strategies[oi][idx[j]]
                oi += 1
        total += payoff_matrix[tuple(idx)] * prob
    return total


def _compute_payoffs(payoff_matrices, strategies):
    n = len(strategies)
    if n == 2:
        p, q = strategies
        return [float(p @ payoff_matrices[0] @ q), float(p @ payoff_matrices[1] @ q)]
    # General: tensor contraction
    payoffs = []
    for player in range(n):
        val = _marginalise(payoff_matrices[player], player, 0,
                          [strategies[j] for j in range(n) if j != player], n)
        # Weighted sum over player's own actions
        total = sum(
            strategies[player][a] * _marginalise(
                payoff_matrices[player], player, a,
                [strategies[j] for j in range(n) if j != player], n)
            for a in range(len(strategies[player]))
        )
        payoffs.append(total)
    return payoffs


def _try_support(A, B, s1, s2):
    """Try to find NE on given support pair."""
    m, n = A.shape
    k = len(s1)

    # Solve for q: A[s1,:] @ q gives equal payoffs for player 1
    sub_A = A[np.ix_(s1, s2)]
    sub_B = B[np.ix_(s1, s2)]

    # Player 2's strategy: solve indifference for player 1
    # sub_A @ q_sub = v1 * 1  (equal payoffs)
    # 1' q_sub = 1
    lhs = np.vstack([sub_A, np.ones(k)])
    rhs = np.zeros(k + 1)
    rhs[-1] = 1.0

    try:
        # Solve augmented system
        M = np.vstack([
            np.hstack([sub_A, -np.ones((k, 1))]),
            np.hstack([np.ones((1, k)), np.zeros((1, 1))]),
        ])
        b = np.zeros(k + 1)
        b[-1] = 1.0
        x = np.linalg.solve(M, b)
        q_sub = x[:k]
        v1 = x[k]
    except np.linalg.LinAlgError:
        return None

    if np.any(q_sub < -1e-8):
        return None

    # Player 1's strategy: solve indifference for player 2
    try:
        M2 = np.vstack([
            np.hstack([sub_B.T, -np.ones((k, 1))]),
            np.hstack([np.ones((1, k)), np.zeros((1, 1))]),
        ])
        b2 = np.zeros(k + 1)
        b2[-1] = 1.0
        y = np.linalg.solve(M2, b2)
        p_sub = y[:k]
    except np.linalg.LinAlgError:
        return None

    if np.any(p_sub < -1e-8):
        return None

    # Map to full strategies
    p = np.zeros(m)
    q = np.zeros(n)
    p[list(s1)] = np.maximum(p_sub, 0)
    q[list(s2)] = np.maximum(q_sub, 0)

    # Normalise
    if p.sum() > 0:
        p /= p.sum()
    if q.sum() > 0:
        q /= q.sum()

    # Check best response
    gap = _best_response_gap(A, B, p, q)
    return (p, q, gap)


def _best_response_gap(A, B, p, q):
    """How far strategies are from best response."""
    br1_payoff = float(np.max(A @ q))
    actual1 = float(p @ A @ q)
    br2_payoff = float(np.max(B.T @ p))
    actual2 = float(p @ B @ q)
    return abs(br1_payoff - actual1) + abs(br2_payoff - actual2)
