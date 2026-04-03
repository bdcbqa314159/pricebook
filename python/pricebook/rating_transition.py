"""
Credit rating transition models.

Generator matrix Q: transition intensities between rating states.
Transition probability: P(t) = exp(Q*t) via matrix exponential.
Jarrow-Lando-Turnbull: risky ZCB per rating.
MC simulation of rating migration paths.

    from pricebook.rating_transition import RatingTransitionMatrix

    Q = RatingTransitionMatrix(
        ratings=["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"],
        generator=[[...], ...],
    )
    P = Q.transition_prob(t=1.0)
    default_prob = Q.default_probability("BBB", t=5.0)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm


class RatingTransitionMatrix:
    """Credit rating transition model.

    Args:
        ratings: list of rating labels (last = default/absorbing).
        generator: n×n generator matrix Q where Q[i,j] ≥ 0 for i≠j
            and Q[i,i] = -sum(Q[i,j] for j≠i). Last row = 0 (absorbing).
    """

    def __init__(self, ratings: list[str], generator: list[list[float]] | np.ndarray):
        self.ratings = ratings
        self.n = len(ratings)
        self.Q = np.asarray(generator, dtype=float)

        if self.Q.shape != (self.n, self.n):
            raise ValueError(f"Generator must be {self.n}x{self.n}")

    @property
    def default_state(self) -> int:
        """Index of the default (absorbing) state."""
        return self.n - 1

    def rating_index(self, rating: str) -> int:
        return self.ratings.index(rating)

    def transition_prob(self, t: float) -> np.ndarray:
        """Transition probability matrix P(t) = exp(Q*t). Shape: (n, n)."""
        return expm(self.Q * t)

    def default_probability(self, rating: str, t: float) -> float:
        """Probability of default by time t starting from given rating."""
        idx = self.rating_index(rating)
        P = self.transition_prob(t)
        return float(P[idx, self.default_state])

    def survival_probability(self, rating: str, t: float) -> float:
        """P(no default by t) = 1 - P(default by t)."""
        return 1.0 - self.default_probability(rating, t)

    def spread_term_structure(
        self, rating: str, tenors: list[float], recovery: float = 0.4,
    ) -> list[float]:
        """Implied spread for each tenor.

        spread ≈ -ln(survival) * (1 - R) / T (simplified).
        """
        spreads = []
        for T in tenors:
            surv = self.survival_probability(rating, T)
            if surv > 0 and T > 0:
                spread = -math.log(surv) / T * (1 - recovery)
            else:
                spread = 0.0
            spreads.append(spread)
        return spreads

    def simulate_paths(
        self, initial_rating: str, T: float, n_steps: int, n_paths: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate rating paths via embedded Markov chain.

        Returns: int array of shape (n_paths, n_steps+1), values are rating indices.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        P_dt = self.transition_prob(dt)
        # Normalize rows to sum to exactly 1 (matrix exp can have tiny numerical error)
        P_dt = np.maximum(P_dt, 0.0)
        P_dt = P_dt / P_dt.sum(axis=1, keepdims=True)

        idx0 = self.rating_index(initial_rating)
        paths = np.zeros((n_paths, n_steps + 1), dtype=int)
        paths[:, 0] = idx0

        for i in range(n_steps):
            for p in range(n_paths):
                current = paths[p, i]
                if current == self.default_state:
                    paths[p, i + 1] = self.default_state  # absorbing
                else:
                    paths[p, i + 1] = rng.choice(self.n, p=P_dt[current])

        return paths

    def simulate_default_times(
        self, initial_rating: str, T: float, n_steps: int, n_paths: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate default times. Returns inf for paths that don't default.

        Shape: (n_paths,).
        """
        paths = self.simulate_paths(initial_rating, T, n_steps, n_paths, seed)
        dt = T / n_steps
        default_times = np.full(n_paths, np.inf)

        for p in range(n_paths):
            for i in range(1, n_steps + 1):
                if paths[p, i] == self.default_state:
                    default_times[p] = i * dt
                    break

        return default_times


def risky_zcb_jlt(
    transition_matrix: RatingTransitionMatrix,
    rating: str,
    T: float,
    risk_free_rate: float,
    recovery: float = 0.4,
) -> float:
    """Jarrow-Lando-Turnbull risky ZCB price.

    P_risky = P_riskfree * survival + P_riskfree * (1 - survival) * recovery

    Simplified: P = df * [survival + (1-survival) * recovery]
    """
    df = math.exp(-risk_free_rate * T)
    surv = transition_matrix.survival_probability(rating, T)
    return df * (surv + (1 - surv) * recovery)


# Standard 1Y generator (simplified, based on S&P historical averages)
def standard_generator() -> RatingTransitionMatrix:
    """A simplified 8-state generator matrix (AAA to D).

    Based on historical 1Y transition rates (annualised).
    """
    ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
    # Off-diagonal transition intensities (diagonal computed to ensure rows sum to 0)
    Q = np.array([
        [0, 0.06, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0005],
        [0.01, 0, 0.07, 0.01, 0.005, 0.003, 0.001, 0.001],
        [0.001, 0.02, 0, 0.04, 0.01, 0.005, 0.002, 0.002],
        [0.0005, 0.005, 0.03, 0, 0.03, 0.01, 0.003, 0.0015],
        [0.0001, 0.001, 0.005, 0.02, 0, 0.05, 0.015, 0.009],
        [0.0001, 0.0005, 0.002, 0.005, 0.03, 0, 0.05, 0.0325],
        [0.0001, 0.0001, 0.001, 0.002, 0.01, 0.03, 0, 0.157],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    # Set diagonal so rows sum to 0
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return RatingTransitionMatrix(ratings, Q)
