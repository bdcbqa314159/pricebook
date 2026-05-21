"""Regime-switching credit model.

Hidden Markov Model (HMM) with different hazard rates per economic state.
The credit spread and default intensity depend on an unobservable regime
(e.g. expansion vs recession), with Markov transition probabilities.

    from pricebook.credit.regime_switching import (
        RegimeSwitchingCredit, RegimeState, calibrate_regime_model,
    )

    model = RegimeSwitchingCredit(
        hazard_rates=[0.005, 0.03],       # expansion, recession
        transition_matrix=[[0.98, 0.02],   # P(stay expansion), P(→ recession)
                           [0.10, 0.90]],  # P(→ expansion), P(stay recession)
    )
    q = model.survival(t=5.0)
    spread = model.implied_spread(t=5.0, recovery=0.40)

References:
    Jarrow, Lando & Turnbull (1997). A Markov Model for the Term Structure
    of Credit Risk Spreads. Review of Financial Studies.
    Hamilton (1989). A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass
class RegimeState:
    """Definition of a single regime state."""
    name: str
    hazard_rate: float          # annualised default intensity in this state
    description: str = ""


class RegimeSwitchingCredit:
    """Regime-switching default intensity model.

    The default intensity λ(t) takes one of N values depending on the
    current regime state. Regime transitions follow a continuous-time
    Markov chain with generator matrix Q.

    The survival probability accounts for regime uncertainty:
        Q(t) = π₀ × exp((Q - Λ) × t) × 1

    where Q is the generator, Λ = diag(λ₁, ..., λₙ), π₀ is the
    initial state distribution, and 1 is the ones vector.

    Args:
        hazard_rates: hazard rate for each regime state.
        transition_matrix: N×N annual transition probability matrix P.
            P[i][j] = probability of moving from state i to state j in one year.
        state_names: optional names for each state.
        initial_probs: initial state probabilities (default: stationary).
    """

    def __init__(
        self,
        hazard_rates: list[float],
        transition_matrix: list[list[float]],
        state_names: list[str] | None = None,
        initial_probs: list[float] | None = None,
    ):
        self.n_states = len(hazard_rates)
        self.hazard_rates = np.array(hazard_rates)

        P = np.array(transition_matrix, dtype=float)
        if P.shape != (self.n_states, self.n_states):
            raise ValueError(f"Transition matrix must be {self.n_states}×{self.n_states}")
        if (P < 0).any() or (P > 1).any():
            raise ValueError("Transition matrix entries must be in [0, 1]")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Transition matrix rows must sum to 1")

        # Convert annual transition probability matrix to generator
        # Q = logm(P) — but for stability, use Q = P - I for small dt
        # More precisely: Q_{ij} = P_{ij}/dt for off-diagonal, Q_{ii} = -(sum off-diag)
        # We approximate: generator from 1-year transition matrix
        self._P = P
        self._Q = self._transition_to_generator(P)
        self._Lambda = np.diag(self.hazard_rates)

        if state_names is None:
            state_names = [f"state_{i}" for i in range(self.n_states)]
        self.state_names = state_names

        if initial_probs is not None:
            self._pi0 = np.array(initial_probs)
        else:
            self._pi0 = self._stationary_distribution(P)

        self.states = [
            RegimeState(name=state_names[i], hazard_rate=hazard_rates[i])
            for i in range(self.n_states)
        ]

    def survival(self, t: float, initial_state: int | None = None) -> float:
        """Survival probability to time t, accounting for regime switching.

        If initial_state is given, conditions on starting in that state.
        Otherwise uses the initial distribution (default: stationary).
        """
        if t <= 0:
            return 1.0

        # Matrix exponential: exp((Q - Λ) × t)
        A = self._Q - self._Lambda
        M = expm(A * t)

        if initial_state is not None:
            return float(np.sum(M[initial_state, :]))
        else:
            return float(self._pi0 @ M @ np.ones(self.n_states))

    def survival_term_structure(
        self, times: list[float], initial_state: int | None = None,
    ) -> list[float]:
        """Survival probabilities at multiple times."""
        return [self.survival(t, initial_state) for t in times]

    def implied_hazard(self, t: float, initial_state: int | None = None) -> float:
        """Implied flat hazard rate from regime-switching survival.

        h_implied = -ln(Q(t)) / t
        """
        q = self.survival(t, initial_state)
        if q <= 0 or t <= 0:
            return 0.0
        return -math.log(q) / t

    def implied_spread(
        self, t: float, recovery: float = 0.40, initial_state: int | None = None,
    ) -> float:
        """Implied CDS-like spread from regime-switching survival.

        spread ≈ h_implied × (1 - R)
        """
        h = self.implied_hazard(t, initial_state)
        return h * (1.0 - recovery)

    def regime_probabilities(self, t: float) -> np.ndarray:
        """Probability of being in each regime at time t (no default)."""
        if t <= 0:
            return self._pi0.copy()
        M = expm(self._Q * t)
        return self._pi0 @ M

    def expected_hazard(self, t: float) -> float:
        """Expected hazard rate at time t, weighted by regime probabilities."""
        probs = self.regime_probabilities(t)
        return float(probs @ self.hazard_rates)

    def spread_term_structure(
        self, times: list[float], recovery: float = 0.40,
        initial_state: int | None = None,
    ) -> list[float]:
        """Implied spread at multiple tenors."""
        return [self.implied_spread(t, recovery, initial_state) for t in times]

    def stationary_distribution(self) -> np.ndarray:
        """Return the stationary distribution of the Markov chain."""
        return self._pi0.copy()

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "hazard_rates": self.hazard_rates.tolist(),
            "state_names": self.state_names,
            "transition_matrix": self._P.tolist(),
            "initial_probs": self._pi0.tolist(),
            "stationary_dist": self._stationary_distribution(self._P).tolist(),
        }

    @staticmethod
    def _transition_to_generator(P: np.ndarray) -> np.ndarray:
        """Convert 1-year transition probability matrix to generator.

        Uses the matrix logarithm approach: Q = log(P).
        For numerical stability, falls back to Q ≈ P - I.
        """
        n = P.shape[0]
        try:
            from scipy.linalg import logm
            Q = np.real(logm(P))
            # Ensure off-diagonal non-negative, diagonal negative
            for i in range(n):
                for j in range(n):
                    if i != j:
                        Q[i, j] = max(Q[i, j], 0.0)
                Q[i, i] = -np.sum(Q[i, :]) + Q[i, i]
            return Q
        except Exception:
            # Fallback: approximate
            return P - np.eye(n)

    @staticmethod
    def _stationary_distribution(P: np.ndarray) -> np.ndarray:
        """Compute stationary distribution of transition matrix P."""
        n = P.shape[0]
        # Solve π P = π, Σ πᵢ = 1
        A = np.vstack([(P.T - np.eye(n)), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.maximum(pi, 0.0)
        pi /= pi.sum()
        return pi


def calibrate_regime_model(
    observed_spreads: list[float],
    tenors: list[float],
    recovery: float = 0.40,
    n_states: int = 2,
) -> RegimeSwitchingCredit:
    """Calibrate a regime-switching model from observed CDS spread term structure.

    Simple calibration: fit 2-state model where:
    - State 0 (expansion): low hazard
    - State 1 (recession): high hazard

    Uses the spread level and slope to infer hazard rates and transition probs.

    Args:
        observed_spreads: CDS spreads in decimal (e.g. 0.01 = 100bp).
        tenors: corresponding tenors in years.
        recovery: assumed recovery rate.
        n_states: number of regime states (2 or 3).

    Returns:
        Calibrated RegimeSwitchingCredit model.
    """
    if len(observed_spreads) < 2:
        raise ValueError("Need at least 2 spread observations for calibration")

    # Average hazard from mid-tenor
    mid_idx = len(observed_spreads) // 2
    h_avg = observed_spreads[mid_idx] / (1.0 - recovery)

    # Short-term vs long-term spread ratio gives regime information
    h_short = observed_spreads[0] / (1.0 - recovery)
    h_long = observed_spreads[-1] / (1.0 - recovery)

    if n_states == 2:
        # 2-state: expansion and recession
        h_low = min(h_short, h_long) * 0.7
        h_high = max(h_short, h_long) * 1.5

        # Transition: mean reversion strength from spread slope
        slope = (h_long - h_short) / max(tenors[-1] - tenors[0], 1.0)
        p_stay = min(max(0.90, 1.0 - abs(slope) * 5), 0.99)
        p_switch = 1.0 - p_stay

        P = [[p_stay, p_switch],
             [p_switch * 2, 1.0 - p_switch * 2]]
        # Normalize rows
        P = [[p / sum(row) for p in row] for row in P]

        names = ["expansion", "recession"]
        return RegimeSwitchingCredit([h_low, h_high], P, names)

    elif n_states == 3:
        h_low = h_avg * 0.5
        h_mid = h_avg
        h_high = h_avg * 2.5

        P = [[0.92, 0.07, 0.01],
             [0.05, 0.90, 0.05],
             [0.02, 0.08, 0.90]]
        names = ["expansion", "normal", "recession"]
        return RegimeSwitchingCredit([h_low, h_mid, h_high], P, names)

    else:
        raise ValueError(f"n_states must be 2 or 3, got {n_states}")
