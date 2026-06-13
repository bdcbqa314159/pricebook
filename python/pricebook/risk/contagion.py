"""Default cascade and contagion on financial networks.

    from pricebook.risk.contagion import (
        DefaultCascade, CascadeResult, settlement_fail_cascade,
    )

References:
    Eisenberg & Noe (2001). Systemic Risk in Financial Systems.
    Furfine (2003). Interbank Exposures: Quantifying the Risk of Contagion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CascadeResult:
    """Result of default cascade simulation."""
    initial_defaults: list[str]
    final_defaults: list[str]
    cascade_rounds: int
    total_losses: float
    losses_by_node: dict[str, float]
    clearing_payments: np.ndarray | None

    def contagion_multiplier(self) -> float:
        """How many additional defaults were caused by contagion."""
        initial = len(self.initial_defaults)
        final = len(self.final_defaults)
        return (final - initial) / max(initial, 1)

    def to_dict(self) -> dict:
        return {
            "initial_defaults": self.initial_defaults,
            "final_defaults": self.final_defaults,
            "cascade_rounds": self.cascade_rounds,
            "total_losses": self.total_losses,
            "contagion_multiplier": self.contagion_multiplier(),
        }


class DefaultCascade:
    """Eisenberg-Noe clearing vector model with cascade rounds.

    Given a network of bilateral exposures and capital buffers,
    simulate the cascade of defaults when one or more nodes fail.

    Args:
        nodes: list of node names.
        exposure_matrix: (N, N) bilateral exposures. exp[i][j] = i's exposure to j.
        capital_buffers: (N,) capital available to absorb losses.
    """

    def __init__(
        self,
        nodes: list[str],
        exposure_matrix: np.ndarray,
        capital_buffers: np.ndarray,
    ):
        self.nodes = nodes
        self.exposures = np.asarray(exposure_matrix, dtype=float)
        self.buffers = np.asarray(capital_buffers, dtype=float)
        self.n = len(nodes)
        self._node_idx = {name: i for i, name in enumerate(nodes)}

    def simulate(
        self,
        initial_defaults: list[str],
        max_rounds: int = 20,
        recovery_rate: float = 0.40,
    ) -> CascadeResult:
        """Simulate default cascade.

        Round 0: initial defaults occur.
        Round k: losses from round k-1 defaults hit surviving nodes.
                 Any node whose losses exceed buffer also defaults.
        Repeat until no new defaults.

        Args:
            initial_defaults: list of initially defaulting node names.
            max_rounds: maximum cascade rounds.
            recovery_rate: recovery on defaulted exposures.
        """
        defaulted = set()
        for name in initial_defaults:
            if name in self._node_idx:
                defaulted.add(self._node_idx[name])

        remaining_buffer = self.buffers.copy()
        losses = np.zeros(self.n)
        total_cascade_rounds = 0
        # Fix T4-RISK17: pre-fix used ``remaining_buffer[d] = -1`` as both
        # "node has defaulted" and "node's outward losses already
        # propagated".  But a creditor who defaults mid-cascade ALSO has
        # ``remaining_buffer < 0`` (from absorbed losses), so the
        # ``if remaining_buffer[d] < 0: continue`` check at the next
        # round skipped propagating their losses to *their* creditors.
        # Second-order contagion was silently dropped — the cascade
        # effectively terminated after the initial defaults' direct
        # creditors.  Now uses a separate ``processed`` set as the
        # "outward losses propagated" marker.
        processed: set[int] = set()

        for round_num in range(max_rounds):
            new_defaults: set[int] = set()

            to_process = defaulted - processed
            if not to_process:
                break

            for d in to_process:
                processed.add(d)

                # Losses to creditors of d
                for creditor in range(self.n):
                    if creditor in defaulted:
                        continue
                    exposure = self.exposures[creditor, d]
                    loss = exposure * (1 - recovery_rate)
                    losses[creditor] += loss
                    remaining_buffer[creditor] -= loss

                    if remaining_buffer[creditor] < 0:
                        new_defaults.add(creditor)

            if not new_defaults:
                break

            defaulted |= new_defaults
            total_cascade_rounds = round_num + 1

        # Results
        default_names = [self.nodes[i] for i in sorted(defaulted)]
        losses_by_node = {self.nodes[i]: float(losses[i]) for i in range(self.n) if losses[i] > 0}

        return CascadeResult(
            initial_defaults=initial_defaults,
            final_defaults=default_names,
            cascade_rounds=total_cascade_rounds,
            total_losses=float(losses.sum()),
            losses_by_node=losses_by_node,
            clearing_payments=None,
        )

    def stress_test(
        self,
        shock_scenarios: list[list[str]],
        recovery_rate: float = 0.40,
    ) -> list[dict]:
        """Run multiple cascade scenarios."""
        results = []
        for scenario in shock_scenarios:
            r = self.simulate(scenario, recovery_rate=recovery_rate)
            results.append({
                "scenario": scenario,
                "n_initial": len(scenario),
                "n_final": len(r.final_defaults),
                "contagion_multiplier": r.contagion_multiplier(),
                "total_losses": r.total_losses,
            })
        return results

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "n_nodes": self.n,
            "total_exposure": float(self.exposures.sum()),
            "total_capital": float(self.buffers.sum()),
        }
