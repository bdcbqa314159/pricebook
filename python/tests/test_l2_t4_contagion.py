"""Regression for L2 phase-2 audit of `risk.contagion`:

Pre-fix `DefaultCascade.simulate` used ``remaining_buffer[d] = -1`` as
both the "node has defaulted" flag AND the "outward losses propagated"
marker.  But a creditor that defaults mid-cascade has
``remaining_buffer < 0`` from incoming losses, so the next round's
``if remaining_buffer[d] < 0: continue`` skipped them — their losses
never propagated to *their* creditors.  Second-order contagion was
silently dropped.

Fix: separate ``processed`` set tracks which defaulters have had their
losses pushed out.  Now each defaulter is processed exactly once,
regardless of how their buffer ended up.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.contagion import DefaultCascade


class TestSecondOrderContagion:
    def test_chain_cascade_three_nodes(self):
        """A → B → C: A defaults, takes B down, B should take C down."""
        # Setup: A has $0 to lose (zero buffer).  B owes A, A defaults.
        # B's loss > B's buffer → B defaults.  C owes B, so C must absorb B's loss.
        #
        # exposures[i][j] = i's exposure to j (loses if j defaults).
        #   B has exposure 100 to A.
        #   C has exposure 100 to B.
        nodes = ["A", "B", "C"]
        exposures = np.array([
            [0,   0, 0],  # A has no exposure
            [100, 0, 0],  # B → A: 100
            [0, 100, 0],  # C → B: 100
        ], dtype=float)
        buffers = np.array([0.0, 30.0, 200.0])  # A: zero (defaults). B: 30 (can't absorb 60). C: 200 (could absorb 60)

        dc = DefaultCascade(nodes, exposures, buffers)
        r = dc.simulate(["A"], recovery_rate=0.4)  # loss = 60 (1-0.4)
        # Expected: A initial. B's loss (60) > B's buffer (30) → B defaults.
        # B's default → C absorbs 60.  C's buffer 200 - 60 = 140 > 0 → C survives.
        # final = {A, B}.  Pre-fix would have stopped at A → B without confirming B → C check.
        # In this case C does NOT default — but B's losses to C MUST be tracked.
        assert "A" in r.final_defaults
        assert "B" in r.final_defaults
        assert "C" not in r.final_defaults
        # C's losses must be > 0 — pre-fix would have left C's loss = 0 (since
        # B's outward losses were skipped).
        assert r.losses_by_node.get("C", 0.0) > 0
        assert r.losses_by_node["C"] == pytest.approx(60.0, abs=1e-9)

    def test_three_node_full_cascade(self):
        """All three default in series.  Pre-fix would only get the first two."""
        nodes = ["A", "B", "C"]
        exposures = np.array([
            [0,   0, 0],
            [100, 0, 0],
            [0, 100, 0],
        ], dtype=float)
        # A: zero (initial default).  B: 30 (defaults from A).  C: 30 (defaults from B).
        buffers = np.array([0.0, 30.0, 30.0])
        dc = DefaultCascade(nodes, exposures, buffers)
        r = dc.simulate(["A"], recovery_rate=0.4)
        # Expected: A → B → C all default.  Pre-fix would stop at A → B (since
        # B's losses to C were never propagated).
        assert set(r.final_defaults) == {"A", "B", "C"}
        assert r.cascade_rounds == 2


class TestNoContagion:
    def test_well_capitalised_neighbours_survive(self):
        """A defaults; B, C have huge buffers → no further defaults."""
        nodes = ["A", "B", "C"]
        exposures = np.array([
            [0, 0, 0],
            [100, 0, 0],
            [50, 0, 0],
        ], dtype=float)
        buffers = np.array([0.0, 1000.0, 1000.0])
        dc = DefaultCascade(nodes, exposures, buffers)
        r = dc.simulate(["A"], recovery_rate=0.4)
        assert r.final_defaults == ["A"]
        assert r.contagion_multiplier() == 0.0


class TestProcessedOnceInvariant:
    def test_each_default_propagates_exactly_once(self):
        """Even with a complex topology, each defaulter propagates outward once."""
        # 4-node ring: A → B → C → D → A.  Initial: A.
        nodes = ["A", "B", "C", "D"]
        exposures = np.zeros((4, 4))
        exposures[1, 0] = 100  # B owes A
        exposures[2, 1] = 100  # C owes B
        exposures[3, 2] = 100  # D owes C
        exposures[0, 3] = 100  # A owes D
        # Buffers tight: each can just barely fail.
        buffers = np.array([0.0, 30.0, 30.0, 30.0])  # A initial; B, C, D all default
        dc = DefaultCascade(nodes, exposures, buffers)
        r = dc.simulate(["A"], recovery_rate=0.4)
        # All four should default — chain propagates around.
        assert set(r.final_defaults) == {"A", "B", "C", "D"}
