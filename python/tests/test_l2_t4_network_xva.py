"""Regression for L2 phase-2 audit of `risk.network_xva`:

Pre-fix `compute_network_cva` and `systemic_cva_adjustment` had
``max(contagion_multiplier, 1.0)`` flooring with a comment claiming
"no contagion → no adjustment (multiplicative identity)".  But the
floor at 1.0 actually forces a *non-zero* adjustment of
``α · centrality`` even when multiplier = 0 — contradicting both
the comment AND the docstring formula
``CVA × (1 + α × centrality × multiplier)``.

For an isolated counterparty (no outgoing exposures, no contagion),
multiplier is 0 and the network CVA should equal the standalone CVA.
Pre-fix it was bumped by ``α · centrality`` × CVA.

Fix: removed the floor.  multiplier = 0 → adjustment = 0 → CVA unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.network_xva import (
    NetworkXVAEngine, systemic_cva_adjustment,
)


class TestNoContagionNoAdjustment:
    def test_isolated_counterparty_unchanged(self):
        """Isolated CP (no outgoing exposures) → multiplier=0 → CVA unchanged."""
        # CounterpartyZ has no outgoing exposures — its default causes no contagion.
        nodes = ["BankA", "BankB", "Z"]
        exposures = np.array([
            [0,  100, 0],   # BankA exposed to BankB
            [50, 0,   0],   # BankB exposed to BankA
            [0,  0,   0],   # Z exposed to nobody, and nobody exposed to Z (col 2 = 0)
        ], dtype=float)
        buffers = np.array([200.0, 200.0, 100.0])

        engine = NetworkXVAEngine(nodes, exposures, buffers)
        # Z's default: no one is exposed to Z, so no contagion → multiplier = 0.
        result = engine.compute_network_cva("Z", standalone_cva=1_000_000.0,
                                              alpha=0.5)
        assert result.contagion_multiplier == pytest.approx(0.0, abs=1e-12)
        # Network CVA must equal standalone (multiplier=0 → adjustment=0).
        assert result.network_cva == pytest.approx(result.standalone_cva, abs=1e-6)

    def test_systemic_cva_adjustment_multiplier_zero(self):
        cva = systemic_cva_adjustment(
            standalone_cva=1_000_000,
            centrality=0.5,
            contagion_multiplier=0.0,
            alpha=0.5,
        )
        assert cva == pytest.approx(1_000_000, abs=1e-6)


class TestContagionRaisesCVA:
    def test_systemic_counterparty_pays_more(self):
        """A counterparty whose default cascades should have CVA > standalone."""
        nodes = ["BankA", "BankB", "BankC", "BankD"]
        # BankA owes B and C; defaults trigger contagion via D.
        exposures = np.array([
            [0,   80, 60, 0],
            [80,  0,  0,  0],
            [60,  0,  0,  0],
            [0,  50, 50, 0],
        ], dtype=float)
        # Tight buffers so contagion propagates.
        buffers = np.array([30.0, 30.0, 30.0, 30.0])
        engine = NetworkXVAEngine(nodes, exposures, buffers)

        r = engine.compute_network_cva("BankA", standalone_cva=500_000.0,
                                         alpha=0.5)
        assert r.network_cva > r.standalone_cva  # contagion → uplift


class TestAlphaZeroNoChange:
    """alpha=0 must always recover standalone."""

    def test_alpha_zero_isolated(self):
        nodes = ["X", "Y"]
        exposures = np.array([[0, 50], [50, 0]], dtype=float)
        buffers = np.array([100.0, 100.0])
        engine = NetworkXVAEngine(nodes, exposures, buffers)
        r = engine.compute_network_cva("X", standalone_cva=100_000.0, alpha=0.0)
        assert r.network_cva == pytest.approx(100_000.0, abs=1e-6)


class TestSystemicCvaAdjustmentClosedForm:
    def test_formula_matches_docstring(self):
        """CVA_adj = CVA × (1 + α × centrality × multiplier) exactly."""
        cva = 100_000
        alpha = 0.5
        c = 0.3
        m = 2.0
        result = systemic_cva_adjustment(cva, c, m, alpha)
        expected = cva * (1 + alpha * c * m)
        assert result == pytest.approx(expected, abs=1e-9)
