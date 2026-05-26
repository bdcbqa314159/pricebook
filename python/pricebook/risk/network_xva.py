"""Network-enhanced XVA — systemic risk adjustments to CVA/DVA.

Integrates financial network topology and contagion cascades into
counterparty risk valuation adjustments.

    from pricebook.risk.network_xva import (
        NetworkXVAEngine, NetworkCVAResult,
        systemic_cva_adjustment, contagion_cva_stress,
    )

Key ideas:
- Counterparty CVA should reflect not just bilateral default risk,
  but contagion exposure through the financial network
- A counterparty connected to systemically important nodes has
  higher effective default probability
- Network centrality measures proxy for "too-connected-to-fail"
- Eisenberg-Noe cascade simulation quantifies contagion multiplier

References:
    Eisenberg & Noe (2001). Systemic Risk in Financial Systems.
    Cont et al. (2013). Running for the Exit: Distressed Selling and
        Endogenous Correlation in Financial Markets.
    Glasserman & Young (2016). Contagion in Financial Networks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkCVAResult:
    """CVA adjusted for network/systemic effects."""
    standalone_cva: float
    network_cva: float
    systemic_adjustment: float
    adjustment_ratio: float
    counterparty: str
    centrality_score: float
    contagion_multiplier: float
    n_connected_defaults: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class SystemicStressResult:
    """Result of network-wide CVA stress test."""
    total_standalone_cva: float
    total_network_cva: float
    systemic_surcharge: float
    counterparty_results: list[dict]
    worst_contagion_scenario: str
    max_contagion_multiplier: float

    def to_dict(self) -> dict:
        return {
            "total_standalone_cva": self.total_standalone_cva,
            "total_network_cva": self.total_network_cva,
            "systemic_surcharge": self.systemic_surcharge,
            "n_counterparties": len(self.counterparty_results),
            "worst_contagion_scenario": self.worst_contagion_scenario,
            "max_contagion_multiplier": self.max_contagion_multiplier,
        }


class NetworkXVAEngine:
    """Computes network-adjusted XVA for a portfolio of counterparties.

    Steps:
    1. Build financial network from bilateral exposures
    2. Compute centrality metrics for each counterparty
    3. Run contagion cascades to estimate conditional default probabilities
    4. Adjust standalone CVA by systemic risk multiplier

    Usage:
        engine = NetworkXVAEngine(counterparties, exposures, buffers)
        result = engine.compute_network_cva(
            counterparty="BankA",
            standalone_cva=1_000_000,
            pd=0.02,
        )
    """

    def __init__(
        self,
        counterparties: list[str],
        exposure_matrix: np.ndarray,
        capital_buffers: np.ndarray,
        recovery_rate: float = 0.4,
    ):
        """
        Args:
            counterparties: list of counterparty names.
            exposure_matrix: (N, N) bilateral exposure matrix.
            capital_buffers: (N,) capital/equity buffer per counterparty.
            recovery_rate: recovery rate assumption.
        """
        self.counterparties = counterparties
        self.exposures = np.array(exposure_matrix, dtype=float)
        self.buffers = np.array(capital_buffers, dtype=float)
        self.recovery = recovery_rate
        self.n = len(counterparties)
        self._idx = {name: i for i, name in enumerate(counterparties)}

        # Pre-compute network metrics
        self._compute_centrality()

    def _compute_centrality(self):
        """Compute centrality scores for all counterparties."""
        from pricebook.risk.network import FinancialNetwork
        self._network = FinancialNetwork(self.counterparties, self.exposures)
        self._network_result = self._network.compute_all()

        # Composite centrality (average of normalised degree, pagerank, eigenvector)
        degree = self._network.degree_centrality()
        pagerank = self._network.pagerank()
        eigenvec = self._network.eigenvector_centrality()

        self._centrality = {}
        for name in self.counterparties:
            d = degree.get(name, 0)
            p = pagerank.get(name, 1 / self.n)
            e = eigenvec.get(name, 0)
            # Normalise pagerank to [0, 1] scale
            self._centrality[name] = (d + p * self.n + e) / 3

    def _contagion_analysis(self, trigger: str) -> dict:
        """Run contagion cascade from a single counterparty default."""
        from pricebook.risk.contagion import DefaultCascade
        cascade = DefaultCascade(self.counterparties, self.exposures, self.buffers)
        result = cascade.simulate([trigger])
        return {
            "trigger": trigger,
            "n_defaults": len(result.final_defaults),
            "defaults": result.final_defaults,
            "multiplier": result.contagion_multiplier(),
            "losses": result.total_losses,
        }

    def compute_network_cva(
        self,
        counterparty: str,
        standalone_cva: float,
        pd: float = 0.02,
        alpha: float = 0.5,
    ) -> NetworkCVAResult:
        """Compute network-adjusted CVA for a single counterparty.

        The adjustment formula:
            CVA_network = CVA_standalone × (1 + α × centrality × contagion_multiplier)

        where:
            α: systemic risk weight (0 = no adjustment, 1 = full)
            centrality: composite centrality score
            contagion_multiplier: from Eisenberg-Noe cascade

        Args:
            counterparty: name of the counterparty.
            standalone_cva: CVA computed without network effects.
            pd: annual default probability (for contagion weighting).
            alpha: systemic risk weight.
        """
        if counterparty not in self._idx:
            raise ValueError(f"Unknown counterparty: {counterparty}")

        centrality = self._centrality[counterparty]
        cascade = self._contagion_analysis(counterparty)

        # Contagion multiplier: how many additional defaults does this trigger?
        cond_defaults = cascade["n_defaults"] - 1  # exclude self
        multiplier = cascade["multiplier"]

        # Network adjustment: higher centrality + more contagion = higher CVA
        adjustment = alpha * centrality * max(multiplier, 1.0)
        network_cva = standalone_cva * (1.0 + adjustment)

        return NetworkCVAResult(
            standalone_cva=standalone_cva,
            network_cva=network_cva,
            systemic_adjustment=network_cva - standalone_cva,
            adjustment_ratio=network_cva / max(standalone_cva, 1e-10),
            counterparty=counterparty,
            centrality_score=centrality,
            contagion_multiplier=multiplier,
            n_connected_defaults=cond_defaults,
        )

    def stress_test(
        self,
        standalone_cvas: dict[str, float],
        alpha: float = 0.5,
    ) -> SystemicStressResult:
        """Run network CVA stress test across all counterparties.

        Args:
            standalone_cvas: {counterparty_name: standalone_cva} dict.
            alpha: systemic risk weight.
        """
        results = []
        worst_scenario = ""
        max_mult = 0.0

        for name, cva in standalone_cvas.items():
            if name not in self._idx:
                continue
            r = self.compute_network_cva(name, cva, alpha=alpha)
            results.append(r.to_dict())
            if r.contagion_multiplier > max_mult:
                max_mult = r.contagion_multiplier
                worst_scenario = name

        total_standalone = sum(standalone_cvas.get(name, 0) for name in self.counterparties)
        total_network = sum(r["network_cva"] for r in results)

        return SystemicStressResult(
            total_standalone_cva=total_standalone,
            total_network_cva=total_network,
            systemic_surcharge=total_network - total_standalone,
            counterparty_results=results,
            worst_contagion_scenario=worst_scenario,
            max_contagion_multiplier=max_mult,
        )

    def systemic_ranking(self) -> list[dict]:
        """Rank counterparties by systemic importance."""
        ranking = []
        for name in self.counterparties:
            cascade = self._contagion_analysis(name)
            ranking.append({
                "counterparty": name,
                "centrality": self._centrality[name],
                "contagion_multiplier": cascade["multiplier"],
                "cascade_defaults": cascade["n_defaults"],
                "systemic_score": self._centrality[name] * cascade["multiplier"],
            })
        ranking.sort(key=lambda x: x["systemic_score"], reverse=True)
        for i, r in enumerate(ranking):
            r["rank"] = i + 1
        return ranking

    def to_dict(self) -> dict:
        return {
            "n_counterparties": self.n,
            "counterparties": self.counterparties,
            "recovery_rate": self.recovery,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def systemic_cva_adjustment(
    standalone_cva: float,
    centrality: float,
    contagion_multiplier: float,
    alpha: float = 0.5,
) -> float:
    """Quick systemic CVA adjustment without full network engine.

    CVA_adj = CVA × (1 + α × centrality × contagion_multiplier)
    """
    return standalone_cva * (1.0 + alpha * centrality * max(contagion_multiplier, 1.0))


def contagion_cva_stress(
    counterparties: list[str],
    exposure_matrix: np.ndarray,
    buffers: np.ndarray,
    standalone_cvas: dict[str, float],
    alpha: float = 0.5,
) -> SystemicStressResult:
    """One-shot network CVA stress test.

    Builds the network, runs cascades, adjusts CVAs.
    """
    engine = NetworkXVAEngine(counterparties, exposure_matrix, buffers)
    return engine.stress_test(standalone_cvas, alpha)
