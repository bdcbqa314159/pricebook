"""Multi-asset hedging: generalised N-instrument × M-target optimiser.

The capstone module — extends the 2×2 delta-vega solver from
:mod:`pricebook.equity_rv` and :mod:`pricebook.duration_management`
to an arbitrary number of hedge instruments and Greek targets.

* :class:`HedgeTarget` — a Greek to flatten (delta, gamma, vega, …).
* :class:`HedgeInstrument` — per-instrument Greek profile.
* :func:`optimal_hedge` — least-squares solve for hedge quantities.
* :func:`hedge_residual` — residual risk after applying hedges.
* :func:`what_if_analysis` — impact of adding/removing a hedge.
* :func:`hedge_recommendation` — recommend optimal hedge + residual report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---- Data classes ----

@dataclass
class HedgeTarget:
    """A Greek target to flatten."""
    name: str       # "delta", "gamma", "vega", "vanna", "volga"
    exposure: float  # current book exposure


@dataclass
class HedgeInstrument:
    """An instrument available for hedging."""
    name: str
    greeks: dict[str, float]  # greek_name → sensitivity per unit
    cost_per_unit: float = 0.0


@dataclass
class HedgeAllocation:
    """Quantity of each instrument in the hedge."""
    instrument: HedgeInstrument
    quantity: float

    @property
    def cost(self) -> float:
        return abs(self.quantity) * self.instrument.cost_per_unit


@dataclass
class HedgeResult:
    """Full hedge result: allocations + residuals."""
    allocations: list[HedgeAllocation]
    residuals: dict[str, float]
    total_cost: float
    max_residual: float


# ---- Optimal hedge ----

def optimal_hedge(
    targets: list[HedgeTarget],
    instruments: list[HedgeInstrument],
    cost_penalty: float = 0.0,
) -> HedgeResult:
    """Solve for hedge quantities to flatten multiple Greek targets.

    Minimises ``||A·x + b||²`` (+ optional cost penalty) where:
    - A is the (n_targets × n_instruments) Greek matrix,
    - b is the book exposure vector,
    - x is the hedge quantity vector.

    When ``cost_penalty > 0``, adds ``cost_penalty × Σ|cost_i × x_i|``
    as a Tikhonov regularisation term (approximated as L2).

    Args:
        targets: list of Greek targets to flatten.
        instruments: list of available hedge instruments.
        cost_penalty: weight on hedge cost (0 = pure risk minimisation).

    Returns:
        :class:`HedgeResult` with allocations and residuals.
    """
    n_targets = len(targets)
    n_inst = len(instruments)

    if n_targets == 0 or n_inst == 0:
        return HedgeResult([], {t.name: t.exposure for t in targets}, 0.0,
                           max(abs(t.exposure) for t in targets) if targets else 0.0)

    target_names = [t.name for t in targets]
    b = np.array([t.exposure for t in targets])

    A = np.zeros((n_targets, n_inst))
    for j, inst in enumerate(instruments):
        for i, name in enumerate(target_names):
            A[i, j] = inst.greeks.get(name, 0.0)

    # Tikhonov regularisation for cost
    if cost_penalty > 0:
        costs = np.array([inst.cost_per_unit for inst in instruments])
        reg = np.diag(cost_penalty * costs)
        A_aug = np.vstack([A, reg])
        b_aug = np.concatenate([-b, np.zeros(n_inst)])
        x, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    else:
        x, _, _, _ = np.linalg.lstsq(A, -b, rcond=None)

    allocations = [
        HedgeAllocation(inst, float(x[j]))
        for j, inst in enumerate(instruments)
    ]

    # Compute residuals
    residuals = {}
    for i, name in enumerate(target_names):
        residual = b[i] + sum(A[i, j] * x[j] for j in range(n_inst))
        residuals[name] = float(residual)

    total_cost = sum(a.cost for a in allocations)
    max_res = max(abs(v) for v in residuals.values()) if residuals else 0.0

    return HedgeResult(allocations, residuals, total_cost, max_res)


# ---- Hedge residual ----

def hedge_residual(
    targets: list[HedgeTarget],
    allocations: list[HedgeAllocation],
) -> dict[str, float]:
    """Compute residual exposure after applying hedge allocations."""
    residuals = {t.name: t.exposure for t in targets}
    for alloc in allocations:
        for name in residuals:
            residuals[name] += alloc.quantity * alloc.instrument.greeks.get(name, 0.0)
    return residuals


# ---- What-if analysis ----

@dataclass
class WhatIfResult:
    """Impact of adding or removing a hedge instrument."""
    instrument_name: str
    action: str  # "add" or "remove"
    residuals_before: dict[str, float]
    residuals_after: dict[str, float]
    improvement: float  # reduction in max residual


def what_if_analysis(
    targets: list[HedgeTarget],
    current_allocations: list[HedgeAllocation],
    candidate: HedgeInstrument,
    candidate_qty: float,
    action: str = "add",
) -> WhatIfResult:
    """Assess the impact of adding/removing a single hedge instrument."""
    before = hedge_residual(targets, current_allocations)

    if action == "add":
        after_allocs = current_allocations + [HedgeAllocation(candidate, candidate_qty)]
    else:
        after_allocs = [a for a in current_allocations if a.instrument.name != candidate.name]

    after = hedge_residual(targets, after_allocs)
    max_before = max(abs(v) for v in before.values()) if before else 0.0
    max_after = max(abs(v) for v in after.values()) if after else 0.0

    return WhatIfResult(candidate.name, action, before, after, max_before - max_after)


# ---- Hedge recommendation ----

@dataclass
class HedgeRecommendation:
    """Optimal hedge recommendation with residual risk report."""
    hedge: HedgeResult
    risk_reduction_pct: float
    largest_residual_greek: str
    largest_residual_value: float


def hedge_recommendation(
    targets: list[HedgeTarget],
    instruments: list[HedgeInstrument],
    cost_penalty: float = 0.0,
) -> HedgeRecommendation:
    """Recommend optimal hedge and report residual risk."""
    result = optimal_hedge(targets, instruments, cost_penalty)

    max_initial = max(abs(t.exposure) for t in targets) if targets else 0.0
    reduction = (
        (1 - result.max_residual / max_initial) * 100
        if max_initial > 0 else 100.0
    )

    largest_greek = max(result.residuals, key=lambda k: abs(result.residuals[k])) if result.residuals else ""
    largest_val = result.residuals.get(largest_greek, 0.0)

    return HedgeRecommendation(result, reduction, largest_greek, largest_val)
