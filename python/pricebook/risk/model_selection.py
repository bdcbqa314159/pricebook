"""Model selection uncertainty: Bayesian model averaging, model committee.

    from pricebook.risk.model_selection import (
        ModelCandidate, model_committee_price,
        bayesian_model_average, model_risk_matrix,
    )

References:
    Hoeting et al. (1999). Bayesian Model Averaging: A Tutorial. Stat. Science.
    Cont (2006). Model Uncertainty and its Impact on Pricing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ModelCandidate:
    """A candidate model for committee pricing."""
    name: str
    pricer: Callable         # callable(**params) → float
    weight: float = 1.0      # prior or posterior weight
    aic: float | None = None
    bic: float | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "weight": self.weight,
                "aic": self.aic, "bic": self.bic}


@dataclass
class CommitteePriceResult:
    """Result of model committee pricing."""
    price: float                     # weighted average
    price_std: float                 # standard deviation across models
    price_range: tuple[float, float]  # (min, max)
    model_prices: dict[str, float]   # per-model prices
    model_uncertainty_reserve: float  # half-range

    def to_dict(self) -> dict:
        return {
            "price": self.price, "price_std": self.price_std,
            "price_range": self.price_range,
            "model_uncertainty_reserve": self.model_uncertainty_reserve,
        }


def model_committee_price(
    candidates: list[ModelCandidate],
    **pricing_kwargs,
) -> CommitteePriceResult:
    """Weighted average price from a committee of models.

    Args:
        candidates: list of ModelCandidate.
        **pricing_kwargs: passed to each pricer.

    Returns:
        CommitteePriceResult with weighted average and dispersion.
    """
    prices = {}
    weights = {}
    total_weight = sum(c.weight for c in candidates)

    for c in candidates:
        p = c.pricer(**pricing_kwargs)
        prices[c.name] = p
        weights[c.name] = c.weight / total_weight

    # Weighted average
    avg = sum(prices[name] * weights[name] for name in prices)

    # Dispersion — Fix T4-RISK18: pre-fix used ``np.std(vals)`` (unweighted)
    # alongside a *weighted* mean.  Inconsistent: a heavily-down-weighted
    # outlier model still contributed full mass to the std.  For BMA
    # committees with one model dominating (e.g. weight 0.9 + 0.05 + 0.05)
    # the pre-fix std overstated uncertainty by an order of magnitude.
    # Now uses the weighted standard deviation
    # ``sqrt(Σ w_i · (p_i − avg)²)`` consistent with the weighted mean.
    if len(prices) > 1:
        weighted_var = sum(weights[name] * (prices[name] - avg) ** 2 for name in prices)
        std = float(math.sqrt(max(weighted_var, 0.0)))
    else:
        std = 0.0
    vals = list(prices.values())
    price_min, price_max = min(vals), max(vals)
    reserve = (price_max - price_min) / 2

    return CommitteePriceResult(avg, std, (price_min, price_max), prices, reserve)


def bayesian_model_average(
    candidates: list[ModelCandidate],
    use_bic: bool = True,
) -> list[ModelCandidate]:
    """Compute posterior model weights from AIC/BIC.

    BMA weights: w_i ∝ exp(-0.5 × IC_i) where IC = AIC or BIC.
    Lower IC → higher weight.

    Args:
        candidates: models with aic/bic fields populated.
        use_bic: use BIC (True) or AIC (False).

    Returns:
        New list of ModelCandidate with updated weights.
    """
    ic_values = []
    for c in candidates:
        ic = c.bic if use_bic else c.aic
        if ic is None:
            # No IC available — assign equal weight by using the mean IC of others
            ic = float("inf")  # will get near-zero weight; overridden below
        ic_values.append(ic)

    # Replace inf entries with mean of finite values (equal prior)
    finite_ics = [v for v in ic_values if math.isfinite(v)]
    mean_ic = sum(finite_ics) / len(finite_ics) if finite_ics else 0.0
    ic_values = [v if math.isfinite(v) else mean_ic for v in ic_values]

    # Shift for numerical stability
    ic_min = min(ic_values)
    log_weights = [-0.5 * (ic - ic_min) for ic in ic_values]
    total = sum(math.exp(lw) for lw in log_weights)

    result = []
    for c, lw in zip(candidates, log_weights):
        new_c = ModelCandidate(c.name, c.pricer, math.exp(lw) / total, c.aic, c.bic)
        result.append(new_c)

    return result


def model_risk_matrix(
    candidates: list[ModelCandidate],
    scenarios: list[dict],
) -> dict:
    """Price each model under each scenario.

    Args:
        candidates: list of models.
        scenarios: list of dicts with pricing kwargs per scenario.

    Returns:
        Dict with model_names, scenario_labels, prices (n_models × n_scenarios).
    """
    n_models = len(candidates)
    n_scenarios = len(scenarios)
    prices = np.zeros((n_models, n_scenarios))

    for i, c in enumerate(candidates):
        for j, sc in enumerate(scenarios):
            try:
                prices[i, j] = c.pricer(**sc)
            except Exception:
                prices[i, j] = float("nan")

    return {
        "model_names": [c.name for c in candidates],
        "n_scenarios": n_scenarios,
        "prices": prices.tolist(),
        "range_per_scenario": [float(np.nanmax(prices[:, j]) - np.nanmin(prices[:, j]))
                                for j in range(n_scenarios)],
    }
