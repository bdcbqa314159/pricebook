"""Model risk reserve calculation.

Computes reserves from parameter uncertainty bands. Integrates with
EBA prudent valuation (AVA) framework.

    from pricebook.risk.model_reserve import (
        compute_model_reserve, reserve_by_risk_factor,
    )

References:
    EBA (2016). RTS on Prudent Valuation, 2016/101.
    Cont (2006). Model Uncertainty and its Impact on Pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pricebook.risk.parameter_uncertainty import ParameterBand, sensitivity_ladder


@dataclass
class ModelReserveResult:
    """Model risk reserve computation result."""
    reserve: float              # total reserve (positive = charge)
    components: dict[str, float]  # per-parameter reserve
    confidence: float
    method: str

    def to_dict(self) -> dict:
        return {"reserve": self.reserve, "confidence": self.confidence,
                "method": self.method, "n_components": len(self.components)}


def compute_model_reserve(
    pricer: Callable,
    base_params: dict[str, float],
    bands: list[ParameterBand],
    confidence: float = 0.90,
    method: str = "worst_case",
) -> ModelReserveResult:
    """Compute model risk reserve from parameter uncertainty.

    Args:
        pricer: callable(params_dict) → float (PV).
        base_params: baseline calibrated parameters.
        bands: confidence bands per parameter.
        confidence: reserve confidence level.
        method:
            "worst_case" — sum of worst-case PV impacts across parameters.
            "quadrature" — sqrt of sum of squared impacts (assumes independence).

    Returns:
        ModelReserveResult with total reserve and per-parameter breakdown.
    """
    ladder = sensitivity_ladder(pricer, base_params, bands)
    base_pv = pricer(base_params)

    components = {}
    for entry in ladder:
        # Reserve = max adverse PV change
        worst = max(abs(entry.low_pv - base_pv), abs(entry.high_pv - base_pv))
        components[entry.param_name] = worst

    if method == "worst_case":
        total = sum(components.values())
    elif method == "quadrature":
        total = sum(v**2 for v in components.values()) ** 0.5
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'worst_case' or 'quadrature'.")

    return ModelReserveResult(total, components, confidence, method)


def reserve_by_risk_factor(
    pricer: Callable,
    base_params: dict[str, float],
    bands: list[ParameterBand],
) -> dict[str, float]:
    """Per-risk-factor reserve breakdown.

    Returns {param_name: reserve_amount} sorted by magnitude.
    """
    result = compute_model_reserve(pricer, base_params, bands)
    return dict(sorted(result.components.items(), key=lambda kv: kv[1], reverse=True))


def model_risk_reserve_ava(
    pricer: Callable,
    base_params: dict[str, float],
    bands: list[ParameterBand],
    confidence: float = 0.90,
) -> dict:
    """Model risk AVA (Additional Valuation Adjustment) format.

    Compatible with EBA prudent valuation framework.

    Returns dict with:
        ava_model_risk: reserve amount
        ava_category: "model_risk"
        confidence_level: confidence
    """
    result = compute_model_reserve(pricer, base_params, bands, confidence, "quadrature")
    return {
        "ava_model_risk": result.reserve,
        "ava_category": "model_risk",
        "confidence_level": confidence,
        "n_parameters": len(bands),
        "components": result.components,
    }
