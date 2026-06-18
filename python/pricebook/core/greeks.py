"""Greeks dataclass: standard option sensitivities.

Extracted to core/ so that instruments (L2) can use it without
importing from risk/ (L3).

    from pricebook.core.greeks import Greeks
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Greeks:
    """Standard option sensitivities.

    Dollar-Greeks (`delta × spot`, `0.5 × gamma × spot² × dS²`) are NOT
    properties on this class because `Greeks` does not carry the spot.
    Compute them externally where spot is known. The previous
    `dollar_delta` and `dollar_gamma` properties were misleading
    (formula did not match docstring) and had zero production callers;
    removed in T-LOW-CLEANUP.
    """
    price: float
    delta: float = 0.0       # ∂V/∂S
    gamma: float = 0.0       # ∂²V/∂S²
    vega: float = 0.0        # ∂V/∂σ (per 1% vol shift, i.e. × 0.01)
    theta: float = 0.0       # ∂V/∂t (per 1 day, i.e. × 1/365)
    rho: float = 0.0         # ∂V/∂r (per 1% rate shift, i.e. × 0.01)
    vanna: float = 0.0       # ∂²V/(∂S∂σ)
    volga: float = 0.0       # ∂²V/∂σ²

    def to_dict(self) -> dict:
        return dict(vars(self))
