"""Greeks dataclass: standard option sensitivities.

Extracted to core/ so that instruments (L2) can use it without
importing from risk/ (L3).

    from pricebook.core.greeks import Greeks
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Greeks:
    """Standard option sensitivities."""
    price: float
    delta: float = 0.0       # ∂V/∂S
    gamma: float = 0.0       # ∂²V/∂S²
    vega: float = 0.0        # ∂V/∂σ (per 1% vol shift, i.e. × 0.01)
    theta: float = 0.0       # ∂V/∂t (per 1 day, i.e. × 1/365)
    rho: float = 0.0         # ∂V/∂r (per 1% rate shift, i.e. × 0.01)
    vanna: float = 0.0       # ∂²V/(∂S∂σ)
    volga: float = 0.0       # ∂²V/∂σ²

    @property
    def dollar_delta(self) -> float:
        """Approximate dollar delta: delta × option_price."""
        return self.delta * self.price

    @property
    def dollar_gamma(self) -> float:
        """Gamma P&L for a 1% spot move: 0.5 × gamma × S² × 0.01²."""
        return 0.5 * self.gamma

    def to_dict(self) -> dict:
        return vars(self)
