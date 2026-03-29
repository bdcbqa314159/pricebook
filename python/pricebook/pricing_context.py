"""
PricingContext — bundles market data for instrument pricing.

A PricingContext is "what date, what curves, what vols" in one object,
replacing the growing parameter lists in pricing methods. Instruments
query the context for what they need.

    ctx = PricingContext(
        valuation_date=date(2024, 1, 15),
        discount_curve=ois_curve,
        projection_curves={"USD.3M": libor_3m_curve},
        vol_surfaces={"ir": swaption_vol},
        credit_curves={"ACME": acme_survival},
        fx_spots={("EUR", "USD"): 1.0850},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


@dataclass
class PricingContext:
    """Immutable snapshot of market data for pricing.

    Attributes:
        valuation_date: the "as-of" date for the pricing run.
        discount_curve: the risk-free (OIS) discount curve.
        projection_curves: named forward projection curves (e.g. "USD.3M").
        vol_surfaces: named volatility surfaces (e.g. "ir", "fx").
            Each must expose a ``vol(expiry, strike)`` method.
        credit_curves: named survival curves (e.g. issuer name).
        fx_spots: spot rates keyed by (base, quote) currency code tuples.
    """

    valuation_date: date
    discount_curve: DiscountCurve | None = None
    projection_curves: dict[str, DiscountCurve] = field(default_factory=dict)
    vol_surfaces: dict[str, object] = field(default_factory=dict)
    credit_curves: dict[str, SurvivalCurve] = field(default_factory=dict)
    fx_spots: dict[tuple[str, str], float] = field(default_factory=dict)

    def get_projection_curve(self, name: str) -> DiscountCurve:
        """Return a projection curve by name, or raise KeyError."""
        if name not in self.projection_curves:
            raise KeyError(f"Projection curve '{name}' not in context")
        return self.projection_curves[name]

    def get_vol_surface(self, name: str) -> object:
        """Return a vol surface by name, or raise KeyError."""
        if name not in self.vol_surfaces:
            raise KeyError(f"Vol surface '{name}' not in context")
        return self.vol_surfaces[name]

    def get_credit_curve(self, name: str) -> SurvivalCurve:
        """Return a credit/survival curve by name, or raise KeyError."""
        if name not in self.credit_curves:
            raise KeyError(f"Credit curve '{name}' not in context")
        return self.credit_curves[name]

    def get_fx_spot(self, base: str, quote: str) -> float:
        """Return an FX spot rate, or raise KeyError."""
        key = (base, quote)
        if key not in self.fx_spots:
            raise KeyError(f"FX spot '{base}/{quote}' not in context")
        return self.fx_spots[key]
