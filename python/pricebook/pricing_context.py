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

Multi-currency support: ``discount_curves``, ``inflation_curves``, and
``repo_curves`` are keyed by currency code. The original ``discount_curve``
field is kept for backward compatibility — ``get_discount_curve(ccy)``
checks ``discount_curves`` first, then falls back to ``discount_curve``.

FX translation: ``fx_translate()`` converts a PV from one currency to
``reporting_currency`` using the ``fx_spots`` dict.
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
        discount_curve: the risk-free (OIS) discount curve (single-ccy shortcut).
        discount_curves: per-currency discount curves (e.g. ``{"USD": usd_ois, "EUR": eur_ois}``).
        projection_curves: named forward projection curves (e.g. "USD.3M").
        vol_surfaces: named volatility surfaces (e.g. "ir", "fx").
            Each must expose a ``vol(expiry, strike)`` method.
        credit_curves: named survival curves (e.g. issuer name).
        fx_spots: spot rates keyed by (base, quote) currency code tuples.
        inflation_curves: per-currency CPI/inflation curves.
        repo_curves: per-currency repo/funding curves.
        reporting_currency: base currency for P&L aggregation.
    """

    valuation_date: date
    discount_curve: DiscountCurve | None = None
    discount_curves: dict[str, DiscountCurve] = field(default_factory=dict)
    projection_curves: dict[str, DiscountCurve] = field(default_factory=dict)
    vol_surfaces: dict[str, object] = field(default_factory=dict)
    credit_curves: dict[str, SurvivalCurve] = field(default_factory=dict)
    fx_spots: dict[tuple[str, str], float] = field(default_factory=dict)
    inflation_curves: dict[str, object] = field(default_factory=dict)
    repo_curves: dict[str, DiscountCurve] = field(default_factory=dict)
    reporting_currency: str = "USD"

    # ---- Curve accessors ----

    def get_discount_curve(self, ccy: str | None = None) -> DiscountCurve:
        """Return a discount curve by currency.

        Looks up ``discount_curves[ccy]`` first; if not found (or *ccy*
        is ``None``), falls back to the single ``discount_curve`` field.
        Raises ``KeyError`` if neither is available.
        """
        if ccy is not None and ccy in self.discount_curves:
            return self.discount_curves[ccy]
        if self.discount_curve is not None:
            return self.discount_curve
        raise KeyError(
            f"No discount curve for '{ccy}' and no default discount_curve"
        )

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

    def get_inflation_curve(self, ccy: str) -> object:
        """Return an inflation/CPI curve by currency, or raise KeyError."""
        if ccy not in self.inflation_curves:
            raise KeyError(f"Inflation curve '{ccy}' not in context")
        return self.inflation_curves[ccy]

    def get_repo_curve(self, ccy: str) -> DiscountCurve:
        """Return a repo/funding curve by currency, or raise KeyError."""
        if ccy not in self.repo_curves:
            raise KeyError(f"Repo curve '{ccy}' not in context")
        return self.repo_curves[ccy]

    # ---- FX translation ----

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        """FX rate to convert ``from_ccy`` to ``to_ccy``.

        Tries direct lookup, then inverse. Returns 1.0 if
        ``from_ccy == to_ccy``.
        """
        if from_ccy == to_ccy:
            return 1.0
        key = (from_ccy, to_ccy)
        if key in self.fx_spots:
            return self.fx_spots[key]
        inv = (to_ccy, from_ccy)
        if inv in self.fx_spots:
            return 1.0 / self.fx_spots[inv]
        raise KeyError(
            f"No FX rate for '{from_ccy}/{to_ccy}' or its inverse"
        )

    def fx_translate(
        self,
        value: float,
        from_ccy: str,
        to_ccy: str | None = None,
    ) -> float:
        """Translate a value from *from_ccy* to *to_ccy*.

        Defaults to ``reporting_currency`` if *to_ccy* is ``None``.
        """
        target = to_ccy or self.reporting_currency
        return value * self.fx_rate(from_ccy, target)

    # ---- Copy with replacements ----

    def replace(self, **kwargs) -> "PricingContext":
        """Return a new PricingContext with specified fields replaced."""
        return PricingContext(
            valuation_date=kwargs.get("valuation_date", self.valuation_date),
            discount_curve=kwargs.get("discount_curve", self.discount_curve),
            discount_curves=kwargs.get("discount_curves", self.discount_curves),
            projection_curves=kwargs.get("projection_curves", self.projection_curves),
            vol_surfaces=kwargs.get("vol_surfaces", self.vol_surfaces),
            credit_curves=kwargs.get("credit_curves", self.credit_curves),
            fx_spots=kwargs.get("fx_spots", self.fx_spots),
            inflation_curves=kwargs.get("inflation_curves", self.inflation_curves),
            repo_curves=kwargs.get("repo_curves", self.repo_curves),
            reporting_currency=kwargs.get("reporting_currency", self.reporting_currency),
        )

    @classmethod
    def simple(
        cls,
        valuation_date: date,
        rate: float = 0.05,
        vol: float | None = None,
        hazard: float | None = None,
    ) -> "PricingContext":
        """Build a simple PricingContext from flat rates.

        Convenience for quick pricing and testing.
        """
        from pricebook.discount_curve import DiscountCurve
        from pricebook.vol_surface import FlatVol

        curve = DiscountCurve.flat(valuation_date, rate)
        vol_surfaces = {}
        if vol is not None:
            vol_surfaces["ir"] = FlatVol(vol)

        credit_curves = {}
        if hazard is not None:
            from pricebook.survival_curve import SurvivalCurve
            credit_curves["default"] = SurvivalCurve.flat(valuation_date, hazard)

        return cls(
            valuation_date=valuation_date,
            discount_curve=curve,
            vol_surfaces=vol_surfaces,
            credit_curves=credit_curves,
        )
