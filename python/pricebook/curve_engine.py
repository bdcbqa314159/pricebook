"""Curve engine: declarative curve definitions, builder, and multi-curve sets.

CurveDefinition describes what a curve is built from.
CurveBuilder takes a definition + market data → DiscountCurve.
CurveSet groups related curves (OIS + projection + basis) for a currency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.market_data import (
    MarketDataSnapshot, QuoteType, Quote, tenor_to_date, tenor_to_years,
    MissingQuoteError,
)
from pricebook.pricing_context import PricingContext


# ---------------------------------------------------------------------------
# Curve Definition
# ---------------------------------------------------------------------------


class CurveRole(Enum):
    DISCOUNT = "discount"
    PROJECTION = "projection"
    BASIS = "basis"


class ExtrapolationPolicy(Enum):
    FLAT_FORWARD = "flat_forward"
    SMITH_WILSON = "smith_wilson"
    NONE = "none"


@dataclass
class InstrumentSpec:
    """A single instrument in the curve definition."""

    quote_type: QuoteType
    tenor: str
    convention: str = ""  # for future use (e.g. "deposit", "future", "swap")


@dataclass
class CurveDefinition:
    """Declarative specification for building one curve.

    Args:
        name: curve identifier (e.g. "USD_OIS", "EUR_ESTR").
        currency: currency code.
        role: discount, projection, or basis.
        instruments: ordered list of instrument specs (short → long end).
        interpolation: method for the resulting curve.
        extrapolation: policy beyond last instrument.
        day_count: day count convention.
    """

    name: str
    currency: str = "USD"
    role: CurveRole = CurveRole.DISCOUNT
    instruments: list[InstrumentSpec] = field(default_factory=list)
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR
    extrapolation: ExtrapolationPolicy = ExtrapolationPolicy.FLAT_FORWARD
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "currency": self.currency,
            "role": self.role.value,
            "instruments": [
                {"quote_type": i.quote_type.value, "tenor": i.tenor}
                for i in self.instruments
            ],
            "interpolation": self.interpolation.value,
            "extrapolation": self.extrapolation.value,
            "day_count": self.day_count.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurveDefinition:
        return cls(
            name=d["name"],
            currency=d.get("currency", "USD"),
            role=CurveRole(d.get("role", "discount")),
            instruments=[
                InstrumentSpec(QuoteType(i["quote_type"]), i["tenor"])
                for i in d.get("instruments", [])
            ],
            interpolation=InterpolationMethod(d.get("interpolation", "log_linear")),
            extrapolation=ExtrapolationPolicy(d.get("extrapolation", "flat_forward")),
            day_count=DayCountConvention(d.get("day_count", "ACT/365F")),
        )

    @classmethod
    def usd_ois(cls) -> CurveDefinition:
        """Standard USD OIS curve definition."""
        return cls(
            name="USD_OIS",
            currency="USD",
            role=CurveRole.DISCOUNT,
            instruments=[
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "1M"),
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "3M"),
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "6M"),
                InstrumentSpec(QuoteType.SWAP_RATE, "1Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "2Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "3Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "5Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "7Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "10Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "15Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "20Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "30Y"),
            ],
        )

    @classmethod
    def eur_estr(cls) -> CurveDefinition:
        """Standard EUR ESTR curve definition."""
        return cls(
            name="EUR_ESTR",
            currency="EUR",
            role=CurveRole.DISCOUNT,
            instruments=[
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "1M"),
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "3M"),
                InstrumentSpec(QuoteType.DEPOSIT_RATE, "6M"),
                InstrumentSpec(QuoteType.SWAP_RATE, "1Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "2Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "5Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "10Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "20Y"),
                InstrumentSpec(QuoteType.SWAP_RATE, "30Y"),
            ],
        )


# ---------------------------------------------------------------------------
# Curve Builder
# ---------------------------------------------------------------------------


import math  # noqa: E402


def build_curve(
    definition: CurveDefinition,
    snapshot: MarketDataSnapshot,
) -> DiscountCurve:
    """Build a DiscountCurve from a definition and market data snapshot.

    Selects quotes matching the definition's instruments from the snapshot,
    then bootstraps via simple zero-rate approach.
    """
    ref = snapshot.snapshot_date
    deposits: list[tuple[date, float]] = []
    swaps: list[tuple[date, float]] = []

    for spec in definition.instruments:
        quotes = snapshot.get_quotes(spec.quote_type, definition.currency)
        match = [q for q in quotes if q.tenor == spec.tenor]
        if not match:
            raise MissingQuoteError(
                f"No {spec.quote_type.value} quote for tenor {spec.tenor} "
                f"in {definition.currency}"
            )
        q = match[0]
        mat = tenor_to_date(ref, spec.tenor)

        if spec.quote_type == QuoteType.DEPOSIT_RATE:
            deposits.append((mat, q.value))
        elif spec.quote_type == QuoteType.SWAP_RATE:
            swaps.append((mat, q.value))

    # Build from deposits + swaps
    pillar_dates = []
    pillar_dfs = []

    # Deposits: df = 1 / (1 + r * tau)
    for mat, rate in sorted(deposits):
        tau = tenor_to_years(
            next(s.tenor for s in definition.instruments
                 if tenor_to_date(ref, s.tenor) == mat)
        )
        df = 1.0 / (1.0 + rate * tau)
        pillar_dates.append(mat)
        pillar_dfs.append(df)

    # Swaps: treat as zero rates, df = exp(-r * t)
    for mat, rate in sorted(swaps):
        t = tenor_to_years(
            next(s.tenor for s in definition.instruments
                 if tenor_to_date(ref, s.tenor) == mat)
        )
        df = math.exp(-rate * t)
        pillar_dates.append(mat)
        pillar_dfs.append(df)

    if not pillar_dates:
        raise MissingQuoteError(f"No quotes matched for curve {definition.name}")

    # Apply Smith-Wilson extrapolation if requested
    if definition.extrapolation == ExtrapolationPolicy.SMITH_WILSON:
        from pricebook.smith_wilson import smith_wilson_curve
        maturities = [tenor_to_years(s.tenor) for s in definition.instruments]
        return smith_wilson_curve(ref, maturities, pillar_dfs)

    return DiscountCurve(
        ref, pillar_dates, pillar_dfs,
        day_count=definition.day_count,
        interpolation=definition.interpolation,
    )


# ---------------------------------------------------------------------------
# Curve Set
# ---------------------------------------------------------------------------


@dataclass
class CurveSet:
    """A named collection of related curves for one currency.

    Typically: one discount curve + one or more projection curves.
    """

    name: str
    currency: str
    curves: dict[str, DiscountCurve] = field(default_factory=dict)
    roles: dict[str, CurveRole] = field(default_factory=dict)

    def add(self, curve_name: str, curve: DiscountCurve, role: CurveRole) -> None:
        self.curves[curve_name] = curve
        self.roles[curve_name] = role

    @property
    def discount_curve(self) -> DiscountCurve | None:
        for name, role in self.roles.items():
            if role == CurveRole.DISCOUNT:
                return self.curves[name]
        return None

    @property
    def projection_curves(self) -> dict[str, DiscountCurve]:
        return {
            name: curve for name, curve in self.curves.items()
            if self.roles.get(name) == CurveRole.PROJECTION
        }

    def to_pricing_context(
        self,
        valuation_date: date | None = None,
        vol_surfaces: dict | None = None,
    ) -> PricingContext:
        """Build a PricingContext from this curve set."""
        disc = self.discount_curve
        val_date = valuation_date
        if val_date is None and disc is not None:
            val_date = disc.reference_date

        return PricingContext(
            valuation_date=val_date,
            discount_curve=disc,
            projection_curves=self.projection_curves,
            vol_surfaces=vol_surfaces or {},
        )

    @classmethod
    def from_definitions(
        cls,
        name: str,
        currency: str,
        definitions: list[CurveDefinition],
        snapshot: MarketDataSnapshot,
    ) -> CurveSet:
        """Build a CurveSet from multiple definitions and one snapshot."""
        cs = cls(name=name, currency=currency)
        for defn in definitions:
            curve = build_curve(defn, snapshot)
            cs.add(defn.name, curve, defn.role)
        return cs
