"""The implied-volatility surface and the floor quotes it is built from (L1). BERHAN-2012 §5.2.

Two market-data value types: a **quoted floor** (the market's observable) and the **implied-vol
surface** (the market's vol view, recovered from a strip of them). Both are frozen data — the
stripping and the inversion live at L3, where the Black analytic is.

**Nodes are keyed by `(payment date, strike)`, with dates and not year fractions.** Tranche 7's
`real_curve_from_breakeven` lost a pointwise oracle to exactly this: a caller saying "2.5 years"
becomes 912 whole days, which is 2.4986 years on the curve's own clock, so the caller's `T` and the
curve's `T` disagreed by a day. A date has one spelling.

**No interpolation, deliberately (§3.4).** The surface stores the nodes that were implied, and
nothing more. Reading between them arrives with a consumer that needs an off-node vol — a smile
model or a vega — and until then an interpolation scheme would be an unexercised choice baked into
market data. Asking for a node it does not hold raises, so the absence is loud rather than silently
filled.

Provenance:
  stock:  _stock/quarry/pricebook/options/inflation_vol.py, vol_surface.py (mined for the node
          SHAPE only — their interpolation, arbitrage repair and SABR fitting are out of scope here)
  source: BERHAN-2012 (docs/references.md) §5.2 — the floorlet vol surface of Figures 5.3/5.4,
          whose axes are (strike, maturity) against an annualised implied volatility
  oracle: tests/L3/test_implied_vol_surface.py — the surface recovers a known SLOPED generating vol
          at every node, two strikes strip independently, and a node it does not hold raises
  crosscheck: none — a frozen (expiry, strike) -> sigma container; the numerics it holds are
          cross-checked where they are computed (models/black.py, models/implied_vol_surface.py)
  slice:  tranche 9(b) — the floor strip and the implied-vol surface
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Money
from pricebook.foundation.schedule import CouponSchedule


@dataclass(frozen=True)
class FloorQuote:
    """A quoted year-on-year inflation floor: its schedule, its strike `kappa`, its price.

    `notional` is carried alongside `price` rather than folded into a per-unit number, because a
    market quote is a price on a size and collapsing the two would hide which convention the
    caller used. Four fields, within the limit."""

    schedule: CouponSchedule
    strike: float
    price: Money
    notional: Money

    def __post_init__(self) -> None:
        if self.price.currency != self.notional.currency:
            raise ValueError(
                f"a floor quote's price and notional must share a currency, got "
                f"{self.price.currency} and {self.notional.currency}"
            )
        if self.notional.amount <= 0.0:
            raise ValueError(f"a floor quote needs a positive notional, got {self.notional}")


@dataclass(frozen=True)
class ImpliedVolSurface:
    """Annualised implied volatilities by `(payment date, strike)`, as at `anchor`.

    `anchor` is carried because a surface is only meaningful as at a valuation date — it is
    constitutive of the object's identity, not a convenience, and the round-trip exercises it."""

    anchor: date
    nodes: Mapping[tuple[date, float], float]

    def vol(self, maturity: date, strike: float) -> float:
        """The implied volatility at a node, or `KeyError` — never an interpolated guess."""
        try:
            return self.nodes[(maturity, strike)]
        except KeyError:
            raise KeyError(
                f"no implied volatility at ({maturity}, strike {strike}); this surface holds "
                f"{len(self.nodes)} nodes and does not interpolate"
            ) from None
