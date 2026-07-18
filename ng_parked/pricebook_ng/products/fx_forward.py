"""FX forward as pure data (L2).

An `FXForward` exchanges `base` (base-currency amount) for `quote`
(quote-currency amount) at `maturity`. `buy_base=True` receives the base leg and
pays the quote leg. Pure data — no `pv` method (CLAUDE.md 2); pricing is L4.

Provenance:
  quarry: python/pricebook/fx/ (fx forward)
  source: covered interest parity
  oracle: par FX forward reprices to zero (fx-forward slice)
  slice:  fx-forward
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Currency, Money


@dataclass(frozen=True)
class FXForward:
    """Exchange `base` for `quote` at `maturity` (buyer receives base, pays quote)."""

    base: Money
    quote: Money
    maturity: date
    buy_base: bool = True


def fx_forward(base: Money, quote_ccy: Currency, strike: float, maturity: date) -> FXForward:
    """Build a buy-base FX forward at `strike` (quote units per base unit): the
    quote leg is `base_notional * strike`."""
    return FXForward(
        base=base,
        quote=Money(base.amount * strike, quote_ccy),
        maturity=maturity,
    )
