"""European FX option as pure data (L2).

An `FXOption` is the right to exchange `base` for `quote` at `maturity`: a call
buys the base (pays quote) if in the money, a put sells it. Pure data — no `pv`
method (CLAUDE.md 2); pricing is L4 (Garman-Kohlhagen).

Provenance:
  quarry: python/pricebook/fx/ (fx option)
  source: Garman & Kohlhagen (1983)
  oracle: put-call parity ties to the FX forward; Black recompute (fx-option slice)
  slice:  fx-option
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Currency, Money


@dataclass(frozen=True)
class FXOption:
    """A European option to exchange `base` for `quote` at `maturity` (call = buy
    the base leg, put = sell it). The strike is `quote.amount / base.amount`."""

    base: Money
    quote: Money
    maturity: date
    is_call: bool


def fx_option(
    base: Money, quote_ccy: Currency, strike: float, maturity: date, is_call: bool
) -> FXOption:
    """Build an FX option at `strike` (quote units per base unit)."""
    return FXOption(
        base=base,
        quote=Money(base.amount * strike, quote_ccy),
        maturity=maturity,
        is_call=is_call,
    )
