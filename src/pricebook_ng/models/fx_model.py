"""FXForwardModel — the model an FX forward is priced under (L3).

Amendment A1: a model carries its economy. The FX forward values in the QUOTE
currency, so `market` is the quote-currency snapshot (its `discount_curve` is the
quote curve). The base-currency `base_curve` and the `spot` (quote units per base)
are the FX-specific market data.

ponytail: base curve + spot ride on the model here, exactly as the CDS survival
curve did before ruling §5.1 promoted it to the snapshot. A multi-currency
snapshot (curves keyed by currency + FX spots) that lets FX greeks flow through
the `Priceable` protocol is the follow-up.

Provenance:
  quarry: python/pricebook/fx/; core/currency.py (forward_rate_from_curves)
  source: covered interest parity
  oracle: par FX forward reprices to zero (fx-forward slice)
  slice:  fx-forward
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.market.snapshot import CurveHandle, MarketSnapshot


@dataclass(frozen=True)
class FXForwardModel:
    """Quote-currency market + base-currency curve + spot (quote per base)."""

    market: MarketSnapshot
    base_curve: CurveHandle
    spot: float
