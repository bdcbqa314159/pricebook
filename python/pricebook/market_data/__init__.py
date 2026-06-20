"""Pricebook market-data layer (DESIGN.md §1.3 L1, §5.1 A2, §6 G1 P2).

This is the canonical raw-market-data layer — separate from L2 curves
(which are *fit* to data) and L3 models (which have dynamics). Quotes are
what you *observe*; curves are what you *infer*. The split forces honest
attribution of every number to its source.

Public API:

    from pricebook.market_data import (
        Quote,
        QuoteId,
        QuoteKind,
        MarketSnapshot,
        FixingHistory,
    )

A `MarketSnapshot` is frozen, dated, has a UUID. Curves built from a
snapshot record `MarketSnapshot.id` in their `CalibrationResult.market_snapshot_id`
field — this is how the audit chain extends from price → calibration →
market snapshot.

This module defines the *types*; integration with the existing curve
bootstrap entry points lands in subsequent slices of G1 P2.
"""

from pricebook.market_data._types import (
    FixingHistory,
    MarketSnapshot,
    MissingQuoteError,
    Quote,
    QuoteId,
    QuoteKind,
    tenor_to_date,
    tenor_to_years,
)

__all__ = [
    "FixingHistory",
    "MarketSnapshot",
    "MissingQuoteError",
    "Quote",
    "QuoteId",
    "QuoteKind",
    "tenor_to_date",
    "tenor_to_years",
]
