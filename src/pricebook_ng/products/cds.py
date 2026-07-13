"""Credit default swap as pure data (L2).

A `CDS` is the protection buyer's contract: a running `spread` paid on a premium
schedule in exchange for `(1 - R)` on default, to `maturity`. Pure data — no `pv`
method (CLAUDE.md 2); the CDS leg math and pricing live in L1/L4.

`buy_protection=False` flips it to the protection seller. Premium accrual is the
CDS ACT/360 convention (applied in the L1 leg math); the builder's `terms`
day-count is unused for CDS — only its frequency + roll set the schedule.

Provenance:
  quarry: python/pricebook/credit/ (CDS)
  source: standard single-name CDS
  oracle: par CDS reprices to zero through the CDSEngine (cds-product slice)
  slice:  cds-product
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.schedule import ScheduleTerms, generate_schedule


@dataclass(frozen=True)
class CDS:
    """A single-name CDS: premium schedule + running spread, buyer or seller."""

    premium_schedule: tuple[date, ...]
    spread: float
    notional: float
    currency: Currency
    buy_protection: bool = True


def cds(
    face: Money,
    spread: float,
    start: date,
    maturity: date,
    terms: ScheduleTerms,
) -> CDS:
    """Build a protection-buyer CDS from `start` to `maturity` (flip with
    `dataclasses.replace(cds, buy_protection=False)` for the seller)."""
    schedule = generate_schedule(start, maturity, terms.frequency, terms.roll)
    return CDS(
        premium_schedule=tuple(schedule),
        spread=spread,
        notional=face.amount,
        currency=face.currency,
    )
