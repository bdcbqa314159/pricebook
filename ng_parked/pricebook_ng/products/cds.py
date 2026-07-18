"""Credit default swap as pure data (L2).

A `CDS` is the protection buyer's contract: a running `spread` paid on a premium
schedule in exchange for `(1 - R)` on default, to `maturity`. Pure data — no `pv`
method (CLAUDE.md 2); the CDS leg math and pricing live in L1/L4.

`buy_protection=False` flips it to the protection seller. The `issuer` names the
reference entity — the engine looks up its survival curve at
`MarketKey(CREDIT, issuer)` (A5, multi-issuer). Premium is the CDS convention:
annual, ACT/360 accrual (applied in the L1 leg math).

Provenance:
  quarry: python/pricebook/credit/ (CDS)
  source: standard single-name CDS
  oracle: par CDS reprices to zero through the CDSEngine (cds-product slice)
  slice:  cds-product; A5 (issuer -> keyed survival lookup)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.schedule import Frequency, generate_schedule


@dataclass(frozen=True)
class CDS:
    """A single-name CDS on `issuer`: premium schedule + running spread on `face`
    (notional + currency), buyer/seller."""

    issuer: str
    premium_schedule: tuple[date, ...]
    spread: float
    face: Money
    buy_protection: bool = True


def cds(face: Money, issuer: str, spread: float, start: date, maturity: date) -> CDS:
    """Build a protection-buyer CDS on `issuer` (annual ACT/360 premiums); flip with
    `dataclasses.replace(cds, buy_protection=False)` for the seller."""
    schedule = generate_schedule(start, maturity, Frequency.ANNUAL)
    return CDS(issuer=issuer, premium_schedule=tuple(schedule), spread=spread, face=face)
