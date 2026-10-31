"""Fixed cashflow (L2) — pure data.

**Products describe; they do not price.** There is no `pv()` here and there will not be one: a
product that prices itself has to reach for market data, which puts valuation in the layer that
defines contracts and makes a model/market mismatch expressible. Pricing lives in the engine.

**Serialisation is here from the first product, deliberately.** The scope line makes *how a
product and a pricing request are represented* the library's durable public contract, and the
predecessor learned that adding serialisation per-product while already inside the module beats
the mass retrofit. A pricing server is the consumer that will prove the format; this is the
format it will consume.

Provenance:
  stock:  — (built fresh; the stock's `products/cash.py` is Deposit/FRA/Future, not a bare flow)
  source: CLAUDE.md-era §2 ("products are pure data") + the scope line's representation contract
  oracle: round-trips through its serialised form; priced against `N·e^{−rt}` to 1e-12
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.cashflow import SCHEMA_VERSION, Cashflow


@dataclass(frozen=True)
class FixedCashflow:
    """A single known payment — the simplest priceable thing there is.

    One field, which looks like a pointless wrapper and is not: the *product* is what a pricing
    request names and what the engine dispatches on, while `Cashflow` is the L0 atom that legs and
    schedules will also be built from. Collapsing them would mean the engine dispatches on an atom
    shared by every future product."""

    cashflow: Cashflow

    @property
    def currency(self):
        """The currency to discount in — read from the flow rather than duplicated as a field, so
        the two cannot disagree."""
        return self.cashflow.amount.currency

    def to_dict(self) -> dict:
        return {"kind": "fixed_cashflow", "schema_version": SCHEMA_VERSION,
                "cashflow": self.cashflow.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> FixedCashflow:
        if d.get("kind") != "fixed_cashflow":
            raise ValueError(f"not a fixed_cashflow payload: {d.get('kind')!r}")
        return cls(Cashflow.from_dict(d["cashflow"]))
