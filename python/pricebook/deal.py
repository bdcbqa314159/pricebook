"""Deal structuring: groups linked instruments with roles, metadata, and aggregate risk.

A Deal is a molecule — instruments are atoms. Deals capture the
relationship between components: a swap hedging a bond, a swaption
embedded in a note, fee legs alongside principal trades.

    deal = Deal("repack_001", counterparty="ACME")
    deal.add("bond", bond_trade, role=DealRole.PRINCIPAL)
    deal.add("hedge", swap_trade, role=DealRole.HEDGE)
    deal.pv(ctx)  # aggregate PV
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

from pricebook.pricing_context import PricingContext
from pricebook.trade import Trade


class DealRole(Enum):
    PRINCIPAL = "principal"
    HEDGE = "hedge"
    FEE = "fee"
    OPTION = "option"
    COLLATERAL = "collateral"


@dataclass
class DealComponent:
    """One instrument within a deal."""

    name: str
    trade: Trade
    role: DealRole = DealRole.PRINCIPAL
    linked_to: str = ""  # name of another component this is linked to


class Deal:
    """A structured deal: collection of linked instruments.

    Args:
        deal_id: unique identifier.
        counterparty: deal counterparty.
        book: trading book.
        desk: trading desk.
    """

    def __init__(
        self,
        deal_id: str,
        counterparty: str = "",
        book: str = "",
        desk: str = "",
    ):
        self.deal_id = deal_id
        self.counterparty = counterparty
        self.book = book
        self.desk = desk
        self._components: dict[str, DealComponent] = {}

    def add(
        self,
        name: str,
        trade: Trade,
        role: DealRole = DealRole.PRINCIPAL,
        linked_to: str = "",
    ) -> None:
        """Add an instrument to the deal."""
        if name in self._components:
            raise ValueError(f"Component '{name}' already exists in deal {self.deal_id}")
        self._components[name] = DealComponent(name, trade, role, linked_to)

    def get(self, name: str) -> DealComponent:
        if name not in self._components:
            raise KeyError(f"Component '{name}' not in deal {self.deal_id}")
        return self._components[name]

    @property
    def components(self) -> dict[str, DealComponent]:
        return dict(self._components)

    @property
    def size(self) -> int:
        return len(self._components)

    def by_role(self, role: DealRole) -> list[DealComponent]:
        return [c for c in self._components.values() if c.role == role]

    # ----- Pricing -----

    def pv(self, ctx: PricingContext) -> float:
        """Aggregate PV across all components."""
        total = 0.0
        for comp in self._components.values():
            try:
                total += comp.trade.pv(ctx)
            except Exception:
                pass
        return total

    def pv_by_component(self, ctx: PricingContext) -> dict[str, float]:
        """PV broken down by component."""
        result = {}
        for name, comp in self._components.items():
            try:
                result[name] = comp.trade.pv(ctx)
            except Exception:
                result[name] = float("nan")
        return result

    def pv_by_role(self, ctx: PricingContext) -> dict[str, float]:
        """PV aggregated by role."""
        result: dict[str, float] = {}
        for comp in self._components.values():
            role_name = comp.role.value
            try:
                pv = comp.trade.pv(ctx)
            except Exception:
                pv = 0.0
            result[role_name] = result.get(role_name, 0.0) + pv
        return result

    # ----- Risk -----

    def dv01(self, ctx: PricingContext, shift: float = 0.0001) -> float:
        """Aggregate parallel DV01."""
        if ctx.discount_curve is None:
            return 0.0
        pv_base = self.pv(ctx)
        bumped = ctx.replace(discount_curve=ctx.discount_curve.bumped(shift))
        return self.pv(bumped) - pv_base

    def risk_report(self, ctx: PricingContext) -> dict[str, Any]:
        """Deal-level risk report."""
        return {
            "deal_id": self.deal_id,
            "counterparty": self.counterparty,
            "n_components": self.size,
            "total_pv": self.pv(ctx),
            "dv01": self.dv01(ctx),
            "pv_by_component": self.pv_by_component(ctx),
            "pv_by_role": self.pv_by_role(ctx),
        }

    # ----- Serialization -----

    def to_dict(self) -> dict[str, Any]:
        from pricebook.serialization import trade_to_dict
        return {
            "deal_id": self.deal_id,
            "counterparty": self.counterparty,
            "book": self.book,
            "desk": self.desk,
            "components": [
                {
                    "name": c.name,
                    "role": c.role.value,
                    "linked_to": c.linked_to,
                    "trade": trade_to_dict(c.trade),
                }
                for c in self._components.values()
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Deal:
        from pricebook.serialization import trade_from_dict
        deal = cls(
            deal_id=d["deal_id"],
            counterparty=d.get("counterparty", ""),
            book=d.get("book", ""),
            desk=d.get("desk", ""),
        )
        for comp in d.get("components", []):
            trade = trade_from_dict(comp["trade"])
            deal.add(
                name=comp["name"],
                trade=trade,
                role=DealRole(comp.get("role", "principal")),
                linked_to=comp.get("linked_to", ""),
            )
        return deal

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> Deal:
        return cls.from_dict(json.loads(s))
