"""Loan portfolio analytics: aggregation, risk decomposition, WAM/WAL.

    from pricebook.loan_portfolio import LoanBook

    book = LoanBook("CLO_1")
    book.add(participation_1)
    book.add(trs_1)
    print(book.pv(ctx))
    print(book.wal(disc))

References:
    LSTA (2022). The Handbook of Loan Syndications and Trading, Ch. 20.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.serialisable import _register


@dataclass
class LoanBookSummary:
    """Summary analytics for a loan book."""
    total_pv: float
    total_notional: float
    total_funded: float
    total_unfunded: float
    n_positions: int
    wam: float                  # weighted average maturity (years)
    wal: float                  # weighted average life (years)
    avg_spread: float           # notional-weighted average spread

    def to_dict(self) -> dict:
        return {
            "total_pv": self.total_pv, "total_notional": self.total_notional,
            "total_funded": self.total_funded, "total_unfunded": self.total_unfunded,
            "n_positions": self.n_positions, "wam": self.wam, "wal": self.wal,
            "avg_spread": self.avg_spread,
        }


class LoanBook:
    """Container for loan positions with portfolio-level analytics.

    Holds any mix of TermLoan, RevolvingFacility, LoanParticipation,
    PartialFundedParticipation, or TRS-on-loan positions.

    Args:
        name: book identifier.
    """

    _SERIAL_TYPE = "loan_book"

    def __init__(self, name: str = "loan_book"):
        self.name = name
        self._positions: list[dict] = []  # {name, instrument, notional}

    def add(
        self,
        instrument,
        name: str = "",
        notional_override: float | None = None,
    ) -> None:
        """Add a position to the book."""
        notional = notional_override or getattr(instrument, "notional", 0) or \
                   getattr(instrument, "funded_amount", 0) or 1_000_000
        self._positions.append({
            "name": name or f"pos_{len(self._positions)}",
            "instrument": instrument,
            "notional": notional,
        })

    @property
    def n_positions(self) -> int:
        return len(self._positions)

    def pv(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        survival_curve: SurvivalCurve | None = None,
    ) -> float:
        """Total PV of all positions.

        Dispatches by checking instrument type explicitly (not try/except).
        """
        from pricebook.loan_participation import LoanParticipation, PartialFundedParticipation

        total = 0.0
        for pos in self._positions:
            inst = pos["instrument"]
            if isinstance(inst, PartialFundedParticipation):
                total += inst.total_pv(discount_curve, projection_curve, survival_curve)
            elif isinstance(inst, LoanParticipation):
                total += inst.pv(discount_curve, projection_curve, survival_curve).pv
            elif hasattr(inst, "pv"):
                # TermLoan, RevolvingFacility, etc.
                try:
                    total += inst.pv(discount_curve, projection_curve)
                except TypeError:
                    total += inst.pv(discount_curve)
        return total

    def summary(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> LoanBookSummary:
        """Compute portfolio-level summary analytics."""
        total_pv = self.pv(discount_curve, projection_curve)
        total_notional = sum(p["notional"] for p in self._positions)

        # Funded vs unfunded
        total_funded = 0.0
        total_unfunded = 0.0
        for p in self._positions:
            inst = p["instrument"]
            if hasattr(inst, "funded_amount"):
                total_funded += inst.funded_amount
            elif hasattr(inst, "cash_outlay"):
                total_funded += inst.cash_outlay
                total_unfunded += inst.notional - inst.cash_outlay
            else:
                total_funded += p["notional"]

        # WAM: weighted average maturity
        wam_num = 0.0
        wam_den = 0.0
        for p in self._positions:
            inst = p["instrument"]
            underlying = getattr(inst, "underlying", inst)
            end = getattr(underlying, "end", None)
            start = getattr(underlying, "start", None)
            if end and start:
                mat = year_fraction(start, end, DayCountConvention.ACT_365_FIXED)
                wam_num += p["notional"] * mat
                wam_den += p["notional"]
        wam = wam_num / wam_den if wam_den > 0 else 0.0

        # WAL: use loan WAL if available
        wal_num = 0.0
        wal_den = 0.0
        for p in self._positions:
            inst = p["instrument"]
            underlying = getattr(inst, "underlying", inst)
            if hasattr(underlying, "weighted_average_life"):
                try:
                    proj = projection_curve or discount_curve
                    w = underlying.weighted_average_life(proj)
                    wal_num += p["notional"] * w
                    wal_den += p["notional"]
                except Exception:
                    pass
        wal = wal_num / wal_den if wal_den > 0 else wam

        # Average spread
        spread_num = 0.0
        spread_den = 0.0
        for p in self._positions:
            inst = p["instrument"]
            underlying = getattr(inst, "underlying", inst)
            spread = getattr(underlying, "spread", getattr(underlying, "drawn_spread", 0))
            if spread:
                spread_num += p["notional"] * spread
                spread_den += p["notional"]
        avg_spread = spread_num / spread_den if spread_den > 0 else 0.0

        return LoanBookSummary(
            total_pv=total_pv, total_notional=total_notional,
            total_funded=total_funded, total_unfunded=total_unfunded,
            n_positions=self.n_positions, wam=wam, wal=wal,
            avg_spread=avg_spread,
        )

    def concentration(self, top_n: int = 10) -> list[dict]:
        """Top-N positions by notional as % of total."""
        total = sum(p["notional"] for p in self._positions) or 1
        sorted_pos = sorted(self._positions, key=lambda p: -p["notional"])
        return [{"name": p["name"], "notional": p["notional"],
                 "pct": p["notional"] / total * 100}
                for p in sorted_pos[:top_n]]

    def pv_ctx(self, ctx) -> float:
        proj = None
        if ctx.projection_curves:
            proj = next(iter(ctx.projection_curves.values()), None)
        sc = None
        if hasattr(ctx, "credit_curves") and ctx.credit_curves:
            sc = next(iter(ctx.credit_curves.values()), None)
        return self.pv(ctx.discount_curve, proj, sc)

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        positions = []
        for p in self._positions:
            positions.append({
                "name": p["name"],
                "instrument": p["instrument"].to_dict(),
                "notional": p["notional"],
            })
        return {"type": self._SERIAL_TYPE, "params": {
            "name": self.name, "positions": positions,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> LoanBook:
        from pricebook.serialisable import from_dict as _fd
        p = d["params"]
        book = cls(name=p.get("name", "loan_book"))
        for pos in p.get("positions", []):
            inst = _fd(pos["instrument"])
            book.add(inst, name=pos.get("name", ""),
                     notional_override=pos.get("notional"))
        return book


_register(LoanBook)
