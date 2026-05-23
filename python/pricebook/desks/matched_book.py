"""Matched-book optimization for IB financing desks.

    from pricebook.desks.matched_book import (
        MatchedBookPosition, matched_book_pnl, matched_book_optimise,
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchedBookPosition:
    """A paired repo + reverse repo."""
    bond_id: str
    repo_rate: float             # rate paid (borrow cash)
    reverse_rate: float          # rate received (lend cash)
    repo_days: int
    reverse_days: int
    notional: float
    currency: str = "USD"

    @property
    def spread(self) -> float:
        """Spread capture (reverse - repo) in decimal."""
        return self.reverse_rate - self.repo_rate

    @property
    def spread_bp(self) -> float:
        return self.spread * 10_000

    @property
    def duration_gap_days(self) -> int:
        """Maturity mismatch (term transformation risk)."""
        return self.reverse_days - self.repo_days

    def pnl(self) -> float:
        """P&L from spread capture over holding period."""
        t = min(self.repo_days, self.reverse_days) / 360.0
        return self.notional * self.spread * t

    def to_dict(self) -> dict:
        return {
            "bond_id": self.bond_id,
            "spread_bp": self.spread_bp,
            "duration_gap_days": self.duration_gap_days,
            "pnl": self.pnl(),
            "notional": self.notional,
        }


def matched_book_pnl(positions: list[MatchedBookPosition]) -> dict:
    """Aggregate matched-book P&L."""
    total_pnl = sum(p.pnl() for p in positions)
    total_notional = sum(p.notional for p in positions)
    avg_spread = sum(p.spread_bp * p.notional for p in positions) / max(total_notional, 1)
    max_gap = max((abs(p.duration_gap_days) for p in positions), default=0)

    return {
        "total_pnl": total_pnl,
        "total_notional": total_notional,
        "avg_spread_bp": avg_spread,
        "max_duration_gap_days": max_gap,
        "n_positions": len(positions),
    }


def matched_book_optimise(
    opportunities: list[dict],
    max_notional: float,
    max_gap_days: int = 30,
    min_spread_bp: float = 1.0,
) -> list[MatchedBookPosition]:
    """Select matched-book trades to maximize spread income.

    Greedy: pick highest spread first, subject to gap and notional limits.

    Args:
        opportunities: list of {bond_id, repo_rate, reverse_rate, repo_days,
                                reverse_days, notional, currency}.
        max_notional: total notional limit.
        max_gap_days: maximum maturity mismatch allowed.
        min_spread_bp: minimum spread to accept.
    """
    # Filter and sort by spread
    valid = [
        o for o in opportunities
        if (o["reverse_rate"] - o["repo_rate"]) * 10_000 >= min_spread_bp
        and abs(o.get("reverse_days", o["repo_days"]) - o["repo_days"]) <= max_gap_days
    ]
    valid.sort(key=lambda o: -(o["reverse_rate"] - o["repo_rate"]))

    selected = []
    remaining = max_notional
    for o in valid:
        use = min(o["notional"], remaining)
        if use <= 0:
            break
        selected.append(MatchedBookPosition(
            bond_id=o["bond_id"],
            repo_rate=o["repo_rate"],
            reverse_rate=o["reverse_rate"],
            repo_days=o["repo_days"],
            reverse_days=o.get("reverse_days", o["repo_days"]),
            notional=use,
            currency=o.get("currency", "USD"),
        ))
        remaining -= use

    return selected
