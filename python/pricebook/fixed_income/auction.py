"""Bond auction theory — uniform/discriminatory pricing, winner's curse.

    from pricebook.fixed_income.auction import (
        BondAuction, AuctionResult, winners_curse_adjustment,
    )

References:
    Milgrom & Weber (1982). A Theory of Auctions and Competitive Bidding.
    Hortacsu & McAdams (2010). Mechanism Choice and Strategic Bidding
    in Divisible Good Auctions. JFE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Bid:
    """A single auction bid."""
    bidder: str
    price: float                 # bid price (per 100 face)
    quantity: float              # face amount bid
    is_competitive: bool = True  # non-competitive bids filled at clearing

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class AuctionResult:
    """Result of a bond auction."""
    clearing_price: float        # uniform: single price; discriminatory: weighted avg
    total_issued: float          # total face amount allocated
    total_bid: float             # total face amount bid
    bid_to_cover: float          # bid/issued ratio
    tail: float                  # clearing - highest accepted (bp)
    allocations: list[dict]      # {bidder, price, quantity_bid, quantity_filled}
    method: str                  # "uniform" or "discriminatory"

    def to_dict(self) -> dict:
        return {
            "clearing_price": self.clearing_price,
            "total_issued": self.total_issued,
            "bid_to_cover": self.bid_to_cover,
            "tail": self.tail,
            "method": self.method,
            "n_bidders": len(self.allocations),
        }


class BondAuction:
    """Bond auction with uniform or discriminatory pricing."""

    def __init__(self, issue_amount: float):
        """Args: issue_amount = total face to be auctioned."""
        if issue_amount <= 0:
            raise ValueError("issue_amount must be positive")
        self.issue_amount = issue_amount

    def uniform_price(self, bids: list[Bid]) -> AuctionResult:
        """Uniform-price (Dutch) auction.

        All accepted bidders pay the clearing price (lowest accepted bid).
        Used by US Treasury, UK DMO.
        """
        return self._run(bids, "uniform")

    def discriminatory_price(self, bids: list[Bid]) -> AuctionResult:
        """Discriminatory-price (pay-your-bid) auction.

        Each bidder pays their own bid price.
        Used by some EM sovereigns.
        """
        return self._run(bids, "discriminatory")

    def _run(self, bids: list[Bid], method: str) -> AuctionResult:
        if not bids:
            raise ValueError("At least one bid required")

        # Separate competitive and non-competitive
        competitive = sorted([b for b in bids if b.is_competitive],
                             key=lambda b: -b.price)  # highest price first
        non_competitive = [b for b in bids if not b.is_competitive]

        # Non-competitive filled first
        nc_total = sum(b.quantity for b in non_competitive)
        remaining = max(self.issue_amount - nc_total, 0)

        # Fill competitive bids from highest price down
        allocations = []
        filled = 0.0
        clearing_price = 0.0

        for b in non_competitive:
            allocations.append({
                "bidder": b.bidder, "price": 0.0,  # filled at clearing
                "quantity_bid": b.quantity, "quantity_filled": b.quantity,
            })

        for b in competitive:
            can_fill = min(b.quantity, remaining - filled)
            if can_fill > 0:
                allocations.append({
                    "bidder": b.bidder, "price": b.price,
                    "quantity_bid": b.quantity, "quantity_filled": can_fill,
                })
                clearing_price = b.price  # last accepted = clearing
                filled += can_fill
            else:
                allocations.append({
                    "bidder": b.bidder, "price": b.price,
                    "quantity_bid": b.quantity, "quantity_filled": 0.0,
                })

        # Update non-competitive allocations with clearing price
        if method == "uniform":
            for a in allocations:
                if a["price"] == 0.0:
                    a["price"] = clearing_price

        total_bid = sum(b.quantity for b in bids)
        btc = total_bid / self.issue_amount if self.issue_amount > 0 else 0.0

        # Tail: highest accepted - clearing
        accepted_prices = [a["price"] for a in allocations if a["quantity_filled"] > 0 and a["price"] > 0]
        tail = (max(accepted_prices) - clearing_price) * 100 if accepted_prices else 0.0

        # Discriminatory: weighted average price
        if method == "discriminatory":
            filled_allocs = [a for a in allocations if a["quantity_filled"] > 0]
            if filled_allocs:
                total_filled = sum(a["quantity_filled"] for a in filled_allocs)
                wavg = sum(a["price"] * a["quantity_filled"] for a in filled_allocs) / total_filled
                clearing_price = wavg

        return AuctionResult(
            clearing_price=clearing_price,
            total_issued=filled + nc_total,
            total_bid=total_bid,
            bid_to_cover=btc,
            tail=tail,
            allocations=allocations,
            method=method,
        )


def winners_curse_adjustment(
    value_estimate: float,
    n_bidders: int,
    value_uncertainty: float,
) -> float:
    """Winner's curse bid shading.

    Optimal bid = value_estimate - adjustment
    adjustment ≈ σ × (n-1)/n × E[max order statistic]

    For uniform values: E[winning bid] = v - σ(n-1)/(n+1)

    Args:
        value_estimate: private value estimate.
        n_bidders: number of competing bidders.
        value_uncertainty: std of private value estimate.
    """
    if n_bidders <= 1:
        return value_estimate
    # Shading factor increases with competition
    shading = value_uncertainty * (n_bidders - 1) / (n_bidders + 1)
    return value_estimate - shading


def expected_revenue(
    n_bidders: int,
    value_mean: float,
    value_std: float,
    method: str = "uniform",
) -> float:
    """Expected auction revenue per unit.

    Revenue equivalence theorem: uniform and discriminatory yield same
    expected revenue under common assumptions.

    E[revenue] ≈ value_mean - value_std / (n_bidders + 1)
    """
    return value_mean - value_std / (n_bidders + 1)
