"""Credit event and auction simulation.

ISDA credit event processing: event determination, two-stage auction
(initial bidding + Dutch auction), and settlement.

* :class:`CreditEvent` — credit event specification.
* :class:`AuctionResult` — auction outcome.
* :func:`simulate_auction` — two-stage ISDA auction simulation.
* :func:`settlement_amount` — CDS settlement from auction price.
* :class:`CreditEventTimeline` — event → determination → auction → settlement.

References:
    ISDA, *Credit Derivatives Definitions*, 2014.
    ISDA, *Credit Event Auction Supplement*, 2009.
    Markit, *Credit Event Processing*, 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import numpy as np


class EventType(Enum):
    """ISDA credit event types."""
    BANKRUPTCY = "bankruptcy"
    FAILURE_TO_PAY = "failure_to_pay"
    RESTRUCTURING = "restructuring"
    OBLIGATION_ACCELERATION = "obligation_acceleration"
    REPUDIATION_MORATORIUM = "repudiation_moratorium"
    GOVERNMENT_INTERVENTION = "government_intervention"


class SettlementType(Enum):
    """CDS settlement method."""
    AUCTION = "auction"
    PHYSICAL = "physical"
    CASH = "cash"


@dataclass
class CreditEvent:
    """Credit event specification."""
    entity_name: str
    event_type: EventType
    event_date: date
    determination_date: date | None = None
    auction_date: date | None = None
    settlement_date: date | None = None
    affected_notional: float = 0.0

    def to_dict(self) -> dict:
        return {
            "entity_name": self.entity_name,
            "event_type": self.event_type.value,
            "event_date": self.event_date.isoformat(),
            "determination_date": self.determination_date.isoformat() if self.determination_date else None,
            "auction_date": self.auction_date.isoformat() if self.auction_date else None,
        }


@dataclass
class AuctionResult:
    """ISDA auction result."""
    initial_midpoint: float     # initial bidding stage midpoint
    final_price: float          # final auction price (% of par)
    open_interest: float        # net open interest from initial stage
    n_dealers: int
    recovery_rate: float        # final_price / 100
    is_buy_side: bool           # True if net open interest is to buy

    def to_dict(self) -> dict:
        return {
            "initial_midpoint": self.initial_midpoint,
            "final_price": self.final_price,
            "open_interest_pct": self.open_interest,
            "n_dealers": self.n_dealers,
            "recovery_rate": self.recovery_rate,
        }


def simulate_auction(
    expected_recovery: float = 0.40,
    recovery_vol: float = 0.10,
    n_dealers: int = 14,
    seed: int | None = None,
) -> AuctionResult:
    """Simulate a two-stage ISDA credit event auction.

    Stage 1 — Initial Bidding:
    Each dealer submits a bid/offer for deliverable obligations.
    The midpoint establishes the initial market price (IMM).

    Stage 2 — Dutch Auction:
    Net open interest (physical settlement requests minus
    limit orders) is filled via a Dutch auction.
    The final price clears the open interest.

    Args:
        expected_recovery: expected recovery rate (0-1).
        recovery_vol: volatility around expected recovery.
        n_dealers: number of participating dealers.
        seed: random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Stage 1: Initial bidding — dealers quote around expected recovery
    expected_price = expected_recovery * 100
    dealer_mids = rng.normal(expected_price, recovery_vol * 100, n_dealers)
    dealer_mids = np.clip(dealer_mids, 0, 100)
    initial_midpoint = float(np.median(dealer_mids))

    # Net open interest: fraction of par
    # Positive = net buy (more physical settlement requests than limit orders)
    oi_mean = 0.0
    oi_vol = 0.15
    open_interest = float(np.clip(rng.normal(oi_mean, oi_vol), -1.0, 1.0))
    is_buy_side = open_interest > 0

    # Stage 2: Dutch auction — price adjusts to clear open interest
    # If net buy: price goes up (sellers attracted at higher price)
    # If net sell: price goes down (buyers attracted at lower price)
    price_impact = open_interest * recovery_vol * 100 * 0.5
    final_price = initial_midpoint + price_impact
    final_price = float(np.clip(final_price, 0, 100))

    return AuctionResult(
        initial_midpoint=round(initial_midpoint, 4),
        final_price=round(final_price, 4),
        open_interest=round(abs(open_interest), 4),
        n_dealers=n_dealers,
        recovery_rate=final_price / 100,
        is_buy_side=is_buy_side,
    )


def settlement_amount(
    notional: float,
    auction_price: float,
    is_protection_buyer: bool = True,
) -> float:
    """CDS settlement amount from auction final price.

    Protection buyer receives: notional × (1 − final_price/100).
    Protection seller pays the same amount.

    Args:
        notional: CDS notional.
        auction_price: final auction price (% of par, 0-100).
        is_protection_buyer: True if calculating for protection buyer.
    """
    recovery = auction_price / 100.0
    payout = notional * (1 - recovery)
    return payout if is_protection_buyer else -payout


@dataclass
class CreditEventTimeline:
    """Full timeline from credit event to settlement."""
    event: CreditEvent
    determination_days: int = 5     # business days to determination
    auction_days: int = 30          # calendar days to auction
    settlement_days: int = 3        # business days after auction

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
            "determination_days": self.determination_days,
            "auction_days": self.auction_days,
            "settlement_days": self.settlement_days,
            "timeline": self.timeline(),
        }

    def timeline(self) -> dict[str, str]:
        """Compute key dates from the event date."""
        event_date = self.event.event_date
        det = event_date + timedelta(days=self.determination_days)
        auction = det + timedelta(days=self.auction_days)
        settle = auction + timedelta(days=self.settlement_days)

        if self.event.determination_date:
            det = self.event.determination_date
        if self.event.auction_date:
            auction = self.event.auction_date
        if self.event.settlement_date:
            settle = self.event.settlement_date

        return {
            "event_date": event_date.isoformat(),
            "determination_date": det.isoformat(),
            "auction_date": auction.isoformat(),
            "settlement_date": settle.isoformat(),
        }


def process_credit_event(
    event: CreditEvent,
    cds_notional: float,
    is_protection_buyer: bool = True,
    expected_recovery: float = 0.40,
    seed: int | None = None,
) -> dict:
    """Process a credit event end-to-end.

    Simulates auction and computes settlement.

    Args:
        event: credit event specification.
        cds_notional: CDS notional.
        is_protection_buyer: True if the party is protection buyer.
        expected_recovery: expected recovery for auction simulation.
        seed: random seed.

    Returns:
        Dictionary with timeline, auction result, and settlement amount.
    """
    timeline = CreditEventTimeline(event)
    auction = simulate_auction(expected_recovery, seed=seed)
    settle = settlement_amount(cds_notional, auction.final_price, is_protection_buyer)

    return {
        "timeline": timeline.timeline(),
        "auction": auction.to_dict(),
        "settlement_amount": settle,
        "is_protection_buyer": is_protection_buyer,
    }
