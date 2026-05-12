"""Treasury quoting conventions: 32nds, reopenings, delivery options.

* :func:`to_32nds` — decimal price → 32nds string (e.g., 99.50 → "99-16").
* :func:`from_32nds` — 32nds string → decimal price.
* :func:`to_32nds_plus` — with half-32nds (e.g., "99-16+" = 99.515625).
* :class:`TreasuryReopen` — reopening vs new issue logic.
* :func:`delivery_option_value` — wild card + quality option for futures.

References:
    Tuckman & Serrat (2012). Fixed Income Securities, Ch. 14-15.
    SIFMA. US Treasury Securities Settlement Practices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


# ---------------------------------------------------------------------------
# 32nds quoting
# ---------------------------------------------------------------------------

def to_32nds(decimal_price: float) -> str:
    """Convert decimal price to 32nds notation.

    99.50    → "99-16"
    100.0    → "100-00"
    99.515625 → "99-16+"
    99.7890625 → "99-25+"  (rounded to nearest half-32nd)

    Treasury convention: handle = integer part, fraction = 32nds.
    A "+" suffix means +1/64 (half a 32nd).
    """
    handle = int(decimal_price)
    remainder = decimal_price - handle
    ticks_64 = round(remainder * 64)
    ticks_32 = ticks_64 // 2
    has_plus = ticks_64 % 2 == 1
    suffix = "+" if has_plus else ""
    return f"{handle}-{ticks_32:02d}{suffix}"


def from_32nds(quote: str) -> float:
    """Convert 32nds notation to decimal price.

    "99-16"  → 99.50
    "99-16+" → 99.515625
    "100-00" → 100.0
    "98-08+" → 98.265625
    """
    quote = quote.strip()
    has_plus = quote.endswith("+")
    if has_plus:
        quote = quote[:-1]

    parts = quote.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid 32nds quote: {quote!r}. Expected 'handle-ticks' format.")
    handle = int(parts[0])
    ticks_32 = int(parts[1])

    if not 0 <= ticks_32 <= 31:
        raise ValueError(f"Ticks must be 0-31, got {ticks_32}")

    decimal = handle + ticks_32 / 32.0
    if has_plus:
        decimal += 1 / 64.0
    return decimal


def tick_value(face: float = 1_000_000) -> float:
    """Dollar value of one 32nd for given face amount.

    1/32 of 1% of face. For $1M face: $312.50.
    """
    return face / 32 / 100


def tick_value_half(face: float = 1_000_000) -> float:
    """Dollar value of one half-32nd (1/64) for given face amount."""
    return face / 64 / 100


# ---------------------------------------------------------------------------
# Reopening logic
# ---------------------------------------------------------------------------

@dataclass
class TreasuryReopen:
    """Distinguishes new issue vs reopening of an existing CUSIP.

    A reopening adds supply to an existing bond (same coupon, maturity).
    The new tranche has the same CUSIP but a different issue date and
    potentially different price (OID/premium).

    Args:
        original_issue: date of the original auction.
        reopen_date: date of the reopening auction.
        original_coupon: coupon rate set at original auction.
        reopen_yield: yield at the reopening auction.
        original_outstanding: face amount outstanding before reopen.
        reopen_amount: face amount added in reopen.
    """
    original_issue: date
    reopen_date: date
    original_coupon: float
    reopen_yield: float
    original_outstanding: float
    reopen_amount: float

    @property
    def is_premium(self) -> bool:
        """Reopen at a premium (yield < coupon → price > par)."""
        return self.reopen_yield < self.original_coupon

    @property
    def is_discount(self) -> bool:
        """Reopen at a discount (yield > coupon → price < par)."""
        return self.reopen_yield > self.original_coupon

    @property
    def reopen_price_approx(self) -> float:
        """Approximate reopen price from yield vs coupon.

        Simplified: P ≈ 100 + (coupon - yield) × duration.
        For exact, use bond.dirty_price(curve).
        """
        # Rough approximation assuming 5Y average duration
        dur_approx = 4.5
        return 100.0 + (self.original_coupon - self.reopen_yield) * dur_approx * 100

    @property
    def total_outstanding(self) -> float:
        return self.original_outstanding + self.reopen_amount

    @property
    def weighted_avg_price(self) -> float:
        """WAP of original (at par) + reopen (at auction price)."""
        orig_price = 100.0
        reopen_price = self.reopen_price_approx
        total = self.total_outstanding
        if total <= 0:
            return 100.0
        return (self.original_outstanding * orig_price +
                self.reopen_amount * reopen_price) / total

    def to_dict(self) -> dict:
        return {
            "original_issue": self.original_issue.isoformat(),
            "reopen_date": self.reopen_date.isoformat(),
            "coupon": self.original_coupon,
            "reopen_yield": self.reopen_yield,
            "is_premium": self.is_premium,
            "reopen_price": self.reopen_price_approx,
            "total_outstanding": self.total_outstanding,
        }


# ---------------------------------------------------------------------------
# Delivery option value (futures)
# ---------------------------------------------------------------------------

@dataclass
class DeliveryOptionResult:
    """Treasury futures delivery option decomposition."""
    quality_option: float        # value of choosing which bond to deliver
    timing_option: float         # value of choosing when in delivery month
    wild_card_option: float      # value of 2:00-8:00 PM price move
    total_option_value: float
    ctd_net_basis: float

    def to_dict(self) -> dict:
        return {
            "quality_option": self.quality_option,
            "timing_option": self.timing_option,
            "wild_card_option": self.wild_card_option,
            "total": self.total_option_value,
            "ctd_net_basis": self.ctd_net_basis,
        }


def delivery_option_value(
    ctd_gross_basis: float,
    ctd_carry: float,
    futures_vol: float = 0.04,
    days_to_delivery: int = 30,
    n_deliverables: int = 5,
    yield_spread_vol: float = 0.003,
) -> DeliveryOptionResult:
    """Estimate delivery option value for Treasury bond futures.

    Net basis = gross basis - carry = delivery option value.
    Decompose into quality + timing + wild card.

    Quality option: value of switching CTD as yields move.
        Approximated as spread_vol × sqrt(T) × duration_diff.
    Timing option: value of choosing delivery date within the month.
        Approximately carry × (days_in_month / 365).
    Wild card: after 2PM futures close, seller has until 8PM to declare.
        Approximately futures_vol × sqrt(6h/252d) × futures_price.

    Args:
        ctd_gross_basis: gross basis of CTD bond (price - CF × futures).
        ctd_carry: carry of CTD over delivery period.
        futures_vol: annualised vol of futures price.
        days_to_delivery: business days to first delivery.
        n_deliverables: number of eligible deliverable bonds.
        yield_spread_vol: vol of yield spread between deliverables.
    """
    net_basis = ctd_gross_basis - ctd_carry

    # Quality option: proportional to number of alternatives and spread vol
    T = days_to_delivery / 252.0
    quality = yield_spread_vol * math.sqrt(max(T, 1e-10)) * 100 * math.log(max(n_deliverables, 1))

    # Timing option: carry accrual choice
    timing = abs(ctd_carry) * 0.1  # ~10% of carry value

    # Wild card: 6-hour window, ~0.024 of a trading day
    wild_card_T = 6.0 / (252 * 6.5)  # 6 hours / annual trading hours
    wild_card = futures_vol * math.sqrt(wild_card_T) * 100  # in price points

    total = quality + timing + wild_card

    return DeliveryOptionResult(
        quality_option=float(quality),
        timing_option=float(timing),
        wild_card_option=float(wild_card),
        total_option_value=float(total),
        ctd_net_basis=float(net_basis),
    )
