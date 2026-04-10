"""Commodity storage and carry plays.

* :class:`CashAndCarry` — classic arbitrage: buy spot, sell forward,
  store. Profit = forward − spot − storage − financing. Also extracts
  implied storage cost and convenience yield from the forward curve.
* :class:`StorageFacility` — a physical storage asset (natgas, oil)
  with injection/withdrawal constraints, intrinsic value from the
  deterministic carry, and extrinsic value from optionality to flex
  the schedule.

References:
    Eydeland & Wolyniec, *Energy and Power Risk Management*, Wiley, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction


# ---- Cash-and-carry ----

@dataclass
class CashAndCarryResult:
    """Outcome of a cash-and-carry trade."""
    spot: float
    forward: float
    storage_cost: float
    financing_cost: float
    profit: float
    implied_storage: float
    implied_convenience_yield: float


def cash_and_carry(
    spot: float,
    forward: float,
    rate: float,
    storage_cost_per_annum: float,
    T: float,
) -> CashAndCarryResult:
    """Cash-and-carry arbitrage analysis.

    The trader buys spot, sells forward, and stores the commodity.

        profit = forward − spot − storage − financing

    where ``financing = spot × (exp(r·T) − 1)`` and
    ``storage = storage_cost_per_annum × T``.

    The *implied storage cost* is the cost that makes the cash-and-carry
    profit exactly zero:

        implied_storage = (forward − spot × exp(r·T)) / T

    The *implied convenience yield* y satisfies ``F = S·exp((r+c−y)·T)``:

        y = r + c/S − ln(F/S) / T

    (where c = storage_cost_per_annum / spot as a rate).

    Args:
        spot: current spot price.
        forward: forward price at maturity T.
        rate: continuously compounded risk-free rate.
        storage_cost_per_annum: storage cost per unit per year.
        T: time to maturity in years.

    Returns:
        :class:`CashAndCarryResult`.
    """
    if T <= 0:
        return CashAndCarryResult(
            spot=spot, forward=forward,
            storage_cost=0.0, financing_cost=0.0, profit=0.0,
            implied_storage=0.0, implied_convenience_yield=0.0,
        )

    financing = spot * (math.exp(rate * T) - 1.0)
    storage = storage_cost_per_annum * T
    profit = forward - spot - storage - financing

    # Implied storage that zeroes the profit
    spot_fv = spot * math.exp(rate * T)
    implied_storage = (forward - spot_fv) / T if T > 0 else 0.0

    # Implied convenience yield
    if spot > 0 and forward > 0:
        c_rate = storage_cost_per_annum / spot
        implied_cy = rate + c_rate - math.log(forward / spot) / T
    else:
        implied_cy = 0.0

    return CashAndCarryResult(
        spot=spot,
        forward=forward,
        storage_cost=storage,
        financing_cost=financing,
        profit=profit,
        implied_storage=implied_storage,
        implied_convenience_yield=implied_cy,
    )


def implied_storage_cost(
    spot: float,
    forward: float,
    rate: float,
    T: float,
) -> float:
    """Storage cost (per annum) implied by the forward-spot spread.

        c = (F − S·e^{rT}) / T
    """
    if T <= 0:
        return 0.0
    return (forward - spot * math.exp(rate * T)) / T


def implied_convenience_yield(
    spot: float,
    forward: float,
    rate: float,
    storage_cost_per_annum: float,
    T: float,
) -> float:
    """Convenience yield implied by the forward curve.

        y = r + c/S − ln(F/S) / T
    """
    if T <= 0 or spot <= 0 or forward <= 0:
        return 0.0
    c_rate = storage_cost_per_annum / spot
    return rate + c_rate - math.log(forward / spot) / T


# ---- Storage facility ----

@dataclass
class StorageScheduleEntry:
    """One period in a storage schedule.

    ``flow > 0`` is injection (buy and store);
    ``flow < 0`` is withdrawal (sell from inventory).
    """
    period_start: date
    period_end: date
    flow: float          # volume injected (+) or withdrawn (−)
    forward_price: float # forward at this delivery date
    cost: float          # injection/withdrawal variable cost


@dataclass
class StorageFacility:
    """A physical commodity storage facility (e.g. salt cavern, tank farm).

    Attributes:
        capacity: maximum inventory (volume units).
        min_inventory: minimum working inventory.
        max_injection_rate: max volume per period to inject.
        max_withdrawal_rate: max volume per period to withdraw (positive).
        injection_cost: variable cost per unit injected.
        withdrawal_cost: variable cost per unit withdrawn.
    """
    capacity: float
    min_inventory: float = 0.0
    max_injection_rate: float = 0.0
    max_withdrawal_rate: float = 0.0
    injection_cost: float = 0.0
    withdrawal_cost: float = 0.0

    def intrinsic_value(
        self,
        forwards: dict[date, float],
        rate: float = 0.0,
        initial_inventory: float = 0.0,
    ) -> float:
        """Deterministic intrinsic value: buy low periods, sell high periods.

        Greedy strategy: inject during the cheapest period and withdraw
        during the most expensive, subject to capacity and rate limits.

        This is a lower bound on the full (extrinsic + intrinsic) value.
        A proper extrinsic calculation requires stochastic modelling.
        """
        sorted_fwds = sorted(forwards.items(), key=lambda kv: kv[0])
        if len(sorted_fwds) < 2:
            return 0.0

        prices = [f for _, f in sorted_fwds]
        n = len(prices)

        # Find cheapest injection period and most expensive withdrawal period
        min_price = min(prices)
        max_price = max(prices)

        if max_price <= min_price:
            return 0.0

        # Simple single-cycle: inject at min, withdraw at max
        injectable = min(
            self.max_injection_rate if self.max_injection_rate > 0 else self.capacity,
            self.capacity - initial_inventory,
        )
        withdrawable = min(
            self.max_withdrawal_rate if self.max_withdrawal_rate > 0 else self.capacity,
            initial_inventory + injectable - self.min_inventory,
        )
        volume = min(injectable, withdrawable)

        if volume <= 0:
            return 0.0

        gross = volume * (max_price - min_price)
        costs = volume * (self.injection_cost + self.withdrawal_cost)
        return gross - costs

    def extrinsic_value(
        self,
        forwards: dict[date, float],
        vol: float,
        rate: float = 0.0,
        initial_inventory: float = 0.0,
    ) -> float:
        """Approximate extrinsic value of the storage optionality.

        Uses a simple spread-option proxy: the storage can be viewed as
        a strip of calendar spread options on each pair of periods.
        The extrinsic premium is proportional to vol × sqrt(T) × spread.

        This is a simplified estimate — a full rolling-intrinsic or
        LSM-based valuation would be more accurate.
        """
        sorted_fwds = sorted(forwards.items(), key=lambda kv: kv[0])
        if len(sorted_fwds) < 2 or vol <= 0:
            return 0.0

        dates = [d for d, _ in sorted_fwds]
        prices = [f for _, f in sorted_fwds]
        ref_date = dates[0]

        total = 0.0
        for i in range(len(prices) - 1):
            spread = abs(prices[i + 1] - prices[i])
            T = year_fraction(
                ref_date, dates[i + 1], DayCountConvention.ACT_365_FIXED,
            )
            if T > 0:
                total += spread * vol * math.sqrt(T)

        volume = min(
            self.max_injection_rate if self.max_injection_rate > 0 else self.capacity,
            self.capacity - initial_inventory,
        )
        return max(volume * total, 0.0)

    def total_value(
        self,
        forwards: dict[date, float],
        vol: float = 0.0,
        rate: float = 0.0,
        initial_inventory: float = 0.0,
    ) -> float:
        """Total storage value = intrinsic + extrinsic."""
        iv = self.intrinsic_value(forwards, rate, initial_inventory)
        ev = self.extrinsic_value(forwards, vol, rate, initial_inventory)
        return iv + ev
