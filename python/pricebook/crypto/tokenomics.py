"""Tokenomics: supply schedules, vesting, inflation, valuation.

* :func:`token_supply_schedule` — circulating supply over time.
* :func:`vesting_schedule` — team/investor token unlocks.
* :func:`token_inflation_rate` — annual inflation from emissions.
* :func:`token_dcf` — DCF-style token valuation from protocol revenue.

References:
    Coingecko, *Understanding Tokenomics*.
    Messari, *Token Terminal — Protocol Revenue*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SupplySchedule:
    """Token supply schedule over time."""
    months: list[int]
    circulating: list[float]
    locked: list[float]
    total: list[float]
    fully_diluted: float

    def dilution_pct(self, month: int) -> float:
        """% of fully diluted supply that's circulating at month."""
        idx = min(month, len(self.circulating) - 1)
        return self.circulating[idx] / self.fully_diluted * 100 if self.fully_diluted > 0 else 0

    def to_dict(self) -> dict:
        return {
            "fully_diluted": self.fully_diluted,
            "current_circulating": self.circulating[-1] if self.circulating else 0,
            "dilution_pct": self.dilution_pct(len(self.months) - 1),
        }


def token_supply_schedule(
    initial_circulating: float,
    fully_diluted: float,
    monthly_emissions: float,
    vesting_unlocks: list[tuple[int, float]] | None = None,
    burn_rate: float = 0.0,
    months: int = 48,
) -> SupplySchedule:
    """Generate token supply schedule.

    Args:
        initial_circulating: tokens in circulation at month 0.
        fully_diluted: maximum possible supply.
        monthly_emissions: new tokens per month (mining/staking rewards).
        vesting_unlocks: list of (month, amount) for vesting cliff unlocks.
        burn_rate: monthly token burn rate (fraction of circulating).
        months: projection horizon.
    """
    unlock_map = {m: a for m, a in (vesting_unlocks or [])}

    circ = [initial_circulating]
    locked_list = [fully_diluted - initial_circulating]

    for m in range(1, months + 1):
        new_circ = circ[-1] + monthly_emissions
        new_circ += unlock_map.get(m, 0)
        new_circ -= circ[-1] * burn_rate
        new_circ = min(new_circ, fully_diluted)
        circ.append(new_circ)
        locked_list.append(fully_diluted - new_circ)

    m_list = list(range(months + 1))
    total = [c + l for c, l in zip(circ, locked_list)]

    return SupplySchedule(m_list, circ, locked_list, total, fully_diluted)


@dataclass
class VestingEvent:
    """Single vesting unlock event."""
    month: int
    amount: float
    recipient: str
    pct_of_supply: float

    def to_dict(self) -> dict:
        return vars(self)


def vesting_schedule(
    allocations: list[dict],
    fully_diluted: float,
) -> list[VestingEvent]:
    """Generate vesting schedule from allocation table.

    Each allocation: {"recipient", "amount", "cliff_months", "vesting_months"}.

    Args:
        allocations: list of allocation dicts.
        fully_diluted: total supply for % calculation.
    """
    events = []
    for alloc in allocations:
        recipient = alloc["recipient"]
        total = alloc["amount"]
        cliff = alloc.get("cliff_months", 12)
        vest = alloc.get("vesting_months", 36)

        if vest <= 0:
            events.append(VestingEvent(cliff, total, recipient, total / fully_diluted * 100))
            continue

        monthly = total / vest
        for m in range(cliff, cliff + vest):
            events.append(VestingEvent(m, monthly, recipient, monthly / fully_diluted * 100))

    events.sort(key=lambda e: e.month)
    return events


def token_inflation_rate(
    circulating: float,
    annual_emissions: float,
    annual_burns: float = 0.0,
) -> dict:
    """Net inflation rate from emissions minus burns.

    net_inflation = (emissions − burns) / circulating × 100.

    Args:
        circulating: current circulating supply.
        annual_emissions: new tokens per year.
        annual_burns: tokens burned per year.
    """
    net = annual_emissions - annual_burns
    rate = net / circulating * 100 if circulating > 0 else 0
    return {
        "gross_inflation_pct": annual_emissions / circulating * 100 if circulating > 0 else 0,
        "burn_rate_pct": annual_burns / circulating * 100 if circulating > 0 else 0,
        "net_inflation_pct": rate,
        "deflationary": net < 0,
    }


@dataclass
class TokenDCFResult:
    """Token DCF valuation result."""
    fair_value_per_token: float
    total_value: float
    terminal_value: float
    discount_rate: float
    revenue_multiple: float

    def to_dict(self) -> dict:
        return vars(self)


def token_dcf(
    annual_revenue: float,
    revenue_growth: float = 0.20,
    discount_rate: float = 0.30,
    terminal_growth: float = 0.03,
    projection_years: int = 5,
    circulating_supply: float = 1_000_000_000,
    revenue_to_token_pct: float = 0.50,
) -> TokenDCFResult:
    """DCF-style token valuation from protocol revenue.

    Discounts future protocol revenue attributable to token holders.
    Higher discount rate than TradFi (30%+ typical for crypto).

    Args:
        annual_revenue: current annual protocol revenue.
        revenue_growth: annual revenue growth rate.
        discount_rate: risk-adjusted discount rate.
        terminal_growth: long-run growth (terminal value).
        projection_years: explicit projection period.
        circulating_supply: tokens in circulation.
        revenue_to_token_pct: fraction of revenue accruing to token holders.
    """
    cashflows = []
    rev = annual_revenue * revenue_to_token_pct
    pv_sum = 0.0

    for y in range(1, projection_years + 1):
        rev *= (1 + revenue_growth)
        df = 1 / (1 + discount_rate) ** y
        pv_sum += rev * df
        cashflows.append(rev)

    # Terminal value
    if discount_rate > terminal_growth:
        tv = cashflows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        tv_pv = tv / (1 + discount_rate) ** projection_years
    else:
        tv_pv = 0

    total = pv_sum + tv_pv
    per_token = total / circulating_supply if circulating_supply > 0 else 0
    multiple = total / (annual_revenue * revenue_to_token_pct) if annual_revenue > 0 else 0

    return TokenDCFResult(per_token, total, tv_pv, discount_rate, multiple)
