"""RFR futures instruments — SOFR SR1/SR3, SONIA, ESTR futures.

Contract date generation, convexity adjustment, and conversion to
forward rates for curve bootstrap.

    from pricebook.fixed_income.rfr_futures import (
        RFRFutureSpec, generate_rfr_contracts, rfr_futures_to_forwards,
    )

References:
    CME (2024). SOFR Futures Contract Specifications.
    ICE (2024). Three Month SONIA Futures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.fixed_income.ir_futures import hw_convexity_adjustment


@dataclass
class RFRFutureSpec:
    """A single RFR futures contract."""
    currency: str               # USD, GBP, EUR
    rfr_name: str               # SOFR, SONIA, ESTR
    contract_type: str           # "1M" or "3M"
    contract_month: date         # first day of contract month
    accrual_start: date          # first day of reference period
    accrual_end: date            # last day of reference period
    price: float = 0.0          # market price (100 - rate)

    @property
    def implied_rate(self) -> float:
        return (100.0 - self.price) / 100.0

    @implied_rate.setter
    def implied_rate(self, rate: float) -> None:
        self.price = 100.0 - rate * 100.0

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "rfr_name": self.rfr_name,
            "contract_type": self.contract_type,
            "contract_month": self.contract_month.isoformat(),
            "accrual_start": self.accrual_start.isoformat(),
            "accrual_end": self.accrual_end.isoformat(),
            "price": self.price,
            "implied_rate": self.implied_rate,
        }


# ═══════════════════════════════════════════════════════════════
# Contract date generation
# ═══════════════════════════════════════════════════════════════


def _imm_date(year: int, month: int) -> date:
    """Third Wednesday of the month (IMM date)."""
    first = date(year, month, 1)
    # Find first Wednesday
    days_to_wed = (2 - first.weekday()) % 7
    first_wed = first + timedelta(days=days_to_wed)
    return first_wed + timedelta(weeks=2)


def _end_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def _next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _add_months(d: date, n: int) -> date:
    year = d.year + (d.month + n - 1) // 12
    month = (d.month + n - 1) % 12 + 1
    return date(year, month, 1)


def generate_rfr_contracts(
    currency: str,
    reference_date: date,
    n_1m: int = 6,
    n_3m: int = 8,
) -> dict[str, list[RFRFutureSpec]]:
    """Generate RFR futures contract specifications.

    Args:
        currency: "USD" (SOFR), "GBP" (SONIA), "EUR" (ESTR).
        reference_date: today's date.
        n_1m: number of 1M serial contracts to generate.
        n_3m: number of 3M quarterly contracts to generate.

    Returns:
        {"1M": [...], "3M": [...]} lists of RFRFutureSpec.
    """
    rfr_name = _CURRENCY_RFR.get(currency.upper())
    if rfr_name is None:
        raise ValueError(f"No RFR futures for {currency}. Available: {list(_CURRENCY_RFR.keys())}")

    contracts_1m = []
    contracts_3m = []

    # 1M serial contracts: next N months
    for i in range(n_1m):
        month_start = _add_months(reference_date, i + 1)
        month_end = _end_of_month(month_start.year, month_start.month)
        contracts_1m.append(RFRFutureSpec(
            currency=currency.upper(),
            rfr_name=rfr_name,
            contract_type="1M",
            contract_month=month_start,
            accrual_start=month_start,
            accrual_end=month_end,
        ))

    # 3M quarterly contracts: next N IMM quarters
    # IMM months: Mar, Jun, Sep, Dec
    imm_months = [3, 6, 9, 12]
    current = reference_date
    found = 0
    # Find next IMM month
    for offset in range(24):
        m = _add_months(current, offset + 1)
        if m.month in imm_months:
            start = m
            end = _end_of_month(_add_months(start, 2).year, _add_months(start, 2).month)
            contracts_3m.append(RFRFutureSpec(
                currency=currency.upper(),
                rfr_name=rfr_name,
                contract_type="3M",
                contract_month=start,
                accrual_start=start,
                accrual_end=end,
            ))
            found += 1
            if found >= n_3m:
                break

    return {"1M": contracts_1m, "3M": contracts_3m}


_CURRENCY_RFR = {
    "USD": "SOFR",
    "GBP": "SONIA",
    "EUR": "ESTR",
    "CHF": "SARON",
    "JPY": "TONA",
}


# ═══════════════════════════════════════════════════════════════
# Convexity adjustment
# ═══════════════════════════════════════════════════════════════


def rfr_futures_convexity(
    spec: RFRFutureSpec,
    reference_date: date,
    hw_a: float = 0.03,
    hw_sigma: float = 0.01,
) -> float:
    """Hull-White convexity adjustment for an RFR future.

    Futures rates are biased upward relative to forward rates due to
    daily margining (marking-to-market). The adjustment:

        forward_rate ≈ futures_rate - convexity_adjustment

    Uses the Hull-White analytical formula from ir_futures.py.

    Args:
        spec: futures contract.
        reference_date: valuation date.
        hw_a: Hull-White mean reversion.
        hw_sigma: Hull-White volatility.
    """
    dc = DayCountConvention.ACT_365_FIXED
    t = 0.0  # valuation time (today)
    t1 = year_fraction(reference_date, spec.accrual_start, dc)
    t2 = year_fraction(reference_date, spec.accrual_end, dc)

    if t1 <= 0:
        return 0.0

    return hw_convexity_adjustment(hw_a, hw_sigma, t, t1, t2)


# ═══════════════════════════════════════════════════════════════
# Conversion to forwards for bootstrap
# ═══════════════════════════════════════════════════════════════


def rfr_futures_to_forwards(
    contracts: list[RFRFutureSpec],
    reference_date: date,
    hw_a: float = 0.03,
    hw_sigma: float = 0.01,
) -> list[tuple[date, date, float]]:
    """Convert RFR futures to forward rate inputs for bootstrap.

    Returns list of (start_date, end_date, forward_rate) tuples,
    suitable for passing to bootstrap() as the 'fras' parameter.

    The forward rate = futures_implied_rate - convexity_adjustment.
    """
    forwards = []
    for spec in contracts:
        if spec.price <= 0:
            continue
        ca = rfr_futures_convexity(spec, reference_date, hw_a, hw_sigma)
        fwd = spec.implied_rate - ca
        forwards.append((spec.accrual_start, spec.accrual_end, fwd))
    return forwards


def list_futures_currencies() -> list[str]:
    """Return currencies with RFR futures support."""
    return sorted(_CURRENCY_RFR.keys())
