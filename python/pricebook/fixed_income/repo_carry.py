"""Multi-currency cost-of-carry breakeven analysis.

    from pricebook.fixed_income.repo_carry import (
        carry_breakeven, xccy_repo_carry, CarryBreakevenResult,
    )

References:
    Tuckman & Serrat (2012). Fixed Income Securities, Ch 15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class CarryBreakevenResult:
    """Cost-of-carry breakeven result."""
    currency: str
    coupon_income: float         # coupon earned over period
    financing_cost_gc: float     # cost at GC rate
    financing_cost_special: float  # cost at special rate
    carry_gc: float              # coupon - gc financing
    carry_special: float         # coupon - special financing
    breakeven_gc_rate: float     # repo rate at which carry = 0
    term_vs_on_pickup_bp: float  # carry advantage of term vs O/N

    def to_dict(self) -> dict:
        return dict(vars(self))


def carry_breakeven(
    bond_price: float,
    coupon_rate: float,
    hold_days: int,
    gc_rate: float,
    special_rate: float | None = None,
    on_rate: float | None = None,
    currency: str = "USD",
    face: float = 100.0,
) -> CarryBreakevenResult:
    """Compute carry breakeven for a bond repo.

    carry = coupon_income - financing_cost
    breakeven = repo_rate where carry = 0

    Args:
        bond_price: dirty price.
        coupon_rate: annual coupon.
        hold_days: holding period in days.
        gc_rate: GC repo rate.
        special_rate: special repo rate (if on special).
        on_rate: overnight repo rate (for term vs O/N comparison).
        currency: for day count conventions.
        face: face value.
    """
    denom = 360.0 if currency.upper() in ("USD", "EUR", "CHF", "MXN") else 365.0
    t = hold_days / denom

    coupon_income = coupon_rate * face * hold_days / denom
    fin_gc = bond_price * gc_rate * t
    fin_special = bond_price * (special_rate or gc_rate) * t

    carry_gc = coupon_income - fin_gc
    carry_special = coupon_income - fin_special

    # Breakeven: coupon_income = bond_price × r × t → r = coupon_income / (bond_price × t)
    breakeven = coupon_income / (bond_price * t) if bond_price * t > 0 else 0.0

    # Term vs O/N pickup
    if on_rate is not None:
        fin_on = bond_price * on_rate * t
        pickup = (fin_on - fin_gc) / (bond_price * t) * 10_000 if t > 0 else 0.0
    else:
        pickup = 0.0

    return CarryBreakevenResult(
        currency=currency.upper(),
        coupon_income=coupon_income,
        financing_cost_gc=fin_gc,
        financing_cost_special=fin_special,
        carry_gc=carry_gc,
        carry_special=carry_special,
        breakeven_gc_rate=breakeven,
        term_vs_on_pickup_bp=pickup,
    )


@dataclass
class XCCYCarryResult:
    """Cross-currency repo carry result."""
    domestic_carry: float
    foreign_carry: float
    xccy_basis_cost: float
    net_xccy_carry: float
    domestic_currency: str
    foreign_currency: str

    def to_dict(self) -> dict:
        return dict(vars(self))


def xccy_repo_carry(
    bond_price: float,
    coupon_rate: float,
    hold_days: int,
    domestic_repo_rate: float,
    foreign_repo_rate: float,
    xccy_basis_bp: float,
    domestic_currency: str = "USD",
    foreign_currency: str = "EUR",
) -> XCCYCarryResult:
    """Cross-currency repo carry with FX basis.

    Compare: funding a UST in USD vs funding it in EUR via xccy repo.
    """
    denom = 360.0
    t = hold_days / denom

    coupon = coupon_rate * bond_price * t
    dom_fin = bond_price * domestic_repo_rate * t
    for_fin = bond_price * foreign_repo_rate * t
    basis_cost = bond_price * xccy_basis_bp / 10_000 * t

    dom_carry = coupon - dom_fin
    for_carry = coupon - for_fin - basis_cost

    return XCCYCarryResult(
        domestic_carry=dom_carry,
        foreign_carry=for_carry,
        xccy_basis_cost=basis_cost,
        net_xccy_carry=for_carry - dom_carry,
        domestic_currency=domestic_currency.upper(),
        foreign_currency=foreign_currency.upper(),
    )


def multi_ccy_carry_comparison(
    bond_price: float,
    coupon_rate: float,
    hold_days: int,
    repo_rates: dict[str, float],
    xccy_basis_vs_usd: dict[str, float] | None = None,
) -> list[dict]:
    """Rank carry across currencies for the same bond.

    Args:
        repo_rates: {currency: gc_repo_rate}.
        xccy_basis_vs_usd: {currency: basis_bp} for non-USD currencies.

    Returns sorted by carry (highest first).
    """
    denom = 360.0
    t = hold_days / denom
    coupon = coupon_rate * bond_price * t

    results = []
    for ccy, rate in repo_rates.items():
        fin = bond_price * rate * t
        basis = 0.0
        if xccy_basis_vs_usd and ccy.upper() != "USD":
            basis = bond_price * xccy_basis_vs_usd.get(ccy, 0) / 10_000 * t
        carry = coupon - fin - basis
        results.append({
            "currency": ccy.upper(),
            "repo_rate": rate,
            "financing_cost": fin,
            "xccy_basis_cost": basis,
            "carry": carry,
        })

    return sorted(results, key=lambda r: -r["carry"])
