"""Bond desk tools: curve fitting from bonds, repo desk, rich/cheap per bond.

Fit a par/zero curve from bond prices, identify rich/cheap bonds
relative to the fitted curve, and manage repo positions.

    from pricebook.bond_desk import (
        fit_curve_from_bonds, bond_rich_cheap, repo_carry,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.solvers import brentq


# ---- Curve fitting from bonds ----

@dataclass
class FittedBond:
    """One bond's fitted vs market price."""
    bond: FixedRateBond
    market_price: float
    fitted_price: float
    residual: float  # market - fitted
    z_spread: float


def fit_curve_from_bonds(
    reference_date: date,
    bonds: list[tuple[FixedRateBond, float]],
    initial_rate: float = 0.05,
) -> tuple[DiscountCurve, list[FittedBond]]:
    """Bootstrap a discount curve from bond prices.

    Sequentially solves for discount factors at each bond's maturity
    such that the bond reprices at market.

    Args:
        bonds: list of (bond, market_clean_price) sorted by maturity.
        initial_rate: starting guess for zero rates.

    Returns:
        (fitted_curve, list of FittedBond with residuals).
    """
    if not bonds:
        raise ValueError("need at least 1 bond")

    # Sort by maturity
    sorted_bonds = sorted(bonds, key=lambda b: b[0].maturity)

    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []

    for bond, mkt_price in sorted_bonds:
        mat = bond.maturity

        def objective(df_guess: float, _bond=bond, _price=mkt_price, _mat=mat) -> float:
            trial_dates = pillar_dates + [_mat]
            trial_dfs = pillar_dfs + [df_guess]
            trial_curve = DiscountCurve(reference_date, trial_dates, trial_dfs)
            fitted = _bond.dirty_price(trial_curve)
            # dirty_price returns per 100 face; compare to clean + accrued
            accrued = _bond.accrued_interest(reference_date)
            return fitted - (_price + accrued / _bond.face_value * 100)

        df_solved = brentq(objective, 0.01, 1.50)
        pillar_dates.append(mat)
        pillar_dfs.append(df_solved)

    fitted_curve = DiscountCurve(reference_date, pillar_dates, pillar_dfs)

    # Compute residuals
    results = []
    for bond, mkt_price in sorted_bonds:
        fitted_price = bond.dirty_price(fitted_curve)
        accrued = bond.accrued_interest(reference_date)
        fitted_clean = fitted_price - accrued / bond.face_value * 100

        # Z-spread (may fail to converge for distressed bonds)
        try:
            from pricebook.risky_bond import z_spread as _zs, RiskyBond
            rb = RiskyBond(bond.issue_date, bond.maturity, bond.coupon_rate,
                          bond.face_value, bond.frequency, bond.day_count)
            zs = _zs(rb, mkt_price, fitted_curve)
        except (ValueError, RuntimeError):
            zs = 0.0

        results.append(FittedBond(
            bond=bond,
            market_price=mkt_price,
            fitted_price=fitted_clean,
            residual=mkt_price - fitted_clean,
            z_spread=zs,
        ))

    return fitted_curve, results


# ---- Bond rich/cheap ----

@dataclass
class BondRichCheap:
    """Rich/cheap analysis for one bond."""
    bond: FixedRateBond
    market_price: float
    model_price: float
    spread: float  # market - model
    signal: str  # "rich", "cheap", "fair"


def bond_rich_cheap(
    bonds: list[tuple[FixedRateBond, float]],
    curve: DiscountCurve,
    threshold: float = 0.5,
) -> list[BondRichCheap]:
    """Identify rich/cheap bonds relative to a fitted curve.

    A bond trading above model price is "rich" (expensive).
    A bond trading below model price is "cheap" (good value).

    Args:
        bonds: list of (bond, market_clean_price).
        curve: fitted discount curve.
        threshold: price difference threshold for signal.
    """
    results = []
    for bond, mkt_price in bonds:
        model = bond.dirty_price(curve)
        accrued = bond.accrued_interest(curve.reference_date)
        model_clean = model - accrued / bond.face_value * 100

        spread = mkt_price - model_clean
        if spread > threshold:
            signal = "rich"
        elif spread < -threshold:
            signal = "cheap"
        else:
            signal = "fair"

        results.append(BondRichCheap(bond, mkt_price, model_clean, spread, signal))

    return results


# ---- Repo desk ----

@dataclass
class RepoPosition:
    """A repo or reverse repo position."""
    bond: FixedRateBond
    bond_price: float  # clean price
    repo_rate: float
    term_days: int
    notional: float
    direction: str  # "repo" (lend bond, borrow cash) or "reverse" (borrow bond, lend cash)


def repo_carry(
    bond_price: float,
    coupon_rate: float,
    repo_rate: float,
    term_days: int,
    face_value: float = 100.0,
) -> dict[str, float]:
    """Compute repo carry for a bond position.

    Carry = coupon_income - repo_cost (for a repo / funded bond position).

    Args:
        bond_price: clean price (for financing cost calculation).
        coupon_rate: annual coupon rate.
        repo_rate: annualised repo rate.
        term_days: repo term in days.
        face_value: bond face value.
    """
    dt = term_days / 365.0
    coupon_income = coupon_rate * face_value * dt
    financing_cost = bond_price * repo_rate * dt
    carry = coupon_income - financing_cost

    return {
        "coupon_income": coupon_income,
        "financing_cost": financing_cost,
        "carry": carry,
        "breakeven_repo": coupon_rate * face_value / bond_price if bond_price > 0 else 0.0,
    }


def securities_lending_fee(
    bond_price: float,
    borrow_fee_bps: float,
    term_days: int,
) -> float:
    """Cost of borrowing a bond for short selling.

    Fee = bond_price × borrow_fee × days / 365.
    """
    return bond_price * (borrow_fee_bps / 10000.0) * (term_days / 365.0)
