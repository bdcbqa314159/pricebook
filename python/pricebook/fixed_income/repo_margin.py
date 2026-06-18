"""Repo margin mechanics — daily VM, margin calls, forecasting.

    from pricebook.fixed_income.repo_margin import (
        calculate_vm, margin_call, margin_forecast,
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarginCallResult:
    """Result of margin call computation."""
    mark_to_market: float        # current MtM of repo
    collateral_value: float      # current collateral value
    exposure: float              # MtM - collateral (positive = under-collateralised)
    threshold: float
    mta: float                   # minimum transfer amount
    call_amount: float           # amount to call (0 if below MTA)
    rounding: float
    direction: str               # "receive" or "deliver"

    def to_dict(self) -> dict:
        return dict(vars(self))


def calculate_vm(
    repo_notional: float,
    repo_rate: float,
    days_elapsed: int,
    collateral_price_current: float,
    collateral_price_initial: float,
    collateral_quantity: float,
) -> float:
    """Daily variation margin.

    VM = change in exposure = (cash_owed - collateral_value) today
       - (cash_owed - collateral_value) yesterday

    Simplified: VM ≈ collateral_quantity × (price_initial - price_current)
    """
    cash_owed = repo_notional * (1 + repo_rate * days_elapsed / 360.0)
    coll_value_now = collateral_quantity * collateral_price_current
    coll_value_init = collateral_quantity * collateral_price_initial
    return coll_value_init - coll_value_now


def margin_call(
    exposure: float,
    threshold: float = 0.0,
    mta: float = 100_000.0,
    rounding: float = 10_000.0,
) -> MarginCallResult:
    """Compute margin call amount.

    Call is made when exposure exceeds threshold + MTA.
    Amount is rounded to nearest rounding increment.

    Args:
        exposure: current under-collateralisation (positive = need more collateral).
        threshold: CSA threshold (exposure below this is tolerated).
        mta: minimum transfer amount.
        rounding: rounding increment.
    """
    excess = exposure - threshold
    if excess <= mta:
        call = 0.0
        direction = "none"
    else:
        # Round up to nearest rounding
        if rounding > 0:
            call = rounding * ((excess + rounding - 1) // rounding)
        else:
            call = excess
        direction = "receive" if exposure > 0 else "deliver"

    return MarginCallResult(
        mark_to_market=0.0,
        collateral_value=0.0,
        exposure=exposure,
        threshold=threshold,
        mta=mta,
        call_amount=call,
        rounding=rounding,
        direction=direction,
    )


def margin_forecast(
    current_exposure: float,
    collateral_daily_vol: float,
    days_ahead: int = 2,
    confidence: float = 0.95,
) -> dict:
    """Forecast margin call for liquidity planning.

    Args:
        current_exposure: current exposure level.
        collateral_daily_vol: daily price vol of collateral (in $ terms).
        days_ahead: forecast horizon (default 2 days for MPOR).
        confidence: confidence level for VaR.
    """
    import math
    from scipy.stats import norm

    z = norm.ppf(confidence)
    expected_move = collateral_daily_vol * math.sqrt(days_ahead) * z

    forecast_exposure = current_exposure + expected_move
    forecast_max = current_exposure + expected_move * 1.5  # tail scenario

    return {
        "current_exposure": current_exposure,
        "forecast_exposure": forecast_exposure,
        "forecast_max_exposure": forecast_max,
        "days_ahead": days_ahead,
        "confidence": confidence,
        "potential_call": max(forecast_exposure, 0),
    }
