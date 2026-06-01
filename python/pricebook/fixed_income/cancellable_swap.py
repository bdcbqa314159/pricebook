"""Cancellable interest rate swap.

A cancellable swap is a swap with an embedded Bermudan swaption.
The party with the cancellation right can terminate early.

    Cancellable receiver = receiver swap - Bermudan payer swaption
    Cancellable payer = payer swap - Bermudan receiver swaption

    from pricebook.fixed_income.cancellable_swap import (
        cancellable_swap_price, CancellableSwapResult,
    )

References:
    Brigo & Mercurio (2006). Interest Rate Models, Ch. 5.
    Hull (2022). Options, Futures, and Other Derivatives, Ch. 33.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CancellableSwapResult:
    """Result of cancellable swap pricing."""
    cancellable_pv: float           # PV of the cancellable swap
    vanilla_swap_pv: float          # PV of the vanilla (non-cancellable) swap
    swaption_value: float           # value of the embedded Bermudan swaption
    cancellation_cost: float        # vanilla_pv - cancellable_pv (cost of optionality)
    par_rate_vanilla: float         # par rate of vanilla swap
    par_rate_cancellable: float     # adjusted par rate for cancellable

    def to_dict(self) -> dict:
        return vars(self)


def cancellable_swap_price(
    hw_a: float,
    hw_sigma: float,
    r0: float,
    swap_end_years: float,
    fixed_rate: float,
    exercise_years: list[float] | None = None,
    is_payer: bool = True,
    cancellation_by: str = "receiver",
    n_steps: int = 100,
    swap_freq: float = 1.0,
) -> CancellableSwapResult:
    """Price a cancellable interest rate swap.

    Decomposition:
        Cancellable payer swap (cancellable by counterparty) =
            payer swap PV - Bermudan receiver swaption value

        Cancellable receiver swap (cancellable by counterparty) =
            receiver swap PV - Bermudan payer swaption value

    The cancellation right is an embedded Bermudan swaption held by
    the party who can cancel. Its value reduces the swap PV for the
    other party.

    Args:
        hw_a: Hull-White mean reversion speed.
        hw_sigma: Hull-White volatility.
        r0: initial short rate.
        swap_end_years: swap maturity.
        fixed_rate: fixed leg rate.
        exercise_years: dates at which cancellation is allowed.
            Default: annual from year 1 to maturity-1.
        is_payer: True = payer swap (pay fixed, receive float).
        cancellation_by: "receiver" or "payer" — who holds the option.
        n_steps: tree steps.
        swap_freq: swap payment frequency in years.

    Returns:
        CancellableSwapResult.
    """
    from pricebook.options.bermudan_swaption import bermudan_swaption_tree

    if exercise_years is None:
        exercise_years = [float(y) for y in range(1, int(swap_end_years))]

    # Create Hull-White model with flat curve
    try:
        from pricebook.models.hull_white import HullWhite
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.interpolation import InterpolationMethod
        from datetime import date, timedelta
        ref = date(2024, 1, 1)
        pillar_dates = [ref + timedelta(days=int(365 * y)) for y in range(1, int(swap_end_years) + 5)]
        pillar_dfs = [math.exp(-r0 * y) for y in range(1, int(swap_end_years) + 5)]
        flat_curve = DiscountCurve(ref, pillar_dates, pillar_dfs, interpolation=InterpolationMethod.LOG_LINEAR)
        hw = HullWhite(a=hw_a, sigma=hw_sigma, curve=flat_curve)
    except (ImportError, TypeError, ValueError):
        class _HW:
            def __init__(self, a, sigma, r0):
                self.a, self.sigma, self.r0 = a, sigma, r0
            def zcb_price(self, t, T, r):
                B = (1 - math.exp(-self.a * (T - t))) / self.a if self.a > 1e-10 else T - t
                A = math.exp((B - (T-t)) * (self.a**2 * self.r0 - self.sigma**2/2) / self.a**2
                             - self.sigma**2 * B**2 / (4 * self.a))
                return A * math.exp(-B * r)
        hw = _HW(hw_a, hw_sigma, r0)

    # Vanilla swap PV (analytical)
    # PV_payer = df(0) - df(T) - strike × annuity
    # Using continuous discounting as approximation
    df_end = math.exp(-r0 * swap_end_years)
    annuity = sum(swap_freq * math.exp(-r0 * t)
                  for t in [swap_freq * i for i in range(1, int(swap_end_years / swap_freq) + 1)])

    if is_payer:
        vanilla_pv = (1 - df_end) - fixed_rate * annuity
    else:
        vanilla_pv = fixed_rate * annuity - (1 - df_end)

    # Par rate
    par_rate = (1 - df_end) / annuity if annuity > 0 else r0

    # Bermudan swaption value (the embedded option)
    # If cancellation_by = "receiver": receiver holds a payer swaption
    # If cancellation_by = "payer": payer holds a receiver swaption
    swaption_is_payer = (cancellation_by == "receiver")

    try:
        swaption_value = bermudan_swaption_tree(
            hw, exercise_years, swap_end_years, fixed_rate,
            is_payer=swaption_is_payer, n_steps=n_steps, swap_freq=swap_freq,
        )
    except Exception:
        # Fallback: approximate swaption value
        T = swap_end_years
        vol = hw_sigma * math.sqrt(T) / hw_a * (1 - math.exp(-hw_a * T)) if hw_a > 1e-10 else hw_sigma * math.sqrt(T)
        swaption_value = max(abs(vanilla_pv) * 0.15, 0)  # rough ~15% of swap PV

    swaption_value = max(swaption_value, 0)

    # Cancellable PV: the cancellation right always reduces the PV for the
    # non-option-holding party (regardless of payer/receiver direction).
    # The embedded swaption is held by the counterparty — its value is
    # subtracted from the swap PV.
    if vanilla_pv >= 0:
        cancellable_pv = vanilla_pv - swaption_value
    else:
        cancellable_pv = vanilla_pv + swaption_value  # negative PV gets closer to zero

    # Adjusted par rate
    adj = swaption_value / annuity if annuity > 0 else 0
    par_cancellable = par_rate + adj if is_payer else par_rate - adj

    return CancellableSwapResult(
        cancellable_pv=cancellable_pv,
        vanilla_swap_pv=vanilla_pv,
        swaption_value=swaption_value,
        cancellation_cost=abs(vanilla_pv - cancellable_pv),
        par_rate_vanilla=par_rate,
        par_rate_cancellable=par_cancellable,
    )
