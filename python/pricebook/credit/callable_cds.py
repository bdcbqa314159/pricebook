"""Callable CDS: CDS with seller termination right.

Unlike a CDS swaption (option to enter a CDS), a callable CDS is
an existing CDS where the protection seller can terminate early.
The seller exercises when credit improves (spread tightens).

    from pricebook.credit.callable_cds import (
        callable_cds_price, CallableCDSResult,
    )

Decomposition:
    Callable CDS PV = Vanilla CDS PV - Termination Option Value

References:
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class CallableCDSResult:
    """Callable CDS pricing result."""
    callable_pv: float          # PV with termination option
    vanilla_pv: float           # PV without option
    termination_option: float   # value of seller's termination right
    vanilla_spread: float       # par spread of vanilla CDS
    callable_spread: float      # par spread of callable CDS

    def to_dict(self) -> dict:
        return vars(self)


def callable_cds_price(
    reference_date: date,
    maturity: date,
    spread: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
    notional: float = 10_000_000.0,
    call_dates: list[date] | None = None,
    n_steps: int = 50,
) -> CallableCDSResult:
    """Price a callable CDS via backward induction.

    The protection seller can terminate at call dates. Seller terminates
    when the CDS has positive MTM for the seller (credit improved).

    Args:
        spread: running premium (e.g. 0.01 = 100bp).
        call_dates: dates at which seller can terminate.
            Default: annual from year 1.

    Returns:
        CallableCDSResult.
    """
    from pricebook.credit.cds import protection_leg_pv, premium_leg_pv

    dc = DayCountConvention.ACT_365_FIXED
    T = year_fraction(reference_date, maturity, dc)

    if call_dates is None:
        from dateutil.relativedelta import relativedelta
        call_dates = [reference_date + relativedelta(years=y)
                      for y in range(1, int(T))]

    # Vanilla CDS PV (protection buyer perspective)
    prot_pv = protection_leg_pv(reference_date, maturity, discount_curve,
                                 survival_curve, recovery, notional)
    prem_pv = premium_leg_pv(reference_date, maturity, spread,
                              discount_curve, survival_curve, notional)
    vanilla_pv = prot_pv - prem_pv  # positive = protection is valuable

    # Vanilla par spread
    if prem_pv != 0:
        # Par spread: spread that makes PV = 0
        # prot_pv = par_spread × (prem_pv / spread)
        vanilla_par = prot_pv / (prem_pv / spread) if spread > 0 else 0
    else:
        vanilla_par = 0

    # Backward induction for callable CDS
    # The seller's termination option is valuable when credit improves
    # (survival increases → protection leg shrinks → CDS PV decreases)
    #
    # At each call date, the seller compares:
    #   - Continue: remaining CDS PV (could be negative for seller)
    #   - Terminate: 0 (walk away)
    # Seller terminates if continuation PV < 0 for seller (= positive for buyer)

    dt = T / n_steps
    call_steps = set()
    for cd in call_dates:
        step = round(year_fraction(reference_date, cd, dc) / dt)
        step = max(1, min(step, n_steps - 1))
        call_steps.add(step)

    # Compute per-step values
    values = [0.0] * (n_steps + 1)  # value at each step (buyer perspective)

    for step in range(n_steps - 1, -1, -1):
        t = step * dt
        t_next = (step + 1) * dt

        # Survival probabilities
        t_date = date.fromordinal(reference_date.toordinal() + int(t * 365))
        t_next_date = date.fromordinal(reference_date.toordinal() + int(t_next * 365))

        q_t = survival_curve.survival(t_date)
        q_next = survival_curve.survival(t_next_date)
        p_survive = min(q_next / max(q_t, 1e-10), 1.0)
        p_default = max(1 - p_survive, 0.0)

        df = math.exp(-discount_curve.zero_rate(t_next_date) * dt) if t_next < T else 1.0

        # Continuation value
        cont = values[step + 1] * p_survive * df

        # Period cashflows (protection - premium)
        protection_cf = (1 - recovery) * notional * p_default * df
        premium_cf = spread * dt * notional * p_survive * df
        period_pv = protection_cf - premium_cf

        cont += period_pv

        # Seller's termination decision at call dates
        if step in call_steps:
            # Seller terminates if CDS is valuable to buyer (cont > 0)
            # This caps the buyer's value at 0 (seller walks away)
            cont = min(cont, 0)  # seller won't continue losing money

        values[step] = cont

    callable_pv = values[0]
    termination_value = max(vanilla_pv - callable_pv, 0)

    # Callable par spread (wider than vanilla — seller wants compensation for giving up option)
    callable_par = vanilla_par + termination_value / max(abs(prem_pv / spread), 1e-10) if spread > 0 else vanilla_par

    return CallableCDSResult(
        callable_pv=callable_pv,
        vanilla_pv=vanilla_pv,
        termination_option=termination_value,
        vanilla_spread=vanilla_par,
        callable_spread=callable_par,
    )
