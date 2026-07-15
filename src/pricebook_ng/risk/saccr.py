"""SA-CCR — Basel standardised approach for counterparty credit risk (L5).

Computes the regulatory Exposure at Default of a derivative netting set, the input
to counterparty RWA and to the KVA capital profile:

    EAD = alpha * (RC + PFE)

  - RC (replacement cost) = max(V, 0) for an uncollateralised set (V = current mark).
  - PFE (potential future exposure) = multiplier * AddOn.
      AddOn (interest rate, one trade) = SF * adjusted_notional * maturity_factor
        adjusted_notional = notional * SD,  SD = (e^{-r S} - e^{-r E}) / r   (r = 5%)
        maturity_factor   = sqrt(min(M, 1yr))                                 (unmargined)
      multiplier = min(1, floor + (1-floor) * exp(V / (2 (1-floor) AddOn)))    (floor 5%)
  - alpha = 1.4.

RWA = EAD * counterparty risk weight; regulatory capital = 8% * RWA.

The current single-date EAD and the forward EAD *runoff* profile both land here:
`forward_ead_profile` reprices SA-CCR at each future coupon date on the shrinking
remaining trade, and `capital_profile` scales it to `8% * RWA` — the capital `K(t)`
that `kva` charges the cost of capital on, closing the SA-CCR -> KVA loop.

Scope (CLAUDE.md 6b): a single interest-rate trade in one hedging set, unmargined,
no collateral, supervisory delta |1| (its sign only matters for netting, which
washes out for one trade). `forward_ead_profile` uses the ATM assumption (mark = 0);
`stochastic_ead_profile` upgrades the replacement cost to the MC expected positive
exposure. Netting-set aggregation (maturity buckets + correlations), margined MF, other
asset classes, and collateral haircuts are later refinements.

Provenance:
  quarry: python/pricebook/risk/ (regulatory capital)
  source: BCBS d424 / CRE52 — SA-CCR; ISDA SA-CCR worked examples
  oracle: 10y ATM $100mm IRS EAD ~ 5.5% notional; RC/multiplier-floor limits;
          EAD runoff -> KVA annuity; stochastic EAD = forward_ead + alpha*EPE
  slice:  saccr; forward-ead-kva (runoff -> KVA); stochastic-ead (EPE as RC)
"""

from __future__ import annotations

import math
from datetime import date

from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swap import VanillaSwap
from pricebook_ng.risk.exposure import exposure_profiles
from pricebook_ng.risk.xva import ExposureProfile

_ALPHA = 1.4               # supervisory EAD multiplier
_SF_IR = 0.005            # interest-rate supervisory factor
_MULT_FLOOR = 0.05        # PFE multiplier floor
_SD_DECAY = 0.05          # supervisory-duration exponential decay
_DC = DayCountConvention.ACT_365_FIXED


def _ead_ir(notional: float, start_years: float, end_years: float, mark: float) -> float:
    """Core SA-CCR IR EAD from raw params: `alpha * (RC + multiplier * AddOn)` with
    `S`/`E` the years to the trade's start/end. Shared by the single-date `saccr_ead`
    and the forward runoff profile."""
    supervisory_duration = (
        math.exp(-_SD_DECAY * start_years) - math.exp(-_SD_DECAY * end_years)
    ) / _SD_DECAY
    maturity_factor = math.sqrt(min(end_years, 1.0))                    # unmargined MF
    add_on = _SF_IR * notional * supervisory_duration * maturity_factor

    replacement_cost = max(mark, 0.0)
    if add_on == 0.0:
        return _ALPHA * replacement_cost
    multiplier = min(
        1.0,
        _MULT_FLOOR + (1.0 - _MULT_FLOOR) * math.exp(mark / (2.0 * (1.0 - _MULT_FLOOR) * add_on)),
    )
    return _ALPHA * (replacement_cost + multiplier * add_on)


def saccr_ead(swap: VanillaSwap, mark: float, valuation_date: date) -> float:
    """SA-CCR Exposure at Default for a single-trade IR netting set, `alpha*(RC+PFE)`.
    `mark` is the current value V of the swap (from pricing); uncollateralised."""
    schedule = swap.float_leg.schedule
    start = max(year_fraction(valuation_date, schedule[0], _DC), 0.0)   # S: years to start
    end = year_fraction(valuation_date, schedule[-1], _DC)              # E: years to maturity
    return _ead_ir(swap.float_leg.face.amount, start, end, mark)


def forward_ead_profile(swap: VanillaSwap, valuation_date: date) -> ExposureProfile:
    """SA-CCR EAD runoff: the as-of-`t_j` EAD of the still-live trade at each future
    coupon date. Under the ATM assumption (expected mark = 0 -> RC = 0, multiplier = 1)
    this is the deterministic supervisory PFE, shrinking as the remaining maturity runs
    off — the capital profile KVA integrates. (A stochastic-mark version would set RC to
    the expected positive exposure; that is the exposure engine's job, a later refinement.)"""
    notional = swap.float_leg.face.amount
    schedule = swap.float_leg.schedule
    swap_start, maturity = schedule[0], schedule[-1]
    grid = [valuation_date, *(d for d in schedule if valuation_date < d < maturity)]
    ead = [
        _ead_ir(
            notional,
            year_fraction(t_j, swap_start, _DC) if swap_start > t_j else 0.0,  # S: 0 once started
            year_fraction(t_j, maturity, _DC),                                 # E: remaining maturity
            0.0,
        )
        for t_j in grid
    ]
    return ExposureProfile(tuple(grid), tuple(ead))


def stochastic_ead_profile(
    swap: VanillaSwap, model: HullWhite, numerics: NumericalConfig
) -> ExposureProfile:
    """SA-CCR EAD runoff with a *stochastic* replacement cost: RC(t_j) is the MC expected
    positive exposure EPE(t_j) (from the exposure engine) rather than the ATM zero, so
    `EAD(t_j) = alpha * (EPE(t_j) + AddOn_remaining(t_j))` — the simulated exposure unified
    with the supervisory PFE. Passing EPE (>= 0) as the SA-CCR mark pins the multiplier at 1,
    so this is `forward_ead_profile` with `mark = EPE(t_j)`; it decomposes as
    `forward_ead(t_j) + alpha * EPE(t_j)`."""
    epe = exposure_profiles(swap, model, numerics).epe
    notional = swap.float_leg.face.amount
    swap_start, maturity = swap.float_leg.schedule[0], swap.float_leg.schedule[-1]
    ead = tuple(
        _ead_ir(
            notional,
            year_fraction(t_j, swap_start, _DC) if swap_start > t_j else 0.0,
            year_fraction(t_j, maturity, _DC),
            expected_positive_exposure,   # mark = EPE >= 0 -> RC = EPE, multiplier pinned at 1
        )
        for t_j, expected_positive_exposure in zip(epe.grid, epe.ee)
    )
    return ExposureProfile(epe.grid, ead)


def capital_profile(ead_profile: ExposureProfile, risk_weight: float) -> ExposureProfile:
    """Turn an EAD runoff into the regulatory capital profile `8% * RWA = 8% * EAD * RW`
    at each date — the `K(t)` that `kva` charges the cost of capital on."""
    return ExposureProfile(
        ead_profile.grid,
        tuple(saccr_capital(risk_weighted_assets(e, risk_weight)) for e in ead_profile.ee),
    )


def risk_weighted_assets(ead: float, risk_weight: float) -> float:
    """Counterparty RWA = EAD * risk weight (standardised credit risk — e.g. 100%
    corporate, 20% bank)."""
    return ead * risk_weight


def saccr_capital(rwa: float) -> float:
    """Regulatory capital = 8% of RWA."""
    return 0.08 * rwa
