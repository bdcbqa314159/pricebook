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

Scope (CLAUDE.md 6b): a single interest-rate trade in one hedging set, unmargined,
no collateral, supervisory delta |1| (its sign only matters for netting, which
washes out for one trade). Netting-set aggregation (maturity buckets + correlations),
margined MF, other asset classes, and collateral haircuts are later refinements. The
*current* EAD lands here; the forward EAD profile that feeds KVA capital is upstream
work on top of this.

Provenance:
  quarry: python/pricebook/risk/ (regulatory capital)
  source: BCBS d424 / CRE52 — SA-CCR; ISDA SA-CCR worked examples
  oracle: 10y ATM $100mm IRS EAD ~ 5.5% notional; RC/multiplier-floor limits
  slice:  saccr
"""

from __future__ import annotations

import math
from datetime import date

from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.products.swap import VanillaSwap

_ALPHA = 1.4               # supervisory EAD multiplier
_SF_IR = 0.005            # interest-rate supervisory factor
_MULT_FLOOR = 0.05        # PFE multiplier floor
_SD_DECAY = 0.05          # supervisory-duration exponential decay
_DC = DayCountConvention.ACT_365_FIXED


def saccr_ead(swap: VanillaSwap, mark: float, valuation_date: date) -> float:
    """SA-CCR Exposure at Default for a single-trade IR netting set, `alpha*(RC+PFE)`.
    `mark` is the current value V of the swap (from pricing); uncollateralised."""
    notional = swap.float_leg.face.amount
    schedule = swap.float_leg.schedule
    start = max(year_fraction(valuation_date, schedule[0], _DC), 0.0)   # S: years to start
    end = year_fraction(valuation_date, schedule[-1], _DC)              # E: years to maturity

    supervisory_duration = (math.exp(-_SD_DECAY * start) - math.exp(-_SD_DECAY * end)) / _SD_DECAY
    maturity_factor = math.sqrt(min(end, 1.0))                          # unmargined MF
    add_on = _SF_IR * notional * supervisory_duration * maturity_factor

    replacement_cost = max(mark, 0.0)
    if add_on == 0.0:
        return _ALPHA * replacement_cost
    multiplier = min(
        1.0,
        _MULT_FLOOR + (1.0 - _MULT_FLOOR) * math.exp(mark / (2.0 * (1.0 - _MULT_FLOOR) * add_on)),
    )
    pfe = multiplier * add_on
    return _ALPHA * (replacement_cost + pfe)


def risk_weighted_assets(ead: float, risk_weight: float) -> float:
    """Counterparty RWA = EAD * risk weight (standardised credit risk — e.g. 100%
    corporate, 20% bank)."""
    return ead * risk_weight


def saccr_capital(rwa: float) -> float:
    """Regulatory capital = 8% of RWA."""
    return 0.08 * rwa
