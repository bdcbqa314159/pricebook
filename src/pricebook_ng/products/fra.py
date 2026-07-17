"""ForwardRateAgreement — a single-period FRA as pure data (L2).

Pay `fixed_rate`, receive the simply-compounded forward over `[accrual_start,
accrual_end]`, settled at the end. Pure data — no `pv` (pricing lives in the L4
`FRAEngine`, which resolves the forward from the curve; the float amount is not
knowable without one, so — like the swap's float leg — it stays structural).

Provenance:
  quarry: python/pricebook/fixed_income/fra.py
  source: standard single-curve FRA
  oracle: par FRA (K = forward) reprices to zero; closed-form off-par PV
  slice:  fra-spine (CP-2b #3 — fixed_income spine to parity)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.time import DayCountConvention


@dataclass(frozen=True)
class ForwardRateAgreement:
    """A vanilla FRA: `pay_fixed` pays `fixed_rate` and receives the forward over
    `[accrual_start, accrual_end]` on `notional`, accrued with `day_count`."""

    notional: float
    fixed_rate: float
    accrual_start: date
    accrual_end: date
    day_count: DayCountConvention
    currency: Currency
    pay_fixed: bool = True
