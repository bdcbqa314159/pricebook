"""The forward-starting CDS (L2). `OKANE-2008` ch 9.2.

Protection and premium both begin at `forward_start` and run to the schedule's maturity. Frozen
data: it describes the contract and prices nothing.

**Why the schedule spans `[t, T]` and the forward start is a date within it.** A forward CDS's
value is a *difference of spot quantities* — `(S(t,tF,T) − S₀)·[RPV01(t,T) − RPV01(t,tF)]` — so
pricing needs both the near and the far premium schedules, and they must share period boundaries
or the difference is not the forward annuity. Carrying one schedule plus a date *on* it makes that
alignment a property of the representation rather than something a pricer has to hope for: the
near schedule is this one truncated at `forward_start`. A separate `near_schedule` field could be
handed two unaligned grids, which is a wrong number nothing would catch.

Provenance:
  stock:  none — the archived predecessor had no forward CDS
  source: OKANE-2008 ch 9.2 (forward starting CDS)
  oracle: tests/L4/test_forward_cds_default_swaption.py — the forward CDS is the difference of the
          two spot CDS prices through tranche 38's engine, for every contractual spread
  crosscheck: none — frozen data, no numerics
  slice:  tranche 40 — the forward CDS and the default swaption
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Money
from pricebook.foundation.schedule import CouponSchedule
from pricebook.products.credit_default_swap import ProtectionDirection


@dataclass(frozen=True)
class ForwardCDS:
    """A CDS entered now, with protection running from `forward_start` to the schedule's end.

    `direction` is tranche 38's `ProtectionDirection`, reused rather than redefined: a forward CDS
    buyer and a spot CDS buyer are the same side of the same contract, so the two share a *shape*
    and not merely a word (§6).
    """

    schedule: CouponSchedule
    forward_start: date
    spread: float
    notional: Money
    direction: ProtectionDirection

    def __post_init__(self) -> None:
        if self.forward_start not in self.schedule.dates:
            raise ValueError(
                f"the forward start {self.forward_start} must be a schedule date, so the near and "
                f"far premium legs share period boundaries and their difference is the forward "
                f"annuity; the schedule runs {self.schedule.dates[0]} → {self.schedule.dates[-1]}")
        if self.forward_start >= self.schedule.dates[-1]:
            raise ValueError(
                f"the forward start {self.forward_start} must fall before the maturity "
                f"{self.schedule.dates[-1]}, or there is no forward window to protect")

    def near_dates(self) -> tuple[date, ...]:
        """The schedule truncated at `forward_start` — the legs that are *subtracted*."""
        index = self.schedule.dates.index(self.forward_start)
        return self.schedule.dates[: index + 1]
