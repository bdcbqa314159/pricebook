"""The forward-starting CDS (L2). `OKANE-2008` ch 9.2.

Protection and premium both begin at the forward period's start and run to its maturity.
Frozen data: it describes the contract and prices nothing.

**Why the schedule spans `[t, T]` and the forward start is a date within it.** A forward CDS's
value is a *difference of spot quantities* — `(S(t,tF,T) − S₀)·[RPV01(t,T) − RPV01(t,tF)]` — so
pricing needs both the near and the far premium schedules, and they must share period boundaries
or the difference is not the forward annuity. Carrying one schedule plus a date *on* it makes that
alignment a property of the representation rather than something a pricer has to hope for: the
near schedule is this one truncated at the split. A separate `near_schedule` field could be handed
two unaligned grids, which is a wrong number nothing would catch. `ForwardPeriod` holds that pair,
shared with the default swaption (tranche 41b).

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
from pricebook.market.survival_curve import ReferenceEntity
from pricebook.products.credit_default_swap import ProtectionDirection


@dataclass(frozen=True)
class ForwardPeriod:
    """A premium schedule spanning `[t, T]`, split at a date **on** it where the forward starts.

    Both forward products carry exactly this pair — the forward CDS calls the split its *forward
    start*, the swaption calls it the *option expiry*, and structurally it is the same date doing
    the same job. Two present consumers sharing a shape, not a word (§6), and the bundle is
    **compelled**: tranche 41b's `reference_entity` would otherwise put both products at six
    fields, over §7's limit, which is un-bundled vocabulary rather than a reason to raise it.

    It also removes a real duplication: the two validations below were written out twice, once per
    product, which is the duplicated-guard shape §4 warns about (a fixture exercising one copy
    leaves the other unguarded). One definition now, so one fixture suffices.
    """

    schedule: CouponSchedule
    start: date

    def __post_init__(self) -> None:
        if self.start not in self.schedule.dates:
            raise ValueError(
                f"the forward start {self.start} must be a schedule date, so the near and far "
                f"premium legs share period boundaries and their difference is the forward "
                f"annuity; the schedule runs {self.schedule.dates[0]} → {self.schedule.dates[-1]}")
        if self.start >= self.schedule.dates[-1]:
            raise ValueError(
                f"the forward start {self.start} must fall before the maturity "
                f"{self.schedule.dates[-1]}, or there is no forward window")

    @property
    def maturity(self) -> date:
        return self.schedule.dates[-1]

    def near_dates(self) -> tuple[date, ...]:
        """The schedule truncated at the split — the legs that are *subtracted*."""
        return self.schedule.dates[: self.schedule.dates.index(self.start) + 1]


@dataclass(frozen=True)
class ForwardCDS:
    """A CDS entered now, with protection running over `period`.

    `direction` is tranche 38's `ProtectionDirection`, reused rather than redefined: a forward CDS
    buyer and a spot CDS buyer are the same side of the same contract, so the two share a *shape*
    and not merely a word (§6).
    """

    period: ForwardPeriod
    spread: float
    notional: Money
    direction: ProtectionDirection
    reference_entity: ReferenceEntity
