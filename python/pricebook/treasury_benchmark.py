"""Treasury benchmark: OTR tracking, auction schedule, specialness.

A TreasuryBenchmark represents a specific tenor point (2Y, 5Y, 10Y, 30Y)
and tracks which bond is currently on-the-run, its repo specialness,
and the next auction date.

    from pricebook.treasury_benchmark import TreasuryBenchmark, AUCTION_SCHEDULE

    bm = TreasuryBenchmark("10Y", otr_cusip="91282CJL2",
                            otr_bond=bond, otr_yield=0.042)
    bm.update_repo(specialness_bps=15)
    print(bm.next_auction(date(2024, 7, 15)))

References:
    US Treasury, Quarterly Refunding Schedule.
    SIFMA, US Treasury Securities Settlement Procedures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from pricebook.bond import FixedRateBond


# US Treasury auction schedule (approximate months for each tenor)
AUCTION_SCHEDULE = {
    "4W":  list(range(1, 13)),  # weekly
    "8W":  list(range(1, 13)),  # weekly
    "13W": list(range(1, 13)),  # weekly
    "26W": list(range(1, 13)),  # weekly
    "52W": list(range(1, 13)),  # every 4 weeks
    "2Y":  list(range(1, 13)),  # monthly
    "3Y":  list(range(1, 13)),  # monthly
    "5Y":  list(range(1, 13)),  # monthly
    "7Y":  list(range(1, 13)),  # monthly
    "10Y": [2, 5, 8, 11],       # quarterly (Feb, May, Aug, Nov)
    "20Y": [2, 5, 8, 11],       # quarterly
    "30Y": [2, 5, 8, 11],       # quarterly
}

# Standard tenors for T-Lock trading
TLOCK_TENORS = ["2Y", "5Y", "10Y", "30Y"]


@dataclass
class TreasuryBenchmark:
    """A Treasury benchmark point for T-Lock trading.

    Args:
        tenor: benchmark label (e.g. "10Y").
        otr_cusip: CUSIP of the on-the-run bond.
        otr_bond: the on-the-run FixedRateBond.
        otr_yield: current OTR yield.
        ofr_yield: most recent off-the-run yield (for spread).
        specialness_bps: repo specialness (OTR repo below GC, in bps).
        last_auction: date of the most recent auction.
    """
    tenor: str
    otr_cusip: str = ""
    otr_bond: FixedRateBond | None = None
    otr_yield: float = 0.0
    ofr_yield: float = 0.0
    specialness_bps: float = 0.0
    last_auction: date | None = None

    @property
    def otr_ofr_spread_bps(self) -> float:
        """OTR-OFR yield spread in bp (positive = OTR trades rich)."""
        return (self.ofr_yield - self.otr_yield) * 10_000

    @property
    def is_special(self) -> bool:
        """Bond is on special if specialness > 25bp."""
        return self.specialness_bps > 25

    def update_repo(self, specialness_bps: float) -> None:
        """Update specialness from repo market."""
        self.specialness_bps = specialness_bps

    def next_auction(self, from_date: date) -> date | None:
        """Estimate next auction date for this tenor.

        Uses the AUCTION_SCHEDULE months. Within each month,
        auctions typically occur mid-month (around the 15th).
        """
        months = AUCTION_SCHEDULE.get(self.tenor)
        if not months:
            return None

        year = from_date.year
        for attempt in range(24):  # search up to 2 years
            m = from_date.month + attempt
            y = year + (m - 1) // 12
            m = ((m - 1) % 12) + 1
            if m in months:
                auction_date = date(y, m, 15)
                if auction_date > from_date:
                    return auction_date
        return None

    def days_to_next_auction(self, from_date: date) -> int | None:
        """Days until next auction for this tenor."""
        nxt = self.next_auction(from_date)
        return (nxt - from_date).days if nxt else None

    def adjusted_repo_rate(self, gc_rate: float) -> float:
        """GC rate minus specialness → effective OTR repo rate."""
        return gc_rate - self.specialness_bps / 10_000

    def to_dict(self) -> dict:
        d = {
            "tenor": self.tenor,
            "otr_cusip": self.otr_cusip,
            "otr_yield": self.otr_yield,
            "ofr_yield": self.ofr_yield,
            "specialness_bps": self.specialness_bps,
        }
        if self.otr_bond:
            d["otr_bond"] = self.otr_bond.to_dict()
        if self.last_auction:
            d["last_auction"] = self.last_auction.isoformat()
        return d


def create_benchmark_set(
    tenors: list[str] | None = None,
) -> dict[str, TreasuryBenchmark]:
    """Create a set of empty benchmarks for standard T-Lock tenors."""
    if tenors is None:
        tenors = TLOCK_TENORS
    return {t: TreasuryBenchmark(tenor=t) for t in tenors}
