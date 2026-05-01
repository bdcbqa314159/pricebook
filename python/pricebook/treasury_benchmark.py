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


# ---------------------------------------------------------------------------
# When-issued bond
# ---------------------------------------------------------------------------

def when_issued_bond(
    auction_date: date,
    tenor_years: int,
    estimated_coupon: float | None = None,
    wi_yield: float | None = None,
    face_value: float = 100.0,
) -> FixedRateBond:
    """Construct a when-issued Treasury note before the auction.

    Pre-auction, the coupon is unknown. Convention:
    - If estimated_coupon given, use it directly.
    - If wi_yield given, round to nearest 1/8% (Treasury convention)
      to estimate coupon: coupon = round(wi_yield * 8) / 8.
    - Issue date = auction_date (dated date).
    - Maturity = auction_date + tenor_years.
    - Uses treasury_note conventions (ACT/ACT, T+1).

    Args:
        auction_date: expected auction settlement date.
        tenor_years: maturity in years (2, 5, 10, 30).
        estimated_coupon: expected coupon (if known from WI trading).
        wi_yield: when-issued yield (coupon inferred by rounding).
    """
    if estimated_coupon is not None:
        coupon = estimated_coupon
    elif wi_yield is not None:
        # Treasury coupon set at nearest 1/8% below WI yield
        coupon = int(wi_yield * 800) / 800.0  # floor to nearest 1/8%
    else:
        raise ValueError("Must provide either estimated_coupon or wi_yield")

    from dateutil.relativedelta import relativedelta
    maturity = auction_date + relativedelta(years=tenor_years)

    return FixedRateBond.treasury_note(
        issue_date=auction_date,
        maturity=maturity,
        coupon_rate=coupon,
        face_value=face_value,
    )


# ---------------------------------------------------------------------------
# Specialness dynamics
# ---------------------------------------------------------------------------

@dataclass
class SpecialnessProfile:
    """Specialness term structure at a point in time."""
    tenor: str
    gc_rate: float
    special_rates: dict[int, float]  # {days: special_rate}
    reference_date: date | None = None

    @property
    def overnight_specialness_bps(self) -> float:
        """O/N specialness in bp."""
        on_rate = self.special_rates.get(1, self.gc_rate)
        return (self.gc_rate - on_rate) * 10_000

    def specialness_at(self, days: int) -> float:
        """Specialness at a given tenor in bp."""
        if days in self.special_rates:
            return (self.gc_rate - self.special_rates[days]) * 10_000
        # Linear interpolation
        tenors = sorted(self.special_rates.keys())
        if not tenors:
            return 0.0
        if days <= tenors[0]:
            return (self.gc_rate - self.special_rates[tenors[0]]) * 10_000
        if days >= tenors[-1]:
            return (self.gc_rate - self.special_rates[tenors[-1]]) * 10_000
        for i in range(len(tenors) - 1):
            if tenors[i] <= days <= tenors[i + 1]:
                frac = (days - tenors[i]) / (tenors[i + 1] - tenors[i])
                r = self.special_rates[tenors[i]] + frac * (
                    self.special_rates[tenors[i + 1]] - self.special_rates[tenors[i]]
                )
                return (self.gc_rate - r) * 10_000
        return 0.0

    def forward_specialness(self, start_days: int, end_days: int) -> float:
        """Forward specialness between two tenors in bp.

        Inferred from no-arbitrage: the forward special rate is the
        rate that makes a term repo = rolling O/N repos.
        """
        if end_days <= start_days:
            return self.specialness_at(start_days)
        # df(0,T2) = df(0,T1) × df(T1,T2)
        # Forward rate from two discount factors
        r1 = self.gc_rate - self.specialness_at(start_days) / 10_000
        r2 = self.gc_rate - self.specialness_at(end_days) / 10_000
        df1 = 1.0 / (1.0 + r1 * start_days / 360.0)
        df2 = 1.0 / (1.0 + r2 * end_days / 360.0)
        dt = (end_days - start_days) / 360.0
        if dt <= 0 or df2 <= 0:
            return 0.0
        fwd_rate = (df1 / df2 - 1.0) / dt
        fwd_gc = self.gc_rate  # assume flat GC for forward
        return (fwd_gc - fwd_rate) * 10_000

    def expected_specialness_decay(
        self,
        days_to_auction: int,
        current_specialness_bps: float,
        decay_rate: float = 0.8,
    ) -> list[dict[str, float]]:
        """Project specialness decay as auction approaches.

        Post-auction, the OTR rolls to new issue and specialness
        of the old OTR typically collapses. Model:
        specialness(t) = current × decay_rate^(t / days_to_auction)

        Returns list of {day, specialness_bps} projections.
        """
        projections = []
        for d in range(0, days_to_auction + 1, max(1, days_to_auction // 10)):
            frac = d / max(days_to_auction, 1)
            spec = current_specialness_bps * (decay_rate ** frac)
            projections.append({"day": d, "specialness_bps": spec})
        # Post-auction: specialness collapses
        projections.append({
            "day": days_to_auction,
            "specialness_bps": current_specialness_bps * 0.1,  # ~90% collapse
        })
        return projections

    def to_dict(self) -> dict:
        return {
            "tenor": self.tenor,
            "gc_rate": self.gc_rate,
            "special_rates": {str(k): v for k, v in self.special_rates.items()},
            "overnight_specialness_bps": self.overnight_specialness_bps,
        }


# ---------------------------------------------------------------------------
# Dynamic CTD switching
# ---------------------------------------------------------------------------

@dataclass
class CTDTransition:
    """A potential CTD switch event."""
    current_ctd: str
    new_ctd: str
    yield_trigger: float    # yield level where switch occurs
    basis_impact_bps: float  # P&L impact of the switch
    probability: str         # "high", "medium", "low"


def ctd_switch_analysis(
    deliverables: list[dict],
    yield_range: tuple[float, float] = (-0.01, 0.01),
    n_steps: int = 20,
) -> list[CTDTransition]:
    """Analyse CTD switching across a yield range.

    For each yield shift, recompute implied repo for all deliverables
    and identify where the CTD switches.

    Args:
        deliverables: list of {name, price, cf, coupon, repo_rate, days}.
        yield_range: (min_shift, max_shift) in absolute yield change.
        n_steps: number of yield steps.

    Returns:
        List of CTDTransition events (yield levels where CTD changes).
    """
    if not deliverables:
        return []

    transitions = []
    step = (yield_range[1] - yield_range[0]) / n_steps

    prev_ctd = None
    for i in range(n_steps + 1):
        shift = yield_range[0] + i * step

        # Compute implied repo at each yield shift (price change)
        best_name = None
        best_repo = -999.0
        for d in deliverables:
            # Price change from yield shift: ΔP ≈ -duration × Δy × price
            duration = d.get("duration", 5.0)
            shifted_price = d["price"] * (1 - duration * shift)
            dt = d["days"] / 360.0
            if dt > 0 and shifted_price > 0:
                implied = (d["coupon"] * d["days"] / 360 - (shifted_price - d["cf"] * d.get("futures_price", 99.0))) / (shifted_price * dt)
            else:
                implied = 0.0
            if implied > best_repo:
                best_repo = implied
                best_name = d["name"]

        if prev_ctd is not None and best_name != prev_ctd:
            # CTD switch detected
            basis_impact = abs(shift) * 10_000  # rough bp impact
            prob = "high" if abs(shift) < 0.005 else ("medium" if abs(shift) < 0.01 else "low")
            transitions.append(CTDTransition(
                current_ctd=prev_ctd,
                new_ctd=best_name,
                yield_trigger=shift,
                basis_impact_bps=basis_impact,
                probability=prob,
            ))
        prev_ctd = best_name

    return transitions
