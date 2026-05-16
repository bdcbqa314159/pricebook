"""Bond position management: BondBook, aggregation by issuer/sector/tenor, limits.

Mirrors the equity/commodity desk pattern for fixed income. Positions
are tracked by issuer, sector (govt/IG/HY/EM), maturity bucket, and
currency, with DV01 and duration aggregation.

    book = BondBook("USD_Govts", valuation_date=date(2024, 1, 15))
    book.add(trade, issuer="UST", sector="govt", currency="USD",
             face_amount=10_000_000, dirty_price=98.5, coupon_rate=0.04,
             maturity=date(2034, 1, 15), dv01_per_million=85.0,
             duration=8.2)
    positions = book.positions_by_issuer()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.trade import Trade


# ---- Position dataclasses ----

@dataclass
class BondPosition:
    """Net position in a single issuer."""
    issuer: str
    sector: str
    currency: str
    net_face: float
    long_face: float
    short_face: float
    net_market_value: float
    net_dv01: float
    weighted_duration: float
    trade_count: int


@dataclass
class BondSectorExposure:
    """Aggregate exposure by sector."""
    sector: str
    net_face: float
    long_face: float
    short_face: float
    net_market_value: float
    net_dv01: float
    n_issuers: int


@dataclass
class BondTenorBucket:
    """Exposure within a maturity bucket."""
    bucket_label: str
    net_face: float
    net_market_value: float
    net_dv01: float
    n_positions: int


@dataclass
class BondLimitBreach:
    """Limit violation for a bond book."""
    limit_type: str
    limit_name: str
    limit_value: float
    actual_value: float
    details: str = ""


@dataclass
class BondLimits:
    """Position limits for a bond book.

    Attributes:
        max_face_per_issuer: per-issuer face cap.
        max_face_per_sector: per-sector face cap.
        max_dv01: absolute DV01 limit for the book.
        max_dv01_per_tenor: DV01 limit per maturity bucket.
        max_duration: maximum weighted-average duration.
    """
    max_face_per_issuer: dict[str, float] = field(default_factory=dict)
    max_face_per_sector: dict[str, float] = field(default_factory=dict)
    max_dv01: float | None = None
    max_dv01_per_tenor: dict[str, float] = field(default_factory=dict)
    max_duration: float | None = None


# ---- Trade entry ----

@dataclass
class BondTradeEntry:
    """A trade with bond-specific metadata."""
    trade: Trade
    issuer: str
    sector: str = "govt"
    currency: str = "USD"
    face_amount: float = 0.0
    dirty_price: float = 100.0
    coupon_rate: float = 0.0
    maturity: date | None = None
    rating: str = "BBB"
    dv01_per_million: float = 0.0
    duration: float = 0.0


# ---- Tenor bucketing ----

BOND_TENOR_BOUNDARIES = [1, 2, 3, 5, 7, 10, 20, 30]
BOND_TENOR_LABELS = [
    "≤1Y", "1-2Y", "2-3Y", "3-5Y", "5-7Y", "7-10Y", "10-20Y", "20-30Y", "30Y+",
]


def bond_tenor_bucket(valuation_date: date, maturity: date | None) -> str:
    """Bucket a bond maturity into a coarse tenor label."""
    if maturity is None:
        return "unknown"
    years = (maturity - valuation_date).days / 365.25
    for boundary, label in zip(BOND_TENOR_BOUNDARIES, BOND_TENOR_LABELS):
        if years <= boundary:
            return label
    return BOND_TENOR_LABELS[-1]


# ---- Bond book ----

class BondBook:
    """A named container of bond trades with per-issuer, per-sector,
    and per-tenor aggregation, DV01/duration tracking, and limit checking.

    Args:
        name: book name (e.g. "USD_Govts", "EUR_IG").
        valuation_date: as-of date for tenor bucketing.
        limits: optional :class:`BondLimits`.
        currency: book reporting currency.
    """

    def __init__(
        self,
        name: str,
        valuation_date: date,
        limits: BondLimits | None = None,
        currency: str = "USD",
    ):
        self.name = name
        self.valuation_date = valuation_date
        self.limits = limits or BondLimits()
        self.currency = currency
        self._entries: list[BondTradeEntry] = []

    def add(
        self,
        trade: Trade,
        issuer: str,
        sector: str = "govt",
        currency: str | None = None,
        face_amount: float = 0.0,
        dirty_price: float = 100.0,
        coupon_rate: float = 0.0,
        maturity: date | None = None,
        rating: str = "BBB",
        dv01_per_million: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        """Add a bond trade with metadata."""
        self._entries.append(BondTradeEntry(
            trade=trade, issuer=issuer, sector=sector,
            currency=currency or self.currency,
            face_amount=face_amount, dirty_price=dirty_price,
            coupon_rate=coupon_rate, maturity=maturity,
            rating=rating, dv01_per_million=dv01_per_million,
            duration=duration,
        ))

    @property
    def entries(self) -> list[BondTradeEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def n_issuers(self) -> int:
        return len({e.issuer for e in self._entries})

    @property
    def n_sectors(self) -> int:
        return len({e.sector for e in self._entries})

    # ---- Helpers ----

    @staticmethod
    def _signed(entry: BondTradeEntry) -> float:
        return entry.trade.direction * entry.trade.notional_scale

    @staticmethod
    def _market_value(entry: BondTradeEntry) -> float:
        return entry.face_amount * entry.dirty_price / 100.0

    @staticmethod
    def _trade_dv01(entry: BondTradeEntry) -> float:
        return entry.face_amount * entry.dv01_per_million / 1_000_000.0

    # ---- Aggregations ----

    def positions_by_issuer(self) -> list[BondPosition]:
        """Aggregate net positions per issuer."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            sign = self._signed(e)
            face = sign * e.face_amount
            mv = sign * self._market_value(e)
            dv01 = sign * self._trade_dv01(e)

            if e.issuer not in agg:
                agg[e.issuer] = {
                    "issuer": e.issuer, "sector": e.sector,
                    "currency": e.currency,
                    "net_face": 0.0, "long": 0.0, "short": 0.0,
                    "mv": 0.0, "dv01": 0.0,
                    "dur_num": 0.0, "dur_den": 0.0, "count": 0,
                }
            d = agg[e.issuer]
            d["net_face"] += face
            d["mv"] += mv
            d["dv01"] += dv01
            abs_mv = abs(sign * self._market_value(e))
            d["dur_num"] += abs_mv * e.duration
            d["dur_den"] += abs_mv
            if face > 0:
                d["long"] += abs(face)
            elif face < 0:
                d["short"] += abs(face)
            d["count"] += 1

        return [
            BondPosition(
                issuer=d["issuer"], sector=d["sector"], currency=d["currency"],
                net_face=d["net_face"], long_face=d["long"], short_face=d["short"],
                net_market_value=d["mv"], net_dv01=d["dv01"],
                weighted_duration=d["dur_num"] / d["dur_den"] if d["dur_den"] > 0 else 0.0,
                trade_count=d["count"],
            )
            for d in sorted(agg.values(), key=lambda x: x["issuer"])
        ]

    def positions_by_sector(self) -> list[BondSectorExposure]:
        """Aggregate exposure by sector."""
        agg: dict[str, dict] = {}
        issuers_seen: dict[str, set] = {}
        for e in self._entries:
            sign = self._signed(e)
            face = sign * e.face_amount
            mv = sign * self._market_value(e)
            dv01 = sign * self._trade_dv01(e)

            if e.sector not in agg:
                agg[e.sector] = {
                    "net_face": 0.0, "long": 0.0, "short": 0.0,
                    "mv": 0.0, "dv01": 0.0,
                }
                issuers_seen[e.sector] = set()
            d = agg[e.sector]
            d["net_face"] += face
            d["mv"] += mv
            d["dv01"] += dv01
            if face > 0:
                d["long"] += abs(face)
            elif face < 0:
                d["short"] += abs(face)
            issuers_seen[e.sector].add(e.issuer)

        return [
            BondSectorExposure(
                sector=sec,
                net_face=d["net_face"], long_face=d["long"], short_face=d["short"],
                net_market_value=d["mv"], net_dv01=d["dv01"],
                n_issuers=len(issuers_seen[sec]),
            )
            for sec, d in sorted(agg.items())
        ]

    def positions_by_tenor(self) -> list[BondTenorBucket]:
        """Aggregate exposure by maturity bucket."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            sign = self._signed(e)
            face = sign * e.face_amount
            mv = sign * self._market_value(e)
            dv01 = sign * self._trade_dv01(e)
            bucket = bond_tenor_bucket(self.valuation_date, e.maturity)

            if bucket not in agg:
                agg[bucket] = {"face": 0.0, "mv": 0.0, "dv01": 0.0, "count": 0}
            d = agg[bucket]
            d["face"] += face
            d["mv"] += mv
            d["dv01"] += dv01
            d["count"] += 1

        return [
            BondTenorBucket(
                bucket_label=b,
                net_face=d["face"], net_market_value=d["mv"],
                net_dv01=d["dv01"], n_positions=d["count"],
            )
            for b, d in sorted(agg.items())
        ]

    def net_dv01(self) -> float:
        """Total signed DV01 across all positions."""
        return sum(
            self._signed(e) * self._trade_dv01(e) for e in self._entries
        )

    def net_market_value(self) -> float:
        """Total signed market value across all positions."""
        return sum(
            self._signed(e) * self._market_value(e) for e in self._entries
        )

    def weighted_duration(self) -> float:
        """Market-value-weighted average duration across all positions."""
        num = 0.0
        den = 0.0
        for e in self._entries:
            abs_mv = abs(self._signed(e) * self._market_value(e))
            num += abs_mv * e.duration
            den += abs_mv
        return num / den if den > 0 else 0.0

    # ---- Limits ----

    def check_limits(self) -> list[BondLimitBreach]:
        """Check all configured limits and return any breaches."""
        breaches: list[BondLimitBreach] = []

        if self.limits.max_face_per_issuer:
            issuer_face = {
                p.issuer: abs(p.net_face) for p in self.positions_by_issuer()
            }
            for issuer, lim in self.limits.max_face_per_issuer.items():
                actual = issuer_face.get(issuer, 0.0)
                if actual > lim:
                    breaches.append(BondLimitBreach(
                        "per_issuer", issuer, lim, actual,
                        f"{issuer} face {actual:,.0f} exceeds {lim:,.0f}",
                    ))

        if self.limits.max_face_per_sector:
            sector_face = {
                s.sector: abs(s.net_face) for s in self.positions_by_sector()
            }
            for sector, lim in self.limits.max_face_per_sector.items():
                actual = sector_face.get(sector, 0.0)
                if actual > lim:
                    breaches.append(BondLimitBreach(
                        "per_sector", sector, lim, actual,
                        f"sector {sector} face {actual:,.0f} exceeds {lim:,.0f}",
                    ))

        if self.limits.max_dv01 is not None:
            actual = abs(self.net_dv01())
            if actual > self.limits.max_dv01:
                breaches.append(BondLimitBreach(
                    "dv01", f"book:{self.name}",
                    self.limits.max_dv01, actual,
                    f"DV01 {actual:,.0f} exceeds {self.limits.max_dv01:,.0f}",
                ))

        if self.limits.max_dv01_per_tenor:
            tenor_dv01 = {
                b.bucket_label: abs(b.net_dv01) for b in self.positions_by_tenor()
            }
            for tenor, lim in self.limits.max_dv01_per_tenor.items():
                actual = tenor_dv01.get(tenor, 0.0)
                if actual > lim:
                    breaches.append(BondLimitBreach(
                        "per_tenor_dv01", tenor, lim, actual,
                        f"tenor {tenor} DV01 {actual:,.0f} exceeds {lim:,.0f}",
                    ))

        if self.limits.max_duration is not None:
            actual = self.weighted_duration()
            if actual > self.limits.max_duration:
                breaches.append(BondLimitBreach(
                    "duration", f"book:{self.name}",
                    self.limits.max_duration, actual,
                    f"duration {actual:.1f} exceeds {self.limits.max_duration:.1f}",
                ))

        return breaches
