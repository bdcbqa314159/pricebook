"""CDS Index product: CDX/iTraxx as tradeable instrument.

Wraps CDSIndex with roll mechanics, intrinsic vs market spread,
index basis decomposition, and serialisation.

    from pricebook.cds_index_product import CDSIndexProduct

    product = CDSIndexProduct.from_spec("CDX.NA.IG", series=42,
                                         market_spread=0.005, notional=10_000_000)
    result = product.price(discount_curve, survival_curves)

References:
    O'Kane, D. (2008). Modelling Single-name and Multi-name Credit
    Derivatives. Wiley, Ch. 8 — CDS Indices.
    Markit. iTraxx and CDX Index Mechanics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.cds_conventions import (
    CDSIndexSpec, get_index_spec, cds_index_roll_date, next_imm_date,
    standard_cds_dates, STANDARD_COUPONS_BPS, STANDARD_RECOVERY,
)
from pricebook.cds_index import CDSIndex
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.serialisable import _register, _serialise_atom


@dataclass
class IndexResult:
    """CDS index pricing result."""
    pv: float
    intrinsic_spread: float      # weighted avg of constituent par spreads
    market_spread: float         # quoted market spread
    index_basis: float           # market - intrinsic (bp)
    n_constituents: int
    upfront_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "intrinsic_spread": self.intrinsic_spread,
            "market_spread": self.market_spread, "index_basis": self.index_basis,
            "n_constituents": self.n_constituents, "upfront_pct": self.upfront_pct,
        }


class CDSIndexProduct:
    """CDS Index as a tradeable instrument.

    Wraps a portfolio of equally-weighted single-name CDS with index
    conventions. Tracks series, roll dates, and index basis.

    Args:
        index_name: e.g. "CDX.NA.IG".
        series: series number (e.g. 42).
        market_spread: quoted market spread (decimal, e.g. 0.005 = 50bp).
        start: effective date.
        end: maturity date.
        notional: index notional.
        standard_coupon: standard running coupon.
        recovery: standard recovery rate.
        n_names: number of constituents.
    """

    _SERIAL_TYPE = "cds_index_product"

    def __init__(
        self,
        index_name: str,
        series: int = 1,
        market_spread: float = 0.005,
        start: date | None = None,
        end: date | None = None,
        notional: float = 10_000_000.0,
        standard_coupon: float = 0.01,
        recovery: float = 0.4,
        n_names: int = 125,
    ):
        self.index_name = index_name
        self.series = series
        self.market_spread = market_spread
        self.notional = notional
        self.standard_coupon = standard_coupon
        self.recovery = recovery
        self.n_names = n_names
        self.start = start
        self.end = end

    @classmethod
    def from_spec(
        cls,
        index_name: str,
        series: int = 1,
        market_spread: float = 0.005,
        reference_date: date | None = None,
        maturity_years: int = 5,
        notional: float = 10_000_000.0,
    ) -> CDSIndexProduct:
        """Build from a registered index spec."""
        spec = get_index_spec(index_name)
        coupon = spec.standard_coupon_bps / 10_000
        recovery = spec.standard_recovery

        start = None
        end = None
        if reference_date is not None:
            dates = standard_cds_dates(reference_date, maturity_years)
            start = dates[0]
            end = dates[-1]

        return cls(
            index_name=index_name, series=series,
            market_spread=market_spread, start=start, end=end,
            notional=notional, standard_coupon=coupon,
            recovery=recovery, n_names=spec.n_names,
        )

    def price(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
    ) -> IndexResult:
        """Price the index from constituent survival curves.

        Each constituent is equally weighted. Intrinsic spread is the
        average par spread. Index basis = market - intrinsic.
        """
        if len(survival_curves) != self.n_names:
            raise ValueError(
                f"Expected {self.n_names} survival curves, got {len(survival_curves)}"
            )

        ref = discount_curve.reference_date
        start = self.start or ref
        end = self.end or (ref + timedelta(days=1825))

        # Build constituent CDS at market spread
        per_name_notional = self.notional / self.n_names

        # Compute par spread for each constituent
        par_spreads = []
        total_pv = 0.0
        for sc in survival_curves:
            cds = CDS(start, end, spread=self.market_spread,
                      notional=per_name_notional, recovery=self.recovery)
            par = cds.par_spread(discount_curve, sc)
            par_spreads.append(par)
            total_pv += cds.pv(discount_curve, sc)

        intrinsic = sum(par_spreads) / len(par_spreads)
        basis = (self.market_spread - intrinsic) * 10_000  # in bp

        # Upfront
        from pricebook.cds import risky_annuity
        ann = risky_annuity(start, end, discount_curve,
                            survival_curves[0])  # approximate with first
        upfront_pct = (self.market_spread - self.standard_coupon) * ann

        return IndexResult(
            pv=total_pv,
            intrinsic_spread=intrinsic,
            market_spread=self.market_spread,
            index_basis=basis,
            n_constituents=self.n_names,
            upfront_pct=upfront_pct,
        )

    def intrinsic_spread(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
    ) -> float:
        """Intrinsic spread: equally-weighted average of constituent par spreads."""
        ref = discount_curve.reference_date
        start = self.start or ref
        end = self.end or (ref + timedelta(days=1825))
        total = 0.0
        for sc in survival_curves:
            cds = CDS(start, end, spread=0.01, notional=1.0, recovery=self.recovery)
            total += cds.par_spread(discount_curve, sc)
        return total / len(survival_curves)

    def cheapest_to_protect(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
        names: list[str] | None = None,
    ) -> dict[str, float]:
        """Find the widest-spread name in the index.

        Returns dict with name (or index), par_spread, rank.
        """
        ref = discount_curve.reference_date
        start = self.start or ref
        end = self.end or (ref + timedelta(days=1825))
        spreads = []
        for i, sc in enumerate(survival_curves):
            cds = CDS(start, end, spread=0.01, notional=1.0, recovery=self.recovery)
            par = cds.par_spread(discount_curve, sc)
            name = names[i] if names and i < len(names) else f"name_{i}"
            spreads.append((name, par))

        spreads.sort(key=lambda x: -x[1])
        return {
            "widest_name": spreads[0][0],
            "widest_spread_bp": spreads[0][1] * 10_000,
            "tightest_name": spreads[-1][0],
            "tightest_spread_bp": spreads[-1][1] * 10_000,
        }

    def next_roll_date(self, reference_date: date) -> date:
        """Next index roll date."""
        return cds_index_roll_date(self.index_name, reference_date)

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — needs survival_curves."""
        if ctx.credit_curves:
            scs = list(ctx.credit_curves.values())
            # Pad or truncate to n_names
            while len(scs) < self.n_names:
                scs.append(scs[-1])
            scs = scs[:self.n_names]
            return self.price(ctx.discount_curve, scs).pv
        return 0.0

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self._SERIAL_TYPE, "params": {
            "index_name": self.index_name, "series": self.series,
            "market_spread": self.market_spread,
            "notional": self.notional,
            "standard_coupon": self.standard_coupon,
            "recovery": self.recovery, "n_names": self.n_names,
        }}
        if self.start:
            d["params"]["start"] = self.start.isoformat()
        if self.end:
            d["params"]["end"] = self.end.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CDSIndexProduct:
        p = d["params"]
        return cls(
            index_name=p["index_name"], series=p.get("series", 1),
            market_spread=p["market_spread"],
            start=date.fromisoformat(p["start"]) if "start" in p else None,
            end=date.fromisoformat(p["end"]) if "end" in p else None,
            notional=p.get("notional", 10_000_000.0),
            standard_coupon=p.get("standard_coupon", 0.01),
            recovery=p.get("recovery", 0.4),
            n_names=p.get("n_names", 125),
        )


_register(CDSIndexProduct)
