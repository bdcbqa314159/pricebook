"""Composite conventions for exotic structured products.

Conventions that nest other conventions — for products like TRS-on-SPV,
convertible bonds, or any instrument that combines multiple building blocks.

Each composite convention serialises to a nested JSON tree. The Serialisable
mixin handles recursion automatically via to_dict()/from_dict().

    from pricebook.models.composite_convention import (
        BondTRSConvention, SPVNoteConvention, CouponCapSpec,
        create_trs_on_spv,
    )

Usage:
    # Build from Python (interactive)
    conv = BondTRSConvention(
        bond_market="UST",
        funding_index="SOFR",
        funding_spread_bps=50,
        haircut=0.05,
        recovery=0.4,
    )
    trs = conv.create(issue_date, maturity, coupon=0.04, notional=10e6)

    # Build from JSON (static)
    conv = BondTRSConvention.from_dict(json.load(f))
    trs = conv.create(...)

    # Exotic: TRS on SPV note with dual credit + capped coupon
    spv_conv = SPVNoteConvention(
        bond_market="EUR_CORPORATE",
        issuer_recovery=0.30,
        coupon_cap=CouponCapSpec(strike=0.06),
        collateral_type="ABS",
    )
    trs_conv = BondTRSConvention(
        bond_market="BUND",  # reference convention for freq/dc
        funding_index="ESTR",
        funding_spread_bps=150,
        haircut=0.12,
        recovery=0.4,
        spv=spv_conv,  # nested SPV convention
    )

References:
    Burgess (2024). Bond Total Return Swaps.
    Lou (2018). TRS Pricing Framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.core.serialisable import serialisable_convention


@serialisable_convention("coupon_cap_spec")
@dataclass(frozen=True)
class CouponCapSpec:
    """Coupon cap specification for structured notes."""
    strike: float               # cap strike (e.g. 0.06 = 6%)
    cap_type: str = "european"  # "european" or "sticky"
    floor: float = 0.0          # optional floor (0 = no floor)


@serialisable_convention("funding_convention")
@dataclass(frozen=True)
class FundingConvention:
    """Funding leg convention for TRS or repo."""
    rate_index: str = "SOFR"
    spread_bps: float = 0.0
    day_count: str = "ACT/360"
    frequency: str = "quarterly"
    compounding: str = "in_arrears"
    payment_delay_days: int = 2
    observation_shift_days: int = 0


@serialisable_convention("collateral_convention")
@dataclass(frozen=True)
class CollateralConvention:
    """Collateral/margin convention."""
    haircut: float = 0.05
    margin_frequency: str = "daily"
    margin_currency: str = "USD"
    rehypothecation: bool = False
    eligible_collateral: list[str] = field(default_factory=lambda: ["govies", "cash"])


@serialisable_convention("spv_note_convention")
@dataclass(frozen=True)
class SPVNoteConvention:
    """SPV structured note convention — nests bond + credit conventions.

    An SPV note is a bond issued by a special purpose vehicle, typically
    backed by a collateral pool. It has:
    - Bond-like coupon structure (from a market convention)
    - Issuer credit risk (SPV default → collateral recovery)
    - Optional coupon cap/floor
    - Collateral type classification
    """
    bond_market: str = "BUND"           # sovereign convention for freq/dc
    issuer_recovery: float = 0.30       # SPV recovery rate
    collateral_type: str = "corporate"  # "ABS", "corporate", "sovereign", "covered"
    coupon_cap: CouponCapSpec | None = None
    notes: str = ""


@serialisable_convention("bond_trs_convention")
@dataclass(frozen=True)
class BondTRSConvention:
    """TRS convention — optionally nests an SPV convention for structured underlyings.

    For vanilla TRS:
        conv = BondTRSConvention(bond_market="UST", funding_index="SOFR", ...)

    For TRS on SPV note (exotic):
        conv = BondTRSConvention(bond_market="BUND", spv=SPVNoteConvention(...), ...)
    """
    bond_market: str = "UST"
    funding_index: str = "SOFR"
    funding_spread_bps: float = 0.0
    haircut: float = 0.05
    recovery: float = 0.4
    margin_frequency: str = "daily"
    spv: SPVNoteConvention | None = None    # nested SPV convention for exotic
    notes: str = ""

    def create(
        self,
        issue_date: date,
        maturity: date,
        coupon_rate: float,
        notional: float = 1_000_000.0,
        start: date | None = None,
        end: date | None = None,
    ) -> dict:
        """Create a TRS instrument from this convention.

        Returns a dict of kwargs suitable for TotalReturnSwap construction.
        If an SPV convention is present, builds the SPV note as the underlying.
        """
        from pricebook.fixed_income.sovereign_bonds import get_conventions
        from pricebook.fixed_income.bond import FixedRateBond

        # Build the underlying bond
        bond_conv = get_conventions(self.bond_market)

        if self.spv is not None:
            # SPV note: use SPV's bond market conventions, apply credit + cap
            spv_conv = get_conventions(self.spv.bond_market)
            underlying = FixedRateBond.from_convention(
                spv_conv, issue_date, maturity, coupon_rate,
            )
            credit_info = {
                "issuer_recovery": self.spv.issuer_recovery,
                "collateral_type": self.spv.collateral_type,
                "coupon_cap": self.spv.coupon_cap.to_dict() if self.spv.coupon_cap else None,
            }
        else:
            underlying = FixedRateBond.from_convention(
                bond_conv, issue_date, maturity, coupon_rate,
            )
            credit_info = None

        return {
            "underlying": underlying,
            "notional": notional,
            "start": start or issue_date,
            "end": end or maturity,
            "repo_spread": self.funding_spread_bps / 10_000,
            "haircut": self.haircut,
            "recovery": self.recovery,
            "credit_info": credit_info,
            "convention": self.to_dict(),
        }

    @property
    def is_exotic(self) -> bool:
        return self.spv is not None


def create_trs_on_spv(
    spv_convention: SPVNoteConvention,
    funding_index: str = "SOFR",
    funding_spread_bps: float = 150,
    haircut: float = 0.12,
    recovery: float = 0.4,
) -> BondTRSConvention:
    """Build a BondTRSConvention for a TRS on SPV note.

    Convenience function that wires the SPV convention into the TRS convention.
    """
    return BondTRSConvention(
        bond_market=spv_convention.bond_market,
        funding_index=funding_index,
        funding_spread_bps=funding_spread_bps,
        haircut=haircut,
        recovery=recovery,
        spv=spv_convention,
    )
