"""Asset swaps: par/par and proceeds conventions.

**Par/par asset swap** (ParAssetSwap):
  Buyer pays par (100), receives bond coupons, pays floating + ASW spread.
  Upfront = 100 − market_price.

**Proceeds asset swap** (ProceedsAssetSwap):
  Buyer pays market dirty price (no upfront exchange).
  Spread compensates credit/liquidity from dirty price, not par.

**Post-LIBOR:** Both use OIS discounting. The floating leg references
SOFR/ESTR/SONIA (not LIBOR). The ASW spread is over the RFR.

    from pricebook.par_asset_swap import (
        ParAssetSwap, ProceedsAssetSwap,
        asw_vs_zspread,
    )

References:
    Choudhry, *The Bond and Money Markets*, Butterworth-Heinemann, 2001, Ch. 14.
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*, Wiley, 2008.
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012, Ch. 17.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


@dataclass
class AssetSwapResult:
    """Par asset swap pricing result."""
    asw_spread: float          # spread over floating
    bond_pv: float             # PV of bond cashflows (risk-free)
    annuity: float             # PV of 1bp on the floating leg
    market_price: float
    upfront: float             # par - market = upfront payment


def forward_asw_spread(
    bond_rf_price: float,
    bond_risky_price: float,
    annuity: float,
) -> float:
    """Forward asset swap spread (Pucci 2012a, Eq 2).

    R^asw = (B^rf - B) / A

    Args:
        bond_rf_price: risk-free bond price B^rf (same coupons, risk-free curve).
        bond_risky_price: risky bond market price B.
        annuity: risk-free annuity A = sum(y_i * D_{0,T_i}).

    Returns:
        Forward ASW spread. Positive when risky bond trades below risk-free
        (credit spread). Can be negative for "premium bonds".
    """
    if abs(annuity) < 1e-15:
        return 0.0
    return (bond_rf_price - bond_risky_price) / annuity


class ParAssetSwap:
    """Par asset swap package.

    Args:
        bond: the underlying fixed-rate bond.
        settlement: trade settlement date.
        market_price: bond clean market price (per 100 face).
    """

    def __init__(
        self,
        bond: FixedRateBond,
        settlement: date,
        market_price: float,
    ):
        self.bond = bond
        self.settlement = settlement
        self.market_price = market_price

    def price(self, curve: DiscountCurve) -> AssetSwapResult:
        """Compute par asset swap spread.

        ASW = (100 - market_price + PV_coupons - 100 × df_maturity) / (annuity × 100)

        Equivalently: the spread that makes the floating leg PV equal the
        difference between the bond's risk-free PV and its market price.
        """
        # PV of remaining fixed coupons
        pv_coupons = sum(
            cf.amount * curve.df(cf.payment_date)
            for cf in self.bond._future_cashflows(self.settlement)
        )

        # PV of principal
        pv_principal = self.bond.face_value * curve.df(self.bond.maturity)

        # Annuity (sum of year_frac × df for each period)
        annuity = 0.0
        for cf in self.bond._future_cashflows(self.settlement):
            yf = year_fraction(cf.accrual_start, cf.accrual_end, self.bond.day_count)
            annuity += yf * curve.df(cf.payment_date)

        if annuity <= 0:
            return AssetSwapResult(0.0, 0.0, 0.0, self.market_price, 0.0)

        # Risk-free PV per 100 face
        rf_pv = (pv_coupons + pv_principal) / self.bond.face_value * 100.0

        # Upfront: buyer pays par, receives bond worth market_price
        upfront = 100.0 - self.market_price

        # ASW spread = (rf_pv - market_price) / (annuity × face)
        asw = (rf_pv - self.market_price) / (annuity * 100.0)

        return AssetSwapResult(
            asw_spread=asw,
            bond_pv=rf_pv,
            annuity=annuity,
            market_price=self.market_price,
            upfront=upfront,
        )

    _SERIAL_TYPE = "par_asset_swap"

    def to_dict(self) -> dict:
        return {"type": self._SERIAL_TYPE, "params": {
            "bond": self.bond.to_dict(),
            "settlement": self.settlement.isoformat(),
            "market_price": self.market_price,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> "ParAssetSwap":
        from pricebook.serialisable import from_dict as _fd
        p = d["params"]
        return cls(
            bond=_fd(p["bond"]),
            settlement=date.fromisoformat(p["settlement"]),
            market_price=p["market_price"],
        )


class ProceedsAssetSwap:
    """Proceeds (market-value) asset swap.

    Buyer pays dirty market price (not par). No upfront exchange.
    The swap spread compensates for the difference between the
    bond's risk-free PV and its market dirty price.

    Spread = (rf_dirty − market_dirty) / annuity

    For par bonds, proceeds ASW = par ASW. For off-par bonds they differ:
    par ASW embeds the upfront (100 − market) in the spread;
    proceeds ASW prices from dirty directly.

    Args:
        bond: the underlying fixed-rate bond.
        settlement: trade settlement date.
        market_dirty_price: bond dirty market price (per 100 face).
    """

    def __init__(
        self,
        bond: FixedRateBond,
        settlement: date,
        market_dirty_price: float,
    ):
        self.bond = bond
        self.settlement = settlement
        self.market_dirty_price = market_dirty_price

    def price(self, curve: DiscountCurve) -> AssetSwapResult:
        """Compute proceeds asset swap spread.

        ASW_proceeds = (B^rf_dirty − B_dirty) / (annuity × face)
        """
        pv_coupons = sum(
            cf.amount * curve.df(cf.payment_date)
            for cf in self.bond._future_cashflows(self.settlement)
        )
        pv_principal = self.bond.face_value * curve.df(self.bond.maturity)

        annuity = 0.0
        for cf in self.bond._future_cashflows(self.settlement):
            yf = year_fraction(cf.accrual_start, cf.accrual_end, self.bond.day_count)
            annuity += yf * curve.df(cf.payment_date)

        if annuity <= 0:
            return AssetSwapResult(0.0, 0.0, 0.0, self.market_dirty_price, 0.0)

        rf_dirty = (pv_coupons + pv_principal) / self.bond.face_value * 100.0
        asw = (rf_dirty - self.market_dirty_price) / (annuity * 100.0)

        return AssetSwapResult(
            asw_spread=asw,
            bond_pv=rf_dirty,
            annuity=annuity,
            market_price=self.market_dirty_price,
            upfront=0.0,  # no upfront in proceeds convention
        )

    _SERIAL_TYPE = "proceeds_asset_swap"

    def to_dict(self) -> dict:
        return {"type": self._SERIAL_TYPE, "params": {
            "bond": self.bond.to_dict(),
            "settlement": self.settlement.isoformat(),
            "market_dirty_price": self.market_dirty_price,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> "ProceedsAssetSwap":
        from pricebook.serialisable import from_dict as _fd
        p = d["params"]
        return cls(
            bond=_fd(p["bond"]),
            settlement=date.fromisoformat(p["settlement"]),
            market_dirty_price=p["market_dirty_price"],
        )


from pricebook.serialisable import _register as _reg_asw
_reg_asw(ParAssetSwap)
_reg_asw(ProceedsAssetSwap)


# ---------------------------------------------------------------------------
# Z-spread / ASW bridge
# ---------------------------------------------------------------------------

@dataclass
class SpreadComparison:
    """Comparison of z-spread vs asset swap spreads."""
    z_spread: float
    par_asw_spread: float
    proceeds_asw_spread: float
    par_asw_basis: float       # par_asw - z_spread (bp)
    proceeds_asw_basis: float  # proceeds_asw - z_spread (bp)

    def to_dict(self) -> dict:
        return {
            "z_spread": self.z_spread,
            "par_asw_spread": self.par_asw_spread,
            "proceeds_asw_spread": self.proceeds_asw_spread,
            "par_asw_basis_bp": self.par_asw_basis,
            "proceeds_asw_basis_bp": self.proceeds_asw_basis,
        }


def asw_vs_zspread(
    bond: FixedRateBond,
    market_clean_price: float,
    discount_curve: DiscountCurve,
    settlement: date | None = None,
) -> SpreadComparison:
    """Compare z-spread vs par and proceeds ASW spreads.

    At par (market = 100): all three spreads are approximately equal.
    Off par: par ASW and z-spread diverge; proceeds ASW stays closer.

    Args:
        bond: fixed-rate bond.
        market_clean_price: clean market price per 100 face.
        discount_curve: OIS discount curve.
        settlement: settlement date (default: curve reference date).
    """
    settle = settlement or discount_curve.reference_date

    # Z-spread via risky_bond module
    from pricebook.risky_bond import z_spread as _zs, RiskyBond
    rb = RiskyBond(
        bond.issue_date, bond.maturity, bond.coupon_rate,
        notional=bond.face_value,
        frequency=bond.frequency,
        day_count=bond.day_count,
    )
    zs = _zs(rb, market_clean_price, discount_curve, settle)

    # Par ASW
    par = ParAssetSwap(bond, settle, market_clean_price)
    par_result = par.price(discount_curve)

    # Proceeds ASW (using clean as proxy for dirty at settlement)
    proceeds = ProceedsAssetSwap(bond, settle, market_clean_price)
    proceeds_result = proceeds.price(discount_curve)

    return SpreadComparison(
        z_spread=zs,
        par_asw_spread=par_result.asw_spread,
        proceeds_asw_spread=proceeds_result.asw_spread,
        par_asw_basis=(par_result.asw_spread - zs) * 10_000,
        proceeds_asw_basis=(proceeds_result.asw_spread - zs) * 10_000,
    )
