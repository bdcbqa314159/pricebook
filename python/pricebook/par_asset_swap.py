"""Par asset swap: bond + IRS to convert fixed to floating.

The asset swap buyer:
- Buys the bond at par (paying 100 regardless of market price).
- Receives the bond's fixed coupons.
- Pays floating + ASW spread to the swap counterparty.

At inception, the swap is structured so the package has zero NPV.
The ASW spread compensates for any credit/liquidity premium in the bond.

    from pricebook.par_asset_swap import ParAssetSwap

    asw = ParAssetSwap(bond, settlement, market_price)
    result = asw.price(discount_curve)

References:
    Choudhry, *The Bond and Money Markets*, Butterworth-Heinemann, 2001, Ch. 14.
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*, Wiley, 2008.
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
