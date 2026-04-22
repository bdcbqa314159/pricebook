"""Unified API — price anything in 5 lines.

    import pricebook.api as pb

    # Quick start: flat curve
    curve = pb.flat_curve(0.04)

    # Or from market quotes (tenor strings accepted)
    curve = pb.build_curve("USD", deposits={"3M": 0.043, "6M": 0.042},
                           swaps={"2Y": 0.039, "5Y": 0.038, "10Y": 0.040})

    # Price anything — all return PV
    pb.irs("5Y", 0.04, curve)
    pb.bond("10Y", 0.04, curve)
    pb.swaption("1Y", "5Y", 0.04, vol=0.30, curve=curve)
    pb.cap("5Y", 0.04, vol=0.30, curve=curve)

    # Analytics
    pb.par_rate("5Y", curve)
    pb.swap_dv01("5Y", 0.04, curve)
    pb.bond_ytm("10Y", 0.04, market_price=98.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.bootstrap import bootstrap
from pricebook.capfloor import CapFloor
from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.commodity import CommodityForwardCurve, CommoditySwap
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.equity_forward import EquityForward
from pricebook.fra import FRA
from pricebook.fx_forward import FXForward
from pricebook.greeks import Greeks
from pricebook.inflation import CPICurve, zc_inflation_swap_pv, zc_inflation_par_rate
from pricebook.ois import bootstrap_ois
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.swaption import Swaption, SwaptionType
from pricebook.vol_surface import FlatVol
from pricebook.black76 import OptionType


# ============================================================================
# Tenor parsing — accepts "5Y", "6M", "2W", "30D" or a date
# ============================================================================

def _parse_tenor(ref: date, tenor: str | date) -> date:
    """Parse tenor string to date. Pass-through if already a date."""
    if isinstance(tenor, date):
        return tenor
    tenor = tenor.upper().strip()
    if tenor.endswith("Y"):
        return ref + relativedelta(years=int(tenor[:-1]))
    elif tenor.endswith("M"):
        return ref + relativedelta(months=int(tenor[:-1]))
    elif tenor.endswith("W"):
        return ref + relativedelta(weeks=int(tenor[:-1]))
    elif tenor.endswith("D"):
        return ref + relativedelta(days=int(tenor[:-1]))
    raise ValueError(
        f"Cannot parse tenor '{tenor}'. "
        f"Use '5Y', '6M', '2W', '30D', or pass a date object."
    )


def _today() -> date:
    return date.today()


# ============================================================================
# Convention engine
# ============================================================================

@dataclass(frozen=True)
class CcyConventions:
    """Market conventions for a currency."""
    calendar: str
    fixed_dc: DayCountConvention
    float_dc: DayCountConvention
    fixed_freq: Frequency
    float_freq: Frequency
    settlement_days: int


_CCY_CONVENTIONS = {
    "USD": CcyConventions("US", DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
                           Frequency.SEMI_ANNUAL, Frequency.QUARTERLY, 2),
    "EUR": CcyConventions("TARGET", DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
                           Frequency.ANNUAL, Frequency.QUARTERLY, 2),
    "GBP": CcyConventions("UK", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                           Frequency.SEMI_ANNUAL, Frequency.QUARTERLY, 0),
    "JPY": CcyConventions("JP", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_360,
                           Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL, 2),
    "CHF": CcyConventions("CH", DayCountConvention.THIRTY_360, DayCountConvention.ACT_360,
                           Frequency.ANNUAL, Frequency.QUARTERLY, 2),
    "CAD": CcyConventions("CA", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                           Frequency.SEMI_ANNUAL, Frequency.QUARTERLY, 1),
    "AUD": CcyConventions("AU", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                           Frequency.SEMI_ANNUAL, Frequency.QUARTERLY, 2),
}

_DEFAULT_CONV = _CCY_CONVENTIONS["USD"]


def conventions(ccy: str) -> CcyConventions:
    """Get market conventions for a currency.

        conv = pb.conventions("EUR")
        print(conv.fixed_freq)  # Frequency.ANNUAL
    """
    return _CCY_CONVENTIONS.get(ccy.upper(), _DEFAULT_CONV)


# ============================================================================
# Curve building
# ============================================================================

def flat_curve(rate: float = 0.04, reference_date: date | None = None) -> DiscountCurve:
    """Build a flat discount curve — the fastest way to get started.

        curve = pb.flat_curve(0.04)
    """
    ref = reference_date or _today()
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    dates = [ref + relativedelta(months=int(t * 12)) for t in tenors]
    dfs = [math.exp(-rate * t) for t in tenors]
    return DiscountCurve(ref, dates, dfs)


def build_curve(
    ccy: str = "USD",
    reference_date: date | None = None,
    deposits: dict[str, float] | list[tuple[date, float]] | None = None,
    swaps: dict[str, float] | list[tuple[date, float]] | None = None,
    ois_rates: dict[str, float] | list[tuple[date, float]] | None = None,
) -> DiscountCurve:
    """Build a discount curve from market quotes.

    Accepts tenor strings OR dates for flexibility:

        # Tenor strings (most convenient)
        curve = pb.build_curve("USD", deposits={"3M": 0.043}, swaps={"5Y": 0.039})

        # Explicit dates
        curve = pb.build_curve("USD", deposits=[(date(2026,7,21), 0.043)])

        # OIS rates
        curve = pb.build_curve("USD", ois_rates={"1Y": 0.04, "5Y": 0.042})
    """
    ref = reference_date or _today()
    conv = conventions(ccy)

    def _to_pairs(data) -> list[tuple[date, float]]:
        if data is None:
            return []
        if isinstance(data, dict):
            return sorted([(_parse_tenor(ref, t), r) for t, r in data.items()])
        return sorted(data)

    if ois_rates:
        return bootstrap_ois(ref, _to_pairs(ois_rates),
                             day_count=conv.float_dc, fixed_frequency=conv.fixed_freq)

    return bootstrap(ref, _to_pairs(deposits), _to_pairs(swaps),
                     fixed_day_count=conv.fixed_dc, float_day_count=conv.float_dc,
                     fixed_frequency=conv.fixed_freq, float_frequency=conv.float_freq)


def build_credit_curve(
    spreads: dict[str, float],
    discount_curve: DiscountCurve,
    recovery: float = 0.4,
    reference_date: date | None = None,
) -> SurvivalCurve:
    """Build a credit curve from CDS par spreads.

        surv = pb.build_credit_curve({"1Y": 0.005, "5Y": 0.01}, curve)
    """
    ref = reference_date or discount_curve.reference_date
    cds_inputs = sorted(
        [(_parse_tenor(ref, t), s) for t, s in spreads.items()],
        key=lambda x: x[0],
    )
    return bootstrap_credit_curve(ref, cds_inputs, discount_curve, recovery=recovery)


# ============================================================================
# IR Products
# ============================================================================

def irs(
    tenor: str | date, fixed_rate: float, curve: DiscountCurve,
    start: date | None = None, projection_curve: DiscountCurve | None = None,
    ccy: str = "USD", notional: float = 1_000_000.0, direction: str = "payer",
) -> float:
    """PV of a vanilla interest rate swap.

        pb.irs("5Y", 0.04, curve)
        pb.irs("10Y", 0.035, curve, direction="receiver")
    """
    s = start or curve.reference_date
    conv = conventions(ccy)
    d = SwapDirection.PAYER if direction.lower() == "payer" else SwapDirection.RECEIVER
    return InterestRateSwap(
        s, _parse_tenor(s, tenor), fixed_rate, direction=d, notional=notional,
        fixed_frequency=conv.fixed_freq, float_frequency=conv.float_freq,
        fixed_day_count=conv.fixed_dc, float_day_count=conv.float_dc,
    ).pv(curve, projection_curve)


def par_rate(
    tenor: str | date, curve: DiscountCurve,
    start: date | None = None, projection_curve: DiscountCurve | None = None,
    ccy: str = "USD",
) -> float:
    """Par swap rate.

        rate = pb.par_rate("5Y", curve)
    """
    s = start or curve.reference_date
    conv = conventions(ccy)
    return InterestRateSwap(
        s, _parse_tenor(s, tenor), 0.0, notional=1.0,
        fixed_frequency=conv.fixed_freq, float_frequency=conv.float_freq,
        fixed_day_count=conv.fixed_dc, float_day_count=conv.float_dc,
    ).par_rate(curve, projection_curve)


def swap_dv01(
    tenor: str | date, fixed_rate: float, curve: DiscountCurve,
    start: date | None = None, projection_curve: DiscountCurve | None = None,
    ccy: str = "USD", notional: float = 1_000_000.0,
) -> float:
    """Parallel DV01 of an interest rate swap.

        dv01 = pb.swap_dv01("5Y", 0.04, curve)
    """
    s = start or curve.reference_date
    conv = conventions(ccy)
    return InterestRateSwap(
        s, _parse_tenor(s, tenor), fixed_rate, notional=notional,
        fixed_frequency=conv.fixed_freq, float_frequency=conv.float_freq,
        fixed_day_count=conv.fixed_dc, float_day_count=conv.float_dc,
    ).dv01(curve, projection_curve)


def fra(
    start_tenor: str | date, end_tenor: str | date, strike: float,
    curve: DiscountCurve, projection_curve: DiscountCurve | None = None,
    notional: float = 1_000_000.0, reference_date: date | None = None,
) -> float:
    """PV of a forward rate agreement.

        pb.fra("3M", "6M", 0.04, curve)
    """
    ref = reference_date or curve.reference_date
    s = _parse_tenor(ref, start_tenor)
    e = _parse_tenor(ref, end_tenor)
    return FRA(s, e, strike, notional).pv(curve, projection_curve)


def cap(
    tenor: str | date, strike: float, vol: float, curve: DiscountCurve,
    start: date | None = None, notional: float = 1_000_000.0,
) -> float:
    """PV of an interest rate cap.

        pb.cap("5Y", 0.04, 0.30, curve)
    """
    s = start or curve.reference_date
    return CapFloor(s, _parse_tenor(s, tenor), strike,
                    OptionType.CALL, notional).pv(curve, FlatVol(vol))


def floor(
    tenor: str | date, strike: float, vol: float, curve: DiscountCurve,
    start: date | None = None, notional: float = 1_000_000.0,
) -> float:
    """PV of an interest rate floor.

        pb.floor("5Y", 0.03, 0.30, curve)
    """
    s = start or curve.reference_date
    return CapFloor(s, _parse_tenor(s, tenor), strike,
                    OptionType.PUT, notional).pv(curve, FlatVol(vol))


def swaption(
    expiry: str | date, swap_tenor: str | date, strike: float,
    vol: float, curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    swaption_type: str = "payer", notional: float = 1_000_000.0,
    return_greeks: bool = False,
) -> float | Greeks:
    """Price a European swaption. Set return_greeks=True for Greeks.

        pb.swaption("1Y", "5Y", 0.04, 0.30, curve)
        pb.swaption("1Y", "5Y", 0.04, 0.30, curve, return_greeks=True)
    """
    ref = curve.reference_date
    exp = _parse_tenor(ref, expiry)
    swap_end = _parse_tenor(exp, swap_tenor)
    st = SwaptionType.PAYER if swaption_type.lower() == "payer" else SwaptionType.RECEIVER
    swpn = Swaption(exp, swap_end, strike, swaption_type=st, notional=notional)
    if return_greeks:
        return swpn.greeks(curve, FlatVol(vol), projection_curve)
    return swpn.pv(curve, FlatVol(vol), projection_curve)


# ============================================================================
# Bonds
# ============================================================================

def bond(
    maturity: str | date, coupon_rate: float, curve: DiscountCurve,
    start: date | None = None, face_value: float = 100.0,
) -> float:
    """Dirty price of a fixed-rate bond (per 100 face).

        pb.bond("10Y", 0.04, curve)
    """
    s = start or curve.reference_date
    return FixedRateBond(s, _parse_tenor(s, maturity), coupon_rate,
                         face_value=face_value).dirty_price(curve)


def bond_ytm(
    maturity: str | date, coupon_rate: float, market_price: float,
    start: date | None = None, settlement: date | None = None,
) -> float:
    """Yield to maturity of a bond.

        ytm = pb.bond_ytm("10Y", 0.04, 98.5)
    """
    s = start or _today()
    b = FixedRateBond(s, _parse_tenor(s, maturity), coupon_rate)
    return b.yield_to_maturity(market_price, settlement or s)


def bond_duration(
    maturity: str | date, coupon_rate: float, market_price: float,
    start: date | None = None, settlement: date | None = None,
) -> float:
    """Modified duration of a bond.

        dur = pb.bond_duration("10Y", 0.04, 100.0)
    """
    s = start or _today()
    settle = settlement or s
    b = FixedRateBond(s, _parse_tenor(s, maturity), coupon_rate)
    ytm = b.yield_to_maturity(market_price, settle)
    return b.modified_duration(ytm, settle)


def bond_clean_price(
    maturity: str | date, coupon_rate: float, curve: DiscountCurve,
    start: date | None = None, settlement: date | None = None,
    face_value: float = 100.0,
) -> float:
    """Clean (quoted) price of a fixed-rate bond.

        pb.bond_clean_price("10Y", 0.04, curve)
    """
    s = start or curve.reference_date
    return FixedRateBond(s, _parse_tenor(s, maturity), coupon_rate,
                         face_value=face_value).clean_price(curve, settlement)


def z_spread(
    maturity: str | date, coupon_rate: float, market_price: float,
    curve: DiscountCurve, start: date | None = None,
) -> float:
    """Z-spread of a bond over the risk-free curve.

        zs = pb.z_spread("10Y", 0.05, 95.0, curve)
    """
    from pricebook.risky_bond import RiskyBond, z_spread as _z_spread
    s = start or curve.reference_date
    rb = RiskyBond(s, _parse_tenor(s, maturity), coupon_rate)
    return _z_spread(rb, market_price, curve)


# ============================================================================
# Credit
# ============================================================================

def cds(
    tenor: str | date, spread: float,
    discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    start: date | None = None, recovery: float = 0.4,
    notional: float = 1_000_000.0,
) -> float:
    """PV of a CDS (protection buyer).

        pb.cds("5Y", 0.01, curve, surv)
    """
    s = start or discount_curve.reference_date
    return CDS(s, _parse_tenor(s, tenor), spread,
               notional=notional, recovery=recovery).pv(discount_curve, survival_curve)


def cds_par_spread(
    tenor: str | date,
    discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    start: date | None = None, recovery: float = 0.4,
) -> float:
    """Par CDS spread.

        spread = pb.cds_par_spread("5Y", curve, surv)
    """
    s = start or discount_curve.reference_date
    return CDS(s, _parse_tenor(s, tenor), 0.01,
               notional=1.0, recovery=recovery).par_spread(discount_curve, survival_curve)


# ============================================================================
# FX
# ============================================================================

def fx_forward_rate(
    spot: float, tenor: str | date,
    base_curve: DiscountCurve, quote_curve: DiscountCurve,
    reference_date: date | None = None,
) -> float:
    """FX forward rate via CIP.

        pb.fx_forward_rate(1.10, "6M", eur_curve, usd_curve)
    """
    ref = reference_date or base_curve.reference_date
    return FXForward.forward_rate(spot, _parse_tenor(ref, tenor), base_curve, quote_curve)


def fx_option(
    spot: float, strike: float, tenor: str | date,
    vol: float, base_curve: DiscountCurve, quote_curve: DiscountCurve,
    option_type: str = "call", reference_date: date | None = None,
    return_greeks: bool = False,
) -> float | Greeks:
    """European FX option price (Garman-Kohlhagen).

        pb.fx_option(1.10, 1.12, "6M", 0.08, eur_curve, usd_curve)
    """
    from pricebook.fx_option import fx_option_price, fx_greeks
    ref = reference_date or base_curve.reference_date
    mat = _parse_tenor(ref, tenor)
    from pricebook.day_count import year_fraction as _yf
    T = _yf(ref, mat, DayCountConvention.ACT_365_FIXED)
    r_d = -math.log(quote_curve.df(mat)) / T if T > 0 else 0.0
    r_f = -math.log(base_curve.df(mat)) / T if T > 0 else 0.0
    ot = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT
    if return_greeks:
        return fx_greeks(spot, strike, r_d, r_f, vol, T, ot)
    return fx_option_price(spot, strike, r_d, r_f, vol, T, ot)


# ============================================================================
# Equity
# ============================================================================

def equity_forward(
    spot: float, maturity: str | date, curve: DiscountCurve,
    div_yield: float = 0.0, borrow_cost: float = 0.0,
    reference_date: date | None = None,
) -> float:
    """Equity forward price.

        pb.equity_forward(100, "1Y", curve, div_yield=0.02)
    """
    ref = reference_date or curve.reference_date
    mat = _parse_tenor(ref, maturity)
    return EquityForward(spot, mat, ref, div_yield=div_yield,
                         borrow_cost=borrow_cost).forward_price(curve)


def equity_option(
    spot: float, strike: float, maturity: str | date,
    vol: float, curve: DiscountCurve,
    option_type: str = "call", div_yield: float = 0.0,
    reference_date: date | None = None,
    return_greeks: bool = False,
) -> float | Greeks:
    """European equity option price (Black-Scholes).

        pb.equity_option(100, 105, "1Y", 0.20, curve)
        pb.equity_option(100, 105, "1Y", 0.20, curve, return_greeks=True)
    """
    from pricebook.equity_option import equity_option_price, equity_greeks
    ref = reference_date or curve.reference_date
    mat = _parse_tenor(ref, maturity)
    from pricebook.day_count import year_fraction as _yf
    T = _yf(ref, mat, DayCountConvention.ACT_365_FIXED)
    r = -math.log(curve.df(mat)) / T if T > 0 else 0.0
    ot = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT
    if return_greeks:
        return equity_greeks(spot, strike, r, vol, T, ot, div_yield)
    return equity_option_price(spot, strike, r, vol, T, ot, div_yield)


# ============================================================================
# Inflation
# ============================================================================

def inflation_breakeven(
    tenor: str | date, cpi_curve: CPICurve,
    reference_date: date | None = None,
) -> float:
    """Breakeven inflation rate at a given tenor.

        be = pb.inflation_breakeven("5Y", cpi_curve)
    """
    ref = reference_date or cpi_curve.reference_date
    return cpi_curve.breakeven_rate(_parse_tenor(ref, tenor))


# ============================================================================
# Commodity
# ============================================================================

def commodity_swap_pv(
    fixed_price: float, forward_curve: CommodityForwardCurve,
    discount_curve: DiscountCurve, tenor: str | date | None = None,
    start: date | None = None, quantity: float = 1.0,
) -> float:
    """PV of a commodity swap.

        pb.commodity_swap_pv(72.0, fwd_curve, disc_curve, "1Y")
    """
    s = start or discount_curve.reference_date
    if tenor:
        end = _parse_tenor(s, tenor)
    else:
        end = s + relativedelta(years=1)
    return CommoditySwap(s, end, fixed_price, quantity).pv(forward_curve, discount_curve)


# ============================================================================
# Market Environment — bundle all market data in one object
# ============================================================================

@dataclass
class MarketEnv:
    """Complete market data for pricing.

        env = pb.market_env(ref,
            curves={"USD": usd_curve, "EUR": eur_curve},
            credit_curves={"ACME": surv},
            fx_spots={"EUR/USD": 1.10})
        pv = pb.irs("5Y", 0.04, env.curve("USD"))
    """
    reference_date: date
    discount_curves: dict[str, DiscountCurve]
    projection_curves: dict[str, DiscountCurve] | None = None
    credit_curves: dict[str, SurvivalCurve] | None = None
    fx_spots: dict[str, float] | None = None
    cpi_curves: dict[str, CPICurve] | None = None
    commodity_curves: dict[str, CommodityForwardCurve] | None = None

    def curve(self, ccy: str = "USD") -> DiscountCurve:
        """Get discount curve for a currency."""
        if ccy in self.discount_curves:
            return self.discount_curves[ccy]
        raise ValueError(f"No discount curve for {ccy}. Available: {list(self.discount_curves)}")

    def projection(self, ccy: str = "USD") -> DiscountCurve:
        """Get projection curve. Falls back to discount if not provided."""
        if self.projection_curves and ccy in self.projection_curves:
            return self.projection_curves[ccy]
        return self.curve(ccy)

    def credit(self, name: str) -> SurvivalCurve:
        if self.credit_curves and name in self.credit_curves:
            return self.credit_curves[name]
        raise ValueError(f"No credit curve for {name}")

    def fx(self, pair: str) -> float:
        if self.fx_spots and pair in self.fx_spots:
            return self.fx_spots[pair]
        raise ValueError(f"No FX spot for {pair}")


def market_env(
    reference_date: date | None = None,
    curves: dict[str, DiscountCurve] | None = None,
    credit_curves: dict[str, SurvivalCurve] | None = None,
    fx_spots: dict[str, float] | None = None,
    cpi_curves: dict[str, CPICurve] | None = None,
    commodity_curves: dict[str, CommodityForwardCurve] | None = None,
) -> MarketEnv:
    """Create a market environment.

        env = pb.market_env(ref, curves={"USD": curve, "EUR": eur_curve})
    """
    ref = reference_date or _today()
    return MarketEnv(
        reference_date=ref,
        discount_curves=curves or {},
        credit_curves=credit_curves,
        fx_spots=fx_spots,
        cpi_curves=cpi_curves,
        commodity_curves=commodity_curves,
    )


# ============================================================================
# Additional products
# ============================================================================

def ois(
    tenor: str | date, fixed_rate: float, curve: DiscountCurve,
    start: date | None = None, ccy: str = "USD",
    notional: float = 1_000_000.0,
) -> float:
    """PV of an overnight index swap.

        pb.ois("2Y", 0.04, curve)
    """
    from pricebook.ois import OISSwap
    s = start or curve.reference_date
    conv = conventions(ccy)
    return OISSwap(s, _parse_tenor(s, tenor), fixed_rate,
                   notional=notional, day_count=conv.float_dc,
                   fixed_frequency=conv.fixed_freq).pv(curve)


def basis_swap(
    tenor: str | date, spread: float, curve: DiscountCurve,
    proj1: DiscountCurve, proj2: DiscountCurve,
    start: date | None = None, notional: float = 1_000_000.0,
) -> float:
    """PV of a basis swap (float vs float).

        pb.basis_swap("5Y", 0.001, disc, proj_3m, proj_6m)
    """
    from pricebook.basis_swap import BasisSwap
    s = start or curve.reference_date
    return BasisSwap(s, _parse_tenor(s, tenor), spread,
                     notional=notional).pv(curve, proj1, proj2)


def frn(
    maturity: str | date, spread: float, curve: DiscountCurve,
    start: date | None = None, notional: float = 1_000_000.0,
) -> float:
    """Dirty price of a floating-rate note.

        pb.frn("5Y", 0.005, curve)
    """
    from pricebook.frn import FloatingRateNote
    s = start or curve.reference_date
    return FloatingRateNote(s, _parse_tenor(s, maturity), spread,
                            notional=notional).dirty_price(curve)


def ndf(
    pair: str, contracted_rate: float, tenor: str | date,
    spot: float, base_curve: DiscountCurve, quote_curve: DiscountCurve,
    notional: float = 1_000_000.0, reference_date: date | None = None,
) -> float:
    """PV of a non-deliverable forward.

        pb.ndf("USD/CNY", 7.25, "1Y", 7.20, usd_curve, cny_curve)
    """
    from pricebook.ndf import NDF
    ref = reference_date or base_curve.reference_date
    return NDF(pair, _parse_tenor(ref, tenor), contracted_rate,
               notional).pv(spot, base_curve, quote_curve)


def risky_bond(
    maturity: str | date, coupon_rate: float,
    discount_curve: DiscountCurve, survival_curve: SurvivalCurve,
    start: date | None = None, recovery: float = 0.4,
) -> float:
    """Dirty price of a credit-risky bond.

        pb.risky_bond("10Y", 0.05, curve, surv)
    """
    from pricebook.risky_bond import RiskyBond
    s = start or discount_curve.reference_date
    return RiskyBond(s, _parse_tenor(s, maturity), coupon_rate,
                     recovery=recovery).dirty_price(discount_curve, survival_curve)


def variance_swap(
    fair_var: float, strike_var: float,
    notional_vega: float, T: float, df: float,
) -> float:
    """PV of a variance swap.

        pb.variance_swap(0.04, 0.035, 100_000, 0.5, 0.98)
    """
    from pricebook.variance_swap import variance_swap_pv
    return variance_swap_pv(fair_var, strike_var, notional_vega, T, df).pv


# ============================================================================
# XVA
# ============================================================================

def cva(
    expected_exposures: list[float],
    default_probs: list[float],
    recovery: float = 0.4,
    discount_factors: list[float] | None = None,
) -> float:
    """Unilateral CVA from exposure profile.

    CVA = (1-R) × Σ EE_i × ΔPD_i × DF_i

    Args:
        expected_exposures: expected positive exposure at each time step.
        default_probs: CUMULATIVE default probabilities (e.g. [0.01, 0.02, 0.03]).
            Marginal PD is computed internally as PD_i - PD_{i-1}.

        cva = pb.cva(exposures, cumulative_default_probs, recovery=0.4)
    """
    lgd = 1.0 - recovery
    total = 0.0
    for i, (ee, pd) in enumerate(zip(expected_exposures, default_probs)):
        df = discount_factors[i] if discount_factors else 1.0
        dpd = pd - (default_probs[i - 1] if i > 0 else 0.0)
        total += lgd * ee * dpd * df
    return total


def simm(
    sensitivities: list[dict],
) -> float:
    """ISDA SIMM initial margin.

    Each sensitivity is a dict with keys: risk_class, bucket, tenor, delta.

        margin = pb.simm([
            {"risk_class": "GIRR", "bucket": "USD", "tenor": "10Y", "delta": 1_000_000},
            {"risk_class": "FX", "bucket": "EUR/USD", "tenor": "spot", "delta": 5_000_000},
        ])
    """
    from pricebook.simm import SIMMCalculator, SIMMSensitivity
    sens = [SIMMSensitivity(**s) for s in sensitivities]
    return SIMMCalculator().compute(sens).total_margin


# ============================================================================
# Portfolio
# ============================================================================

@dataclass
class PortfolioResult:
    """Portfolio-level aggregation."""
    total_pv: float
    trade_pvs: dict[str, float]
    n_trades: int


def portfolio_pv(
    trades: dict[str, float],
) -> PortfolioResult:
    """Aggregate pre-computed trade PVs into a portfolio.

        result = pb.portfolio_pv({"swap_5y": 15000, "bond_10y": -3000, "cds_5y": 800})
        print(result.total_pv)  # 12800
    """
    return PortfolioResult(
        total_pv=sum(trades.values()),
        trade_pvs=dict(trades),
        n_trades=len(trades),
    )
