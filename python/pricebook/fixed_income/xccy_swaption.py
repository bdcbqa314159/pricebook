"""Cross-currency swaption pricing via Black-76 and Bachelier (normal) models.

Functions
---------
- xccy_forward_spread        — forward xccy basis spread from curves and FX
- xccy_swaption_black        — Black-76 pricing on forward xccy basis spread
- xccy_swaption_bachelier    — Bachelier (normal) pricing; handles negative spreads
- xccy_swaption_greeks       — numerical Greeks: delta_dom, delta_for, vega, gamma, fx_delta

References
----------
- Brigo & Mercurio, "Interest Rate Models", Ch. 13 (cross-currency extensions)
- Piterbarg, "Interest Rate Modelling", Vol. III (multi-currency swaptions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class XCCYSwaptionResult:
    """Pricing result for a cross-currency swaption.

    Attributes
    ----------
    price               : option price in domestic currency units
    price_domestic      : same as price (alias kept for explicitness)
    delta_domestic      : dPrice/d(domestic parallel shift) in domestic units per bp
    delta_foreign       : dPrice/d(foreign parallel shift) in domestic units per bp
    vega                : dPrice/d(vol) per 1 vol unit
    gamma               : d²Price/d(spread)²
    fx_delta            : dPrice/d(fx_spot)
    annuity             : domestic annuity (sum of df * tau for underlying swap)
    forward_spread      : forward xccy basis spread (domestic minus foreign, annual)
    implied_vol_input   : vol input as supplied by the caller
    """

    price: float
    price_domestic: float
    delta_domestic: float
    delta_foreign: float
    vega: float
    gamma: float
    fx_delta: float
    annuity: float
    forward_spread: float
    implied_vol_input: float

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _payment_dates(reference_date: date, start_years: float, tenor_years: float, frequency: int) -> list[date]:
    """Generate payment dates for the underlying xccy swap leg."""
    months_per_period = 12 // frequency
    start = reference_date + relativedelta(months=round(start_years * 12))
    end = start + relativedelta(months=round(tenor_years * 12))
    dates: list[date] = []
    d = start + relativedelta(months=months_per_period)
    while d <= end:
        dates.append(d)
        d += relativedelta(months=months_per_period)
    if not dates or dates[-1] != end:
        dates.append(end)
    return dates


def _annuity(
    reference_date: date,
    curve: DiscountCurve,
    start_years: float,
    tenor_years: float,
    frequency: int,
) -> float:
    """Domestic annuity: sum of df(t_i) * tau_i over the swap fixed leg."""
    pay_dates = _payment_dates(reference_date, start_years, tenor_years, frequency)
    start_date = reference_date + relativedelta(months=round(start_years * 12))
    prev = start_date
    ann = 0.0
    for d in pay_dates:
        tau = year_fraction(prev, d, DayCountConvention.ACT_360)
        ann += curve.df(d) * tau
        prev = d
    return ann


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _black76_price(F: float, K: float, vol: float, T: float, df: float, is_call: bool) -> float:
    """Black-76 call or put price."""
    if T <= 0.0 or vol <= 0.0:
        intrinsic = F - K if is_call else K - F
        return df * max(intrinsic, 0.0)
    sigma_sqrt_T = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    if is_call:
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def _bachelier_price(F: float, K: float, vol_n: float, T: float, df: float, is_call: bool) -> float:
    """Bachelier (normal) call or put price."""
    if T <= 0.0 or vol_n <= 0.0:
        intrinsic = F - K if is_call else K - F
        return df * max(intrinsic, 0.0)
    sigma_sqrt_T = vol_n * math.sqrt(T)
    d = (F - K) / sigma_sqrt_T
    if is_call:
        return df * ((F - K) * _norm_cdf(d) + sigma_sqrt_T * _norm_pdf(d))
    return df * ((K - F) * _norm_cdf(-d) + sigma_sqrt_T * _norm_pdf(d))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def xccy_forward_spread(
    reference_date: date,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    fx_spot: float,
    start_years: float,
    tenor_years: float,
    frequency: int = 4,
) -> float:
    """Compute the forward xccy basis spread.

    The forward spread is the spread S such that the present value of the
    domestic leg (floating flat) equals the present value of the foreign leg
    converted to domestic at forward FX, with spread S added to the domestic
    leg to make the swap at-market.

    Formula (simplified, no collateral):
        S = (df_for(T_end) / df_dom(T_end) * fx_fwd_ratio - 1) / annuity_dom

    where annuity_dom is the domestic annuity of the underlying swap.

    Args:
        reference_date  : pricing date.
        domestic_curve  : domestic discount curve.
        foreign_curve   : foreign discount curve.
        fx_spot         : spot FX (domestic per foreign unit).
        start_years     : forward start in years from reference_date.
        tenor_years     : swap tenor in years.
        frequency       : payment frequency (4 = quarterly).

    Returns:
        Forward xccy basis spread (annualised, e.g. 0.002 = 20 bp).
    """
    start_date = reference_date + relativedelta(months=round(start_years * 12))
    end_date = start_date + relativedelta(months=round(tenor_years * 12))

    df_dom_start = domestic_curve.df(start_date)
    df_dom_end = domestic_curve.df(end_date)
    df_for_start = foreign_curve.df(start_date)
    df_for_end = foreign_curve.df(end_date)

    # Forward FX at start and end (domestic per foreign)
    # Covered interest parity: fwd_fx(T) = spot * df_for(T) / df_dom(T)
    fwd_fx_start = fx_spot * df_for_start / df_dom_start
    fwd_fx_end = fx_spot * df_for_end / df_dom_end

    ann = _annuity(reference_date, domestic_curve, start_years, tenor_years, frequency)

    # Full floating-leg PV replication (Brigo & Mercurio Ch. 13).
    #
    # Domestic floating leg PV (par floater): df_dom(start) - df_dom(end).
    # Foreign floating leg PV in domestic currency (notional exchanged at spot):
    #   FX_spot × N_for × (df_for(start) - df_for(end))
    # where N_for = 1/FX_spot (unit domestic notional → foreign notional).
    # So foreign PV in domestic = df_for(start) - df_for(end).
    #
    # The basis spread S on the domestic leg makes PV = 0:
    #   df_dom(start) - df_dom(end) + S × ann = df_for(start) - df_for(end)
    dom_float_pv = df_dom_start - df_dom_end
    for_float_pv = df_for_start - df_for_end
    spread = (dom_float_pv - for_float_pv) / ann if ann > 1e-12 else 0.0
    return spread


def xccy_swaption_black(
    reference_date: date,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    fx_spot: float,
    strike_spread: float,
    vol: float,
    expiry_years: float,
    swap_tenor_years: float,
    notional: float = 1_000_000.0,
    is_payer: bool = True,
    frequency: int = 4,
) -> XCCYSwaptionResult:
    """Price a cross-currency swaption using Black-76 on the forward xccy spread.

    A payer swaption gives the right to enter as payer on the domestic spread leg.
    This is equivalent to a call on the forward xccy basis spread.

    Args:
        reference_date   : pricing date.
        domestic_curve   : domestic discount curve.
        foreign_curve    : foreign discount curve.
        fx_spot          : spot FX (domestic per foreign).
        strike_spread    : strike xccy basis spread (e.g. 0.002 = 20 bp).
        vol              : lognormal vol on the forward spread (e.g. 0.30 = 30%).
        expiry_years     : option expiry in years.
        swap_tenor_years : underlying swap tenor in years.
        notional         : notional in domestic currency.
        is_payer         : True = payer swaption (call on spread).
        frequency        : payment frequency (4 = quarterly).

    Returns:
        XCCYSwaptionResult with price and Greeks.
    """
    fwd_spread = xccy_forward_spread(
        reference_date, domestic_curve, foreign_curve, fx_spot,
        expiry_years, swap_tenor_years, frequency,
    )
    ann = _annuity(reference_date, domestic_curve, expiry_years, swap_tenor_years, frequency)
    df_expiry = domestic_curve.df(
        reference_date + relativedelta(months=round(expiry_years * 12))
    )

    option_price_per_unit = _black76_price(
        F=fwd_spread, K=strike_spread, vol=vol, T=expiry_years,
        df=df_expiry, is_call=is_payer,
    )
    price = notional * ann * option_price_per_unit

    greeks = xccy_swaption_greeks(
        reference_date, domestic_curve, foreign_curve, fx_spot,
        strike_spread, vol, expiry_years, swap_tenor_years,
        notional, is_payer, frequency,
    )

    return XCCYSwaptionResult(
        price=price,
        price_domestic=price,
        delta_domestic=greeks.delta_domestic,
        delta_foreign=greeks.delta_foreign,
        vega=greeks.vega,
        gamma=greeks.gamma,
        fx_delta=greeks.fx_delta,
        annuity=ann * notional,
        forward_spread=fwd_spread,
        implied_vol_input=vol,
    )


def xccy_swaption_bachelier(
    reference_date: date,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    fx_spot: float,
    strike_spread: float,
    vol_normal: float,
    expiry_years: float,
    swap_tenor_years: float,
    notional: float = 1_000_000.0,
    is_payer: bool = True,
    frequency: int = 4,
) -> XCCYSwaptionResult:
    """Price a cross-currency swaption using the Bachelier (normal) model.

    Preferred when xccy basis spreads can be negative (common for EUR/USD basis).
    Prices a call (payer) or put (receiver) on the forward xccy spread assuming
    a normally distributed spread under the annuity measure.

    Args:
        reference_date   : pricing date.
        domestic_curve   : domestic discount curve.
        foreign_curve    : foreign discount curve.
        fx_spot          : spot FX (domestic per foreign).
        strike_spread    : strike xccy basis spread.
        vol_normal       : normal vol on the forward spread (absolute, e.g. 0.0005 = 5 bp/yr).
        expiry_years     : option expiry in years.
        swap_tenor_years : underlying swap tenor in years.
        notional         : notional in domestic currency.
        is_payer         : True = payer swaption (call on spread).
        frequency        : payment frequency (4 = quarterly).

    Returns:
        XCCYSwaptionResult with price and Greeks.
    """
    fwd_spread = xccy_forward_spread(
        reference_date, domestic_curve, foreign_curve, fx_spot,
        expiry_years, swap_tenor_years, frequency,
    )
    ann = _annuity(reference_date, domestic_curve, expiry_years, swap_tenor_years, frequency)
    df_expiry = domestic_curve.df(
        reference_date + relativedelta(months=round(expiry_years * 12))
    )

    option_price_per_unit = _bachelier_price(
        F=fwd_spread, K=strike_spread, vol_n=vol_normal, T=expiry_years,
        df=df_expiry, is_call=is_payer,
    )
    price = notional * ann * option_price_per_unit

    greeks = xccy_swaption_greeks(
        reference_date, domestic_curve, foreign_curve, fx_spot,
        strike_spread, vol_normal, expiry_years, swap_tenor_years,
        notional, is_payer, frequency,
        _use_bachelier=True,
    )

    return XCCYSwaptionResult(
        price=price,
        price_domestic=price,
        delta_domestic=greeks.delta_domestic,
        delta_foreign=greeks.delta_foreign,
        vega=greeks.vega,
        gamma=greeks.gamma,
        fx_delta=greeks.fx_delta,
        annuity=ann * notional,
        forward_spread=fwd_spread,
        implied_vol_input=vol_normal,
    )


@dataclass
class _GreeksResult:
    delta_domestic: float
    delta_foreign: float
    vega: float
    gamma: float
    fx_delta: float


def xccy_swaption_greeks(
    reference_date: date,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    fx_spot: float,
    strike_spread: float,
    vol: float,
    expiry_years: float,
    swap_tenor_years: float,
    notional: float = 1_000_000.0,
    is_payer: bool = True,
    frequency: int = 4,
    bump: float = 0.0001,
    _use_bachelier: bool = False,
) -> _GreeksResult:
    """Compute numerical Greeks for a cross-currency swaption.

    All curve bumps are parallel shifts of 1 bp (bump=0.0001 by default).

    Args:
        reference_date   : pricing date.
        domestic_curve   : domestic discount curve.
        foreign_curve    : foreign discount curve.
        fx_spot          : spot FX (domestic per foreign).
        strike_spread    : strike xccy basis spread.
        vol              : vol input (lognormal or normal per _use_bachelier).
        expiry_years     : option expiry in years.
        swap_tenor_years : underlying swap tenor in years.
        notional         : notional in domestic currency.
        is_payer         : True = payer swaption.
        frequency        : payment frequency.
        bump             : parallel rate bump for finite differences (default 1 bp).
        _use_bachelier   : if True use Bachelier model internally.

    Returns:
        _GreeksResult with delta_domestic, delta_foreign, vega, gamma, fx_delta.
    """

    def _price(dom_curve: DiscountCurve, for_curve: DiscountCurve, fx: float, v: float) -> float:
        fwd = xccy_forward_spread(
            reference_date, dom_curve, for_curve, fx,
            expiry_years, swap_tenor_years, frequency,
        )
        ann = _annuity(reference_date, dom_curve, expiry_years, swap_tenor_years, frequency)
        df_exp = dom_curve.df(reference_date + relativedelta(months=round(expiry_years * 12)))
        if _use_bachelier:
            opt = _bachelier_price(fwd, strike_spread, v, expiry_years, df_exp, is_payer)
        else:
            opt = _black76_price(fwd, strike_spread, v, expiry_years, df_exp, is_payer)
        return notional * ann * opt

    def _bump_curve(curve: DiscountCurve, shift: float) -> DiscountCurve:
        """Return a new DiscountCurve with all zero rates shifted by shift."""
        ref = curve.reference_date
        # Collect the original pillar dates and dfs, apply parallel zero shift
        pillar_dates = [ref + relativedelta(months=m) for m in [3, 6, 12, 24, 36, 60, 84, 120, 180, 240, 360]]
        new_dfs = []
        for d in pillar_dates:
            t = year_fraction(ref, d, DayCountConvention.ACT_365_FIXED)
            if t <= 0:
                new_dfs.append(1.0)
            else:
                z = curve.zero_rate(d) + shift
                new_dfs.append(math.exp(-z * t))
        return DiscountCurve(ref, pillar_dates, new_dfs)

    base = _price(domestic_curve, foreign_curve, fx_spot, vol)

    # Delta domestic: parallel bump of domestic curve by 1 bp
    dom_up = _bump_curve(domestic_curve, bump)
    dom_dn = _bump_curve(domestic_curve, -bump)
    p_dom_up = _price(dom_up, foreign_curve, fx_spot, vol)
    p_dom_dn = _price(dom_dn, foreign_curve, fx_spot, vol)
    delta_dom = (p_dom_up - p_dom_dn) / (2.0 * bump)

    # Delta foreign: parallel bump of foreign curve by 1 bp
    for_up = _bump_curve(foreign_curve, bump)
    for_dn = _bump_curve(foreign_curve, -bump)
    p_for_up = _price(domestic_curve, for_up, fx_spot, vol)
    p_for_dn = _price(domestic_curve, for_dn, fx_spot, vol)
    delta_for = (p_for_up - p_for_dn) / (2.0 * bump)

    # Vega: bump vol by 1 vol point (0.01)
    vega_bump = 0.01
    p_vega_up = _price(domestic_curve, foreign_curve, fx_spot, vol + vega_bump)
    p_vega_dn = _price(domestic_curve, foreign_curve, fx_spot, vol - vega_bump)
    vega = (p_vega_up - p_vega_dn) / (2.0 * vega_bump)

    # Gamma: second derivative w.r.t. domestic curve
    gamma = (p_dom_up - 2.0 * base + p_dom_dn) / (bump * bump)

    # FX delta: bump spot FX by 0.1%
    fx_bump = fx_spot * 0.001
    p_fx_up = _price(domestic_curve, foreign_curve, fx_spot + fx_bump, vol)
    p_fx_dn = _price(domestic_curve, foreign_curve, fx_spot - fx_bump, vol)
    fx_delta = (p_fx_up - p_fx_dn) / (2.0 * fx_bump)

    return _GreeksResult(
        delta_domestic=delta_dom,
        delta_foreign=delta_for,
        vega=vega,
        gamma=gamma,
        fx_delta=fx_delta,
    )
