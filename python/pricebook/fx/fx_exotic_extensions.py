"""FX exotic extensions: digitals, quantos, vol swaps, compound, chooser.

Fills remaining gaps in the FX product suite with explicit products
rather than approximation proxies.

* :func:`fx_digital_option` — European digital (binary) FX option.
* :func:`fx_double_barrier_option` — analytical double knock-out/knock-in.
* :func:`fx_quanto_option` — FX quanto with correlation adjustment.
* :func:`fx_variance_swap` — FX variance/vol swap pricing.
* :func:`fx_compound_option` — option on an FX option.
* :func:`fx_chooser_option` — call or put decided at future date.
* :func:`fx_local_vol` — Dupire local vol surface from implied vols.

References:
    Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017.
    Clark, *FX Option Pricing*, Wiley, 2011.
    Dupire, *Pricing with a Smile*, Risk, 1994.
    Rubinstein, *Options for the Undecided*, Risk, 1991.
    Geske, *The Valuation of Compound Options*, JFE, 1979.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.models.black76 import OptionType, _norm_cdf, _norm_pdf


# ═══════════════════════════════════════════════════════════════
# FX1: Digital (Binary) Options
# ═══════════════════════════════════════════════════════════════

@dataclass
class FXDigitalResult:
    """FX digital option result."""
    price: float            # in domestic currency
    delta: float
    vega: float
    payout: float
    barrier_shift: float    # overhedge shift applied

    def to_dict(self) -> dict:
        return vars(self)


def fx_digital_option(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
    payout: float = 1.0,
    option_type: str = "call",
    payout_currency: str = "domestic",
    overhedge_shift: float = 0.0,
) -> FXDigitalResult:
    """European digital (binary) FX option.

    Cash-or-nothing: pays fixed amount if S_T > K (call) or S_T < K (put).

    Domestic digital call: payout × N(d₂) × df_d
    Foreign digital call: payout × N(d₁) × df_f × S/S  (pays in foreign)

    Overhedge: shifts strike by half bid-ask to manage pin risk.

    Args:
        spot: FX spot rate.
        strike: digital barrier/strike.
        r_d: domestic risk-free rate.
        r_f: foreign risk-free rate.
        vol: implied vol.
        T: time to expiry (years).
        payout: fixed payout amount.
        option_type: "call" (pays if S>K) or "put" (pays if S<K).
        payout_currency: "domestic" or "foreign".
        overhedge_shift: shift strike for overhedge (in spot units).
    """
    if T <= 0 or vol <= 0:
        fwd = spot * math.exp((r_d - r_f) * T)
        if option_type == "call":
            return FXDigitalResult(payout if fwd > strike else 0, 0, 0, payout, 0)
        return FXDigitalResult(payout if fwd < strike else 0, 0, 0, payout, 0)

    K = strike + overhedge_shift if option_type == "call" else strike - overhedge_shift
    fwd = spot * math.exp((r_d - r_f) * T)
    df_d = math.exp(-r_d * T)
    df_f = math.exp(-r_f * T)
    sqrt_t = math.sqrt(T)

    d1 = (math.log(fwd / K) + 0.5 * vol**2 * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    if payout_currency == "foreign":
        # Asset-or-nothing in foreign currency
        if option_type == "call":
            price = payout * df_f * _norm_cdf(d1)
        else:
            price = payout * df_f * _norm_cdf(-d1)
    else:
        # Cash-or-nothing in domestic currency
        if option_type == "call":
            price = payout * df_d * _norm_cdf(d2)
        else:
            price = payout * df_d * _norm_cdf(-d2)

    # Greeks
    # Delta: d(price)/d(spot)
    dprice_dspot = payout * df_d * _norm_pdf(d2) / (spot * vol * sqrt_t)
    if option_type == "put":
        dprice_dspot = -dprice_dspot

    # Vega: d(price)/d(vol) per 1%
    vega = -payout * df_d * _norm_pdf(d2) * d1 / vol * 0.01
    if option_type == "put":
        vega = -vega

    return FXDigitalResult(
        price=price,
        delta=dprice_dspot,
        vega=vega,
        payout=payout,
        barrier_shift=overhedge_shift,
    )


# ═══════════════════════════════════════════════════════════════
# FX2: Quanto Options
# ═══════════════════════════════════════════════════════════════

@dataclass
class FXQuantoResult:
    """FX quanto option result."""
    price: float            # in quanto (domestic) currency
    price_no_quanto: float  # without quanto adjustment
    quanto_adjustment: float
    delta: float
    vega: float
    correlation: float

    def to_dict(self) -> dict:
        return vars(self)


def fx_quanto_option(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol_asset: float,
    vol_fx: float,
    correlation: float,
    T: float,
    option_type: str = "call",
    fx_rate: float = 1.0,
) -> FXQuantoResult:
    """FX quanto option: option on foreign asset paid in domestic currency.

    The quanto adjustment modifies the drift:
    r_quanto = r_f − ρ × σ_asset × σ_FX

    Price = FX_rate × BS(S, K, r_d, r_quanto, σ_asset, T)

    When ρ > 0 (asset and FX move together), the quanto drift is
    lower → call is cheaper, put is more expensive.

    Args:
        spot: foreign asset spot price (in foreign currency).
        strike: strike in foreign currency.
        r_d: domestic risk-free rate.
        r_f: foreign risk-free rate.
        vol_asset: asset volatility.
        vol_fx: FX vol (domestic per foreign).
        correlation: asset-FX correlation.
        T: time to expiry.
        fx_rate: current FX rate (domestic per foreign).
    """
    # Quanto-adjusted foreign rate
    r_quanto = r_f - correlation * vol_asset * vol_fx

    # Forward with quanto adjustment
    fwd_quanto = spot * math.exp((r_d - r_quanto) * T)
    fwd_no_quanto = spot * math.exp((r_d - r_f) * T)
    df_d = math.exp(-r_d * T)

    otype = OptionType.CALL if option_type == "call" else OptionType.PUT

    if T <= 0 or vol_asset <= 0:
        intrinsic = max(spot - strike, 0) if option_type == "call" else max(strike - spot, 0)
        return FXQuantoResult(fx_rate * df_d * intrinsic, fx_rate * df_d * intrinsic, 0, 0, 0, correlation)

    sqrt_t = math.sqrt(T)

    # Quanto price
    d1_q = (math.log(fwd_quanto / strike) + 0.5 * vol_asset**2 * T) / (vol_asset * sqrt_t)
    d2_q = d1_q - vol_asset * sqrt_t

    if option_type == "call":
        price_q = df_d * (fwd_quanto * _norm_cdf(d1_q) - strike * _norm_cdf(d2_q))
    else:
        price_q = df_d * (strike * _norm_cdf(-d2_q) - fwd_quanto * _norm_cdf(-d1_q))

    # Non-quanto price (for comparison)
    d1_nq = (math.log(fwd_no_quanto / strike) + 0.5 * vol_asset**2 * T) / (vol_asset * sqrt_t)
    d2_nq = d1_nq - vol_asset * sqrt_t

    if option_type == "call":
        price_nq = df_d * (fwd_no_quanto * _norm_cdf(d1_nq) - strike * _norm_cdf(d2_nq))
    else:
        price_nq = df_d * (strike * _norm_cdf(-d2_nq) - fwd_no_quanto * _norm_cdf(-d1_nq))

    price_q *= fx_rate
    price_nq *= fx_rate

    # Delta
    delta = fx_rate * df_d * (_norm_cdf(d1_q) if option_type == "call" else _norm_cdf(d1_q) - 1)

    # Vega per 1%
    vega = fx_rate * df_d * fwd_quanto * sqrt_t * _norm_pdf(d1_q) * 0.01

    return FXQuantoResult(
        price=price_q,
        price_no_quanto=price_nq,
        quanto_adjustment=price_q - price_nq,
        delta=delta,
        vega=vega,
        correlation=correlation,
    )


# ═══════════════════════════════════════════════════════════════
# FX3: FX Variance/Vol Swaps
# ═══════════════════════════════════════════════════════════════

@dataclass
class FXVarianceSwapResult:
    """FX variance swap result."""
    fair_variance: float        # annualised fair strike variance
    fair_vol: float             # √variance (in vol points)
    pv: float                   # MTM PV
    vega_notional: float
    realised_vol: float

    def to_dict(self) -> dict:
        return vars(self)


def fx_variance_swap(
    atm_vol: float,
    rr25: float = 0.0,
    bf25: float = 0.0,
    T: float = 1.0,
    r_d: float = 0.04,
    vega_notional: float = 100_000.0,
    realised_vol: float | None = None,
) -> FXVarianceSwapResult:
    """FX variance swap fair strike from smile.

    Fair variance ≈ ATM² + (convexity adjustment from wings).

    The convexity adjustment from the smile:
    σ²_var ≈ σ²_ATM + 2 × BF × σ_ATM + higher order

    Butterfly captures the smile's impact on variance.

    Args:
        atm_vol: ATM implied vol.
        rr25: 25-delta risk reversal (vol_call − vol_put).
        bf25: 25-delta butterfly (avg wing − ATM).
        T: time to expiry.
        r_d: domestic rate.
        vega_notional: notional for P&L.
        realised_vol: if given, compute MTM.
    """
    # Fair variance with convexity from smile
    # σ²_fair ≈ σ²_ATM + bf25 × 2 × σ_ATM (convexity from wings)
    fair_var = atm_vol**2 + 2 * bf25 * atm_vol
    fair_var = max(fair_var, 0)
    fair_vol = math.sqrt(fair_var)

    # MTM
    if realised_vol is not None:
        realised_var = realised_vol**2
        pv = vega_notional / (2 * fair_vol) * (realised_var - fair_var) if fair_vol > 0 else 0
    else:
        pv = 0.0

    return FXVarianceSwapResult(
        fair_variance=fair_var,
        fair_vol=fair_vol * 100,  # as percentage
        pv=pv,
        vega_notional=vega_notional,
        realised_vol=(realised_vol or 0) * 100,
    )


# ═══════════════════════════════════════════════════════════════
# FX4: Dupire Local Vol
# ═══════════════════════════════════════════════════════════════

@dataclass
class LocalVolSurface:
    """Dupire local volatility surface."""
    strikes: np.ndarray
    expiries: np.ndarray
    local_vols: np.ndarray      # 2D: [expiry_idx, strike_idx]

    def vol(self, T: float, K: float) -> float:
        """Interpolate local vol at (T, K)."""
        # Find bracketing indices
        ei = max(0, min(np.searchsorted(self.expiries, T) - 1, len(self.expiries) - 2))
        si = max(0, min(np.searchsorted(self.strikes, K) - 1, len(self.strikes) - 2))

        e0, e1 = self.expiries[ei], self.expiries[min(ei + 1, len(self.expiries) - 1)]
        s0, s1 = self.strikes[si], self.strikes[min(si + 1, len(self.strikes) - 1)]

        we = (T - e0) / (e1 - e0) if e1 > e0 else 0.0
        ws = (K - s0) / (s1 - s0) if s1 > s0 else 0.0

        v00 = self.local_vols[ei, si]
        v01 = self.local_vols[ei, min(si + 1, self.local_vols.shape[1] - 1)]
        v10 = self.local_vols[min(ei + 1, self.local_vols.shape[0] - 1), si]
        v11 = self.local_vols[min(ei + 1, self.local_vols.shape[0] - 1),
                               min(si + 1, self.local_vols.shape[1] - 1)]

        return (1 - we) * ((1 - ws) * v00 + ws * v01) + we * ((1 - ws) * v10 + ws * v11)

    def to_dict(self) -> dict:
        return {
            "n_expiries": len(self.expiries),
            "n_strikes": len(self.strikes),
            "expiry_range": [float(self.expiries[0]), float(self.expiries[-1])],
        }


def fx_local_vol(
    spot: float,
    r_d: float,
    r_f: float,
    strikes: list[float],
    expiries: list[float],
    implied_vols: list[list[float]],
) -> LocalVolSurface:
    """Dupire local volatility from implied vol surface.

    σ_LV²(K,T) = (∂C/∂T + (r_d − r_f)K ∂C/∂K + ½(r_d − r_f)²K²...)
                  / (½K² ∂²C/∂K²)

    Simplified numerical Dupire using finite differences on
    the call price surface.

    Args:
        spot: FX spot.
        r_d: domestic rate.
        r_f: foreign rate.
        strikes: strike grid.
        expiries: expiry grid (years).
        implied_vols: 2D implied vol grid [expiry][strike].
    """
    K_arr = np.array(strikes)
    T_arr = np.array(expiries)
    n_T = len(T_arr)
    n_K = len(K_arr)

    # Build call price surface
    C = np.zeros((n_T, n_K))
    for i, T in enumerate(T_arr):
        fwd = spot * math.exp((r_d - r_f) * T)
        df = math.exp(-r_d * T)
        for j, K in enumerate(K_arr):
            sigma = implied_vols[i][j]
            if T > 0 and sigma > 0 and K > 0 and fwd > 0:
                sqrt_t = math.sqrt(T)
                d1 = (math.log(fwd / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_t)
                d2 = d1 - sigma * sqrt_t
                C[i, j] = df * (fwd * _norm_cdf(d1) - K * _norm_cdf(d2))
            else:
                C[i, j] = df * max(fwd - K, 0)

    # Compute local vol via Dupire
    local_vols = np.zeros((n_T, n_K))
    for i in range(n_T):
        for j in range(n_K):
            K = K_arr[j]
            T = T_arr[i]

            # dC/dT
            if i == 0:
                dCdT = (C[1, j] - C[0, j]) / (T_arr[1] - T_arr[0]) if n_T > 1 else 0
            elif i == n_T - 1:
                dCdT = (C[i, j] - C[i - 1, j]) / (T_arr[i] - T_arr[i - 1])
            else:
                dCdT = (C[i + 1, j] - C[i - 1, j]) / (T_arr[i + 1] - T_arr[i - 1])

            # dC/dK
            if j == 0:
                dCdK = (C[i, 1] - C[i, 0]) / (K_arr[1] - K_arr[0]) if n_K > 1 else 0
            elif j == n_K - 1:
                dCdK = (C[i, j] - C[i, j - 1]) / (K_arr[j] - K_arr[j - 1])
            else:
                dCdK = (C[i, j + 1] - C[i, j - 1]) / (K_arr[j + 1] - K_arr[j - 1])

            # d²C/dK²
            if 0 < j < n_K - 1:
                dK = (K_arr[j + 1] - K_arr[j - 1]) / 2
                d2CdK2 = (C[i, j + 1] - 2 * C[i, j] + C[i, j - 1]) / (dK**2)
            else:
                d2CdK2 = 0

            # Dupire formula
            numerator = dCdT + (r_d - r_f) * K * dCdK
            denominator = 0.5 * K**2 * d2CdK2

            if denominator > 1e-15:
                lv2 = numerator / denominator
                local_vols[i, j] = math.sqrt(max(lv2, 0))
            else:
                local_vols[i, j] = implied_vols[i][j]

    return LocalVolSurface(K_arr, T_arr, local_vols)


# ═══════════════════════════════════════════════════════════════
# FX5: Double-Barrier Options (Analytical)
# ═══════════════════════════════════════════════════════════════

@dataclass
class FXDoubleBarrierResult:
    """FX double-barrier option result."""
    price: float
    vanilla_price: float
    barrier_discount: float     # vanilla − barrier price
    lower_barrier: float
    upper_barrier: float
    knock_type: str             # "out" or "in"

    def to_dict(self) -> dict:
        return vars(self)


def fx_double_barrier_option(
    spot: float,
    strike: float,
    lower_barrier: float,
    upper_barrier: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
    option_type: str = "call",
    knock_type: str = "out",
    n_sims: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> FXDoubleBarrierResult:
    """Double knock-out/knock-in FX option via Monte Carlo.

    Simulates GBM paths and checks barrier breaches at each step.
    knock-in + knock-out = vanilla (parity).

    Args:
        lower_barrier: lower barrier (L < S < U).
        upper_barrier: upper barrier.
        knock_type: "out" or "in".
        n_sims: Monte Carlo paths.
        n_steps: time steps per path.
    """
    from pricebook.fx.fx_option import fx_option_price

    otype = OptionType.CALL if option_type == "call" else OptionType.PUT
    vanilla = fx_option_price(spot, strike, r_d, r_f, vol, T, otype)

    if spot <= lower_barrier or spot >= upper_barrier:
        if knock_type == "out":
            return FXDoubleBarrierResult(0, vanilla, vanilla, lower_barrier, upper_barrier, knock_type)
        return FXDoubleBarrierResult(vanilla, vanilla, 0, lower_barrier, upper_barrier, knock_type)

    if T <= 0 or vol <= 0:
        return FXDoubleBarrierResult(vanilla, vanilla, 0, lower_barrier, upper_barrier, knock_type)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r_d - r_f - 0.5 * vol**2) * dt
    diffusion = vol * math.sqrt(dt)
    df_d = math.exp(-r_d * T)

    # Vectorised MC
    Z = rng.standard_normal((n_sims, n_steps))
    log_S = np.full(n_sims, math.log(spot))
    alive = np.ones(n_sims, dtype=bool)  # not knocked out

    for t in range(n_steps):
        log_S += drift + diffusion * Z[:, t]
        S_t = np.exp(log_S)
        hit = (S_t <= lower_barrier) | (S_t >= upper_barrier)
        alive &= ~hit

    # Terminal payoff
    S_T = np.exp(log_S)
    if option_type == "call":
        payoff = np.maximum(S_T - strike, 0)
    else:
        payoff = np.maximum(strike - S_T, 0)

    # Knock-out: pay only if alive
    ko_payoffs = payoff * alive
    price_ko = float(np.mean(ko_payoffs)) * df_d

    if knock_type == "in":
        price = vanilla - price_ko
    else:
        price = price_ko

    return FXDoubleBarrierResult(
        price=price,
        vanilla_price=vanilla,
        barrier_discount=vanilla - price,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        knock_type=knock_type,
    )


# ═══════════════════════════════════════════════════════════════
# FX6: Compound Options (option on option)
# ═══════════════════════════════════════════════════════════════

@dataclass
class FXCompoundResult:
    """FX compound option result."""
    price: float
    underlying_option_price: float
    compound_type: str          # "call_on_call", etc.
    inner_strike: float
    outer_strike: float

    def to_dict(self) -> dict:
        return vars(self)


def fx_compound_option(
    spot: float,
    inner_strike: float,
    outer_strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T_outer: float,
    T_inner: float,
    outer_type: str = "call",
    inner_type: str = "call",
    n_sims: int = 50_000,
    seed: int = 42,
) -> FXCompoundResult:
    """Compound FX option: option on an FX option.

    At T_outer, holder exercises if underlying option value > outer_strike.
    The underlying option expires at T_inner (> T_outer).

    call-on-call: right to buy a call at premium = outer_strike.
    put-on-call: right to sell a call.

    Args:
        inner_strike: strike of the underlying option.
        outer_strike: premium paid/received at T_outer to enter underlying.
        T_outer: compound option expiry.
        T_inner: underlying option expiry (T_inner > T_outer).
        outer_type: "call" or "put" on the underlying option.
        inner_type: "call" or "put" for the underlying option.
    """
    from pricebook.fx.fx_option import fx_option_price

    inner_otype = OptionType.CALL if inner_type == "call" else OptionType.PUT

    rng = np.random.default_rng(seed)
    df_outer = math.exp(-r_d * T_outer)

    # Simulate spot at T_outer
    fwd = spot * math.exp((r_d - r_f) * T_outer)
    sqrt_t = math.sqrt(T_outer)

    total_pv = 0.0
    for _ in range(n_sims):
        z = rng.standard_normal()
        S_outer = fwd * math.exp(-0.5 * vol**2 * T_outer + vol * sqrt_t * z)

        # Value of underlying option at T_outer
        remaining = T_inner - T_outer
        inner_value = fx_option_price(S_outer, inner_strike, r_d, r_f, vol, remaining, inner_otype)

        # Compound payoff
        if outer_type == "call":
            payoff = max(inner_value - outer_strike, 0)
        else:
            payoff = max(outer_strike - inner_value, 0)

        total_pv += payoff

    price = total_pv / n_sims * df_outer

    # Current underlying option price
    underlying = fx_option_price(spot, inner_strike, r_d, r_f, vol, T_inner, inner_otype)

    return FXCompoundResult(
        price=price,
        underlying_option_price=underlying,
        compound_type=f"{outer_type}_on_{inner_type}",
        inner_strike=inner_strike,
        outer_strike=outer_strike,
    )


# ═══════════════════════════════════════════════════════════════
# FX7: Chooser Options
# ═══════════════════════════════════════════════════════════════

@dataclass
class FXChooserResult:
    """FX chooser option result."""
    price: float
    call_value: float           # value of call component
    put_value: float            # value of put component
    choose_date_years: float
    prob_choose_call: float

    def to_dict(self) -> dict:
        return vars(self)


def fx_chooser_option(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T_choose: float,
    T_expiry: float,
    n_sims: int = 50_000,
    seed: int = 42,
) -> FXChooserResult:
    """FX chooser option: at T_choose, holder picks call or put.

    At T_choose: value = max(call(S, K, T_expiry − T_choose),
                              put(S, K, T_expiry − T_choose))

    For simple chooser (same strike, same expiry), Rubinstein (1991)
    gives closed form. Here we use MC for generality.

    Args:
        T_choose: date at which call/put choice is made.
        T_expiry: option expiry date (T_expiry > T_choose).
    """
    from pricebook.fx.fx_option import fx_option_price

    rng = np.random.default_rng(seed)
    df_choose = math.exp(-r_d * T_choose)
    fwd = spot * math.exp((r_d - r_f) * T_choose)
    sqrt_t = math.sqrt(T_choose)

    total_pv = 0.0
    n_call = 0

    for _ in range(n_sims):
        z = rng.standard_normal()
        S_choose = fwd * math.exp(-0.5 * vol**2 * T_choose + vol * sqrt_t * z)

        remaining = T_expiry - T_choose
        call_val = fx_option_price(S_choose, strike, r_d, r_f, vol, remaining, OptionType.CALL)
        put_val = fx_option_price(S_choose, strike, r_d, r_f, vol, remaining, OptionType.PUT)

        if call_val >= put_val:
            total_pv += call_val
            n_call += 1
        else:
            total_pv += put_val

    price = total_pv / n_sims * df_choose

    # Current call and put values
    call_now = fx_option_price(spot, strike, r_d, r_f, vol, T_expiry, OptionType.CALL)
    put_now = fx_option_price(spot, strike, r_d, r_f, vol, T_expiry, OptionType.PUT)

    return FXChooserResult(
        price=price,
        call_value=call_now,
        put_value=put_now,
        choose_date_years=T_choose,
        prob_choose_call=n_call / n_sims,
    )
