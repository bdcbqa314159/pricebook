"""Extended equity exotics and dividend derivatives.

Products commonly traded on equity desks:
    forward-start, chooser, quanto, Himalaya, outperformance, accumulator,
    dividend futures, dividend swaps, dividend options.

    from pricebook.equity.equity_exotic_extended import (
        forward_start_option, chooser_option, quanto_equity_option,
        himalaya_option, outperformance_option, equity_accumulator,
        dividend_future, dividend_swap, dividend_option,
    )

References:
    Rubinstein (1991). Pay Now, Choose Later. Risk.
    Zhang (1998). Exotic Options, Ch. 15-18.
    Bouzoubaa & Osseiran (2010). Exotic Options and Hybrids, Ch. 7-9.
    Wystup (2006). FX Options and Structured Products, Ch. 3.
    Bos et al. (2003). Valuation of Dividend Derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.models.black76 import black76_price, OptionType


# ---------------------------------------------------------------------------
# 1. Forward-start option
# ---------------------------------------------------------------------------

@dataclass
class ForwardStartResult:
    """Forward-start option result."""
    price: float
    forward_vol: float
    moneyness: float
    start_date_frac: float

    def to_dict(self) -> dict:
        return {"price": self.price, "forward_vol": self.forward_vol,
                "moneyness": self.moneyness, "start_date_frac": self.start_date_frac}


def forward_start_option(
    spot: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T_start: float,
    T_end: float,
    moneyness: float = 1.0,
    is_call: bool = True,
    notional: float = 1.0,
) -> ForwardStartResult:
    """Forward-start option: strike set at T_start as moneyness x S(T_start).

    At T_start, strike K = moneyness x S(T_start).
    Payoff at T_end: max(S(T_end) - K, 0) for call.

    Under GBM, the price equals a standard option on S with
    adjusted forward and time-to-expiry = T_end - T_start.

    Key property: delta at inception = 0 (no directional exposure
    until strike is set). This makes it the building block for cliquets.

    Args:
        T_start: time when strike is set (years).
        T_end: expiry (years).
        moneyness: K/S ratio at T_start (1.0 = ATM forward-start).
    """
    if T_end <= T_start:
        raise ValueError(f"T_end ({T_end}) must be after T_start ({T_start})")

    tau = T_end - T_start
    F = spot * math.exp((rate - dividend_yield) * tau)

    # Under GBM: forward-start = e^{-q*T_start} x BS(S, moneyness*S, vol, tau)
    scale = math.exp(-dividend_yield * T_start)
    opt = OptionType.CALL if is_call else OptionType.PUT
    unit_price = black76_price(F / spot, moneyness, vol, tau, math.exp(-rate * tau), opt)
    price = spot * scale * unit_price * notional

    return ForwardStartResult(
        price=float(price), forward_vol=vol,
        moneyness=moneyness, start_date_frac=T_start,
    )


# ---------------------------------------------------------------------------
# 2. Chooser option
# ---------------------------------------------------------------------------

@dataclass
class ChooserResult:
    """Chooser option result."""
    price: float
    call_component: float
    put_component: float
    choice_date_frac: float

    def to_dict(self) -> dict:
        return {"price": self.price, "call_component": self.call_component,
                "put_component": self.put_component, "choice_date": self.choice_date_frac}


def chooser_option(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T_choose: float,
    T_expiry: float,
    notional: float = 1.0,
) -> ChooserResult:
    """Simple chooser: at T_choose, holder picks call or put (same strike, same expiry).

    Rubinstein (1991): chooser = call(K, T) + put(K_adj, T_choose)
    where K_adj = K x exp(-(r-q)(T-T_choose)).

    Decomposition: the chooser is always worth at least as much as a
    call, because the holder can always choose call. The extra value
    comes from the put component (insurance against the wrong choice).

    Args:
        T_choose: time when holder chooses call or put.
        T_expiry: option expiry.
    """
    if T_expiry <= T_choose:
        raise ValueError(f"T_expiry must be after T_choose")

    tau = T_expiry - T_choose
    F = spot * math.exp((rate - dividend_yield) * T_expiry)
    df = math.exp(-rate * T_expiry)

    # Full call value
    call_price = black76_price(F, strike, vol, T_expiry, df, OptionType.CALL)

    # Put component: put with adjusted strike at T_choose
    K_adj = strike * math.exp(-(rate - dividend_yield) * tau)
    F_choose = spot * math.exp((rate - dividend_yield) * T_choose)
    df_choose = math.exp(-rate * T_choose)
    put_component = black76_price(F_choose, K_adj, vol, T_choose, df_choose, OptionType.PUT)

    price = (call_price + put_component) * notional

    return ChooserResult(
        price=float(price),
        call_component=float(call_price * notional),
        put_component=float(put_component * notional),
        choice_date_frac=T_choose,
    )


# ---------------------------------------------------------------------------
# 3. Quanto equity option
# ---------------------------------------------------------------------------

@dataclass
class QuantoEquityResult:
    """Quanto equity option result."""
    price: float
    quanto_forward: float
    quanto_adjustment: float
    correlation: float

    def to_dict(self) -> dict:
        return {"price": self.price, "quanto_forward": self.quanto_forward,
                "quanto_adjustment": self.quanto_adjustment,
                "correlation": self.correlation}


def quanto_equity_option(
    spot: float,
    strike: float,
    rate_domestic: float,
    rate_foreign: float,
    dividend_yield: float,
    vol_equity: float,
    vol_fx: float,
    correlation: float,
    T: float,
    is_call: bool = True,
    notional: float = 1.0,
) -> QuantoEquityResult:
    """Quanto equity option: equity payoff converted at fixed FX rate.

    Payoff in domestic currency = max(S_T - K, 0) x fixed_fx_rate.
    The FX rate is fixed at inception — no FX risk for the holder.

    Quanto adjustment (Reiner 1992):
        F_quanto = F_equity x exp(-rho x sigma_eq x sigma_fx x T)

    Negative correlation (equity up → FX down) reduces the quanto forward.

    Args:
        rate_domestic: domestic risk-free rate (payoff currency).
        rate_foreign: foreign rate (equity currency).
        vol_fx: FX volatility.
        correlation: correlation between equity and FX returns.
    """
    # Quanto drift adjustment
    quanto_adj = math.exp(-correlation * vol_equity * vol_fx * T)
    F_quanto = spot * math.exp((rate_foreign - dividend_yield) * T) * quanto_adj

    df = math.exp(-rate_domestic * T)
    opt = OptionType.CALL if is_call else OptionType.PUT
    price = black76_price(F_quanto, strike, vol_equity, T, df, opt) * notional

    return QuantoEquityResult(
        price=float(price),
        quanto_forward=float(F_quanto),
        quanto_adjustment=float(quanto_adj - 1.0),
        correlation=correlation,
    )


# ---------------------------------------------------------------------------
# 4. Himalaya option (mountain range)
# ---------------------------------------------------------------------------

@dataclass
class HimalayaResult:
    """Himalaya option result."""
    price: float
    std_error: float
    n_paths: int
    n_assets: int

    def to_dict(self) -> dict:
        return {"price": self.price, "std_error": self.std_error,
                "n_paths": self.n_paths, "n_assets": self.n_assets}


def himalaya_option(
    spots: list[float],
    rate: float,
    dividend_yields: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    n_periods: int | None = None,
    is_call: bool = True,
    notional: float = 1.0,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> HimalayaResult:
    """Himalaya option: at each period, remove the best performer and lock in its return.

    Payoff = average of locked-in returns (one per period).

    Period 1: best of N assets removed, return locked.
    Period 2: best of remaining N-1 removed, return locked.
    ...
    Period N: last asset's return locked.
    Final payoff = max(average of all locked returns, 0).

    This is attractive because diversification across assets AND time.
    Cheaper than best-of because good performers are removed early.

    Args:
        n_periods: number of observation periods (default = n_assets).
    """
    n = len(spots)
    if n_periods is None:
        n_periods = n
    n_periods = min(n_periods, n)

    from pricebook.models.mc_migrate import correlated_gbm_paths

    mus = [rate - q for q in dividend_yields]
    paths = correlated_gbm_paths(spots, mus, vols, correlations, T, n_periods, n_paths, seed or 42)
    # paths: (n_paths, n_periods+1, n_assets)

    df = math.exp(-rate * T)
    spots_arr = np.array(spots)

    locked_returns = np.zeros(n_paths)
    available = np.ones((n_paths, n), dtype=bool)

    for period in range(1, n_periods + 1):
        # Returns for this period: S(t)/S(0) - 1
        returns = paths[:, period, :] / spots_arr - 1.0

        # Mask unavailable assets with -inf
        masked = np.where(available, returns, -np.inf)

        # Best performer index per path
        best_idx = np.argmax(masked, axis=1)

        # Lock in the best return
        best_return = returns[np.arange(n_paths), best_idx]
        locked_returns += best_return

        # Remove best from available
        available[np.arange(n_paths), best_idx] = False

    avg_return = locked_returns / n_periods
    if is_call:
        payoff = np.maximum(avg_return, 0.0)
    else:
        payoff = np.maximum(-avg_return, 0.0)

    discounted = df * payoff * notional
    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / math.sqrt(n_paths))

    return HimalayaResult(price=price, std_error=stderr, n_paths=n_paths, n_assets=n)


# ---------------------------------------------------------------------------
# 5. Outperformance option
# ---------------------------------------------------------------------------

@dataclass
class OutperformanceResult:
    """Outperformance option result."""
    price: float
    expected_outperformance: float

    def to_dict(self) -> dict:
        return {"price": self.price, "expected_outperformance": self.expected_outperformance}


def outperformance_option(
    spot1: float,
    spot2: float,
    rate: float,
    div1: float,
    div2: float,
    vol1: float,
    vol2: float,
    correlation: float,
    T: float,
    notional: float = 1.0,
    is_call: bool = True,
) -> OutperformanceResult:
    """Outperformance option: payoff on relative performance S1/S2.

    Call payoff = notional x max(S1(T)/S1(0) - S2(T)/S2(0), 0).

    Reduces to a Margrabe exchange option on normalised assets.
    Vol of spread: sigma_spread = sqrt(v1^2 + v2^2 - 2*rho*v1*v2).

    Args:
        spot1, spot2: initial prices of the two assets.
        div1, div2: dividend yields.
        vol1, vol2: volatilities.
        correlation: return correlation between the two assets.
    """
    # Spread vol
    sigma = math.sqrt(max(vol1**2 + vol2**2 - 2 * correlation * vol1 * vol2, 0.0))

    # Margrabe: option to exchange asset2 for asset1
    F1 = math.exp((rate - div1) * T)  # normalised forward (S1/S1_0)
    F2 = math.exp((rate - div2) * T)  # normalised forward (S2/S2_0)

    if sigma * math.sqrt(T) < 1e-10:
        if is_call:
            price = max(F1 - F2, 0.0) * math.exp(-rate * T) * notional
        else:
            price = max(F2 - F1, 0.0) * math.exp(-rate * T) * notional
        return OutperformanceResult(float(price), float(F1 - F2))

    d1 = (math.log(F1 / F2) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    df = math.exp(-rate * T)
    if is_call:
        price = df * (F1 * norm.cdf(d1) - F2 * norm.cdf(d2)) * notional
    else:
        price = df * (F2 * norm.cdf(-d2) - F1 * norm.cdf(-d1)) * notional

    return OutperformanceResult(
        price=float(price),
        expected_outperformance=float(F1 - F2),
    )


# ---------------------------------------------------------------------------
# 6. Equity accumulator
# ---------------------------------------------------------------------------

@dataclass
class EquityAccumulatorResult:
    """Equity accumulator result."""
    price: float
    knockout_probability: float
    expected_accumulation: float
    strike: float
    barrier: float

    def to_dict(self) -> dict:
        return {"price": self.price, "ko_prob": self.knockout_probability,
                "expected_accumulation": self.expected_accumulation,
                "strike": self.strike, "barrier": self.barrier}


def equity_accumulator(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    daily_quantity: float = 1.0,
    leverage: float = 2.0,
    n_paths: int = 20_000,
    n_steps: int = 252,
    seed: int | None = 42,
) -> EquityAccumulatorResult:
    """Equity accumulator (KODA): buy shares daily at discount, knockout above barrier.

    Each trading day:
    - If S(t) >= barrier: contract terminates (knockout).
    - If S(t) >= strike: buy daily_quantity shares at strike (profit).
    - If S(t) < strike: buy daily_quantity x leverage shares at strike (loss, leveraged).

    Popular retail structured product. Seller profits from premium
    and time value; buyer profits from discount to spot.

    Args:
        strike: accumulation strike (typically 85-95% of spot).
        barrier: knockout level (typically 103-110% of spot).
        daily_quantity: shares accumulated per day above strike.
        leverage: multiplier for shares below strike (typically 2x).
    """
    from pricebook.models.mc_migrate import gbm_paths

    paths = gbm_paths(spot, rate - dividend_yield, vol, T, n_steps, n_paths, seed or 42)
    dt = T / n_steps

    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    accumulated = np.zeros(n_paths)

    for step in range(1, n_steps + 1):
        S = paths[:, step]
        t = step * dt
        df_t = math.exp(-rate * t)

        # Knockout check
        ko = alive & (S >= barrier)
        alive &= ~ko

        # Accumulation for alive paths
        above = alive & (S >= strike)
        below = alive & (S < strike)

        # P&L: (S - strike) x quantity
        qty_above = daily_quantity
        qty_below = daily_quantity * leverage

        pv += np.where(above, (S - strike) * qty_above * df_t, 0.0)
        pv += np.where(below, (S - strike) * qty_below * df_t, 0.0)  # negative
        accumulated += np.where(above, qty_above, 0.0)
        accumulated += np.where(below, qty_below, 0.0)

    ko_prob = float(1 - alive.mean())
    price = float(pv.mean())
    expected_acc = float(accumulated.mean())

    return EquityAccumulatorResult(
        price=price, knockout_probability=ko_prob,
        expected_accumulation=expected_acc,
        strike=strike, barrier=barrier,
    )


# ---------------------------------------------------------------------------
# Dividend future
# ---------------------------------------------------------------------------

@dataclass
class DividendFutureResult:
    """Dividend future pricing result."""
    price: float
    implied_dividend: float
    forward_price: float
    present_value: float

    def to_dict(self) -> dict:
        return vars(self)


def dividend_future(
    spot: float,
    rate: float,
    dividend_yield: float,
    T: float,
    notional: float = 1.0,
) -> DividendFutureResult:
    """Dividend future: forward contract on expected dividends over [0, T].

    The fair price is the PV of expected dividends:
        D = S × (1 - e^{-q×T})

    where q is the continuous dividend yield.

    Args:
        spot: current stock/index price.
        dividend_yield: continuous dividend yield (e.g. 0.02 for 2%).
        T: time to maturity in years.
    """
    expected_div = spot * (1 - math.exp(-dividend_yield * T))
    df = math.exp(-rate * T)
    fwd = expected_div  # no-arbitrage: futures ≈ expected dividends
    pv = df * expected_div * notional

    return DividendFutureResult(
        price=float(fwd * notional),
        implied_dividend=float(expected_div),
        forward_price=float(fwd),
        present_value=float(pv),
    )


# ---------------------------------------------------------------------------
# Dividend swap
# ---------------------------------------------------------------------------

@dataclass
class DividendSwapResult:
    """Dividend swap pricing result."""
    pv: float
    fixed_rate: float
    implied_dividend: float
    notional: float

    def to_dict(self) -> dict:
        return vars(self)


def dividend_swap(
    spot: float,
    rate: float,
    dividend_yield: float,
    fixed_dividend: float,
    T: float,
    notional: float = 1_000_000,
    n_periods: int | None = None,
) -> DividendSwapResult:
    """Dividend swap: float realised dividends vs fixed.

    Receiver pays fixed_dividend per period,
    receives implied_dividend per period.

    PV = Σ df(t_i) × notional × (implied_div_i - fixed_div_i).

    Args:
        fixed_dividend: fixed annual dividend (in index points, e.g. 85).
        dividend_yield: implied continuous dividend yield.
    """
    if n_periods is None:
        n_periods = max(int(T), 1)
    dt = T / n_periods

    implied_annual = spot * dividend_yield

    pv = 0.0
    for i in range(n_periods):
        t = (i + 1) * dt
        df = math.exp(-rate * t)
        pv += df * notional * (implied_annual - fixed_dividend) * dt

    return DividendSwapResult(
        pv=float(pv),
        fixed_rate=fixed_dividend,
        implied_dividend=float(implied_annual),
        notional=notional,
    )


# ---------------------------------------------------------------------------
# Dividend option
# ---------------------------------------------------------------------------

@dataclass
class DividendOptionResult:
    """Dividend option pricing result."""
    price: float
    implied_dividend: float
    strike: float
    is_call: bool

    def to_dict(self) -> dict:
        return vars(self)


def dividend_option(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    div_vol: float,
    T: float,
    is_call: bool = True,
    notional: float = 1_000_000,
) -> DividendOptionResult:
    """Option on realised dividends via Black-76.

    Underlying = expected cumulative dividend over [0, T].
    Strike = fixed dividend level.

    Args:
        strike: dividend strike (in same units as spot, e.g. index points).
        div_vol: volatility of the dividend index.
    """
    from pricebook.models.black76 import black76_price as _b76, OptionType

    if dividend_yield <= 0 or T <= 0:
        return DividendOptionResult(0.0, 0.0, strike, is_call)

    implied_div = spot * (1 - math.exp(-dividend_yield * T))
    if implied_div <= 0 or strike <= 0:
        return DividendOptionResult(0.0, float(implied_div), strike, is_call)

    df = math.exp(-rate * T)
    opt_type = OptionType.CALL if is_call else OptionType.PUT
    unit_price = _b76(implied_div, strike, div_vol, T, df, opt_type)

    return DividendOptionResult(
        price=float(unit_price * notional),
        implied_dividend=float(implied_div),
        strike=float(strike),
        is_call=is_call,
    )
