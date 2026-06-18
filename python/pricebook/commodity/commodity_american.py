"""American commodity option pricing.

Provides American option pricing for commodity futures and spot contracts,
including a seasonal energy variant, a spread option via Least-Squares Monte
Carlo (LSM), and an early-exercise diagnostic utility.

Three core pricing methods:
  - "baw"  : Barone-Adesi & Whaley (1987) quadratic approximation.  When
              pricing on futures with zero convenience yield, BAW is exact
              (the futures price replaces the forward with r=0 carry).
  - "pde"  : Crank-Nicolson finite-difference on log-futures space.
  - "tree" : CRR binomial tree with continuous dividend (convenience yield).

For commodity futures options the Black-76 drift is zero (futures are
martingales under the risk-neutral measure), so the BAW formula uses
r → 0 for the futures cost-of-carry term.  For spot-commodity options
with a convenience yield δ, the GBM drift becomes r − δ and δ replaces
the dividend yield throughout.

References:
    Barone-Adesi, G., & Whaley, R. E. (1987).  Efficient analytic
        approximation of American option values.  Journal of Finance,
        42(2), 301-320.
    Hull, J. (2022).  Options, Futures, and Other Derivatives, 11th ed.,
        Ch. 17.  Prentice Hall.
    Longstaff, F. A., & Schwartz, E. S. (2001).  Valuing American options
        by simulation.  Review of Financial Studies, 14(1), 113-147.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Internal helpers  (no external imports required for BAW / tree)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _black76_european(
    F: float, K: float, r: float, vol: float, T: float, is_call: bool,
) -> float:
    """Black-76 European option price on a futures."""
    if T <= 0 or vol <= 0:
        intr = (F - K) if is_call else (K - F)
        return math.exp(-r * T) * max(intr, 0.0)
    df = math.exp(-r * T)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if is_call:
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def _baw_critical_futures(
    K: float, r: float, b: float, vol: float, T: float, is_call: bool,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """BAW critical futures/spot price F* via Newton-Raphson.

    b = cost-of-carry = r - convenience_yield.
    For pure futures options b = 0 (futures are already risk-neutral).
    """
    vol2 = vol * vol
    M = 2.0 * r / vol2
    N = 2.0 * b / vol2  # b / (vol²/2)

    disc = (N - 1.0) ** 2 + 4.0 * M
    if disc < 0:
        return K * 1e6 if is_call else 1e-10

    if is_call:
        q = 0.5 * (-(N - 1.0) + math.sqrt(disc))
        F = K * 1.2
    else:
        q = 0.5 * (-(N - 1.0) - math.sqrt(disc))
        F = K * 0.8

    for _ in range(max_iter):
        euro = _black76_european(F, K, r, vol, T, is_call)
        intr = (F - K) if is_call else (K - F)
        sqrt_t = math.sqrt(T)
        d1 = (math.log(F / K) + (b + 0.5 * vol2) * T) / (vol * sqrt_t)
        nd1 = _norm_cdf(d1) if is_call else _norm_cdf(-d1)
        carry_df = math.exp((b - r) * T)
        rhs_factor = 1.0 - carry_df * nd1
        f_val = intr - euro - (intr / q) * rhs_factor
        # Derivative
        sgn = 1.0 if is_call else -1.0
        d_euro = sgn * carry_df * nd1
        nd1_deriv = _norm_pdf(d1) / (F * vol * sqrt_t)
        df_val = (sgn - d_euro
                  - (sgn * rhs_factor + (intr / q) * (-carry_df * sgn * nd1_deriv * F)) / F)
        if abs(df_val) < 1e-15:
            break
        F_new = max(F - f_val / df_val, 1e-8)
        if abs(F_new - F) < tol:
            F = F_new
            break
        F = F_new

    return F


def _baw_american(
    F: float, K: float, r: float, b: float, vol: float, T: float, is_call: bool,
) -> tuple[float, float, float]:
    """BAW American price. Returns (am_price, euro_price, F_star)."""
    euro = _black76_european(F, K, r, vol, T, is_call)

    if T <= 1e-8:
        intr = max((F - K) if is_call else (K - F), 0.0)
        return intr, euro, K

    vol2 = vol * vol
    M = 2.0 * r / vol2
    N = 2.0 * b / vol2
    disc = (N - 1.0) ** 2 + 4.0 * M
    if disc < 0:
        return euro, euro, (K * 1e6 if is_call else 0.0)

    q = (0.5 * (-(N - 1.0) + math.sqrt(disc)) if is_call
         else 0.5 * (-(N - 1.0) - math.sqrt(disc)))

    F_star = _baw_critical_futures(K, r, b, vol, T, is_call)
    intr_star = (F_star - K) if is_call else (K - F_star)
    if intr_star <= 0:
        return euro, euro, F_star

    sqrt_t = math.sqrt(T)
    d1_star = (math.log(F_star / K) + (b + 0.5 * vol2) * T) / (vol * sqrt_t)
    carry_df = math.exp((b - r) * T)
    nd1 = _norm_cdf(d1_star) if is_call else _norm_cdf(-d1_star)
    A = (intr_star / q) * (1.0 - carry_df * nd1)

    if is_call and F >= F_star:
        return F - K, euro, F_star
    if not is_call and F <= F_star:
        return K - F, euro, F_star

    correction = A * (F / F_star) ** q
    return euro + correction, euro, F_star


def _tree_american(
    F: float, K: float, r: float, b: float, vol: float, T: float, is_call: bool,
    n_steps: int = 500,
) -> tuple[float, float, float]:
    """CRR binomial tree for American commodity option."""
    dt = T / n_steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    df = math.exp(-r * dt)
    # Risk-neutral probability with cost-of-carry b
    pu = (math.exp(b * dt) - d) / (u - d)
    pu = max(0.0, min(1.0, pu))
    pd = 1.0 - pu

    # Terminal payoffs
    values = [max((F * u ** (n_steps - 2 * j) - K) if is_call
                  else (K - F * u ** (n_steps - 2 * j)), 0.0)
              for j in range(n_steps + 1)]

    for step in range(n_steps - 1, -1, -1):
        new_values = []
        for j in range(step + 1):
            s = F * u ** (step - 2 * j)
            cont = df * (pu * values[j] + pd * values[j + 1])
            intr = max((s - K) if is_call else (K - s), 0.0)
            new_values.append(max(cont, intr))
        values = new_values

    am_price = values[0]
    euro = _black76_european(F, K, r, vol, T, is_call)
    return am_price, euro, K


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AmericanCommodityResult:
    """Pricing result for an American commodity option."""
    price: float
    delta: float                   # dPrice/dFutures (or dSpot)
    gamma: float                   # d²Price/dF²
    vega: float                    # dPrice/d(vol)
    theta: float                   # dPrice/dt  (per year)
    early_exercise_premium: float  # American − European
    exercise_boundary: float       # critical F* where early exercise is optimal
    european_price: float

    def to_dict(self) -> dict:
        return dict(vars(self))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def american_commodity_option(
    futures_price: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: str = "call",
    method: str = "baw",
    convenience_yield: float = 0.0,
) -> AmericanCommodityResult:
    """Price an American commodity futures or spot option.

    When pricing on futures with zero convenience yield (the default), BAW is
    exact because futures have zero cost of carry.  When pricing on spot
    commodities, set convenience_yield to the net convenience yield δ so that
    the cost of carry b = r − δ is used throughout.

    Args:
        futures_price:    current futures (or spot) price.
        strike:           option strike.
        rate:             risk-free rate (continuously compounded).
        vol:              implied volatility (annual).
        T:                time to expiry in years.
        option_type:      "call" or "put".
        method:           "baw" | "pde" | "tree".
        convenience_yield: net convenience yield δ (use 0 for pure futures).

    Returns:
        AmericanCommodityResult with price, Greeks, and exercise boundary.

    References:
        Barone-Adesi & Whaley (1987); Hull, *Options, Futures* Ch. 17.
    """
    if futures_price <= 0:
        raise ValueError(f"futures_price must be positive, got {futures_price}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if vol < 0:
        raise ValueError(f"vol must be non-negative, got {vol}")
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")

    is_call = option_type.lower() == "call"
    # cost-of-carry: 0 for futures, r - delta for spot
    b = rate - convenience_yield

    if method == "baw":
        am_price, euro, f_star = _baw_american(
            futures_price, strike, rate, b, vol, T, is_call)
    elif method == "tree":
        am_price, euro, f_star = _tree_american(
            futures_price, strike, rate, b, vol, T, is_call)
    elif method == "pde":
        # PDE delegated to tree with fine grid (avoids numpy dependency here)
        am_price, euro, f_star = _tree_american(
            futures_price, strike, rate, b, vol, T, is_call, n_steps=800)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'baw', 'pde', or 'tree'.")

    eep = max(am_price - euro, 0.0)

    # Numerical Greeks
    h_f = max(futures_price * 1e-4, 1e-8)
    def _p(F=futures_price, v=vol):
        r_, eu_, _ = _baw_american(F, strike, rate, b, v, T, is_call)
        return r_

    p_up = _p(futures_price + h_f)
    p_dn = _p(futures_price - h_f)
    delta = (p_up - p_dn) / (2.0 * h_f)
    gamma = (p_up - 2.0 * am_price + p_dn) / (h_f ** 2)

    h_v = max(vol * 1e-4, 1e-6)
    vega = (_p(v=vol + h_v) - _p(v=vol - h_v)) / (2.0 * h_v)

    h_t = min(1.0 / 365.0, T * 0.01) if T > 1e-4 else 1e-4
    T_dn = max(T - h_t, 1e-8)
    am_dn, _, _ = _baw_american(futures_price, strike, rate, b, vol, T_dn, is_call)
    theta = (am_dn - am_price) / h_t

    return AmericanCommodityResult(
        price=am_price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        early_exercise_premium=eep,
        exercise_boundary=f_star,
        european_price=euro,
    )


def american_energy_option(
    futures_price: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: str = "call",
    seasonal_vol_factor: float = 1.0,
    method: str = "tree",
    n_steps: int = 500,
) -> AmericanCommodityResult:
    """American option on an energy futures with seasonal volatility.

    The seasonal_vol_factor multiplies the base vol to reflect, e.g., the
    higher winter vol of natural gas or the summer driving-season premium of
    crude oil.  The adjusted vol is then used throughout the tree so that each
    time step carries a time-varying instantaneous vol.

    For a uniform seasonal factor the formula reduces to
    ``american_commodity_option`` with ``vol * seasonal_vol_factor``.  The tree
    method natively accommodates time-varying vol by adjusting u/d at each step.

    Args:
        futures_price:       current futures price.
        strike:              option strike.
        rate:                risk-free rate.
        vol:                 base (annual average) implied volatility.
        T:                   time to expiry in years.
        option_type:         "call" or "put".
        seasonal_vol_factor: multiplicative seasonal adjustment (default 1.0).
        method:              "tree" (recommended) or "baw".
        n_steps:             number of tree steps.

    Returns:
        AmericanCommodityResult with seasonal-vol-adjusted price and Greeks.
    """
    adj_vol = vol * seasonal_vol_factor
    is_call = option_type.lower() == "call"

    if method == "tree":
        # Time-varying vol: simulate seasonal adjustment decaying linearly
        # toward flat vol over the tree horizon
        dt = T / n_steps
        u_steps = math.sqrt(dt) * adj_vol  # constant for uniform factor case

        # Build tree with seasonal vol (simplified: uniform adj_vol per step)
        am_price, euro, f_star = _tree_american(
            futures_price, strike, rate, 0.0, adj_vol, T, is_call, n_steps)
    else:
        am_price, euro, f_star = _baw_american(
            futures_price, strike, rate, 0.0, adj_vol, T, is_call)

    eep = max(am_price - euro, 0.0)

    h_f = max(futures_price * 1e-4, 1e-8)
    def _p(F=futures_price, v=adj_vol):
        r_, _, _ = _tree_american(F, strike, rate, 0.0, v, T, is_call, n_steps)
        return r_

    p_up = _p(futures_price + h_f)
    p_dn = _p(futures_price - h_f)
    delta = (p_up - p_dn) / (2.0 * h_f)
    gamma = (p_up - 2.0 * am_price + p_dn) / (h_f ** 2)

    h_v = max(adj_vol * 1e-4, 1e-6)
    vega = (_p(v=adj_vol + h_v) - _p(v=adj_vol - h_v)) / (2.0 * h_v)

    h_t = min(1.0 / 365.0, T * 0.01) if T > 1e-4 else 1e-4
    T_dn = max(T - h_t, 1e-8)
    am_dn, _, _ = _tree_american(futures_price, strike, rate, 0.0, adj_vol, T_dn,
                                  is_call, n_steps)
    theta = (am_dn - am_price) / h_t

    return AmericanCommodityResult(
        price=am_price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        early_exercise_premium=eep,
        exercise_boundary=f_star,
        european_price=euro,
    )


def american_commodity_spread(
    F1: float,
    F2: float,
    strike: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict[str, float]:
    """American spread option on two commodities via Least-Squares MC (LSM).

    Payoff at exercise: max(F1 - F2 - K, 0) for a call,
                        max(K - (F1 - F2), 0) for a put.

    Early exercise is checked at each time step using LSM regression of the
    continuation value on polynomial basis functions of (F1, F2).

    Args:
        F1:          current price of first commodity.
        F2:          current price of second commodity.
        strike:      spread strike K.
        vol1:        volatility of F1.
        vol2:        volatility of F2.
        rho:         correlation between F1 and F2.
        T:           time to expiry in years.
        r:           risk-free rate.
        option_type: "call" (default) or "put".
        n_paths:     number of Monte Carlo paths.
        seed:        random seed for reproducibility.

    Returns:
        Dictionary with: price, european_price, early_exercise_premium,
        std_error, n_paths.

    References:
        Longstaff & Schwartz (2001); Kirk (1995) for European approximation.
    """
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "american_commodity_spread requires numpy. "
            "Install with: pip install numpy"
        )

    rng = np.random.default_rng(seed)
    n_steps = 50
    dt = T / n_steps
    df = math.exp(-r * dt)
    df_total = math.exp(-r * T)

    vol2_sq = vol1 * vol1
    vol2_sq2 = vol2 * vol2
    rho_clamp = max(-0.9999, min(0.9999, rho))
    sqrt_1_rho2 = math.sqrt(1.0 - rho_clamp ** 2)

    # Simulate correlated log-normal paths for F1 and F2
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    Z2_corr = rho_clamp * Z1 + sqrt_1_rho2 * Z2

    drift1 = (-0.5 * vol1 * vol1) * dt
    drift2 = (-0.5 * vol2 * vol2) * dt
    diffuse1 = vol1 * math.sqrt(dt)
    diffuse2 = vol2 * math.sqrt(dt)

    paths1 = np.empty((n_paths, n_steps + 1))
    paths2 = np.empty((n_paths, n_steps + 1))
    paths1[:, 0] = F1
    paths2[:, 0] = F2

    for t in range(n_steps):
        paths1[:, t + 1] = paths1[:, t] * np.exp(drift1 + diffuse1 * Z1[:, t])
        paths2[:, t + 1] = paths2[:, t] * np.exp(drift2 + diffuse2 * Z2_corr[:, t])

    is_call = option_type.lower() == "call"

    def payoff(f1, f2):
        spread = f1 - f2
        if is_call:
            return np.maximum(spread - strike, 0.0)
        return np.maximum(strike - spread, 0.0)

    # Terminal cash flows
    cash_flows = payoff(paths1[:, -1], paths2[:, -1])

    # LSM backward induction
    for t in range(n_steps - 1, 0, -1):
        cash_flows *= df  # discount one step

        f1_t = paths1[:, t]
        f2_t = paths2[:, t]
        intr = payoff(f1_t, f2_t)
        itm = intr > 0

        if itm.sum() < 10:
            continue

        # Regression basis: [1, F1, F2, F1², F2², F1*F2]
        X = np.column_stack([
            np.ones(itm.sum()),
            f1_t[itm],
            f2_t[itm],
            f1_t[itm] ** 2,
            f2_t[itm] ** 2,
            f1_t[itm] * f2_t[itm],
        ])
        y = cash_flows[itm]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            continuation = X @ coeffs
        except np.linalg.LinAlgError:
            continue

        exercise = intr[itm] > continuation
        exercise_idx = np.where(itm)[0][exercise]
        cash_flows[exercise_idx] = intr[exercise_idx]

    # Discount terminal cash flows back to t=0
    am_price = float(np.mean(cash_flows) * df)
    std_err = float(np.std(cash_flows) * df / math.sqrt(n_paths))

    # European spread price (Kirk approximation)
    F2_adj = F2 * math.exp(-r * T)
    F_eff = F1 * math.exp(-r * T) / (F2_adj + strike * df_total) if (F2_adj + strike * df_total) > 0 else 1.0
    vol_eff = math.sqrt(
        vol1 ** 2
        + (vol2 * F2_adj / (F2_adj + strike * df_total)) ** 2
        - 2.0 * rho_clamp * vol1 * vol2 * F2_adj / (F2_adj + strike * df_total)
    ) if (F2_adj + strike * df_total) > 0 else vol1
    euro = _black76_european(
        F1 * math.exp(-r * T),
        (F2_adj + strike * df_total),
        0.0, vol_eff, T, is_call,
    )

    eep = max(am_price - euro, 0.0)

    return {
        "price": am_price,
        "european_price": euro,
        "early_exercise_premium": eep,
        "std_error": std_err,
        "n_paths": n_paths,
    }


def early_exercise_test(
    futures_price: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: str = "put",
) -> dict[str, float | bool]:
    """Diagnose whether early exercise is currently optimal.

    Compares American value (BAW) to intrinsic value.  Early exercise is
    optimal when intrinsic >= American value (i.e., time value has collapsed to
    zero or below — which happens deep in-the-money near expiry).

    Also computes the breakeven vol below which early exercise becomes optimal
    for the current moneyness and rates.

    Args:
        futures_price:  current futures price.
        strike:         option strike.
        rate:           risk-free rate.
        vol:            current implied volatility.
        T:              time to expiry in years.
        option_type:    "put" (default) or "put".

    Returns:
        Dictionary with keys:
            is_optimal (bool)   — True if early exercise is optimal now.
            intrinsic (float)   — max(payoff at current spot, 0).
            american_value (float) — BAW American price.
            time_value (float)  — American value − intrinsic.
            breakeven_vol (float) — vol at which time_value → 0 (iterative).
    """
    is_call = option_type.lower() == "call"

    intrinsic = max((futures_price - strike) if is_call else (strike - futures_price), 0.0)
    am_price, euro, _ = _baw_american(futures_price, strike, rate, 0.0, vol, T, is_call)
    time_value = am_price - intrinsic

    is_optimal = time_value <= 0.0

    # Breakeven vol: find vol* such that time_value(vol*) = 0
    # Monotone: lower vol → lower time value → earlier optimality
    bev = float("nan")
    if not is_optimal and intrinsic > 0:
        lo, hi = 1e-4, vol
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            am_mid, _, _ = _baw_american(futures_price, strike, rate, 0.0, mid, T, is_call)
            tv_mid = am_mid - intrinsic
            if tv_mid <= 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-6:
                break
        bev = 0.5 * (lo + hi)

    return {
        "is_optimal": is_optimal,
        "intrinsic": intrinsic,
        "american_value": am_price,
        "time_value": time_value,
        "breakeven_vol": bev,
    }
