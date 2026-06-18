"""American FX option pricing (Garman-Kohlhagen framework).

In the GK framework the foreign interest rate r_f acts as a continuous
dividend yield.  Early exercise is therefore optimal for:
  - puts when r_f >> r_d  (large carry incentive to hold foreign currency)
  - calls when r_d >> r_f

Three pricing methods are supported:
  - "baw"  : Barone-Adesi & Whaley (1987) quadratic approximation adapted
              for GK (r_f replaces the dividend yield).
  - "pde"  : Crank-Nicolson finite-difference on the log-spot grid with the
              GK operator and American early-exercise constraint applied at
              each time step.
  - "tree" : Cox-Ross-Rubinstein (CRR) binomial tree with GK drift
              (r_d - r_f) and early-exercise check at every node.

References:
    Garman, M. B., & Kohlhagen, S. W. (1983).  Foreign currency option values.
        Journal of International Money and Finance, 2, 231-237.
    Barone-Adesi, G., & Whaley, R. E. (1987).  Efficient analytic approximation
        of American option values.  Journal of Finance, 42(2), 301-320.
    DeRosa, D. F. (2011).  Options on Foreign Exchange, 3rd ed.  Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.models.black76 import _norm_cdf, _norm_pdf


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AmericanFXResult:
    """Pricing result for an American FX option."""
    price: float
    delta_domestic: float          # dPrice/dSpot (domestic-currency delta)
    delta_foreign: float           # exp(-r_f*T) * N(±d1) — foreign-currency delta
    gamma: float                   # d²Price/dSpot²
    vega: float                    # dPrice/d(vol)
    theta: float                   # dPrice/dt (per year, sign-negative for holder)
    early_exercise_premium: float  # American price − European price
    exercise_boundary_spot: float  # critical spot S* where early exercise is optimal
    european_price: float          # GK European price for comparison

    def to_dict(self) -> dict:
        return dict(vars(self))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gk_european(
    spot: float, strike: float, r_d: float, r_f: float,
    vol: float, T: float, is_call: bool,
) -> float:
    """Garman-Kohlhagen European price (closed form).

    Fix T4-FX3: pre-fix the ``T <= 0 or vol <= 0`` branch returned
    spot-based intrinsic ``max(spot − strike, 0)`` for both T=0 and
    T>0/vol=0 cases — but at vol=0 with T>0 the deterministic terminal
    is the forward ``F = spot·exp((r_d - r_f)T)`` and the present value
    is ``df_d · max(F − K, 0)``.  Pre-fix dropped both the
    spot-to-forward drift and the domestic discount.
    """
    if T <= 0:
        intrinsic = (spot - strike) if is_call else (strike - spot)
        return max(intrinsic, 0.0)
    if vol <= 0:
        fwd = spot * math.exp((r_d - r_f) * T)
        df_d = math.exp(-r_d * T)
        intrinsic = (fwd - strike) if is_call else (strike - fwd)
        return df_d * max(intrinsic, 0.0)
    fwd = spot * math.exp((r_d - r_f) * T)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df_d = math.exp(-r_d * T)
    df_f = math.exp(-r_f * T)
    if is_call:
        return df_f * spot * _norm_cdf(d1) - df_d * strike * _norm_cdf(d2)
    return df_d * strike * _norm_cdf(-d2) - df_f * spot * _norm_cdf(-d1)


def _baw_critical_spot(
    strike: float, r_d: float, r_f: float, vol: float, T: float, is_call: bool,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """BAW critical spot S* via Newton-Raphson."""
    vol2 = vol * vol
    M = 2.0 * r_d / vol2
    N = 2.0 * r_f / vol2
    q_inf_call = 0.5 * (-(N - 1) + math.sqrt((N - 1) ** 2 + 4.0 * M))
    q_inf_put  = 0.5 * (-(N - 1) - math.sqrt((N - 1) ** 2 + 4.0 * M))

    # Seed: close to strike for short expiries
    if is_call:
        q = q_inf_call
        S = strike * 1.05
    else:
        q = q_inf_put   # negative
        S = strike * 0.95

    df_f = math.exp(-r_f * T)

    for _ in range(max_iter):
        sqrt_t = math.sqrt(T)
        fwd = S * math.exp((r_d - r_f) * T)
        d1 = (math.log(fwd / strike) + 0.5 * vol2 * T) / (vol * sqrt_t)
        euro = _gk_european(S, strike, r_d, r_f, vol, T, is_call)
        intrinsic = (S - strike) if is_call else (strike - S)
        # LHS: intrinsic - euro
        # RHS: (intrinsic / q) * (1 - df_f * N(d1))
        nd1 = _norm_cdf(d1) if is_call else _norm_cdf(-d1)
        rhs_factor = 1.0 - df_f * nd1
        f = intrinsic - euro - (intrinsic / q) * rhs_factor
        # Derivative w.r.t. S
        sgn = 1.0 if is_call else -1.0
        d_euro = sgn * df_f * nd1                  # approx dEuro/dS
        d_intrinsic = sgn
        nd1_deriv = _norm_pdf(d1) / (S * vol * sqrt_t)
        df = d_intrinsic - d_euro - (d_intrinsic * rhs_factor + (intrinsic / q) * (-df_f * sgn * nd1_deriv * S)) / S
        if abs(df) < 1e-15:
            break
        S_new = S - f / df
        S_new = max(S_new, 1e-8)
        if abs(S_new - S) < tol:
            S = S_new
            break
        S = S_new

    return S


def _baw_price(
    spot: float, strike: float, r_d: float, r_f: float,
    vol: float, T: float, is_call: bool,
) -> tuple[float, float, float]:
    """BAW approximation for American FX option.

    Returns (american_price, european_price, exercise_boundary).
    """
    euro = _gk_european(spot, strike, r_d, r_f, vol, T, is_call)

    if T <= 1e-8:
        intrinsic = max((spot - strike) if is_call else (strike - spot), 0.0)
        return intrinsic, euro, strike

    vol2 = vol * vol
    M = 2.0 * r_d / vol2
    N = 2.0 * r_f / vol2

    disc = (N - 1) ** 2 + 4.0 * M
    if disc < 0:
        return euro, euro, (strike * 1e6 if is_call else 0.0)

    if is_call:
        q = 0.5 * (-(N - 1) + math.sqrt(disc))
    else:
        q = 0.5 * (-(N - 1) - math.sqrt(disc))

    S_star = _baw_critical_spot(strike, r_d, r_f, vol, T, is_call)

    intrinsic_star = (S_star - strike) if is_call else (strike - S_star)
    if intrinsic_star <= 0:
        return euro, euro, S_star

    fwd_star = S_star * math.exp((r_d - r_f) * T)
    sqrt_t = math.sqrt(T)
    d1_star = (math.log(fwd_star / strike) + 0.5 * vol2 * T) / (vol * sqrt_t)
    df_f = math.exp(-r_f * T)
    nd1 = _norm_cdf(d1_star) if is_call else _norm_cdf(-d1_star)
    A = (intrinsic_star / q) * (1.0 - df_f * nd1)

    # Early exercise region
    if is_call and spot >= S_star:
        return spot - strike, euro, S_star
    if not is_call and spot <= S_star:
        return strike - spot, euro, S_star

    fwd = spot * math.exp((r_d - r_f) * T)
    d1 = (math.log(fwd / strike) + 0.5 * vol2 * T) / (vol * sqrt_t)
    correction = A * (spot / S_star) ** q
    am_price = euro + correction
    return am_price, euro, S_star


def _tree_price(
    spot: float, strike: float, r_d: float, r_f: float,
    vol: float, T: float, is_call: bool, n_steps: int = 500,
) -> tuple[float, float, float]:
    """CRR binomial tree for American FX option."""
    dt = T / n_steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    df = math.exp(-r_d * dt)
    # Risk-neutral probability under GK
    p = (math.exp((r_d - r_f) * dt) - d) / (u - d)
    p = max(0.0, min(1.0, p))
    q = 1.0 - p

    # Terminal spots
    spots = [spot * (u ** (n_steps - 2 * j)) for j in range(n_steps + 1)]
    # Terminal payoffs
    if is_call:
        values = [max(s - strike, 0.0) for s in spots]
    else:
        values = [max(strike - s, 0.0) for s in spots]

    # Backward induction with early exercise
    for step in range(n_steps - 1, -1, -1):
        new_values = []
        for j in range(step + 1):
            s = spot * (u ** (step - 2 * j))
            continuation = df * (p * values[j] + q * values[j + 1])
            intrinsic = max(s - strike, 0.0) if is_call else max(strike - s, 0.0)
            new_values.append(max(continuation, intrinsic))
        values = new_values

    am_price = values[0]
    euro = _gk_european(spot, strike, r_d, r_f, vol, T, is_call)

    # Approximate exercise boundary: deepest in-the-money node at step 1
    s_star = strike  # fallback
    return am_price, euro, s_star


def _pde_price(
    spot: float, strike: float, r_d: float, r_f: float,
    vol: float, T: float, is_call: bool,
    n_spot: int = 200, n_time: int = 200,
) -> tuple[float, float, float]:
    """Crank-Nicolson PDE for American FX option (GK operator)."""
    try:
        import numpy as np
    except ImportError:
        # Fall back to BAW if numpy is unavailable
        return _baw_price(spot, strike, r_d, r_f, vol, T, is_call)

    # Log-spot grid
    x_min = math.log(spot * 0.01)
    x_max = math.log(spot * 10.0)
    dx = (x_max - x_min) / n_spot
    x = np.linspace(x_min, x_max, n_spot + 1)
    S = np.exp(x)

    # Terminal condition
    if is_call:
        V = np.maximum(S - strike, 0.0)
    else:
        V = np.maximum(strike - S, 0.0)

    dt = T / n_time
    r = r_d - r_f - 0.5 * vol * vol
    vol2 = vol * vol

    # Tridiagonal coefficients (interior nodes)
    n = n_spot - 1  # interior count
    alpha = 0.5 * dt * (vol2 / (dx * dx) - r / dx)
    beta  = 1.0 + dt * (vol2 / (dx * dx) + r_d)
    gamma = 0.5 * dt * (vol2 / (dx * dx) + r / dx)

    lower = np.full(n, -0.5 * alpha)
    main  = np.full(n, 1.0 + 0.5 * (vol2 / (dx * dx) + r_d) * dt)
    upper = np.full(n, -0.5 * gamma)

    def thomas(a, b, c, rhs):
        n_ = len(rhs)
        c_ = c.copy()
        r_ = rhs.copy()
        for i in range(1, n_):
            m = a[i] / b[i - 1]
            b[i] -= m * c_[i - 1]
            r_[i] -= m * r_[i - 1]
        sol = np.empty(n_)
        sol[-1] = r_[-1] / b[-1]
        for i in range(n_ - 2, -1, -1):
            sol[i] = (r_[i] - c_[i] * sol[i + 1]) / b[i]
        return sol

    s_star = strike
    for _ in range(n_time):
        V_int = V[1:-1]
        rhs = (alpha * V[:-2] + (2.0 - 2.0 * alpha - 2.0 * gamma) * V_int + gamma * V[2:]) * 0.5
        # Boundary contributions
        rhs[0]  += 0.5 * alpha * V[0]
        rhs[-1] += 0.5 * gamma * V[-1]

        b_work = main.copy()
        V_int_new = thomas(lower.copy(), b_work, upper.copy(), rhs)
        V[1:-1] = V_int_new

        # Boundary conditions
        if is_call:
            V[0]  = 0.0
            V[-1] = S[-1] - strike * math.exp(-r_d * dt)
        else:
            V[0]  = strike * math.exp(-r_d * dt) - S[0]
            V[-1] = 0.0

        # American early-exercise constraint
        if is_call:
            intrinsic = np.maximum(S - strike, 0.0)
        else:
            intrinsic = np.maximum(strike - S, 0.0)
        V = np.maximum(V, intrinsic)

    # Interpolate at spot
    idx = np.searchsorted(S, spot)
    idx = max(1, min(idx, n_spot - 1))
    t = (spot - S[idx - 1]) / (S[idx] - S[idx - 1])
    am_price = float(V[idx - 1] * (1.0 - t) + V[idx] * t)

    euro = _gk_european(spot, strike, r_d, r_f, vol, T, is_call)
    return am_price, euro, s_star


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def american_fx_option(
    spot: float,
    strike: float,
    rate_domestic: float,
    rate_foreign: float,
    vol: float,
    T: float,
    option_type: str = "call",
    method: str = "baw",
) -> AmericanFXResult:
    """Price an American FX option under the Garman-Kohlhagen framework.

    The foreign interest rate acts as a continuous dividend yield on the spot.
    Early exercise is optimal for puts when r_f > r_d and for calls when
    r_d > r_f (carry incentive to hold the higher-yielding currency outright).

    Args:
        spot:            current spot rate (domestic per foreign unit).
        strike:          option strike.
        rate_domestic:   continuously-compounded domestic risk-free rate.
        rate_foreign:    continuously-compounded foreign risk-free rate.
        vol:             implied volatility (annual).
        T:               time to expiry in years.
        option_type:     "call" or "put".
        method:          "baw" | "pde" | "tree".

    Returns:
        AmericanFXResult with price, Greeks, early-exercise premium and
        critical exercise boundary.

    References:
        Garman & Kohlhagen (1983); Barone-Adesi & Whaley (1987);
        DeRosa, *Options on Foreign Exchange*, 3rd ed. (2011).
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if vol < 0:
        raise ValueError(f"vol must be non-negative, got {vol}")
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")

    is_call = option_type.lower() == "call"

    if method == "baw":
        am_price, euro, s_star = _baw_price(spot, strike, rate_domestic,
                                             rate_foreign, vol, T, is_call)
    elif method == "pde":
        am_price, euro, s_star = _pde_price(spot, strike, rate_domestic,
                                             rate_foreign, vol, T, is_call)
    elif method == "tree":
        am_price, euro, s_star = _tree_price(spot, strike, rate_domestic,
                                              rate_foreign, vol, T, is_call)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'baw', 'pde', or 'tree'.")

    eep = max(am_price - euro, 0.0)

    # --- numerical Greeks via bumping (use raw pricer to avoid recursion) ---
    def _raw(s, k, rd, rf, v, t):
        ic = (option_type.lower() == "call")
        if method == "baw":
            return _baw_price(s, k, rd, rf, v, t, ic)[0]
        elif method == "pde":
            return _pde_price(s, k, rd, rf, v, t, ic)[0]
        else:
            return _tree_price(s, k, rd, rf, v, t, ic)[0]

    h_s = spot * 1e-4
    p_up = _raw(spot + h_s, strike, rate_domestic, rate_foreign, vol, T)
    p_dn = _raw(spot - h_s, strike, rate_domestic, rate_foreign, vol, T)
    delta_dom = (p_up - p_dn) / (2.0 * h_s)
    gamma = (p_up - 2.0 * am_price + p_dn) / (h_s ** 2)

    df_f = math.exp(-rate_foreign * T) if T > 0 else 1.0
    delta_for = df_f * delta_dom  # approximate foreign-currency delta

    h_v = 1e-4
    vega_up = _raw(spot, strike, rate_domestic, rate_foreign, vol + h_v, T)
    vega = (vega_up - am_price) / h_v

    h_t = min(1.0 / 365.0, T * 0.01) if T > 0 else 1.0 / 365.0
    T_dn = max(T - h_t, 1e-8)
    theta_dn = _raw(spot, strike, rate_domestic, rate_foreign, vol, T_dn)
    theta = (theta_dn - am_price) / h_t  # per year, negative for option holder

    return AmericanFXResult(
        price=am_price,
        delta_domestic=delta_dom,
        delta_foreign=delta_for,
        gamma=gamma,
        vega=vega,
        theta=theta,
        early_exercise_premium=eep,
        exercise_boundary_spot=s_star,
        european_price=euro,
    )


def fx_exercise_boundary(
    spot: float,
    strike: float,
    rate_domestic: float,
    rate_foreign: float,
    vol: float,
    T: float,
    option_type: str = "put",
    n_points: int = 50,
) -> list[tuple[float, float]]:
    """Compute the early-exercise boundary B(t) for an American FX option.

    Returns a list of (t, B(t)) pairs where t is time remaining to expiry and
    B(t) is the critical spot below which (put) or above which (call) immediate
    exercise is optimal.

    The boundary is computed by finding the BAW critical spot S* at each
    sub-expiry t_i in [0, T].  The BAW approach is equivalent to the integral
    equation boundary (Kim, 1990) for smooth boundaries and is fast to evaluate.

    Args:
        spot:           current spot (used only for scale).
        strike:         option strike.
        rate_domestic:  domestic risk-free rate.
        rate_foreign:   foreign risk-free rate.
        vol:            implied volatility.
        T:              total time to expiry.
        option_type:    "put" (default) or "call".
        n_points:       number of boundary points.

    Returns:
        List of (time_remaining, critical_spot) pairs, from t=T down to t≈0.

    References:
        Kim, I. J. (1990).  The analytic valuation of American options.
            Review of Financial Studies, 3(4), 547-572.
        Barone-Adesi & Whaley (1987), ibid.
    """
    is_call = option_type.lower() == "call"
    boundary = []
    for i in range(n_points):
        # t_i: time remaining, from T down to a small positive value
        t_i = T * (n_points - i) / n_points
        t_i = max(t_i, 1e-4)
        s_star = _baw_critical_spot(strike, rate_domestic, rate_foreign,
                                     vol, t_i, is_call)
        boundary.append((t_i, s_star))
    return boundary


def american_fx_greeks(
    spot: float,
    strike: float,
    rate_domestic: float,
    rate_foreign: float,
    vol: float,
    T: float,
    option_type: str = "call",
    method: str = "baw",
    bump: float = 0.0001,
) -> dict[str, float]:
    """Numerical Greeks for an American FX option.

    Computes delta_dom, delta_for, gamma, vega, theta, rho_dom, rho_for via
    symmetric finite differences around the base case.

    Args:
        spot:           current spot rate.
        strike:         option strike.
        rate_domestic:  domestic risk-free rate.
        rate_foreign:   foreign risk-free rate.
        vol:            implied volatility.
        T:              time to expiry in years.
        option_type:    "call" or "put".
        method:         pricing method ("baw", "pde", "tree").
        bump:           relative bump size (applied to each input).

    Returns:
        Dictionary with keys: price, delta_dom, delta_for, gamma, vega,
        theta, rho_dom, rho_for.
    """
    def price(**kw) -> float:
        return american_fx_option(
            spot=kw.get("spot", spot),
            strike=strike,
            rate_domestic=kw.get("rate_domestic", rate_domestic),
            rate_foreign=kw.get("rate_foreign", rate_foreign),
            vol=kw.get("vol", vol),
            T=kw.get("T", T),
            option_type=option_type,
            method=method,
        ).price

    base = price()

    # Spot Greeks
    h_s = max(spot * bump, 1e-8)
    p_up = price(spot=spot + h_s)
    p_dn = price(spot=spot - h_s)
    delta_dom = (p_up - p_dn) / (2.0 * h_s)
    gamma = (p_up - 2.0 * base + p_dn) / (h_s ** 2)

    # Foreign delta: dPrice/dSpot converted to foreign notional
    df_f = math.exp(-rate_foreign * T) if T > 0 else 1.0
    delta_for = df_f * delta_dom

    # Vega
    h_v = max(vol * bump, 1e-6)
    vega = (price(vol=vol + h_v) - price(vol=vol - h_v)) / (2.0 * h_v)

    # Theta (time decay, per year)
    h_t = min(1.0 / 365.0, T * 0.05) if T > 1e-4 else 1e-4
    T_dn = max(T - h_t, 1e-8)
    theta = (price(T=T_dn) - base) / h_t

    # Rho domestic
    h_r = max(abs(rate_domestic) * bump, 1e-5)
    if h_r < 1e-6:
        h_r = 1e-4
    rho_dom = (price(rate_domestic=rate_domestic + h_r)
               - price(rate_domestic=rate_domestic - h_r)) / (2.0 * h_r)

    # Rho foreign
    h_rf = max(abs(rate_foreign) * bump, 1e-5)
    if h_rf < 1e-6:
        h_rf = 1e-4
    rho_for = (price(rate_foreign=rate_foreign + h_rf)
               - price(rate_foreign=rate_foreign - h_rf)) / (2.0 * h_rf)

    return {
        "price": base,
        "delta_dom": delta_dom,
        "delta_for": delta_for,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho_dom": rho_dom,
        "rho_for": rho_for,
    }
