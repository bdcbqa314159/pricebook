"""
CMS spread pricing under the G2++ two-factor Hull-White model.

Under a single-factor model (HW1F), all zero-coupon bond prices are driven by
one Brownian motion.  Consequently, any two CMS rates (e.g. 2Y and 10Y swap
rates computed from the ZCB curve) are perfectly correlated — implied CMS
spread volatility is zero and spread options cannot be priced meaningfully.

G2++ breaks this degeneracy: the x-factor (fast mean reversion, drives the
short end) and the y-factor (slow mean reversion, drives the long end) load
differently onto CMS rates of different tenors.  The resulting non-trivial
correlation produces realistic spread volatilities and spread option prices.

This module prices three related quantities via Monte Carlo simulation of
the G2++ (x, y) factor paths:

1. **CMS spread** — distribution of CMS_long - CMS_short at time T.
2. **CMS spread option** — cap/floor on the CMS spread: max(spread - K, 0).
3. **Implied CMS correlation** — Pearson correlation between two CMS rates
   under G2++; a key diagnostic (equals 1.0 under HW1F).

Usage::

    from pricebook.structured.cms_spread_g2pp import (
        cms_spread_g2pp, cms_spread_option_g2pp, cms_correlation_g2pp,
    )

    result = cms_spread_g2pp(g2pp, cms_long_tenor=10, cms_short_tenor=2, T=1.0)
    opt    = cms_spread_option_g2pp(g2pp, 10, 2, strike=0.01, T=1.0)
    corr   = cms_correlation_g2pp(g2pp, tenor1=10, tenor2=2, T=1.0)

References:
    Brigo, D. & Mercurio, F., *Interest Rate Models — Theory and Practice*,
    2nd ed., Ch. 13.4, Springer, 2006.
    Piterbarg, V., *Interest Rate Modeling*, Vol. III, Ch. 14, 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.vasicek import G2PlusPlus
from pricebook.core.day_count import date_from_year_fraction


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CMSSpreadG2PPResult:
    """Distribution statistics for the CMS spread under G2++.

    Attributes:
        price: discounted expected spread = E[DF(T) * (CMS_long - CMS_short)].
            For a forward spread this equals the spread forward value.
        cms_long: expected CMS long-tenor rate at time T.
        cms_short: expected CMS short-tenor rate at time T.
        spread: expected spread = cms_long - cms_short (forward value).
        spread_vol: standard deviation of the spread across MC paths.
        correlation_implied: Pearson correlation between CMS_long and CMS_short
            across paths.  Under HW1F this would be ~1.0; under G2++ it is < 1.
    """
    price: float
    cms_long: float
    cms_short: float
    spread: float
    spread_vol: float
    correlation_implied: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CMSSpreadOptionResult:
    """G2++ CMS spread option pricing result.

    Attributes:
        price: option price (discounted expected payoff).
        delta: finite-difference delta with respect to the spread forward.
        vega: finite-difference vega with respect to sigma1 (x-factor vol).
        implied_vol: implied Black vol of the spread option.
        spread_forward: forward value of the CMS spread at T.
    """
    price: float
    delta: float
    vega: float
    implied_vol: float
    spread_forward: float

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cms_rate_from_zcb(
    g2pp: G2PlusPlus,
    x: np.ndarray,
    y: np.ndarray,
    t_start: float,
    cms_tenor: float,
    freq: float = 2.0,
) -> np.ndarray:
    """Compute the par CMS rate from G2++ ZCB prices at time t_start.

    The par swap rate for a swap starting at t_start with tenor cms_tenor is:

        S = (P(t_start) - P(t_start + cms_tenor)) / annuity

    where annuity = sum_{i=1}^{n} (1/freq) * P(t_start + i/freq).

    Because the G2++ ZCB formula conditions on (x, y) at the current time,
    we evaluate all bond prices in the G2++ affine framework.

    Args:
        g2pp: G2PlusPlus model.
        x: x-factor values, shape (n_paths,).
        y: y-factor values, shape (n_paths,).
        t_start: time at which the CMS rate is observed (years).
        cms_tenor: tenor of the underlying swap (years).
        freq: coupon frequency of the underlying swap (default 2 = semi-annual).

    Returns:
        Array of par CMS rates, shape (n_paths,).
    """
    # G2++ ZCB: P(x, y; t_start, T) evaluated at t_start.
    # The zcb_price method gives P(0,T) adjusted for x,y starting from 0 at
    # time 0.  For a simulation at time t_start we need the FORWARD ZCB:
    #   P(x_t, y_t; t, T) = P_mkt(T) / P_mkt(t) * exp(-Bx(T-t)*x - By(T-t)*y + 0.5*V(T-t))
    # where Bx, By, V use the *remaining* horizon tau = T - t_start.

    ref = g2pp.curve.reference_date
    a, b_ = g2pp.a, g2pp.b
    s1, s2, rho = g2pp.sigma1, g2pp.sigma2, g2pp.rho

    def _B(k: float, tau: float) -> float:
        if k == 0 or abs(k) < 1e-10:
            return tau
        return (1 - math.exp(-k * tau)) / k

    def _V(tau: float) -> float:
        Ba = _B(a, tau)
        Bb = _B(b_, tau)
        return (
            s1**2 / a**2 * (tau - 2 * Ba + _B(2 * a, tau))
            + s2**2 / b_**2 * (tau - 2 * Bb + _B(2 * b_, tau))
            + 2 * rho * s1 * s2 / (a * b_) * (tau - Ba - Bb + _B(a + b_, tau))
        )

    def _forward_zcb(tau: float) -> np.ndarray:
        """Forward ZCB P(x, y; t_start, t_start+tau) as array over paths."""
        if tau <= 1e-10:
            return np.ones(len(x))
        T_abs = t_start + tau
        d_t = date_from_year_fraction(ref, t_start)
        d_T = date_from_year_fraction(ref, T_abs)
        p_t = g2pp.curve.df(d_t) if t_start > 1e-10 else 1.0
        p_T = g2pp.curve.df(d_T)

        Bx = _B(a, tau)
        By = _B(b_, tau)
        V = _V(tau)

        return (p_T / p_t) * np.exp(-Bx * x - By * y + 0.5 * V)

    # Annuity: sum over fixed leg payment dates
    period = 1.0 / freq
    n_payments = int(round(cms_tenor * freq))
    annuity = np.zeros(len(x))
    for k in range(1, n_payments + 1):
        tau_k = k * period
        annuity += period * _forward_zcb(tau_k)

    p_near = _forward_zcb(0.0)          # = 1.0 (at t_start)
    p_far = _forward_zcb(cms_tenor)

    # Par swap rate
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(annuity > 1e-12,
                        (p_near - p_far) / annuity,
                        np.zeros(len(x)))
    return rate


def _simulate_factors(
    g2pp: G2PlusPlus,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate G2++ (x, y) paths and compute path-wise discount factors to T.

    Returns:
        (x_T, y_T, df_T): factor values at T, shape (n_paths,); and
        discount factors E[exp(-int_0^T r ds)] per path.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    a, b_ = g2pp.a, g2pp.b
    s1, s2, rho = g2pp.sigma1, g2pp.sigma2, g2pp.rho

    e_a = math.exp(-a * dt)
    e_b = math.exp(-b_ * dt)
    std_x = s1 * math.sqrt((1 - math.exp(-2 * a * dt)) / (2 * a)) if a > 0 else s1 * math.sqrt(dt)
    std_y = s2 * math.sqrt((1 - math.exp(-2 * b_ * dt)) / (2 * b_)) if b_ > 0 else s2 * math.sqrt(dt)

    # Cholesky for correlated increments
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)

    x = np.zeros(n_paths)
    y = np.zeros(n_paths)
    log_df = np.zeros(n_paths)

    # phi(t): forward rate + G2++ correction
    ref = g2pp.curve.reference_date

    def _fwd(t_: float) -> float:
        eps = 1e-5
        if t_ < eps:
            d1 = date_from_year_fraction(ref, eps)
            return -math.log(g2pp.curve.df(d1)) / eps
        d1 = date_from_year_fraction(ref, t_ - eps)
        d2 = date_from_year_fraction(ref, t_ + eps)
        return -math.log(g2pp.curve.df(d2) / g2pp.curve.df(d1)) / (2 * eps)

    def _phi(t_: float) -> float:
        ea = (1 - math.exp(-a * t_)) if a > 0 else t_
        eb = (1 - math.exp(-b_ * t_)) if b_ > 0 else t_
        return (_fwd(t_)
                + s1**2 / (2 * a**2) * ea**2
                + s2**2 / (2 * b_**2) * eb**2
                + rho * s1 * s2 / (a * b_) * ea * eb)

    for step in range(n_steps):
        t_cur = step * dt
        phi_t = _phi(t_cur)

        z = rng.standard_normal((n_paths, 2)) @ L.T
        x_new = x * e_a + std_x * z[:, 0]
        y_new = y * e_b + std_y * z[:, 1]

        r_mid = 0.5 * (x + y + phi_t + x_new + y_new + _phi(t_cur + dt))
        log_df -= r_mid * dt

        x = x_new
        y = y_new

    return x, y, np.exp(log_df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cms_spread_g2pp(
    g2pp: G2PlusPlus,
    cms_long_tenor: float,
    cms_short_tenor: float,
    T: float,
    notional: float = 1_000_000.0,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int = 42,
) -> CMSSpreadG2PPResult:
    """CMS spread distribution under G2++ via Monte Carlo.

    Simulates G2++ (x, y) paths to the observation date T, then computes the
    par CMS rates for the long and short tenors using the analytical G2++ ZCB
    formula at each path's terminal (x_T, y_T) state.

    Key insight: the long CMS rate loads more heavily on the y-factor
    (slow mean reversion, drives the long end) while the short CMS rate loads
    more heavily on the x-factor.  This imperfect loading creates non-trivial
    correlation and positive spread volatility — impossible under HW1F.

    Args:
        g2pp: calibrated G2PlusPlus model.
        cms_long_tenor: tenor of the long CMS rate (years, e.g. 10).
        cms_short_tenor: tenor of the short CMS rate (years, e.g. 2).
        T: observation date (years from today).
        notional: notional amount for the price calculation.
        n_paths: Monte Carlo paths.
        n_steps: time steps in the simulation.
        seed: random seed for reproducibility.

    Returns:
        :class:`CMSSpreadG2PPResult` with distribution statistics.
    """
    x_T, y_T, df_T = _simulate_factors(g2pp, T, n_steps, n_paths, seed)

    rate_long = _cms_rate_from_zcb(g2pp, x_T, y_T, T, cms_long_tenor)
    rate_short = _cms_rate_from_zcb(g2pp, x_T, y_T, T, cms_short_tenor)
    spread = rate_long - rate_short

    # Discounted expected payoffs
    price = float(np.mean(df_T * spread)) * notional
    cms_long_fwd = float(np.mean(rate_long))
    cms_short_fwd = float(np.mean(rate_short))
    spread_fwd = float(np.mean(spread))
    spread_vol = float(np.std(spread))

    # Implied correlation between CMS_long and CMS_short paths
    if np.std(rate_long) > 1e-12 and np.std(rate_short) > 1e-12:
        corr = float(np.corrcoef(rate_long, rate_short)[0, 1])
    else:
        corr = 1.0

    return CMSSpreadG2PPResult(
        price=price,
        cms_long=cms_long_fwd,
        cms_short=cms_short_fwd,
        spread=spread_fwd,
        spread_vol=spread_vol,
        correlation_implied=corr,
    )


def cms_spread_option_g2pp(
    g2pp: G2PlusPlus,
    cms_long_tenor: float,
    cms_short_tenor: float,
    strike: float,
    T: float,
    option_type: str = "call",
    notional: float = 1_000_000.0,
    n_paths: int = 50_000,
    seed: int = 42,
) -> CMSSpreadOptionResult:
    """CMS spread option price under G2++ via Monte Carlo.

    Prices a European option on the CMS spread::

        call payoff = max(CMS_long(T) - CMS_short(T) - K, 0)
        put  payoff = max(K - CMS_long(T) + CMS_short(T), 0)

    Args:
        g2pp: calibrated G2PlusPlus model.
        cms_long_tenor: long CMS tenor (years).
        cms_short_tenor: short CMS tenor (years).
        strike: option strike (decimal, e.g. 0.01 = 100 bp).
        T: option expiry (years).
        option_type: "call" or "put".
        notional: notional amount.
        n_paths: Monte Carlo paths.
        seed: random seed.

    Returns:
        :class:`CMSSpreadOptionResult` with price, greeks, implied vol, and
        spread forward.
    """
    n_steps = max(50, int(T * 100))

    x_T, y_T, df_T = _simulate_factors(g2pp, T, n_steps, n_paths, seed)

    rate_long = _cms_rate_from_zcb(g2pp, x_T, y_T, T, cms_long_tenor)
    rate_short = _cms_rate_from_zcb(g2pp, x_T, y_T, T, cms_short_tenor)
    spread = rate_long - rate_short

    opt = option_type.lower()
    if opt == "call":
        payoff = np.maximum(spread - strike, 0.0)
    elif opt == "put":
        payoff = np.maximum(strike - spread, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    price = float(np.mean(df_T * payoff)) * notional
    spread_fwd = float(np.mean(spread))
    spread_vol = float(np.std(spread))

    # Delta: finite difference on notional (bump spread_fwd via strike shift)
    price_up = float(np.mean(df_T * (np.maximum(spread - (strike - 1e-4), 0.0)
                                     if opt == "call"
                                     else np.maximum((strike - 1e-4) - spread, 0.0)))) * notional
    price_dn = float(np.mean(df_T * (np.maximum(spread - (strike + 1e-4), 0.0)
                                     if opt == "call"
                                     else np.maximum((strike + 1e-4) - spread, 0.0)))) * notional
    delta = (price_up - price_dn) / (2e-4 * notional)

    # Vega: bump sigma1 by 1 bp and reprice (separate simulation)
    g2pp_vega = G2PlusPlus(
        a=g2pp.a, b=g2pp.b,
        sigma1=g2pp.sigma1 + 1e-4, sigma2=g2pp.sigma2,
        rho=g2pp.rho, curve=g2pp.curve,
    )
    x_v, y_v, df_v = _simulate_factors(g2pp_vega, T, n_steps, n_paths, seed)
    rate_long_v = _cms_rate_from_zcb(g2pp_vega, x_v, y_v, T, cms_long_tenor)
    rate_short_v = _cms_rate_from_zcb(g2pp_vega, x_v, y_v, T, cms_short_tenor)
    spread_v = rate_long_v - rate_short_v
    payoff_v = (np.maximum(spread_v - strike, 0.0) if opt == "call"
                else np.maximum(strike - spread_v, 0.0))
    price_vega = float(np.mean(df_v * payoff_v)) * notional
    vega = (price_vega - price) / 1e-4

    # Implied vol via Bachelier (normal model for spread)
    ref = g2pp.curve.reference_date
    d_T = date_from_year_fraction(ref, T)
    df_mkt = g2pp.curve.df(d_T) if T > 1e-10 else 1.0

    implied_vol = 0.0
    if df_mkt > 1e-12 and T > 1e-10 and spread_vol > 1e-10:
        # Bachelier implied vol: price = df * N_vol * sqrt(T) * [phi(d) + d*Phi(d)]
        # Approximate: price / (df * notional) ~ N_vol * sqrt(T) / sqrt(2*pi) for ATM
        intrinsic = df_mkt * max(spread_fwd - strike, 0.0)
        time_value = price / notional - intrinsic
        if time_value > 0 and T > 0:
            implied_vol = time_value / (df_mkt * math.sqrt(T / (2 * math.pi)))

    return CMSSpreadOptionResult(
        price=price,
        delta=delta,
        vega=vega,
        implied_vol=implied_vol,
        spread_forward=spread_fwd,
    )


def cms_correlation_g2pp(
    g2pp: G2PlusPlus,
    tenor1: float,
    tenor2: float,
    T: float,
    n_paths: int = 50_000,
    seed: int = 42,
) -> float:
    """Implied correlation between two CMS rates under G2++.

    Computes the Pearson correlation between CMS rate 1 (tenor `tenor1`) and
    CMS rate 2 (tenor `tenor2`) at time T, measured across G2++ MC paths.

    Key diagnostic: under HW1F this would be 1.0 (single-factor model);
    under G2++ it is strictly less than 1.0 because the x and y factors load
    differently onto different tenors.  The correlation approaches 1.0 when
    rho -> 1 or sigma2 -> 0 (collapsing to HW1F).

    Args:
        g2pp: calibrated G2PlusPlus model.
        tenor1: first CMS tenor (years).
        tenor2: second CMS tenor (years).
        T: observation date (years).
        n_paths: Monte Carlo paths.
        seed: random seed.

    Returns:
        Pearson correlation in [-1, 1].  Typically 0.7 -- 0.99 for typical
        G2++ parameters.
    """
    n_steps = max(50, int(T * 100))
    x_T, y_T, _ = _simulate_factors(g2pp, T, n_steps, n_paths, seed)

    rate1 = _cms_rate_from_zcb(g2pp, x_T, y_T, T, tenor1)
    rate2 = _cms_rate_from_zcb(g2pp, x_T, y_T, T, tenor2)

    std1 = float(np.std(rate1))
    std2 = float(np.std(rate2))
    if std1 < 1e-12 or std2 < 1e-12:
        return 1.0

    return float(np.corrcoef(rate1, rate2)[0, 1])
