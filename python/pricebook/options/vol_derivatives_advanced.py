"""Advanced volatility derivatives: var swap Greeks, forward variance,
vol swaps, dispersion, Bates CF, SVI calibration.

Extends the existing vol_vol_derivatives.py with production-grade
tools for volatility trading desks.

    from pricebook.options.vol_derivatives_advanced import (
        variance_swap_greeks, VarianceSwapGreeks,
        forward_variance_curve, ForwardVarianceCurve,
        volatility_swap_price, VolSwapResult,
        dispersion_trade, DispersionTradeResult,
        bates_characteristic_function, bates_price,
        svi_calibrate, SVIParams,
    )

References:
    Gatheral (2006), The Volatility Surface, Ch. 3 (SVI).
    Carr & Lee (2009), Robust Replication of Volatility Derivatives.
    Bates (1996), Jumps and Stochastic Volatility, RFS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# 1. Variance swap Greeks
# ---------------------------------------------------------------------------

@dataclass
class VarianceSwapGreeks:
    """Greeks of a variance swap position.

    Unlike options, var swap has:
    - Gamma proportional to 1/S² (constant dollar gamma)
    - Vega = 2 × √K_var × vega_notional
    - Theta from variance accrual
    """
    pv: float
    delta: float               # dPV/dS (via portfolio of OTM options)
    gamma: float               # d²PV/dS² ∝ 1/S²
    vega: float                # dPV/d(σ_implied)
    theta: float               # daily variance accrual
    dollar_gamma: float        # S² × gamma (constant for var swap)

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "delta": self.delta, "gamma": self.gamma,
            "vega": self.vega, "theta": self.theta,
            "dollar_gamma": self.dollar_gamma,
        }


def variance_swap_greeks(
    spot: float,
    strike_var: float,
    T: float,
    vol: float,
    notional_vega: float,
    r: float = 0.0,
    realised_var_so_far: float = 0.0,
    elapsed_fraction: float = 0.0,
) -> VarianceSwapGreeks:
    """Compute Greeks for a variance swap.

    Key property: var swap has constant dollar gamma = 2/T × notional_var.
    Delta comes from log-contract replication (strip of 1/K² weighted options).
    """
    # Guard: if strike_var very small, cap notional_var to avoid explosion
    if strike_var > 1e-6:
        notional_var = notional_vega / (2 * math.sqrt(strike_var))
    else:
        notional_var = 0.0

    # PV: (expected_var - strike_var) × notional × DF(remaining-to-maturity)
    remaining = 1.0 - elapsed_fraction
    expected_var = elapsed_fraction * realised_var_so_far + remaining * vol ** 2
    df_remaining = math.exp(-r * T * remaining)
    pv = (expected_var - strike_var) * notional_var * df_remaining

    # Dollar gamma: constant = 2/T × notional_var (Carr & Lee 2009)
    dollar_gamma = 2.0 / max(T, 1e-10) * notional_var * remaining

    # Gamma: dollar_gamma / S²
    gamma = dollar_gamma / (spot ** 2) if spot > 0 else 0.0

    # Delta: from log-contract = -2/T × notional × (S - K_ref) / S²
    # At inception with K_ref = spot: delta ≈ 0
    delta = 0.0  # zero at inception, builds with spot moves

    # Vega: dPV/dσ = 2·σ·notional_var · DF · remaining_fraction.
    # Fix T4-VDA1: pre-fix the formula was
    #     vega = 2 * vol * T * notional_var * remaining
    # which carried a spurious ``T`` factor — a 2y var swap was
    # reported with 2× the vega of a 1y swap at the same vega_notional,
    # but the canonical convention (notional_var = vega_notional /
    # (2·√strike_var)) is designed so vega ≈ vega_notional at ATM
    # inception regardless of T.  The missing discount factor also
    # underweighted long-dated, high-rate vegas.  Both corrected.
    vega = 2.0 * vol * notional_var * df_remaining * remaining

    # Theta: daily variance accrual = -vol² / (365 × T) × notional
    theta = -vol ** 2 / 365 * notional_var * remaining

    return VarianceSwapGreeks(pv, delta, gamma, vega, theta, dollar_gamma)


# ---------------------------------------------------------------------------
# 2. Forward variance curve
# ---------------------------------------------------------------------------

@dataclass
class ForwardVarianceCurve:
    """Term structure of forward variance."""
    dates: list[date]
    forward_vars: list[float]   # forward variance for each period
    total_vars: list[float]     # cumulative total variance σ²T

    def forward_vol(self, idx: int) -> float:
        """Forward vol for period idx."""
        return math.sqrt(max(self.forward_vars[idx], 0.0))

    def term_vol(self, idx: int) -> float:
        """Term vol up to period idx."""
        if idx < 0 or idx >= len(self.total_vars):
            return 0.0
        T = (idx + 1)  # simplified: assume unit periods
        return math.sqrt(max(self.total_vars[idx] / max(T, 1), 0.0))

    def to_dict(self) -> dict:
        return {
            "dates": [d.isoformat() for d in self.dates],
            "forward_vars": self.forward_vars,
            "forward_vols": [math.sqrt(max(v, 0)) for v in self.forward_vars],
        }


def forward_variance_curve(
    atm_vols: list[tuple[float, float]],
) -> ForwardVarianceCurve:
    """Extract forward variance from ATM vol term structure.

    Forward var(T1, T2) = (σ²₂T₂ - σ²₁T₁) / (T₂ - T₁).

    Args:
        atm_vols: list of (time_to_expiry, atm_vol) sorted by time.
    """
    from datetime import timedelta

    ref = date.today()
    dates = []
    total_vars = []
    forward_vars = []

    prev_total_var = 0.0
    prev_T = 0.0

    for T, vol in sorted(atm_vols):
        total_var = vol ** 2 * T
        total_vars.append(total_var)

        dt = T - prev_T
        if dt > 0:
            fwd_var = (total_var - prev_total_var) / dt
        else:
            fwd_var = vol ** 2
        forward_vars.append(max(fwd_var, 0.0))

        dates.append(ref + timedelta(days=int(T * 365)))
        prev_total_var = total_var
        prev_T = T

    return ForwardVarianceCurve(dates, forward_vars, total_vars)


# ---------------------------------------------------------------------------
# 3. Volatility swap (pays √realised_var, not var itself)
# ---------------------------------------------------------------------------

@dataclass
class VolSwapResult:
    """Volatility swap pricing result."""
    fair_vol: float            # fair delivery vol
    pv: float                  # mark-to-market
    convexity_adjustment: float  # difference: E[√V] vs √E[V]

    def to_dict(self) -> dict:
        return {"fair_vol": self.fair_vol, "pv": self.pv,
                "convexity_adj": self.convexity_adjustment}


def volatility_swap_price(
    strike_vol: float,
    atm_vol: float,
    vol_of_vol: float = 0.0,
    T: float = 1.0,
    notional: float = 1_000_000,
    r: float = 0.0,
) -> VolSwapResult:
    """Price a volatility swap (distinct from variance swap).

    Vol swap pays: notional × (realised_vol - strike_vol).
    Fair vol ≈ ATM_vol - vol_of_vol² / (8 × ATM_vol) (convexity adjustment).

    The convexity adjustment arises because E[√V] < √E[V] (Jensen's inequality).
    For Heston: adjustment ≈ ξ² / (8σ) where ξ = vol-of-vol.
    """
    # Convexity adjustment (Carr & Lee 2009, Eq 4.2)
    if atm_vol > 0 and vol_of_vol > 0:
        adjustment = vol_of_vol ** 2 / (8 * atm_vol)
    else:
        adjustment = 0.0

    fair_vol = atm_vol - adjustment
    pv = (fair_vol - strike_vol) * notional * math.exp(-r * T)

    return VolSwapResult(fair_vol, pv, adjustment)


# ---------------------------------------------------------------------------
# 4. Dispersion trading
# ---------------------------------------------------------------------------

@dataclass
class DispersionTradeResult:
    """Dispersion trade: short index var, long constituent var."""
    index_var: float
    avg_constituent_var: float
    implied_correlation: float
    dispersion_pnl: float       # long constituents - short index
    breakeven_correlation: float

    def to_dict(self) -> dict:
        return {
            "index_var": self.index_var,
            "avg_constituent_var": self.avg_constituent_var,
            "implied_corr": self.implied_correlation,
            "pnl": self.dispersion_pnl,
            "breakeven_corr": self.breakeven_correlation,
        }


def dispersion_trade(
    index_vol: float,
    constituent_vols: list[float],
    weights: list[float] | None = None,
    notional: float = 1_000_000,
) -> DispersionTradeResult:
    """Analyse a dispersion trade (short index vol, long constituent vols).

    Dispersion profit when realised correlation < implied correlation.

    Index variance ≈ Σ wᵢ² σᵢ² + 2 Σᵢ<ⱼ wᵢ wⱼ ρ σᵢ σⱼ
    ≈ (Σ wᵢ σᵢ)² × ρ + Σ wᵢ² σᵢ² × (1-ρ)     [equicorrelation approx]

    Implied correlation: ρ = (σ²_index - Σ wᵢ² σᵢ²) / (2 Σᵢ<ⱼ wᵢ wⱼ σᵢ σⱼ)
    """
    n = len(constituent_vols)
    if weights is None:
        weights = [1.0 / n] * n

    index_var = index_vol ** 2
    weighted_var = sum(w ** 2 * v ** 2 for w, v in zip(weights, constituent_vols))
    weighted_vol = sum(w * v for w, v in zip(weights, constituent_vols))
    avg_constituent_var = sum(v ** 2 for v in constituent_vols) / n

    # Implied correlation from index vs constituents
    cross_term = index_var - weighted_var
    denominator = weighted_vol ** 2 - weighted_var
    implied_corr = cross_term / denominator if abs(denominator) > 1e-10 else 0.5

    # Dispersion P&L: long constituent vega - short index vega
    # At realised corr < implied: constituents pay more than index costs
    dispersion_pnl = (avg_constituent_var - index_var) * notional

    # Breakeven: correlation where P&L = 0
    breakeven = implied_corr  # at implied corr, trade is flat

    return DispersionTradeResult(
        index_var, avg_constituent_var, implied_corr,
        dispersion_pnl, breakeven,
    )


# ---------------------------------------------------------------------------
# 5. Bates characteristic function (Heston + jumps for Fourier pricing)
# ---------------------------------------------------------------------------

def bates_characteristic_function(
    u: complex,
    S: float, K: float, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    lam: float, mu_j: float, sigma_j: float,
) -> complex:
    """Bates (1996) characteristic function: Heston + Merton jumps.

    dS/S = (r - q - λk)dt + √v dW₁ + J dN
    dv = κ(θ - v)dt + ξ√v dW₂
    corr(dW₁, dW₂) = ρ
    J ~ lognormal(μ_j, σ_j²)

    k = E[e^J - 1] = exp(μ_j + σ_j²/2) - 1

    References:
        Bates (1996), "Jumps and Stochastic Volatility", RFS.
        Albrecher et al. (2007), "The Little Heston Trap", Wilmott.
    """
    # Heston part (Albrecher et al. 2007 rotation for numerical stability)
    discriminant = (rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u ** 2)
    d = np.sqrt(discriminant)
    # Guard: if denominator near zero, add small epsilon for stability
    denom = kappa - rho * xi * 1j * u + d
    if abs(denom) < 1e-15:
        denom = 1e-15 + 0j
    g = (kappa - rho * xi * 1j * u - d) / denom

    # Guard log argument to avoid log(0) or log(negative)
    log_arg = (1 - g * np.exp(-d * T)) / (1 - g) if abs(1 - g) > 1e-15 else 1.0
    A = (kappa * theta / xi ** 2) * (
        (kappa - rho * xi * 1j * u - d) * T
        - 2 * np.log(log_arg)
    )
    B = ((kappa - rho * xi * 1j * u - d) / xi ** 2) * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )

    # Jump part (Merton)
    k = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1
    jump_cf = lam * T * (
        np.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u ** 2 + 1j * u * np.log(1 + k)) - 1
        - 1j * u * k
    )

    x = np.log(S / K)
    cf = np.exp(A + B * v0 + 1j * u * (x + (r - q) * T) + jump_cf)
    return cf


def bates_price(
    S: float, K: float, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    lam: float, mu_j: float, sigma_j: float,
    option_type: str = "call",
    n_points: int = 256,
) -> float:
    """Price European option under Bates (Heston + jumps) via COS method.

    Uses characteristic function with COS expansion for fast pricing.
    """
    from pricebook.models.cos_method import cos_price

    def cf(u):
        return bates_characteristic_function(
            u, S, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, sigma_j)

    # COS method bounds (Fang-Oosterlee)
    c1 = (r - q) * T + (1 - math.exp(-kappa * T)) * (theta - v0) / (2 * kappa) - 0.5 * theta * T
    c2 = (1 / (2 * kappa)) * (xi ** 2 * T * kappa * theta / 2 + v0 * xi ** 2 * (1 - math.exp(-kappa * T)) / kappa)
    c2 = max(c2, v0 * T)

    L = 12  # truncation range
    a = c1 - L * math.sqrt(abs(c2))
    b = c1 + L * math.sqrt(abs(c2))

    return cos_price(cf, S, K, T, r, a, b, n_points, option_type)


# ---------------------------------------------------------------------------
# 6. SVI calibration (Gatheral 2004)
# ---------------------------------------------------------------------------

@dataclass
class SVIParams:
    """SVI raw parameters: w(k) = a + b(ρ(k-m) + √((k-m)² + σ²)).

    Where w = σ²_BS × T (total implied variance) and k = log(K/F) (log-moneyness).
    """
    a: float       # vertical shift (overall variance level)
    b: float       # slope (positive, controls wings)
    rho: float     # rotation (-1 < ρ < 1, skew direction)
    m: float       # translation (shifts smile horizontally)
    sigma: float   # smoothing (ATM curvature, σ > 0)
    T: float       # time to expiry

    def total_variance(self, k: float) -> float:
        """Total implied variance w(k) at log-moneyness k."""
        return self.a + self.b * (self.rho * (k - self.m) + math.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def implied_vol(self, k: float) -> float:
        """Black-Scholes implied vol at log-moneyness k."""
        w = self.total_variance(k)
        if w <= 0 or self.T <= 0:
            return 0.0
        return math.sqrt(w / self.T)

    def is_arbitrage_free(self) -> bool:
        """Check Gatheral no-arbitrage conditions."""
        # Condition 1: b ≥ 0
        if self.b < 0:
            return False
        # Condition 2: a + b × σ × √(1 - ρ²) ≥ 0 (minimum variance ≥ 0)
        if self.a + self.b * self.sigma * math.sqrt(1 - self.rho ** 2) < -1e-10:
            return False
        # Condition 3: b × (1 + |ρ|) ≤ 4/T (slope bound for no butterfly arb)
        if self.T > 0 and self.b * (1 + abs(self.rho)) > 4.0 / self.T + 1e-10:
            return False
        return True

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "rho": self.rho,
                "m": self.m, "sigma": self.sigma, "T": self.T,
                "arb_free": self.is_arbitrage_free()}


def svi_calibrate(
    log_moneyness: list[float],
    total_variances: list[float],
    T: float,
    initial_guess: dict | None = None,
) -> SVIParams:
    """Calibrate SVI raw parameters to market smile.

    Minimises: Σ (w_market(k) - w_SVI(k))².

    Args:
        log_moneyness: k = log(K/F) for each strike.
        total_variances: w = σ²_BS × T at each strike.
        T: time to expiry.
    """
    from scipy.optimize import minimize

    k = np.array(log_moneyness)
    w_market = np.array(total_variances)

    def objective(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma < 0.001 or abs(rho) >= 1:
            return 1e10
        w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        return float(np.sum((w_model - w_market) ** 2))

    # Initial guess
    if initial_guess is None:
        atm_var = float(np.interp(0, k, w_market))
        x0 = [atm_var, 0.1, -0.3, 0.0, 0.1]
    else:
        x0 = [initial_guess.get("a", 0.04), initial_guess.get("b", 0.1),
               initial_guess.get("rho", -0.3), initial_guess.get("m", 0.0),
               initial_guess.get("sigma", 0.1)]

    bounds = [(-1, 2), (0.001, 5), (-0.999, 0.999), (-2, 2), (0.001, 5)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    a, b, rho, m, sigma = result.x
    return SVIParams(a, b, rho, m, sigma, T)
