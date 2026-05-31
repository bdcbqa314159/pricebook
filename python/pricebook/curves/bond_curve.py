"""Bootstrap discount/yield curves from bond prices alone.

When only government or corporate bond prices are available (no swaps,
no deposits), this module extracts the underlying discount curve.

Three methods:
1. **Sequential stripping**: one bond per pillar, exact fit.
2. **Global fit**: least-squares, robust to noise/illiquidity.
3. **Parametric (Nelson-Siegel / Svensson)**: smooth curve from bond prices.

    from pricebook.curves.bond_curve import (
        bootstrap_curve_from_bonds, BondQuote,
    )

    quotes = [BondQuote(maturity, coupon, dirty_price), ...]
    curve = bootstrap_curve_from_bonds(ref_date, quotes, method="auto")

References:
    Hagan & West (2006). Interpolation Methods for Curve Construction.
    Nelson & Siegel (1987). Parsimonious Modeling of Yield Curves.
    Svensson (1994). Estimating and Interpreting Forward Interest Rates.
    Ferstl & Hayden (2010). Zero-Coupon Yield Curve Estimation with the
        Package 'termstrc' in R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import minimize, brentq

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.schedule import Frequency, generate_schedule


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

@dataclass
class BondQuote:
    """A bond price observation for yield curve construction.

    Args:
        maturity: bond maturity date.
        coupon: annual coupon rate (e.g. 0.05 = 5%).
        dirty_price: observed dirty price per 100 face.
        frequency: coupon payments per year (1, 2, or 4).
        weight: fitting weight (lower for illiquid / off-the-run).
        is_on_the_run: True for benchmark bonds (higher weight in fitting).
    """
    maturity: date
    coupon: float
    dirty_price: float
    frequency: int = 2
    weight: float = 1.0
    is_on_the_run: bool = False

    def to_dict(self) -> dict:
        return {**vars(self), "maturity": self.maturity.isoformat()}


@dataclass
class BondCurveResult:
    """Result of yield curve extraction from bonds."""
    discount_curve: DiscountCurve
    pillar_dates: list[date]
    pillar_zeros: list[float]        # zero rates at pillars
    fitted_prices: list[float]
    market_prices: list[float]
    residuals_bp: list[float]        # (model - market) in bp of par
    rmse_bp: float
    max_error_bp: float
    n_bonds: int
    method: str
    converged: bool
    parameters: dict = field(default_factory=dict)  # for parametric methods

    def to_dict(self) -> dict:
        return {
            "pillar_dates": [d.isoformat() for d in self.pillar_dates],
            "pillar_zeros": self.pillar_zeros,
            "rmse_bp": self.rmse_bp,
            "max_error_bp": self.max_error_bp,
            "n_bonds": self.n_bonds,
            "method": self.method,
            "converged": self.converged,
        }


# ═══════════════════════════════════════════════════════════════
# Bond pricing given a discount curve
# ═══════════════════════════════════════════════════════════════

def _price_bond(
    reference_date: date,
    maturity: date,
    coupon: float,
    frequency: int,
    discount_curve: DiscountCurve,
) -> float:
    """Price a risk-free coupon bond per 100 face.

    PV = Σ (coupon × τ × df(t_i)) + 100 × df(T)
    """
    freq_map = {1: Frequency.ANNUAL, 2: Frequency.SEMI_ANNUAL, 4: Frequency.QUARTERLY}
    freq = freq_map.get(frequency, Frequency.SEMI_ANNUAL)
    schedule = generate_schedule(reference_date, maturity, freq)
    dc = DayCountConvention.ACT_365_FIXED

    pv = 0.0
    for i in range(1, len(schedule)):
        tau = year_fraction(schedule[i - 1], schedule[i], dc)
        df = discount_curve.df(schedule[i])
        pv += coupon * tau * 100 * df

    pv += 100 * discount_curve.df(maturity)
    return pv


# ═══════════════════════════════════════════════════════════════
# Method 1: Sequential stripping
# ═══════════════════════════════════════════════════════════════

def _bootstrap_sequential(
    reference_date: date,
    quotes: list[BondQuote],
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> BondCurveResult:
    """Sequential bond curve bootstrap — one bond per pillar.

    Sorts bonds by maturity. For each bond, solves for the discount factor
    at its maturity that reprices the bond, using already-bootstrapped
    shorter pillars.

    Requires distinct maturities and well-ordered bonds.
    """
    sorted_quotes = sorted(quotes, key=lambda q: q.maturity)
    pillar_dates = []
    pillar_dfs = []

    for q in sorted_quotes:
        freq_map = {1: Frequency.ANNUAL, 2: Frequency.SEMI_ANNUAL, 4: Frequency.QUARTERLY}
        freq = freq_map.get(q.frequency, Frequency.SEMI_ANNUAL)
        schedule = generate_schedule(reference_date, q.maturity, freq)
        dc = DayCountConvention.ACT_365_FIXED

        # PV of known cashflows (coupons before maturity, using already-known DFs)
        known_pv = 0.0
        for i in range(1, len(schedule)):
            t_end = schedule[i]
            tau = year_fraction(schedule[i - 1], schedule[i], dc)
            if t_end < q.maturity or (pillar_dates and t_end <= pillar_dates[-1]):
                # Use already-bootstrapped curve for these cashflows
                if pillar_dates:
                    trial_curve = DiscountCurve(reference_date, pillar_dates, pillar_dfs,
                                                 interpolation=interpolation)
                    df = trial_curve.df(t_end)
                else:
                    # No pillars yet — approximate with flat rate
                    t_yf = year_fraction(reference_date, t_end, dc)
                    df = 1.0  # will be refined
                known_pv += coupon_cf(q.coupon, tau) * df

        # Remaining: coupon at maturity + principal, discounted at unknown df(T)
        # P = known_pv + (coupon_at_mat + 100) × df(T)
        # Solve: df(T) = (P - known_pv) / (coupon_at_mat + 100)
        last_tau = year_fraction(schedule[-2], schedule[-1], dc) if len(schedule) > 1 else 1.0 / q.frequency
        final_cf = q.coupon * last_tau * 100 + 100  # last coupon + principal
        remaining = q.dirty_price - known_pv
        df_T = remaining / final_cf if final_cf > 0 else 0.5

        # Clamp to valid range
        df_T = max(0.001, min(1.0, df_T))

        pillar_dates.append(q.maturity)
        pillar_dfs.append(df_T)

    curve = DiscountCurve(reference_date, pillar_dates, pillar_dfs,
                           interpolation=interpolation)

    # Compute fitted prices and diagnostics
    fitted, market, residuals = _compute_diagnostics(reference_date, quotes, curve)

    zeros = [curve.zero_rate(d) for d in pillar_dates]

    return BondCurveResult(
        curve, pillar_dates, zeros, fitted, market, residuals,
        _rmse(residuals), _max_err(residuals), len(quotes),
        "sequential", True,
    )


def coupon_cf(coupon_rate: float, tau: float) -> float:
    return coupon_rate * tau * 100


# ═══════════════════════════════════════════════════════════════
# Method 2: Global least-squares
# ═══════════════════════════════════════════════════════════════

def _bootstrap_global(
    reference_date: date,
    quotes: list[BondQuote],
    n_pillars: int | None = None,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> BondCurveResult:
    """Global least-squares bond curve bootstrap.

    Fits n_pillars zero rates simultaneously to minimise weighted
    pricing errors across all bonds.
    """
    sorted_mats = sorted(set(q.maturity for q in quotes))
    if n_pillars is None:
        n_pillars = min(len(sorted_mats), 10)

    # Pillar placement: evenly spaced across maturity range
    if n_pillars >= len(sorted_mats):
        pillar_dates = sorted_mats
    else:
        indices = np.linspace(0, len(sorted_mats) - 1, n_pillars, dtype=int)
        pillar_dates = [sorted_mats[i] for i in indices]

    n_p = len(pillar_dates)
    dc = DayCountConvention.ACT_365_FIXED
    pillar_years = [year_fraction(reference_date, d, dc) for d in pillar_dates]

    def _objective(zero_rates):
        """Sum of weighted squared pricing errors."""
        dfs = [math.exp(-zero_rates[k] * pillar_years[k]) for k in range(n_p)]
        curve = DiscountCurve(reference_date, pillar_dates, dfs,
                               interpolation=interpolation)
        total = 0.0
        for q in quotes:
            model_px = _price_bond(reference_date, q.maturity, q.coupon,
                                    q.frequency, curve)
            err = (model_px - q.dirty_price) / 100
            w = q.weight * (2.0 if q.is_on_the_run else 1.0)
            total += w * err**2
        return total

    # Initial guess: flat 4%
    x0 = np.full(n_p, 0.04)
    bounds = [(-0.02, 0.30)] * n_p  # allow negative rates (e.g. EUR, JPY)

    result = minimize(_objective, x0, method="L-BFGS-B", bounds=bounds)

    opt_zeros = result.x.tolist()
    dfs = [math.exp(-opt_zeros[k] * pillar_years[k]) for k in range(n_p)]
    curve = DiscountCurve(reference_date, pillar_dates, dfs,
                           interpolation=interpolation)

    fitted, market, residuals = _compute_diagnostics(reference_date, quotes, curve)

    return BondCurveResult(
        curve, pillar_dates, opt_zeros, fitted, market, residuals,
        _rmse(residuals), _max_err(residuals), len(quotes),
        "global", result.success,
    )


# ═══════════════════════════════════════════════════════════════
# Method 3: Parametric (Nelson-Siegel / Svensson from bond prices)
# ═══════════════════════════════════════════════════════════════

def _ns_zero(t: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
    """Nelson-Siegel zero rate at maturity t."""
    if t < 1e-6:
        return beta0 + beta1
    x = t / max(tau, 0.01)
    exp_x = math.exp(-x)
    return beta0 + beta1 * (1 - exp_x) / x + beta2 * ((1 - exp_x) / x - exp_x)


def _svensson_zero(t: float, beta0: float, beta1: float, beta2: float,
                    beta3: float, tau1: float, tau2: float) -> float:
    """Svensson zero rate at maturity t."""
    if t < 1e-6:
        return beta0 + beta1
    x1 = t / max(tau1, 0.01)
    x2 = t / max(tau2, 0.01)
    e1 = math.exp(-x1)
    e2 = math.exp(-x2)
    return (beta0 + beta1 * (1 - e1) / x1
            + beta2 * ((1 - e1) / x1 - e1)
            + beta3 * ((1 - e2) / x2 - e2))


def _bootstrap_parametric(
    reference_date: date,
    quotes: list[BondQuote],
    model: str = "nelson_siegel",
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> BondCurveResult:
    """Fit Nelson-Siegel or Svensson directly from bond prices.

    Unlike the existing NS/Svensson in nelson_siegel.py (which takes
    zero rates as input), this fits the parametric form directly to
    bond prices — no intermediate zero rate extraction needed.
    """
    dc = DayCountConvention.ACT_365_FIXED
    n_curve_points = 60  # dense output curve

    if model == "nelson_siegel":
        def _make_curve(params):
            beta0, beta1, beta2, tau = params
            max_t = max(year_fraction(reference_date, q.maturity, dc) for q in quotes) + 1
            dates_out = []
            dfs_out = []
            for i in range(1, n_curve_points + 1):
                t = i * max_t / n_curve_points
                z = _ns_zero(t, beta0, beta1, beta2, tau)
                d = date.fromordinal(reference_date.toordinal() + int(t * 365))
                dates_out.append(d)
                dfs_out.append(math.exp(-z * t))
            return DiscountCurve(reference_date, dates_out, dfs_out,
                                  interpolation=interpolation)

        def _objective(params):
            beta0, beta1, beta2, tau = params
            if tau < 0.1:
                return 1e6
            curve = _make_curve(params)
            total = 0.0
            for q in quotes:
                mp = _price_bond(reference_date, q.maturity, q.coupon, q.frequency, curve)
                err = (mp - q.dirty_price) / 100
                w = q.weight * (2.0 if q.is_on_the_run else 1.0)
                total += w * err**2
            return total

        x0 = [0.04, -0.01, 0.01, 2.0]  # β0, β1, β2, τ
        bounds = [(-0.02, 0.20), (-0.10, 0.10), (-0.10, 0.10), (0.1, 10.0)]
        result = minimize(_objective, x0, method="L-BFGS-B", bounds=bounds)
        params_dict = dict(zip(["beta0", "beta1", "beta2", "tau"], result.x.tolist()))
        curve = _make_curve(result.x)

    elif model == "svensson":
        def _make_curve_sv(params):
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            max_t = max(year_fraction(reference_date, q.maturity, dc) for q in quotes) + 1
            dates_out = []
            dfs_out = []
            for i in range(1, n_curve_points + 1):
                t = i * max_t / n_curve_points
                z = _svensson_zero(t, beta0, beta1, beta2, beta3, tau1, tau2)
                d = date.fromordinal(reference_date.toordinal() + int(t * 365))
                dates_out.append(d)
                dfs_out.append(math.exp(-z * t))
            return DiscountCurve(reference_date, dates_out, dfs_out,
                                  interpolation=interpolation)

        def _objective_sv(params):
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            if tau1 < 0.1 or tau2 < 0.1:
                return 1e6
            curve = _make_curve_sv(params)
            total = 0.0
            for q in quotes:
                mp = _price_bond(reference_date, q.maturity, q.coupon, q.frequency, curve)
                err = (mp - q.dirty_price) / 100
                w = q.weight * (2.0 if q.is_on_the_run else 1.0)
                total += w * err**2
            return total

        x0 = [0.04, -0.01, 0.01, 0.01, 2.0, 5.0]
        bounds = [(-0.02, 0.20), (-0.10, 0.10), (-0.10, 0.10),
                  (-0.10, 0.10), (0.1, 10.0), (0.1, 15.0)]
        result = minimize(_objective_sv, x0, method="L-BFGS-B", bounds=bounds)
        params_dict = dict(zip(["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"],
                                result.x.tolist()))
        curve = _make_curve_sv(result.x)
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'nelson_siegel' or 'svensson'.")

    # Diagnostics
    pillar_dates = sorted(set(q.maturity for q in quotes))
    pillar_zeros = [curve.zero_rate(d) for d in pillar_dates]
    fitted, market, residuals = _compute_diagnostics(reference_date, quotes, curve)

    return BondCurveResult(
        curve, pillar_dates, pillar_zeros, fitted, market, residuals,
        _rmse(residuals), _max_err(residuals), len(quotes),
        model, result.success, params_dict,
    )


# ═══════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════

def bootstrap_curve_from_bonds(
    reference_date: date,
    quotes: list[BondQuote],
    method: str = "auto",
    n_pillars: int | None = None,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> BondCurveResult:
    """Extract a discount curve from bond prices.

    Args:
        quotes: list of BondQuote observations.
        method:
            "sequential" — exact fit, one bond per pillar (needs distinct maturities).
            "global" — least-squares, robust to noise, supports n_pillars < n_bonds.
            "nelson_siegel" — 4-parameter smooth curve fitted to bond prices.
            "svensson" — 6-parameter smooth curve (more flexible than NS).
            "auto" — sequential if ≤8 bonds with distinct maturities, else global.
        n_pillars: number of pillars for global fit (default: auto).
        interpolation: interpolation method for the discount curve.

    Returns:
        BondCurveResult with discount_curve, fitted prices, diagnostics.

    Example:
        >>> quotes = [
        ...     BondQuote(date(2025,6,1), 0.0, 98.5),     # 1Y T-Bill
        ...     BondQuote(date(2027,1,1), 0.04, 99.2),     # 3Y note
        ...     BondQuote(date(2029,1,1), 0.0425, 98.0),   # 5Y note
        ...     BondQuote(date(2034,1,1), 0.045, 96.5),    # 10Y note
        ...     BondQuote(date(2054,1,1), 0.0475, 94.0),   # 30Y bond
        ... ]
        >>> result = bootstrap_curve_from_bonds(date(2024,6,1), quotes)
        >>> curve = result.discount_curve
        >>> print(f"5Y zero: {curve.zero_rate(date(2029,6,1))*100:.2f}%")
    """
    if not quotes:
        raise ValueError("Need at least one bond quote")

    if method == "auto":
        n_distinct = len(set(q.maturity for q in quotes))
        if n_distinct == len(quotes) and len(quotes) <= 8:
            method = "sequential"
        else:
            method = "global"

    if method == "sequential":
        return _bootstrap_sequential(reference_date, quotes, interpolation)
    elif method == "global":
        return _bootstrap_global(reference_date, quotes, n_pillars, interpolation)
    elif method in ("nelson_siegel", "svensson"):
        return _bootstrap_parametric(reference_date, quotes, method, interpolation)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sequential', 'global', "
                         f"'nelson_siegel', 'svensson', or 'auto'.")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _compute_diagnostics(
    reference_date: date,
    quotes: list[BondQuote],
    curve: DiscountCurve,
) -> tuple[list[float], list[float], list[float]]:
    """Compute fitted prices, market prices, and residuals in bp."""
    fitted = []
    market = []
    residuals = []
    for q in quotes:
        mp = _price_bond(reference_date, q.maturity, q.coupon, q.frequency, curve)
        fitted.append(mp)
        market.append(q.dirty_price)
        residuals.append((mp - q.dirty_price) * 100)  # bp of par
    return fitted, market, residuals


def _rmse(residuals: list[float]) -> float:
    if not residuals:
        return 0.0
    return math.sqrt(sum(r**2 for r in residuals) / len(residuals))


def _max_err(residuals: list[float]) -> float:
    return max(abs(r) for r in residuals) if residuals else 0.0
