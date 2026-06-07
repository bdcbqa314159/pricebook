"""American options on multiple assets with early exercise.

Covers spread, basket, best-of, and worst-of payoffs.  European multi-asset
options exist elsewhere in the library; this module adds the early-exercise
(American) layer via Longstaff-Schwartz Monte Carlo (LSM).

Payoffs:
    Spread   : max(S1 - S2 - K, 0)  [call] or max(K - (S1 - S2), 0) [put]
    Basket   : max(Σ wᵢ Sᵢ - K, 0)
    Best-of  : max(max(S1, S2) - K, 0)   (call only)
    Worst-of : max(K - min(S1, S2), 0)   (put only, key structured-product risk)

All pricers return dedicated dataclasses and additionally expose the
early-exercise premium = American price - European price.

    from pricebook.options.american_multi_asset import (
        american_spread_option,
        american_basket_option,
        american_best_of,
        american_worst_of_put,
    )

References:
    Longstaff, F. A. & Schwartz, E. S. (2001). Valuing American Options by
        Simulation: A Simple Least-Squares Approach. Review of Financial
        Studies 14(1), 113-147.
    Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering,
        Chapter 8. Springer.
    Kirk, E. (1995). Correlation in the Energy Markets. Managing Energy
        Price Risk. Risk Publications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AmericanMultiAssetResult:
    """Result for a two-asset American spread option."""

    price: float
    """American option price."""

    delta_1: float
    """Finite-difference delta with respect to S1."""

    delta_2: float
    """Finite-difference delta with respect to S2."""

    early_exercise_premium: float
    """American price minus European price."""

    european_price: float
    """European price (Kirk approximation for spreads)."""

    exercise_boundary_description: str
    """Human-readable summary of early exercise region."""

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "early_exercise_premium": self.early_exercise_premium,
            "european_price": self.european_price,
            "exercise_boundary_description": self.exercise_boundary_description,
        }


@dataclass
class AmericanBasketResult:
    """Result for an American basket option."""

    price: float
    delta: list[float]
    """Per-asset deltas."""
    early_exercise_premium: float
    european_price: float
    exercise_boundary_description: str

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "delta": self.delta,
            "early_exercise_premium": self.early_exercise_premium,
            "european_price": self.european_price,
            "exercise_boundary_description": self.exercise_boundary_description,
        }


@dataclass
class AmericanBestOfResult:
    """Result for an American option on max(S1, S2)."""

    price: float
    delta_1: float
    delta_2: float
    early_exercise_premium: float
    european_price: float
    exercise_boundary_description: str

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "early_exercise_premium": self.early_exercise_premium,
            "european_price": self.european_price,
            "exercise_boundary_description": self.exercise_boundary_description,
        }


@dataclass
class AmericanWorstOfResult:
    """Result for an American put on min(S1, S2)."""

    price: float
    delta_1: float
    delta_2: float
    early_exercise_premium: float
    european_price: float
    exercise_boundary_description: str

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "early_exercise_premium": self.early_exercise_premium,
            "european_price": self.european_price,
            "exercise_boundary_description": self.exercise_boundary_description,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _correlated_gbm(
    spots: np.ndarray,
    vols: np.ndarray,
    corr_matrix: np.ndarray,
    r: float,
    dividends: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate correlated GBM paths.

    Returns
    -------
    paths : shape (n_assets, n_paths, n_steps+1)
        Asset prices along each path and time step.
    """
    n_assets = len(spots)
    dt = T / n_steps
    # Cholesky factorisation of correlation matrix
    L = np.linalg.cholesky(corr_matrix)

    paths = np.empty((n_assets, n_paths, n_steps + 1))
    paths[:, :, 0] = spots[:, np.newaxis]

    for step in range(n_steps):
        Z = rng.standard_normal((n_assets, n_paths))
        corr_Z = L @ Z                          # shape (n_assets, n_paths)
        for a in range(n_assets):
            drift = (r - dividends[a] - 0.5 * vols[a] ** 2) * dt
            diffusion = vols[a] * math.sqrt(dt) * corr_Z[a]
            paths[a, :, step + 1] = paths[a, :, step] * np.exp(drift + diffusion)

    return paths


def _lsm(
    payoffs: np.ndarray,
    discount_factor: float,
    basis_fn,
) -> float:
    """Generic LSM backward induction.

    Parameters
    ----------
    payoffs : shape (n_paths, n_steps) exercise value at each step/path.
    discount_factor : per-step discount factor exp(-r dt).
    basis_fn : callable(state_at_step) -> design matrix (n_paths, n_basis).

    Returns
    -------
    Estimated option price (mean discounted cashflow at t=0).
    """
    n_paths, n_steps = payoffs.shape
    cashflow = payoffs[:, -1].copy()

    for step in range(n_steps - 2, -1, -1):
        cashflow *= discount_factor          # discount one step forward
        ev = payoffs[:, step]
        itm = ev > 0.0
        if itm.sum() < 20:
            continue
        X = basis_fn(step, itm)
        y = cashflow[itm]
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            cont = X @ coef                  # continuation for ITM paths
        except np.linalg.LinAlgError:
            continue
        exercise_now = itm.copy()
        exercise_now[itm] = ev[itm] > cont
        cashflow[exercise_now] = ev[exercise_now]

    return float(np.mean(cashflow) * discount_factor)


def _european_price_mc(payoffs_terminal: np.ndarray, df_total: float) -> float:
    """European price from terminal payoffs."""
    return float(np.mean(payoffs_terminal) * df_total)


# ---------------------------------------------------------------------------
# Kirk approximation for European spread option
# ---------------------------------------------------------------------------

def _kirk_spread_european(
    S1: float, S2: float, K: float,
    vol1: float, vol2: float, rho: float,
    T: float, r: float, q1: float, q2: float,
    option_type: str,
) -> float:
    """Kirk (1995) approximation for European spread option."""
    F1 = S1 * math.exp((r - q1) * T)
    F2 = S2 * math.exp((r - q2) * T)
    FK = F2 + K
    sigma = math.sqrt(
        vol1 ** 2
        + (vol2 * F2 / FK) ** 2
        - 2.0 * rho * vol1 * vol2 * F2 / FK
    )
    if sigma < 1e-10 or T < 1e-10:
        if option_type == "call":
            return max(F1 - F2 - K, 0.0) * math.exp(-r * T)
        else:
            return max(K + F2 - F1, 0.0) * math.exp(-r * T)
    d1 = (math.log(F1 / FK) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    def _N(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    df = math.exp(-r * T)
    if option_type == "call":
        return df * (F1 * _N(d1) - FK * _N(d2))
    else:
        return df * (FK * _N(-d2) - F1 * _N(-d1))


# ---------------------------------------------------------------------------
# 1. American spread option
# ---------------------------------------------------------------------------

def american_spread_option(
    S1: float,
    S2: float,
    strike: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    q1: float = 0.0,
    q2: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> AmericanMultiAssetResult:
    """American spread option via LSM.

    Prices max(S1 - S2 - K, 0) [call] or max(K - (S1 - S2), 0) [put]
    with early exercise at any time before expiry.

    Regression basis at each exercise step:
        [1, S1, S2, S1², S2², S1·S2]

    The early-exercise premium is compared against the Kirk (1995)
    European approximation.

    Parameters
    ----------
    S1, S2       : current asset prices.
    strike       : spread strike K.
    vol1, vol2   : asset volatilities.
    rho          : correlation between S1 and S2.
    T            : time to expiry in years.
    r            : risk-free rate (continuously compounded).
    q1, q2       : continuous dividend yields.
    option_type  : "call" or "put".
    n_paths      : number of Monte Carlo paths.
    n_steps      : number of time steps.
    seed         : random seed.

    Returns
    -------
    AmericanMultiAssetResult
    """
    rng = np.random.default_rng(seed)
    spots = np.array([S1, S2])
    vols = np.array([vol1, vol2])
    divs = np.array([q1, q2])
    corr = np.array([[1.0, rho], [rho, 1.0]])
    dt = T / n_steps
    df_step = math.exp(-r * dt)
    df_total = math.exp(-r * T)

    paths = _correlated_gbm(spots, vols, corr, r, divs, T, n_steps, n_paths, rng)
    p1 = paths[0]   # shape (n_paths, n_steps+1)
    p2 = paths[1]

    # Payoffs at each step (excluding t=0)
    if option_type == "call":
        payoffs = np.maximum(p1[:, 1:] - p2[:, 1:] - strike, 0.0)
    else:
        payoffs = np.maximum(strike - (p1[:, 1:] - p2[:, 1:]), 0.0)

    def basis(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = p1[mask, step + 1]
        x2 = p2[mask, step + 1]
        return np.column_stack([
            np.ones(mask.sum()), x1, x2, x1 * x1, x2 * x2, x1 * x2,
        ])

    price = _lsm(payoffs, df_step, basis)

    # European via Kirk approximation
    european_price = _kirk_spread_european(
        S1, S2, strike, vol1, vol2, rho, T, r, q1, q2, option_type
    )
    early_exercise_premium = price - european_price

    # Finite-difference deltas (bump S1 and S2 by 0.5%)
    bump = 0.01
    rng_u = np.random.default_rng(seed)
    paths_u = _correlated_gbm(
        np.array([S1 * (1 + bump), S2]), vols, corr, r, divs, T, n_steps, n_paths, rng_u
    )
    if option_type == "call":
        payoffs_u = np.maximum(paths_u[0, :, 1:] - paths_u[1, :, 1:] - strike, 0.0)
    else:
        payoffs_u = np.maximum(strike - (paths_u[0, :, 1:] - paths_u[1, :, 1:]), 0.0)

    def basis_u(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = paths_u[0, mask, step + 1]
        x2 = paths_u[1, mask, step + 1]
        return np.column_stack([
            np.ones(mask.sum()), x1, x2, x1 * x1, x2 * x2, x1 * x2,
        ])

    price_u1 = _lsm(payoffs_u, df_step, basis_u)

    rng_d = np.random.default_rng(seed)
    paths_d = _correlated_gbm(
        np.array([S1, S2 * (1 + bump)]), vols, corr, r, divs, T, n_steps, n_paths, rng_d
    )
    if option_type == "call":
        payoffs_d = np.maximum(paths_d[0, :, 1:] - paths_d[1, :, 1:] - strike, 0.0)
    else:
        payoffs_d = np.maximum(strike - (paths_d[0, :, 1:] - paths_d[1, :, 1:]), 0.0)

    def basis_d(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = paths_d[0, mask, step + 1]
        x2 = paths_d[1, mask, step + 1]
        return np.column_stack([
            np.ones(mask.sum()), x1, x2, x1 * x1, x2 * x2, x1 * x2,
        ])

    price_d2 = _lsm(payoffs_d, df_step, basis_d)

    delta_1 = (price_u1 - price) / (S1 * bump)
    delta_2 = (price_d2 - price) / (S2 * bump)

    desc = (
        f"Early exercise optimal when spread > {strike + early_exercise_premium:.4f}; "
        f"premium = {early_exercise_premium:.6f}"
    )

    return AmericanMultiAssetResult(
        price=price,
        delta_1=delta_1,
        delta_2=delta_2,
        early_exercise_premium=early_exercise_premium,
        european_price=european_price,
        exercise_boundary_description=desc,
    )


# ---------------------------------------------------------------------------
# 2. American basket option
# ---------------------------------------------------------------------------

def american_basket_option(
    spots: list[float],
    vols: list[float],
    correlations: list[list[float]],
    weights: list[float],
    strike: float,
    T: float,
    r: float,
    dividends: list[float] | None = None,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> AmericanBasketResult:
    """American basket option via LSM.

    Basket payoff: max(Σ wᵢ Sᵢ(t) - K, 0) [call] or max(K - Σ wᵢ Sᵢ(t), 0) [put].
    Regression basis: [1, B, B², B³] where B = Σ wᵢ Sᵢ is the basket value.

    Parameters
    ----------
    spots        : list of initial asset prices.
    vols         : list of asset volatilities.
    correlations : n_assets × n_assets correlation matrix (list of lists).
    weights      : basket weights (need not sum to 1).
    strike       : basket strike.
    T            : time to expiry in years.
    r            : risk-free rate.
    dividends    : continuous dividend yields; defaults to zero.
    option_type  : "call" or "put".
    n_paths      : number of Monte Carlo paths.
    n_steps      : number of time steps.
    seed         : random seed.

    Returns
    -------
    AmericanBasketResult
    """
    rng = np.random.default_rng(seed)
    n_assets = len(spots)
    spots_arr = np.array(spots, dtype=float)
    vols_arr = np.array(vols, dtype=float)
    weights_arr = np.array(weights, dtype=float)
    divs_arr = np.zeros(n_assets) if dividends is None else np.array(dividends, dtype=float)
    corr = np.array(correlations, dtype=float)

    dt = T / n_steps
    df_step = math.exp(-r * dt)
    df_total = math.exp(-r * T)

    paths = _correlated_gbm(spots_arr, vols_arr, corr, r, divs_arr, T, n_steps, n_paths, rng)
    # paths shape: (n_assets, n_paths, n_steps+1)

    # Basket value at each step (excluding t=0)
    basket = np.einsum("a,aps->ps", weights_arr, paths[:, :, 1:])
    # basket shape: (n_paths, n_steps)

    if option_type == "call":
        payoffs = np.maximum(basket - strike, 0.0)
    else:
        payoffs = np.maximum(strike - basket, 0.0)

    def basis(step: int, mask: np.ndarray) -> np.ndarray:
        b = basket[mask, step]
        return np.column_stack([np.ones(mask.sum()), b, b * b, b ** 3])

    price = _lsm(payoffs, df_step, basis)

    # European MC price (terminal payoff)
    european_price = _european_price_mc(payoffs[:, -1], df_total)
    early_exercise_premium = price - european_price

    # Per-asset deltas via finite difference
    bump = 0.01
    delta = []
    for a in range(n_assets):
        spots_b = spots_arr.copy()
        spots_b[a] *= (1 + bump)
        rng_b = np.random.default_rng(seed)
        paths_b = _correlated_gbm(spots_b, vols_arr, corr, r, divs_arr, T, n_steps, n_paths, rng_b)
        basket_b = np.einsum("a,aps->ps", weights_arr, paths_b[:, :, 1:])
        if option_type == "call":
            pay_b = np.maximum(basket_b - strike, 0.0)
        else:
            pay_b = np.maximum(strike - basket_b, 0.0)

        def basis_b(step: int, mask: np.ndarray, _b=basket_b) -> np.ndarray:
            bv = _b[mask, step]
            return np.column_stack([np.ones(mask.sum()), bv, bv * bv, bv ** 3])

        price_b = _lsm(pay_b, df_step, basis_b)
        delta.append((price_b - price) / (spots_arr[a] * bump))

    desc = (
        f"Basket early exercise premium = {early_exercise_premium:.6f}; "
        f"n_assets={n_assets}, weights={list(weights_arr.round(4))}"
    )

    return AmericanBasketResult(
        price=price,
        delta=delta,
        early_exercise_premium=early_exercise_premium,
        european_price=european_price,
        exercise_boundary_description=desc,
    )


# ---------------------------------------------------------------------------
# 3. American best-of option
# ---------------------------------------------------------------------------

def american_best_of(
    S1: float,
    S2: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    q1: float = 0.0,
    q2: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> AmericanBestOfResult:
    """American option on max(S1, S2) with early exercise.

    Payoff: max(S1(t), S2(t)) at the optimal stopping time.
    No strike — the holder receives the better-performing asset value.
    Regression on (S1, S2) with polynomial basis.

    Parameters
    ----------
    S1, S2   : initial asset prices.
    vol1, vol2 : asset volatilities.
    rho      : correlation.
    T        : time to expiry.
    r        : risk-free rate.
    q1, q2   : dividend yields.
    n_paths  : Monte Carlo paths.
    n_steps  : time steps.
    seed     : random seed.

    Returns
    -------
    AmericanBestOfResult
    """
    rng = np.random.default_rng(seed)
    spots = np.array([S1, S2])
    vols = np.array([vol1, vol2])
    divs = np.array([q1, q2])
    corr = np.array([[1.0, rho], [rho, 1.0]])

    dt = T / n_steps
    df_step = math.exp(-r * dt)
    df_total = math.exp(-r * T)

    paths = _correlated_gbm(spots, vols, corr, r, divs, T, n_steps, n_paths, rng)
    p1 = paths[0]   # (n_paths, n_steps+1)
    p2 = paths[1]

    # Payoff: max(S1, S2) — holder always exercises (no strike), so
    # all paths are "in the money"; comparison is to continuation value
    payoffs = np.maximum(p1[:, 1:], p2[:, 1:])

    def basis(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = p1[mask, step + 1]
        x2 = p2[mask, step + 1]
        return np.column_stack([
            np.ones(mask.sum()), x1, x2, x1 * x1, x2 * x2, x1 * x2,
        ])

    price = _lsm(payoffs, df_step, basis)

    # European best-of: Stulz (1982) closed-form would be exact;
    # use MC terminal payoff as European benchmark
    european_price = _european_price_mc(payoffs[:, -1], df_total)
    early_exercise_premium = price - european_price

    # Deltas
    bump = 0.01
    rng_u1 = np.random.default_rng(seed)
    paths_u1 = _correlated_gbm(
        np.array([S1 * (1 + bump), S2]), vols, corr, r, divs, T, n_steps, n_paths, rng_u1
    )
    pay_u1 = np.maximum(paths_u1[0, :, 1:], paths_u1[1, :, 1:])

    def basis_u1(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = paths_u1[0, mask, step + 1]
        x2 = paths_u1[1, mask, step + 1]
        return np.column_stack([np.ones(mask.sum()), x1, x2, x1*x1, x2*x2, x1*x2])

    price_u1 = _lsm(pay_u1, df_step, basis_u1)

    rng_u2 = np.random.default_rng(seed)
    paths_u2 = _correlated_gbm(
        np.array([S1, S2 * (1 + bump)]), vols, corr, r, divs, T, n_steps, n_paths, rng_u2
    )
    pay_u2 = np.maximum(paths_u2[0, :, 1:], paths_u2[1, :, 1:])

    def basis_u2(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = paths_u2[0, mask, step + 1]
        x2 = paths_u2[1, mask, step + 1]
        return np.column_stack([np.ones(mask.sum()), x1, x2, x1*x1, x2*x2, x1*x2])

    price_u2 = _lsm(pay_u2, df_step, basis_u2)

    delta_1 = (price_u1 - price) / (S1 * bump)
    delta_2 = (price_u2 - price) / (S2 * bump)

    desc = (
        f"Best-of: hold for max(S1,S2); early exercise premium = {early_exercise_premium:.6f}; "
        f"optimal when high-asset drift makes waiting suboptimal"
    )

    return AmericanBestOfResult(
        price=price,
        delta_1=delta_1,
        delta_2=delta_2,
        early_exercise_premium=early_exercise_premium,
        european_price=european_price,
        exercise_boundary_description=desc,
    )


# ---------------------------------------------------------------------------
# 4. American worst-of put
# ---------------------------------------------------------------------------

def american_worst_of_put(
    S1: float,
    S2: float,
    strike: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    r: float,
    q1: float = 0.0,
    q2: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> AmericanWorstOfResult:
    """American put on min(S1, S2) via LSM.

    Payoff at exercise: max(K - min(S1(t), S2(t)), 0).

    This product is central to structured notes (worst-of autocallables)
    where investors are short this option.  Early exercise is significant
    when the worse-performing asset has high dividend yield or deep ITM.

    Regression basis: [1, S1, S2, S1², S2², S1·S2].

    Parameters
    ----------
    S1, S2   : initial asset prices.
    strike   : put strike K.
    vol1, vol2 : asset volatilities.
    rho      : correlation between S1 and S2.
    T        : time to expiry.
    r        : risk-free rate.
    q1, q2   : continuous dividend yields.
    n_paths  : Monte Carlo paths.
    n_steps  : time steps.
    seed     : random seed.

    Returns
    -------
    AmericanWorstOfResult
    """
    rng = np.random.default_rng(seed)
    spots = np.array([S1, S2])
    vols = np.array([vol1, vol2])
    divs = np.array([q1, q2])
    corr = np.array([[1.0, rho], [rho, 1.0]])

    dt = T / n_steps
    df_step = math.exp(-r * dt)
    df_total = math.exp(-r * T)

    paths = _correlated_gbm(spots, vols, corr, r, divs, T, n_steps, n_paths, rng)
    p1 = paths[0]
    p2 = paths[1]

    worst = np.minimum(p1[:, 1:], p2[:, 1:])
    payoffs = np.maximum(strike - worst, 0.0)

    def basis(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = p1[mask, step + 1]
        x2 = p2[mask, step + 1]
        return np.column_stack([
            np.ones(mask.sum()), x1, x2, x1 * x1, x2 * x2, x1 * x2,
        ])

    price = _lsm(payoffs, df_step, basis)

    european_price = _european_price_mc(payoffs[:, -1], df_total)
    early_exercise_premium = price - european_price

    # Deltas
    bump = 0.01
    rng_u1 = np.random.default_rng(seed)
    paths_u1 = _correlated_gbm(
        np.array([S1 * (1 + bump), S2]), vols, corr, r, divs, T, n_steps, n_paths, rng_u1
    )
    worst_u1 = np.minimum(paths_u1[0, :, 1:], paths_u1[1, :, 1:])
    pay_u1 = np.maximum(strike - worst_u1, 0.0)

    def basis_u1(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = paths_u1[0, mask, step + 1]
        x2 = paths_u1[1, mask, step + 1]
        return np.column_stack([np.ones(mask.sum()), x1, x2, x1*x1, x2*x2, x1*x2])

    price_u1 = _lsm(pay_u1, df_step, basis_u1)

    rng_u2 = np.random.default_rng(seed)
    paths_u2 = _correlated_gbm(
        np.array([S1, S2 * (1 + bump)]), vols, corr, r, divs, T, n_steps, n_paths, rng_u2
    )
    worst_u2 = np.minimum(paths_u2[0, :, 1:], paths_u2[1, :, 1:])
    pay_u2 = np.maximum(strike - worst_u2, 0.0)

    def basis_u2(step: int, mask: np.ndarray) -> np.ndarray:
        x1 = paths_u2[0, mask, step + 1]
        x2 = paths_u2[1, mask, step + 1]
        return np.column_stack([np.ones(mask.sum()), x1, x2, x1*x1, x2*x2, x1*x2])

    price_u2 = _lsm(pay_u2, df_step, basis_u2)

    delta_1 = (price_u1 - price) / (S1 * bump)
    delta_2 = (price_u2 - price) / (S2 * bump)

    desc = (
        f"Worst-of put: K={strike}; early exercise premium = {early_exercise_premium:.6f}; "
        f"exercise optimal when min(S1,S2) well below strike and high dividend/rate environment"
    )

    return AmericanWorstOfResult(
        price=price,
        delta_1=delta_1,
        delta_2=delta_2,
        early_exercise_premium=early_exercise_premium,
        european_price=european_price,
        exercise_boundary_description=desc,
    )
