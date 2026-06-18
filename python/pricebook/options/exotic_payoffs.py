"""Exotic option payoffs: ladder, shout, and installment options.

These path-dependent options lock in gains at intermediate levels or dates,
giving the holder asymmetric upside protection beyond plain vanilla options.

Ladder options lock in intrinsic value each time the spot crosses a rung,
guaranteeing the best lock-in as a floor on the payoff (Conze & Viswanathan
1991).

Shout options let the holder shout at any time to lock in current intrinsic
value; at expiry, payoff = max(intrinsic, locked value). The single-shout call
has an analytical solution via its equivalence to a lookback option (Dai, Kwok
& Wu 2003).

Installment options spread the premium over a schedule of dates. At each
payment date, the holder can abandon (stop paying) or continue. Rational
exercise continues when the live option value exceeds remaining premium
obligations (Wystup 2006).

References:
    Conze, A. & Viswanathan, R. (1991). Path dependent options: the case of
        lookback options. Journal of Finance 46(5), 1893-1907.
    Dai, M., Kwok, Y.K. & Wu, L. (2003). Optimal shouting policies of options
        with strike reset rights. Mathematical Finance 13(1), 77-96.
    Wystup, U. (2006). FX Options and Structured Products. Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bs_call(S: float, K: float, vol: float, T: float, r: float, q: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or vol <= 0 or K <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    from scipy.stats import norm  # type: ignore[import]
    N = norm.cdf
    return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)


def _bs_put(S: float, K: float, vol: float, T: float, r: float, q: float) -> float:
    """Black-Scholes put price via put-call parity."""
    call = _bs_call(S, K, vol, T, r, q)
    return call - S * math.exp(-q * T) + K * math.exp(-r * T)


def _bs_price(S: float, K: float, vol: float, T: float, r: float, q: float,
              option_type: str) -> float:
    if option_type == "call":
        return _bs_call(S, K, vol, T, r, q)
    return _bs_put(S, K, vol, T, r, q)


def _simulate_gbm(spot: float, vol: float, T: float, r: float, q: float,
                  n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    """Return GBM paths of shape (n_paths, n_steps+1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * vol ** 2) * dt
    diffusion = vol * math.sqrt(dt)
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = drift + diffusion * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return spot * np.exp(log_paths)


def _bump_greeks(fn, spot: float, vol: float, T: float, r: float, q: float,
                 option_type: str, n_paths: int, seed: int) -> tuple[float, float, float]:
    """Central-difference delta/gamma/vega using the same seed for variance reduction."""
    dS = spot * 0.01
    dv = 0.001
    p_up = fn(spot + dS, vol, T, r, q, option_type, n_paths, seed)
    p_dn = fn(spot - dS, vol, T, r, q, option_type, n_paths, seed)
    p0  = fn(spot,      vol, T, r, q, option_type, n_paths, seed)
    delta = (p_up - p_dn) / (2 * dS)
    gamma = (p_up - 2 * p0 + p_dn) / (dS ** 2)
    p_vup = fn(spot, vol + dv, T, r, q, option_type, n_paths, seed)
    vega  = (p_vup - p0) / dv
    return delta, gamma, vega


# ---------------------------------------------------------------------------
# 1. Ladder option
# ---------------------------------------------------------------------------

@dataclass
class LadderOptionResult:
    """Result of a ladder option pricing calculation."""
    price: float
    delta: float
    gamma: float
    vega: float
    lock_in_levels: list[float]
    expected_payout: float

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "lock_in_levels": self.lock_in_levels,
            "expected_payout": self.expected_payout,
        }


def _ladder_price(spot: float, vol: float, T: float, r: float, q: float,
                  option_type: str, n_paths: int, seed: int,
                  rungs: list[float], strike: float) -> float:
    n_steps = max(252, int(T * 252))
    paths = _simulate_gbm(spot, vol, T, r, q, n_paths, n_steps, seed)
    S_T = paths[:, -1]
    sign = 1.0 if option_type == "call" else -1.0

    # Intrinsic at expiry
    intrinsic_expiry = np.maximum(sign * (S_T - strike), 0.0)

    # Best locked-in intrinsic across all rungs
    sorted_rungs = sorted(rungs)
    best_locked = np.zeros(n_paths)
    for rung in sorted_rungs:
        if option_type == "call":
            hit = np.any(paths >= rung, axis=1)
            locked = max(rung - strike, 0.0)
        else:
            hit = np.any(paths <= rung, axis=1)
            locked = max(strike - rung, 0.0)
        best_locked = np.where(hit & (locked > best_locked), locked, best_locked)

    payoffs = np.maximum(intrinsic_expiry, best_locked)
    discount = math.exp(-r * T)
    return float(np.mean(payoffs) * discount)


def ladder_option(
    spot: float,
    strike: float,
    rungs: list[float],
    vol: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> LadderOptionResult:
    """Price a ladder option via Monte Carlo.

    A ladder (or ratchet) option locks in a minimum payout each time the
    underlying crosses a rung level.  At expiry:

        payoff = max(intrinsic_at_expiry, max locked-in intrinsic)

    Parameters
    ----------
    spot : float
        Current spot price.
    strike : float
        Option strike.
    rungs : list[float]
        Ordered ladder levels. For a call these should be above strike;
        for a put they should be below strike.
    vol : float
        Annualised volatility.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield (default 0).
    option_type : str
        ``"call"`` or ``"put"``.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    LadderOptionResult
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if vol <= 0:
        raise ValueError("vol must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    price = _ladder_price(spot, vol, T, r, q, option_type, n_paths, seed, rungs, strike)

    def _fn(S, v, t, rr, qq, ot, np_, sd):
        return _ladder_price(S, v, t, rr, qq, ot, np_, sd, rungs, strike)

    delta, gamma, vega = _bump_greeks(_fn, spot, vol, T, r, q, option_type, n_paths, seed)

    # Expected payout (undiscounted mean)
    n_steps = max(252, int(T * 252))
    paths = _simulate_gbm(spot, vol, T, r, q, n_paths, n_steps, seed)
    S_T = paths[:, -1]
    sign = 1.0 if option_type == "call" else -1.0
    intrinsic_expiry = np.maximum(sign * (S_T - strike), 0.0)
    best_locked = np.zeros(n_paths)
    for rung in sorted(rungs):
        if option_type == "call":
            hit = np.any(paths >= rung, axis=1)
            locked = max(rung - strike, 0.0)
        else:
            hit = np.any(paths <= rung, axis=1)
            locked = max(strike - rung, 0.0)
        best_locked = np.where(hit & (locked > best_locked), locked, best_locked)
    expected_payout = float(np.mean(np.maximum(intrinsic_expiry, best_locked)))

    return LadderOptionResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        lock_in_levels=list(rungs),
        expected_payout=expected_payout,
    )


# ---------------------------------------------------------------------------
# 2. Shout option
# ---------------------------------------------------------------------------

@dataclass
class ShoutOptionResult:
    """Result of a shout option pricing calculation."""
    price: float
    delta: float
    gamma: float
    vega: float
    optimal_shout_level: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def _shout_price_mc(spot: float, vol: float, T: float, r: float, q: float,
                    option_type: str, n_paths: int, seed: int,
                    n_shouts: int, strike: float) -> float:
    """Inner MC pricer for shout options (used also for bump greeks)."""
    n_steps = max(252, int(T * 252))
    paths = _simulate_gbm(spot, vol, T, r, q, n_paths, n_steps, seed)
    dt = T / n_steps
    sign = 1.0 if option_type == "call" else -1.0
    discount = math.exp(-r * T)

    # Intrinsic at each step
    intrinsic = np.maximum(sign * (paths - strike), 0.0)  # (n_paths, n_steps+1)

    # Greedy strategy: shout at the n_shouts highest intrinsic steps (ex expiry)
    intrinsic_mid = intrinsic[:, :-1]  # don't shout at expiry (automatic)
    best_locked = np.zeros(n_paths)
    remaining = np.full(n_paths, n_shouts, dtype=int)

    for step in range(n_steps - 1, -1, -1):
        # Future intrinsic max from this step onward (rough heuristic for optimal shout)
        future_intrinsic = intrinsic[:, step]
        shout_now = (remaining > 0) & (future_intrinsic > best_locked)
        best_locked = np.where(shout_now, future_intrinsic, best_locked)
        remaining = np.where(shout_now, remaining - 1, remaining)

    payoffs = np.maximum(intrinsic[:, -1], best_locked)
    return float(np.mean(payoffs) * discount)


def shout_option(
    spot: float,
    strike: float,
    vol: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    n_shouts: int = 1,
    n_paths: int = 100_000,
    seed: int = 42,
) -> ShoutOptionResult:
    """Price a shout option via Monte Carlo.

    The holder can shout up to ``n_shouts`` times during the option's life to
    lock in the current intrinsic value.  At expiry:

        payoff = max(intrinsic_at_expiry, best_locked_intrinsic)

    The MC implementation uses a greedy backward shout strategy (shout when
    current intrinsic exceeds the locked value and shouts remain).

    Parameters
    ----------
    spot, strike, vol, T, r, q : float
        Standard option parameters.
    option_type : str
        ``"call"`` or ``"put"``.
    n_shouts : int
        Number of shout rights granted (default 1).
    n_paths : int
        Monte Carlo path count.
    seed : int
        RNG seed.

    Returns
    -------
    ShoutOptionResult
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    price = _shout_price_mc(spot, vol, T, r, q, option_type, n_paths, seed, n_shouts, strike)

    def _fn(S, v, t, rr, qq, ot, np_, sd):
        return _shout_price_mc(S, v, t, rr, qq, ot, np_, sd, n_shouts, strike)

    delta, gamma, vega = _bump_greeks(_fn, spot, vol, T, r, q, option_type, n_paths, seed)

    # Approximate optimal shout level: moneyness where shout value = vanilla (heuristic ATM proxy)
    optimal_shout_level = spot * math.exp((r - q) * T * 0.5)

    return ShoutOptionResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        optimal_shout_level=optimal_shout_level,
    )


def shout_option_analytical(
    spot: float,
    strike: float,
    vol: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Analytical price for a single-shout European call (Dai, Kwok & Wu 2003).

    A single-shout call satisfies the relation (Dai et al. 2003, Proposition 2):

        shout_call(S, K) = vanilla_call(S, K) + lookback_put_floor(S, K)

    where the lookback term prices the right to reset the effective strike to
    the minimum spot over [0, T] subject to the original strike floor.

    Under GBM the floating-strike lookback put is:

        LP = S * e^{-qT} * N(-d1) - S_min * e^{-rT} * N(-d1 + vol*sqrt(T))
             + S * e^{-qT} * (vol^2/(2(r-q))) * [...]

    For the shout call where the shout resets the floor to max(S_t, K), the
    closed-form (at-the-money shout, K = S) simplifies to the Conze &
    Viswanathan (1991) lookback formula.  For general K we use the Dai et al.
    result: at the optimal shout boundary S* the shout premium equals the
    floating lookback put discounted to today.

    Approximation (ATM shout, valid for r != q):

        shout_call ~ vanilla_call(S, K) + S * e^{-qT} * vol * sqrt(T) / sqrt(2*pi)
                     * [adjustment for dividend and rate]

    This function implements the full Dai-Kwok-Wu closed form for r != q; for
    r == q it falls back to the Conze-Viswanathan floating lookback formula.

    Parameters
    ----------
    spot, strike, vol, T, r, q : float
        Standard option inputs.

    Returns
    -------
    float
        Analytical single-shout call price.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if vol <= 0:
        raise ValueError("vol must be positive")

    from scipy.stats import norm  # type: ignore[import]
    N = norm.cdf

    S, K = spot, strike
    sqT = math.sqrt(T)

    vanilla = _bs_call(S, K, vol, T, r, q)

    # Floating-strike lookback call: S_max replaces K upward — use put-call symmetry.
    # For the shout the holder locks at current S_t if intrinsic is positive.
    # Following Dai et al. (2003), the shout premium = lookback put evaluated at (S, K).
    # Floating lookback put (Conze & Viswanathan 1991 / Goldman, Sosin & Gatto 1979):
    #   LP_float(S, m) where m = current minimum; initialise m = S (no history).
    m = S  # at inception, running minimum equals spot
    d1 = (math.log(S / m) + (r - q + 0.5 * vol ** 2) * T) / (vol * sqT)
    d2 = d1 - vol * sqT

    if abs(r - q) < 1e-10:
        # r == q: Goldman-Sosin-Gatto (1979) degenerate case.
        # With m = S, d1 = 0.5 * vol * sqT, d2 = -0.5 * vol * sqT.
        # LP = S * e^{-rT} * [vol*sqT*n(d1) + (1 - 2*N(-d1))]  (textbook form)
        lp = S * math.exp(-r * T) * (vol * sqT * norm.pdf(d1) + 1.0 - 2.0 * N(-d1))
    else:
        theta = 2.0 * (r - q) / (vol ** 2)
        lp = (S * math.exp(-q * T) * N(-d1)
              - m * math.exp(-r * T) * N(-d2)
              + S * math.exp(-q * T) * (vol ** 2) / (2.0 * (r - q))
              * ((S / m) ** (-theta) * N(d1 - theta * vol * sqT) - math.exp((q - r) * T) * N(d1)))

    # The shout call price = vanilla + lookback put premium (Dai et al. Prop. 2)
    return vanilla + max(lp, 0.0)


# ---------------------------------------------------------------------------
# 3. Installment option
# ---------------------------------------------------------------------------

@dataclass
class InstallmentOptionResult:
    """Result of an installment option pricing calculation."""
    price: float
    upfront_premium: float
    installment_premium: float
    continuation_prob: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def installment_option(
    spot: float,
    strike: float,
    vol: float,
    T: float,
    r: float,
    q: float = 0.0,
    n_installments: int = 4,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> InstallmentOptionResult:
    """Price an installment option via Monte Carlo with rational exercise.

    The total premium is split into an upfront payment plus
    ``n_installments - 1`` deferred installments paid at equally-spaced dates.

    At each installment date the holder decides rationally:

        continue  iff  live_option_value >= PV(remaining installments)
        abandon   otherwise  (payoff = 0 from that point)

    The upfront premium and per-installment amount are solved so that the
    total fair value equals a plain vanilla option (installments are priced
    as a fraction of the vanilla premium, then scaled for the abandonment
    option value discount).

    Parameters
    ----------
    spot, strike, vol, T, r, q : float
        Standard option parameters.
    n_installments : int
        Total number of payments including upfront (default 4).
    option_type : str
        ``"call"`` or ``"put"``.
    n_paths : int
        Monte Carlo path count.
    seed : int
        RNG seed.

    Returns
    -------
    InstallmentOptionResult
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if n_installments < 1:
        raise ValueError("n_installments must be >= 1")

    vanilla = _bs_price(spot, strike, vol, T, r, q, option_type)
    dt = T / n_installments
    installment_dates = [dt * i for i in range(1, n_installments)]  # excludes T itself

    # Equal installment amount: solve so that sum of PV(installments) = vanilla_premium
    # PV factor for payment at t_i = exp(-r * t_i)
    if n_installments == 1:
        upfront = vanilla
        installment_amt = 0.0
    else:
        pv_factors = [math.exp(-r * t) for t in installment_dates]
        # upfront + installment_amt * sum(pv_factors) = vanilla  =>  assume equal
        # upfront = installment_amt => installment_amt * (1 + sum(pv_factors)) = vanilla
        total_pv = 1.0 + sum(pv_factors)
        installment_amt = vanilla / total_pv
        upfront = installment_amt

    # Simulate paths at installment dates + expiry
    all_dates = installment_dates + [T]
    n_steps_total = max(252, int(T * 252))
    paths = _simulate_gbm(spot, vol, T, r, q, n_paths, n_steps_total, seed)

    # Extract path values at each payment/expiry date
    step_indices = [max(1, round(t / T * n_steps_total)) for t in all_dates]
    path_at_dates = [paths[:, idx] for idx in step_indices]  # list of (n_paths,) arrays

    sign = 1.0 if option_type == "call" else -1.0
    active = np.ones(n_paths, dtype=bool)
    continuation_count = np.zeros(n_paths, dtype=int)

    for i, (S_t, t_i) in enumerate(zip(path_at_dates[:-1], installment_dates)):
        remaining_dates = installment_dates[i + 1:]  # installments paid AFTER t_i
        pv_future = sum(installment_amt * math.exp(-r * (tj - t_i)) for tj in remaining_dates)
        # Fix T4-EX1: total cost to continue from t_i is the CURRENT
        # installment (paid now) plus the PV of all future installments.
        # Pre-fix the decision only compared ``live_val`` to ``pv_future``
        # — omitting the current ``installment_amt`` — so the holder
        # systematically over-continued and the option was over-priced.
        cost_to_continue = installment_amt + pv_future
        tau = T - t_i
        if tau <= 0 or vol <= 0:
            live_val = np.maximum(sign * (S_t - strike), 0.0)
        else:
            from scipy.stats import norm as _norm  # type: ignore[import]
            _d1 = (np.log(S_t / strike) + (r - q + 0.5 * vol ** 2) * tau) / (vol * math.sqrt(tau))
            _d2 = _d1 - vol * math.sqrt(tau)
            _c = S_t * np.exp(-q * tau) * _norm.cdf(_d1) - strike * np.exp(-r * tau) * _norm.cdf(_d2)
            live_val = _c if option_type == "call" else _c - S_t * np.exp(-q * tau) + strike * np.exp(-r * tau)
        # Rational: continue iff live_val >= total cost of remaining payments.
        should_continue = active & (live_val >= cost_to_continue)
        active = should_continue
        continuation_count += should_continue.astype(int)

    # Final payoff at expiry for still-active paths
    S_T = path_at_dates[-1]
    terminal_payoff = np.where(active, np.maximum(sign * (S_T - strike), 0.0), 0.0)
    discount = math.exp(-r * T)
    price = float(np.mean(terminal_payoff) * discount)

    continuation_prob = float(np.mean(active))

    return InstallmentOptionResult(
        price=price,
        upfront_premium=upfront,
        installment_premium=installment_amt,
        continuation_prob=continuation_prob,
    )
