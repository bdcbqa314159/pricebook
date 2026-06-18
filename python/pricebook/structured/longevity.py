"""Longevity and mortality derivatives.

Pricing and risk analytics for q-forwards, longevity swaps, mortality bonds,
survivor indices, Lee-Carter forecasts, and life-contingent annuities.

* :class:`QForwardResult`       — q-forward pricing output.
* :class:`LongevitySwapResult`  — longevity swap analytics.
* :func:`q_forward`             — price a q-forward contract.
* :func:`longevity_swap`        — value a fixed-vs-realised mortality swap.
* :func:`survivor_index`        — project a cohort survivor index.
* :func:`lee_carter_forecast`   — simplified Lee-Carter mortality forecast (SVD).
* :func:`mortality_bond_price`  — mortality/longevity bond with principal at risk.
* :func:`value_of_life_annuity` — PV of a life-contingent annuity.

References:
    Lee, R.D. & Carter, L.R. (1992). Modeling and Forecasting U.S. Mortality.
        *Journal of the American Statistical Association*, 87(419), 659–671.
    Blake, D., Cairns, A.J.G. & Dowd, K. (2006). Living with Mortality:
        Longevity Bonds and Other Mortality-Linked Securities.
        *British Actuarial Journal*, 12(1), 153–228.
    Cairns, A.J.G., Blake, D. & Dowd, K. (2006). A Two-Factor Model for
        Stochastic Mortality with Parameter Uncertainty.
        *Journal of Risk and Insurance*, 73(4), 687–718.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class QForwardResult:
    """q-forward pricing output."""
    price: float                # fair value (positive = receiver of q_forward profits)
    fixed_mortality: float      # agreed fixed mortality rate q_fixed
    forward_mortality: float    # risk-adjusted forward mortality rate q_forward
    notional_at_risk: float     # notional × q_fixed (maximum payout)
    pv: float                   # present value of net payment

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class LongevitySwapResult:
    """Longevity swap valuation output."""
    pv: float                       # net PV (fixed payer perspective)
    fixed_leg_pv: float             # PV of fixed mortality payments
    floating_leg_pv: float          # PV of realised mortality payments
    breakeven_improvement: float    # improvement rate that makes PV = 0
    basis_risk: float               # std-dev of residual (model vs realised)

    def to_dict(self) -> dict:
        return dict(vars(self))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rate: float, t: float) -> float:
    """Continuous discount factor."""
    return math.exp(-rate * t)


def _project_mortality(q0: float, improvement: float, year: int) -> float:
    """Project mortality rate forward using constant improvement factor."""
    return q0 * (1.0 - improvement) ** year


# ---------------------------------------------------------------------------
# 1. q-forward
# ---------------------------------------------------------------------------

def q_forward(
    reference_age: int,
    term_years: int,
    current_mortality_rate: float,
    mortality_improvement_rate: float,
    risk_free_rate: float,
    risk_premium: float = 0.0,
    notional: float = 1_000_000.0,
) -> QForwardResult:
    """Price a q-forward (mortality forward) contract.

    A q-forward exchanges a fixed mortality rate (q_fixed) agreed at inception
    for the realised mortality rate at maturity for a given age cohort.

    Settlement:
        payoff = notional × (q_realised - q_fixed) × df(T)

    The fair fixed rate is the risk-adjusted forward mortality:
        q_fixed = q_forward × (1 + risk_premium)

    where q_forward is projected using the mortality improvement scale.

    Args:
        reference_age:            Age of the reference cohort at maturity.
        term_years:               Term of the contract in years.
        current_mortality_rate:   Current (year-0) mortality rate q(x, 0).
        mortality_improvement_rate: Annual mortality improvement factor (e.g. 0.02).
        risk_free_rate:           Risk-free discount rate.
        risk_premium:             Additional loading for longevity risk premium.
        notional:                 Contract notional.

    Returns:
        QForwardResult.
    """
    # Project the "best-estimate" forward mortality rate
    q_fwd = _project_mortality(current_mortality_rate, mortality_improvement_rate, term_years)

    # Risk-adjusted fixed rate: hedger pays a premium for longevity protection
    q_fixed = q_fwd * (1.0 + risk_premium)

    df_T = _df(risk_free_rate, term_years)
    # From fixed-payer perspective: PV = notional × (q_fixed - q_fwd) × df
    # At fair value, q_fixed is set so PV = 0 (no-arbitrage)
    pv = notional * (q_fixed - q_fwd) * df_T
    notional_at_risk = notional * q_fixed

    return QForwardResult(
        price=pv,
        fixed_mortality=q_fixed,
        forward_mortality=q_fwd,
        notional_at_risk=notional_at_risk,
        pv=pv,
    )


# ---------------------------------------------------------------------------
# 2. Longevity swap
# ---------------------------------------------------------------------------

def longevity_swap(
    ages: list[int],
    initial_mortality_rates: list[float],
    improvement_rates: list[float],
    fixed_rates: list[float],
    risk_free_rate: float,
    notional_per_life: float = 1_000.0,
    n_years: int = 20,
) -> LongevitySwapResult:
    """Value a longevity swap: fixed vs realised mortality for a cohort.

    For each age cohort, the pension plan pays a fixed mortality rate (agreed
    at inception) and receives realised (projected) mortality.  The net payment
    at each year is:

        net_t = notional_per_life × survivors(t) × (q_realised(t) - q_fixed(t))

    Args:
        ages:                   List of reference ages at inception.
        initial_mortality_rates: q(x, 0) for each age.
        improvement_rates:      Annual mortality improvement per age cohort.
        fixed_rates:            Fixed mortality rates agreed in the swap.
        risk_free_rate:         Flat risk-free discount rate.
        notional_per_life:      Notional exposure per surviving life.
        n_years:                Swap term in years.

    Returns:
        LongevitySwapResult.
    """
    if not (len(ages) == len(initial_mortality_rates) == len(improvement_rates) == len(fixed_rates)):
        raise ValueError("ages, initial_mortality_rates, improvement_rates, fixed_rates must be same length")

    n_cohorts = len(ages)
    fixed_leg_pv = 0.0
    floating_leg_pv = 0.0
    residuals: list[float] = []

    for c in range(n_cohorts):
        q0 = initial_mortality_rates[c]
        imp = improvement_rates[c]
        q_fix = fixed_rates[c]
        survivors = 1.0  # start with 1 life normalised

        for t in range(1, n_years + 1):
            q_realised = _project_mortality(q0, imp, t)
            df_t = _df(risk_free_rate, t)

            deaths_fixed = survivors * q_fix * notional_per_life
            deaths_float = survivors * q_realised * notional_per_life

            fixed_leg_pv += deaths_fixed * df_t
            floating_leg_pv += deaths_float * df_t
            residuals.append(deaths_fixed - deaths_float)

            survivors *= (1.0 - q_realised)

    pv = fixed_leg_pv - floating_leg_pv
    basis_risk = float(np.std(residuals)) if residuals else 0.0

    # Breakeven improvement: find rate that makes PV = 0 numerically
    # Approximate: improvement that equates projected q to fixed q
    avg_q0 = float(np.mean(initial_mortality_rates))
    avg_fixed = float(np.mean(fixed_rates))
    if avg_q0 > avg_fixed > 0:
        # q0 * (1 - imp)^n = avg_fixed → imp = 1 - (avg_fixed/avg_q0)^(1/n)
        breakeven_improvement = 1.0 - (avg_fixed / avg_q0) ** (1.0 / n_years)
    else:
        breakeven_improvement = float(np.mean(improvement_rates))

    return LongevitySwapResult(
        pv=pv,
        fixed_leg_pv=fixed_leg_pv,
        floating_leg_pv=floating_leg_pv,
        breakeven_improvement=breakeven_improvement,
        basis_risk=basis_risk,
    )


# ---------------------------------------------------------------------------
# 3. Survivor index
# ---------------------------------------------------------------------------

def survivor_index(
    initial_population: float,
    mortality_rates: list[float],
    improvement_rate: float,
    n_years: int,
) -> np.ndarray:
    """Project a cohort survivor index.

    S(0) = initial_population
    S(t) = S(t-1) × (1 - q(t))

    where q(t) = q(0) × (1 - improvement_rate)^t.

    Args:
        initial_population: Starting population or notional (e.g. 100.0).
        mortality_rates:    Base mortality rates q(x) for each age in cohort.
                            If a single cohort is modelled, provide [q0].
        improvement_rate:   Constant annual mortality improvement.
        n_years:            Projection horizon.

    Returns:
        Array of length n_years + 1 with survivor index values S(0)..S(n_years).
    """
    q0 = float(np.mean(mortality_rates))
    index = np.empty(n_years + 1)
    index[0] = initial_population
    survivors = float(initial_population)
    for t in range(1, n_years + 1):
        q_t = _project_mortality(q0, improvement_rate, t)
        q_t = min(max(q_t, 0.0), 1.0)
        survivors *= (1.0 - q_t)
        index[t] = survivors
    return index


# ---------------------------------------------------------------------------
# 4. Lee-Carter forecast
# ---------------------------------------------------------------------------

def lee_carter_forecast(
    log_mortality_matrix: np.ndarray,
    n_forecast_years: int,
    n_simulations: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Simplified Lee-Carter mortality forecast using SVD.

    Model: ln m(x, t) = a(x) + b(x) × k(t)

    Steps:
        1. Estimate a(x) = row means of log_mortality_matrix.
        2. Centre the matrix and extract leading SVD component for b(x), k(t).
        3. Fit a random walk with drift to k(t) and project forward.
        4. Return mean forecast mortality rates (not log).

    Args:
        log_mortality_matrix: 2-D array (n_ages × n_years) of log mortality rates.
        n_forecast_years:     Number of years to forecast forward.
        n_simulations:        MC paths for k(t) uncertainty (unused in mean forecast).
        seed:                 Random seed.

    Returns:
        2-D array (n_ages × n_forecast_years) of forecast mortality rates (not log).
    """
    M = np.asarray(log_mortality_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("log_mortality_matrix must be 2-D (n_ages × n_years)")
    n_ages, n_hist = M.shape

    # Step 1: a(x) = time-average of log mortality for each age
    a = M.mean(axis=1)

    # Step 2: Centre and SVD (leading component only)
    M_centred = M - a[:, np.newaxis]
    U, s, Vt = np.linalg.svd(M_centred, full_matrices=False)
    b = U[:, 0] * s[0]       # raw b(x)
    k = Vt[0, :]               # k(t) time series

    # Normalise: sum(b) = 1, rescale k accordingly
    b_sum = np.sum(b)
    if abs(b_sum) > 1e-12:
        k = k * b_sum
        b = b / b_sum

    # Step 3: Random walk with drift for k(t)
    rng = np.random.default_rng(seed)
    if n_hist >= 2:
        dk = np.diff(k)
        drift = float(np.mean(dk))
        sigma_k = float(np.std(dk, ddof=1)) if n_hist >= 3 else 0.0
    else:
        drift = 0.0
        sigma_k = 0.0

    k_last = k[-1]
    k_forecast = np.empty(n_forecast_years)
    k_cur = k_last
    for t in range(n_forecast_years):
        shock = rng.normal(0.0, sigma_k) if sigma_k > 0 else 0.0
        k_cur += drift + shock
        k_forecast[t] = k_cur

    # Step 4: Forecast log mortality and exponentiate
    log_forecast = a[:, np.newaxis] + np.outer(b, k_forecast)
    return np.exp(log_forecast)


# ---------------------------------------------------------------------------
# 5. Mortality bond price
# ---------------------------------------------------------------------------

def mortality_bond_price(
    notional: float,
    coupon: float,
    risk_free_rate: float,
    attachment: float,
    exhaustion: float,
    expected_mortality: float,
    mortality_vol: float,
    T: float,
) -> dict[str, float]:
    """Price a mortality (or longevity) bond with principal at risk.

    Mortality indices are modelled as lognormal.  If the mortality index at
    maturity exceeds the attachment point, principal is reduced pro-rata up to
    full loss at the exhaustion point.

    Expected loss = E[payout] approximated via Black-Scholes call spread:
        EL ≈ BS_call(attachment) - BS_call(exhaustion)
             normalised by (exhaustion - attachment).

    Args:
        notional:            Face value.
        coupon:              Annual coupon rate (fraction).
        risk_free_rate:      Risk-free rate.
        attachment:          Mortality index level triggering first principal loss.
        exhaustion:          Mortality index level for total principal loss.
        expected_mortality:  Current expected mortality index (forward level).
        mortality_vol:       Volatility of mortality index.
        T:                   Maturity in years.

    Returns:
        Dictionary with price, expected_loss, spread, coupon_pv, principal_pv.
    """
    from math import log, sqrt, erf  # noqa: PLC0415

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

    def _bs_call_price(F: float, K: float, vol: float, t: float) -> float:
        if t <= 0 or vol <= 0 or K <= 0:
            return max(F - K, 0.0)
        d1 = (log(F / K) + 0.5 * vol**2 * t) / (vol * math.sqrt(t))
        d2 = d1 - vol * math.sqrt(t)
        return F * _norm_cdf(d1) - K * _norm_cdf(d2)

    layer = exhaustion - attachment
    if layer <= 0:
        raise ValueError("exhaustion must be greater than attachment")

    # Expected payout as fraction of notional (call spread)
    df_T = _df(risk_free_rate, T)
    c_attach = _bs_call_price(expected_mortality, attachment, mortality_vol, T)
    c_exhaust = _bs_call_price(expected_mortality, exhaustion, mortality_vol, T)
    expected_loss_frac = (c_attach - c_exhaust) / layer

    # PV of coupons (paid annually, bond survives unless triggered).
    # Fix T4-STRUCT: same shape as v1.035 cat_bond_price.  Pre-fix
    # ``range(1, int(T) + 1)`` silently dropped any non-integer
    # remainder.  T=0.5 (6-month bond) gave zero coupon PV.  Add a
    # fractional final accrual when T has a non-integer part.
    n_full = int(T)
    coupon_pv = sum(
        coupon * notional * _df(risk_free_rate, t) for t in range(1, n_full + 1)
    )
    remainder = T - n_full
    if remainder > 1e-9:
        coupon_pv += coupon * notional * remainder * _df(risk_free_rate, T)

    # PV of principal (reduced by expected loss)
    principal_pv = notional * df_T * (1.0 - expected_loss_frac)
    price = (coupon_pv + principal_pv) / notional * 100.0

    # Spread above risk-free implied by expected loss
    spread = expected_loss_frac / max(T, 1e-6) + coupon - risk_free_rate

    return {
        "price": price,
        "expected_loss": expected_loss_frac,
        "spread": spread,
        "coupon_pv": coupon_pv,
        "principal_pv": principal_pv,
    }


# ---------------------------------------------------------------------------
# 6. Value of a life annuity
# ---------------------------------------------------------------------------

def value_of_life_annuity(
    annual_payment: float,
    age: int,
    mortality_rates: list[float],
    improvement_rate: float,
    risk_free_rate: float,
    max_age: int = 120,
) -> float:
    """Present value of a life-contingent annuity.

    Computes the expected PV of an annuity-due (payments at start of each year)
    contingent on the annuitant surviving each period:

        PV = Σ_t  annual_payment × t_p_x × df(t)

    where t_p_x is the t-year survival probability from age x, projected using
    the Lee-Carter-style improvement factor.

    Args:
        annual_payment:     Annual annuity payment amount.
        age:                Current age of the annuitant.
        mortality_rates:    Mortality rates q(x) for ages [age, age+1, ...].
                            If shorter than (max_age - age), last rate is extended.
        improvement_rate:   Annual mortality improvement applied to all ages.
        risk_free_rate:     Flat discount rate.
        max_age:            Maximum age (projection stops here).

    Returns:
        Present value of the life annuity.
    """
    n_years = max_age - age
    if n_years <= 0:
        return 0.0

    # Extend mortality table if necessary
    rates = list(mortality_rates)
    if len(rates) < n_years:
        rates.extend([rates[-1]] * (n_years - len(rates)))

    pv = 0.0
    survival_prob = 1.0  # t_p_x starts at 1

    for t in range(n_years):
        # Payment at start of year t (annuity-due)
        df_t = _df(risk_free_rate, t)
        pv += annual_payment * survival_prob * df_t

        # Update survival for next year
        q_t = _project_mortality(rates[t], improvement_rate, t)
        q_t = min(max(q_t, 0.0), 1.0)
        survival_prob *= (1.0 - q_t)

        if survival_prob < 1e-10:
            break

    return pv
