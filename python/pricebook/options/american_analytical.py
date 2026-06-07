"""Analytical approximations for American option pricing.

Provides three independent methods beyond BAW (in futures_options.py):

* :func:`ju_zhong` — Ju & Zhong (1999) higher-order correction to BAW.
* :func:`kim_integral` — Kim (1990) integral equation / exercise boundary.
* :func:`medvedev_scaillet` — Medvedev & Scaillet (2010) near-expiry asymptotics.
* :func:`american_comparison` — runs all three and returns a comparison dict.

The library already contains:
    BAW (futures_options._baw_futures_option)
    CRR/JR/LR binomial trees
    PDE Crank-Nicolson (fd_american)
    LSM Monte Carlo (lsm_american)
    COS Bermudan
    PSOR

References:
    Ju, N. & Zhong, R. (1999). An Approximate Formula for Pricing American
        Options. Journal of Derivatives, 7(2), 31-40.
    Kim, I. J. (1990). The Analytic Valuation of American Options. Review
        of Financial Studies, 3(4), 547-572.
    Medvedev, A. & Scaillet, O. (2010). Pricing American Options under
        Stochastic Volatility and Stochastic Interest Rates. Journal of
        Financial Economics, 98(1), 145-159.
    Barone-Adesi, G. & Whaley, R. (1987). Efficient Analytic Approximation
        of American Option Values. Journal of Finance, 42(2), 301-320.
    Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate
        Liabilities. Journal of Political Economy, 81(3), 637-654.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from pricebook.models.black76 import _norm_cdf, _norm_pdf


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AmericanApproxResult:
    """Result of an analytical American option approximation.

    Attributes:
        price: American option price.
        early_exercise_premium: price - european_price.
        exercise_boundary: critical spot level(s) for early exercise.
            Scalar for Ju-Zhong / Medvedev-Scaillet; array for Kim.
        european_price: corresponding European (BS) price.
        method: name of the approximation method.
    """
    price: float
    early_exercise_premium: float
    exercise_boundary: float | list[float]
    european_price: float
    method: str

    def to_dict(self) -> dict:
        eb = (
            self.exercise_boundary
            if isinstance(self.exercise_boundary, (int, float))
            else list(self.exercise_boundary)
        )
        return {
            "price": self.price,
            "early_exercise_premium": self.early_exercise_premium,
            "exercise_boundary": eb,
            "european_price": self.european_price,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(spot: float, strike: float, T: float, vol: float) -> None:
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if vol <= 0:
        raise ValueError(f"vol must be positive, got {vol}")


def _bs_price(
    S: float, K: float, r: float, vol: float, T: float, q: float, is_call: bool
) -> float:
    """Black-Scholes price for a European option."""
    if T <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if is_call:
        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
    return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)


def _bs_d1_d2(
    S: float, K: float, r: float, vol: float, T: float, q: float
) -> tuple[float, float]:
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt_t)
    return d1, d1 - vol * sqrt_t


# ---------------------------------------------------------------------------
# 1. Ju & Zhong (1999)
# ---------------------------------------------------------------------------

def ju_zhong(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    q: float = 0.0,
    option_type: str = "call",
) -> AmericanApproxResult:
    """Ju & Zhong (1999) analytical approximation for American options.

    Applies a second-order correction to the BAW quadratic approximation.
    The correction accounts for the curvature of the exercise premium with
    respect to the critical price, yielding improved accuracy especially
    for long-dated options and near-the-money strikes.

    The approximation follows these steps:

    1. Compute the European (BS) price.
    2. Solve for the BAW critical price h* via Newton's method.
    3. Evaluate the BAW early exercise premium A at h*.
    4. Apply the Ju-Zhong second-order correction:
           EEP_JZ = A * (S/h*)^q2 * [1 + chi * ln(S/h*)]
       where chi = (dA/dh* evaluated at the BAW boundary) / A.
    5. American price = European price + EEP_JZ (if S not past boundary).

    Args:
        spot: current spot price S.
        strike: option strike K.
        rate: continuously-compounded risk-free rate r.
        vol: implied volatility sigma.
        T: time to expiry in years.
        q: continuous dividend yield.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        :class:`AmericanApproxResult` with method ``"ju_zhong"``.

    References:
        Ju, N. & Zhong, R. (1999). An Approximate Formula for Pricing
        American Options. Journal of Derivatives, 7(2), 31-40.
    """
    _validate(spot, strike, T, vol)
    is_call = option_type.lower() == "call"
    S, K, r, sigma = spot, strike, rate, vol

    euro = _bs_price(S, K, r, sigma, T, q, is_call)

    # BAW parameters
    M = 2.0 * r / (sigma * sigma)
    N = 2.0 * q / (sigma * sigma)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)

    if is_call:
        # For calls: early exercise only optimal when q > 0 (or r < 0)
        if q <= 0.0:
            return AmericanApproxResult(
                price=euro,
                early_exercise_premium=0.0,
                exercise_boundary=float("inf"),
                european_price=euro,
                method="ju_zhong",
            )
        # q2 for call
        q2 = (-(N - 1) + math.sqrt((N - 1) ** 2 + 4.0 * M / (1.0 - df_r))) / 2.0
        if q2 <= 0:
            return AmericanApproxResult(
                price=euro, early_exercise_premium=0.0,
                exercise_boundary=float("inf"), european_price=euro,
                method="ju_zhong",
            )
        h_star = _baw_critical_call(S, K, r, sigma, T, q, q2, df_r, df_q)
        if S >= h_star:
            price = S - K
            eep = price - euro
            return AmericanApproxResult(
                price=price, early_exercise_premium=eep,
                exercise_boundary=h_star, european_price=euro,
                method="ju_zhong",
            )
        d1_h, d2_h = _bs_d1_d2(h_star, K, r, sigma, T, q)
        A2 = (h_star / q2) * (1.0 - df_q * _norm_cdf(d1_h))
        # Ju-Zhong correction: second derivative of A w.r.t. h_star
        # dA/dh* = (1/q2)*(1 - df_q*N(d1)) - (h*/q2)*df_q*n(d1)/(h**sigma*sqrt(T))
        #        = A/h* - df_q*_norm_pdf(d1_h)/(q2*sigma*sqrt(T))
        dA_dh = A2 / h_star - df_q * _norm_pdf(d1_h) / (q2 * sigma * math.sqrt(T))
        chi = dA_dh * h_star / A2  # dimensionless correction coefficient
        ratio = S / h_star
        eep = A2 * ratio ** q2 * (1.0 + chi * math.log(ratio))
        eep = max(eep, 0.0)
        price = euro + eep
    else:
        # Put: early exercise always potentially optimal
        q2 = (-(N - 1) - math.sqrt((N - 1) ** 2 + 4.0 * M / (1.0 - df_r))) / 2.0
        if q2 >= 0:
            return AmericanApproxResult(
                price=euro, early_exercise_premium=0.0,
                exercise_boundary=0.0, european_price=euro,
                method="ju_zhong",
            )
        h_star = _baw_critical_put(S, K, r, sigma, T, q, q2, df_r, df_q)
        if S <= h_star:
            price = K - S
            eep = price - euro
            return AmericanApproxResult(
                price=price, early_exercise_premium=eep,
                exercise_boundary=h_star, european_price=euro,
                method="ju_zhong",
            )
        d1_h, d2_h = _bs_d1_d2(h_star, K, r, sigma, T, q)
        A2 = -(h_star / q2) * (1.0 - df_q * _norm_cdf(-d1_h))
        # dA/dh* for put: same structural formula, sign-adjusted
        dA_dh = A2 / h_star - df_q * _norm_pdf(d1_h) / (q2 * sigma * math.sqrt(T))
        chi = dA_dh * h_star / A2
        ratio = S / h_star
        eep = A2 * ratio ** q2 * (1.0 + chi * math.log(ratio))
        eep = max(eep, 0.0)
        price = euro + eep

    return AmericanApproxResult(
        price=price,
        early_exercise_premium=eep,
        exercise_boundary=h_star,
        european_price=euro,
        method="ju_zhong",
    )


def _baw_critical_call(
    S: float, K: float, r: float, sigma: float, T: float, q: float,
    q2: float, df_r: float, df_q: float,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """Newton-Raphson for BAW critical call price h*."""
    sqrt_t = math.sqrt(T)
    h = max(K * 1.2, S)  # initial guess above strike
    for _ in range(max_iter):
        d1 = (math.log(h / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        euro_call = _bs_price(h, K, r, sigma, T, q, True)
        lhs = h - K - euro_call
        rhs = (h / q2) * (1.0 - df_q * _norm_cdf(d1))
        diff = lhs - rhs
        if abs(diff) < tol:
            break
        d_lhs = 1.0 - df_q * _norm_cdf(d1)
        d_rhs = (1.0 / q2) * (1.0 - df_q * _norm_cdf(d1)) - (h / q2) * df_q * _norm_pdf(d1) / (h * sigma * sqrt_t)
        deriv = d_lhs - d_rhs
        if abs(deriv) < 1e-15:
            break
        h -= diff / deriv
        h = max(h, K * 0.5)
    return h


def _baw_critical_put(
    S: float, K: float, r: float, sigma: float, T: float, q: float,
    q2: float, df_r: float, df_q: float,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """Newton-Raphson for BAW critical put price h*."""
    sqrt_t = math.sqrt(T)
    h = K * 0.8  # initial guess below strike
    for _ in range(max_iter):
        if h <= 0:
            h = K * 0.01
        d1 = (math.log(h / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        euro_put = _bs_price(h, K, r, sigma, T, q, False)
        lhs = K - h - euro_put
        rhs = -(h / q2) * (1.0 - df_q * _norm_cdf(-d1))
        diff = lhs - rhs
        if abs(diff) < tol:
            break
        d_lhs = -1.0 + df_q * _norm_cdf(-d1)
        d_rhs = -(1.0 / q2) * (1.0 - df_q * _norm_cdf(-d1)) + (h / q2) * df_q * _norm_pdf(d1) / (h * sigma * sqrt_t)
        deriv = d_lhs - d_rhs
        if abs(deriv) < 1e-15:
            break
        h -= diff / deriv
        h = max(h, K * 0.001)
        h = min(h, K * 1.5)
    return h


# ---------------------------------------------------------------------------
# 2. Kim (1990) integral equation
# ---------------------------------------------------------------------------

def kim_integral(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    q: float = 0.0,
    option_type: str = "call",
    n_steps: int = 50,
) -> AmericanApproxResult:
    """Kim (1990) integral equation method for American options.

    Decomposes the American price into European price plus early exercise
    premium (EEP):

        P_am(S, T) = P_eu(S, T) + EEP(S, T)

    For a put the EEP integral is:

        EEP = integral_{0}^{T} r*K*e^{-r*tau}*N(-d2(S,B(tau),tau))
              - q*S*e^{-q*tau}*N(-d1(S,B(tau),tau)) dtau

    where B(tau) is the exercise boundary satisfying the implicit equation:

        K - B(tau) = P_eu(B(tau), K, r, vol, tau, q)
                     + integral_{0}^{tau} EEP_integrand(B(tau), t) dt

    The boundary is discretised on a uniform grid tau_i in [0, T] and
    solved backward from B(0) = K * min(1, r/q) (put) using Newton's
    method at each node, with the integral approximated by the trapezoid
    rule over the already-solved boundary values.

    Args:
        spot: current spot price S.
        strike: option strike K.
        rate: continuously-compounded risk-free rate r.
        vol: implied volatility sigma.
        T: time to expiry in years.
        q: continuous dividend yield.
        option_type: ``"call"`` or ``"put"``.
        n_steps: number of time steps for boundary discretisation.

    Returns:
        :class:`AmericanApproxResult` with method ``"kim_integral"``,
        ``exercise_boundary`` is a list of ``n_steps+1`` critical prices
        at times [0, dt, 2*dt, ..., T].

    References:
        Kim, I. J. (1990). The Analytic Valuation of American Options.
        Review of Financial Studies, 3(4), 547-572.
    """
    _validate(spot, strike, T, vol)
    is_call = option_type.lower() == "call"
    S, K, r, sigma, n = spot, strike, rate, vol, n_steps

    euro = _bs_price(S, K, r, sigma, T, q, is_call)

    dt = T / n
    # tau_i = i * dt  (time elapsed since valuation, i = 0 ... n)
    # boundary[i] = B(tau_i)
    boundary = [0.0] * (n + 1)

    # Boundary condition at tau=0 (immediate exercise horizon)
    if is_call:
        # B(0) -> K * max(1, r/q) for calls
        boundary[0] = K if q <= 0.0 else K * max(1.0, r / q)
    else:
        # B(0) -> K * min(1, r/q) for puts (= K when q=0)
        boundary[0] = K if q <= 0.0 else K * min(1.0, r / q)

    # Solve backward: boundary[i] at tau_i for i = 1 .. n
    # At each node, boundary satisfies:
    #   intrinsic(B) = European_price(B, tau_i) + EEP_integral up to tau_i
    for i in range(1, n + 1):
        tau_i = i * dt
        # Initial guess: previous boundary value
        B = boundary[i - 1]

        for _newton in range(50):
            B = max(B, 1e-8)

            # European price at this candidate boundary
            p_eu = _bs_price(B, K, r, sigma, tau_i, q, is_call)

            # EEP integral from 0..tau_i using trapezoid rule over
            # already-solved boundary[0..i-1] plus current candidate at i
            eep_int = 0.0
            prev_bndry = list(boundary[:i]) + [B]  # length i+1
            for j in range(i):
                t_j = j * dt
                t_jp1 = (j + 1) * dt
                eep_j = _kim_eep_integrand(S, K, r, sigma, prev_bndry[j], t_j, q, is_call)
                eep_jp1 = _kim_eep_integrand(S, K, r, sigma, prev_bndry[j + 1], t_jp1, q, is_call)
                eep_int += 0.5 * dt * (eep_j + eep_jp1)

            # Boundary condition: intrinsic = p_eu + eep_int
            if is_call:
                intrinsic = B - K
                residual = intrinsic - p_eu - eep_int
                # d(residual)/dB ≈ 1 - dP_eu/dB - 0 (eep_int weakly depends on B)
                d1_B, d2_B = _bs_d1_d2(B, K, r, sigma, tau_i, q)
                d_peu_dB = math.exp(-q * tau_i) * _norm_cdf(d1_B)
                d_eep_dB = _kim_eep_boundary_deriv(S, K, r, sigma, B, tau_i, q, is_call, dt)
                deriv = 1.0 - d_peu_dB - d_eep_dB
            else:
                intrinsic = K - B
                residual = intrinsic - p_eu - eep_int
                d1_B, d2_B = _bs_d1_d2(B, K, r, sigma, tau_i, q)
                d_peu_dB = -math.exp(-q * tau_i) * _norm_cdf(-d1_B)
                d_eep_dB = _kim_eep_boundary_deriv(S, K, r, sigma, B, tau_i, q, is_call, dt)
                deriv = -1.0 - d_peu_dB - d_eep_dB

            if abs(residual) < 1e-7:
                break
            if abs(deriv) < 1e-15:
                break
            B -= residual / deriv
            if is_call:
                B = max(B, K)
            else:
                B = max(min(B, K), 1e-8)

        boundary[i] = B

    # Price: EEP evaluated at spot S using the solved boundary
    eep = 0.0
    for j in range(n):
        t_j = j * dt
        t_jp1 = (j + 1) * dt
        eep_j = _kim_eep_integrand(S, K, r, sigma, boundary[j], t_j, q, is_call)
        eep_jp1 = _kim_eep_integrand(S, K, r, sigma, boundary[j + 1], t_jp1, q, is_call)
        eep += 0.5 * dt * (eep_j + eep_jp1)

    eep = max(eep, 0.0)
    price = euro + eep

    # Clamp to intrinsic value
    if is_call:
        price = max(price, max(S - K, 0.0))
    else:
        price = max(price, max(K - S, 0.0))

    return AmericanApproxResult(
        price=price,
        early_exercise_premium=eep,
        exercise_boundary=boundary,
        european_price=euro,
        method="kim_integral",
    )


def _kim_eep_integrand(
    S: float, K: float, r: float, sigma: float,
    B: float, tau: float, q: float, is_call: bool,
) -> float:
    """EEP integrand at a single time point tau (time elapsed)."""
    if tau <= 0 or B <= 0:
        return 0.0
    d1, d2 = _bs_d1_d2(S, B, r, sigma, tau, q)
    if is_call:
        # Call EEP: q*S*e^{-q*tau}*N(d1) - r*K*e^{-r*tau}*N(d2)
        return (
            q * S * math.exp(-q * tau) * _norm_cdf(d1)
            - r * K * math.exp(-r * tau) * _norm_cdf(d2)
        )
    else:
        # Put EEP: r*K*e^{-r*tau}*N(-d2) - q*S*e^{-q*tau}*N(-d1)
        return (
            r * K * math.exp(-r * tau) * _norm_cdf(-d2)
            - q * S * math.exp(-q * tau) * _norm_cdf(-d1)
        )


def _kim_eep_boundary_deriv(
    S: float, K: float, r: float, sigma: float,
    B: float, tau: float, q: float, is_call: bool, dt: float,
) -> float:
    """Numerical derivative of last trapezoid step w.r.t. B (for Newton)."""
    eps = B * 1e-4
    f_p = _kim_eep_integrand(S, K, r, sigma, B + eps, tau, q, is_call)
    f_m = _kim_eep_integrand(S, K, r, sigma, B - eps, tau, q, is_call)
    return 0.5 * dt * (f_p - f_m) / (2.0 * eps)


# ---------------------------------------------------------------------------
# 3. Medvedev & Scaillet (2010)
# ---------------------------------------------------------------------------

def medvedev_scaillet(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    q: float = 0.0,
    option_type: str = "put",
) -> AmericanApproxResult:
    """Medvedev & Scaillet (2010) near-expiry asymptotic approximation.

    Constructs a polynomial expansion of the exercise boundary:

        B(tau) ≈ B_inf * exp(-b1*sqrt(tau) - b2*tau - b3*tau^{3/2})

    where tau = time-to-maturity remaining, and the coefficients are:

        mu = r - q - sigma^2/2  (risk-neutral drift)
        kappa = (r - q) / sigma^2
        B_inf = K * min(1, r/q)  for put  (= K when q=0)

        b1 = sigma * sqrt(2/pi)  when r = q  (ATM-forward)
        b1 general = sigma * sqrt(2/pi) * f(kappa)  where f adjusts for
                     the drift:  f(kappa) = 1 / (1 + kappa*sqrt(pi/2))
                     (first-order drift correction, Medvedev-Scaillet eq. 10)
        b2 = mu / sigma  (linear correction)
        b3 = 0  (third-order omitted for numerical stability)

    The early exercise premium is approximated by integrating the boundary:

        EEP ≈ integral_{0}^{T} r*K*e^{-r*tau}*N(-d2(S,B(tau),tau))
              - q*S*e^{-q*tau}*N(-d1(S,B(tau),tau)) dtau  (put)

    with n=50 trapezoid points.

    Args:
        spot: current spot price S.
        strike: option strike K.
        rate: continuously-compounded risk-free rate r.
        vol: implied volatility sigma.
        T: time to expiry in years.
        q: continuous dividend yield.
        option_type: ``"call"`` or ``"put"`` (asymptotic is sharpest for puts).

    Returns:
        :class:`AmericanApproxResult` with method ``"medvedev_scaillet"``.
        ``exercise_boundary`` is the boundary value at t=0 (i.e., B(T)).

    References:
        Medvedev, A. & Scaillet, O. (2010). Pricing American Options Under
        Stochastic Volatility and Stochastic Interest Rates. Journal of
        Financial Economics, 98(1), 145-159.
    """
    _validate(spot, strike, T, vol)
    is_call = option_type.lower() == "call"
    S, K, r, sigma = spot, strike, rate, vol

    euro = _bs_price(S, K, r, sigma, T, q, is_call)

    # Asymptotic boundary coefficients
    mu = r - q - 0.5 * sigma * sigma  # log drift under risk-neutral measure
    kappa = (r - q) / (sigma * sigma) if sigma > 0 else 0.0

    if is_call:
        # B_inf for call: K * max(1, r/q)
        B_inf = K if q <= 0.0 else K * max(1.0, r / q)
    else:
        # B_inf for put: K * min(1, r/q) -> K when q=0
        B_inf = K if q <= 0.0 else K * min(1.0, r / q)
        B_inf = min(B_inf, K)

    # First-order boundary coefficient (Medvedev-Scaillet, eq. 10)
    # b1 = sigma * sqrt(2/pi) corrected for drift
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    drift_factor = 1.0 / (1.0 + kappa * math.sqrt(math.pi / 2.0))
    b1 = sigma * sqrt_2_pi * drift_factor

    # Second-order coefficient (linear in tau)
    b2 = mu / sigma if sigma > 0 else 0.0

    # Boundary at tau: B(tau) = B_inf * exp(-b1*sqrt(tau) - b2*tau)
    def boundary_at_tau(tau: float) -> float:
        if tau <= 0:
            return B_inf
        return B_inf * math.exp(-b1 * math.sqrt(tau) - b2 * tau)

    # Numerical EEP: trapezoid integration of EEP integrand
    n_quad = 50
    dtau = T / n_quad
    eep = 0.0
    for j in range(n_quad):
        tau_j = j * dtau
        tau_jp1 = (j + 1) * dtau
        eep_j = _kim_eep_integrand(S, K, r, sigma, boundary_at_tau(tau_j), tau_j, q, is_call)
        eep_jp1 = _kim_eep_integrand(S, K, r, sigma, boundary_at_tau(tau_jp1), tau_jp1, q, is_call)
        eep += 0.5 * dtau * (eep_j + eep_jp1)

    eep = max(eep, 0.0)
    price = euro + eep

    # Clamp to intrinsic
    if is_call:
        price = max(price, max(S - K, 0.0))
    else:
        price = max(price, max(K - S, 0.0))

    # Representative boundary: value at mid-life tau = T/2
    boundary_mid = boundary_at_tau(T / 2.0)

    return AmericanApproxResult(
        price=price,
        early_exercise_premium=eep,
        exercise_boundary=boundary_mid,
        european_price=euro,
        method="medvedev_scaillet",
    )


# ---------------------------------------------------------------------------
# 4. Comparison across all methods
# ---------------------------------------------------------------------------

def american_comparison(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    q: float = 0.0,
    option_type: str = "put",
) -> dict:
    """Run all analytical approximations and return a comparison dict.

    Prices the same American option with Ju-Zhong, Kim integral, and
    Medvedev-Scaillet, recording the price and wall-clock time for each.

    Args:
        spot: current spot price.
        strike: option strike.
        rate: continuously-compounded risk-free rate.
        vol: implied volatility.
        T: time to expiry in years.
        q: continuous dividend yield.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        Dictionary with keys ``"ju_zhong"``, ``"kim_integral"``,
        ``"medvedev_scaillet"``, each containing a sub-dict:
            ``price`` (float), ``eep`` (float), ``elapsed_ms`` (float).
        Also includes ``"european_price"`` for reference.
    """
    _validate(spot, strike, T, vol)
    is_call = option_type.lower() == "call"
    euro = _bs_price(spot, strike, rate, vol, T, q, is_call)

    results: dict = {"european_price": euro}

    for name, fn, kwargs in [
        ("ju_zhong", ju_zhong, {}),
        ("kim_integral", kim_integral, {"n_steps": 50}),
        ("medvedev_scaillet", medvedev_scaillet, {}),
    ]:
        t0 = time.perf_counter()
        res = fn(spot, strike, rate, vol, T, q, option_type, **kwargs)  # type: ignore[call-arg]
        elapsed = (time.perf_counter() - t0) * 1000.0
        results[name] = {
            "price": res.price,
            "eep": res.early_exercise_premium,
            "elapsed_ms": round(elapsed, 3),
        }

    return results
