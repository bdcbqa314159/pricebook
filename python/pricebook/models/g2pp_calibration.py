"""G2++ (two-factor Hull-White) model calibration from swaption volatilities.

Fits G2++ parameters (a, b, sigma1, sigma2, rho) by minimising the RMSE
between model-implied and market swaption Black-76 volatilities.

    from pricebook.models.g2pp_calibration import (
        calibrate_g2pp, G2PPCalibrationResult, g2pp_vs_hw1f,
    )

    result = calibrate_g2pp(curve, swaption_vols)
    g2 = result.model

References:
    Brigo & Mercurio (2006). Interest Rate Models — Theory and Practice,
        Ch. 4.2 (G2++ model, bond option and swaption pricing).
    Hull, J. C. (2018). Options, Futures, and Other Derivatives, Ch. 33.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm as _norm

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import date_from_year_fraction
from pricebook.models.vasicek import G2PlusPlus
from pricebook.models.black76 import OptionType, black76_price

if TYPE_CHECKING:
    from pricebook.market_data import MarketSnapshot


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class G2PPCalibrationResult:
    """Result of G2++ calibration to swaption volatilities.

    `calibration_result` carries the canonical provenance artefact.
    `to_calibration_result()` returns the stored instance when populated,
    or builds one on-demand from the existing fields.
    """

    model: G2PlusPlus
    a: float
    b: float
    sigma1: float
    sigma2: float
    rho: float
    rmse_vol: float                   # RMSE in Black-vol space (not bp)
    per_swaption_errors: list[dict]   # per-instrument diagnostics
    n_swaptions: int
    converged: bool
    # New in G1 P1 Slice 3 — canonical calibration artefact.
    calibration_result: CalibrationResult | None = None

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "sigma1": self.sigma1,
            "sigma2": self.sigma2,
            "rho": self.rho,
            "rmse_vol": self.rmse_vol,
            "n_swaptions": self.n_swaptions,
            "converged": self.converged,
            "calibration_id": (
                str(self.calibration_result.id) if self.calibration_result else None
            ),
        }

    def to_calibration_result(self) -> CalibrationResult:
        """Return the canonical `CalibrationResult`.

        Returns the stored instance when populated by `calibrate_g2pp`.
        Builds one on-demand from the existing fields otherwise.
        """
        if self.calibration_result is not None:
            return self.calibration_result
        residuals = [e["error_bp"] for e in self.per_swaption_errors]
        quotes = [
            f"swaption_{e['expiry']}x{e['tenor']}" for e in self.per_swaption_errors
        ]
        return CalibrationResult.new(
            model_class="g2pp",
            parameters={
                "a": self.a, "b": self.b,
                "sigma1": self.sigma1, "sigma2": self.sigma2, "rho": self.rho,
            },
            residuals=residuals,
            objective=ObjectiveKind.SSE,
            optimiser=OptimiserSpec(algorithm="unknown", tolerance=0.0, max_iterations=0),
            iterations=0,
            converged=self.converged,
            quotes_fitted=quotes,
        )


# ---------------------------------------------------------------------------
# G2++ ZCB helper — mirrors G2PlusPlus.zcb_price but pure-function
# ---------------------------------------------------------------------------

def _g2pp_V(a: float, b: float, s1: float, s2: float, rho: float,
             T: float) -> float:
    """Variance term V(T) used in G2++ ZCB formula (Brigo-Mercurio 4.2)."""
    def Bf(k: float, t: float) -> float:
        return (1.0 - math.exp(-k * t)) / k if k > 0 else t

    Ba = Bf(a, T)
    Bb = Bf(b, T)
    ca = (T - 2 * Ba + Bf(2 * a, T)) / a**2 if a > 1e-12 else T**3 / 3
    cb = (T - 2 * Bb + Bf(2 * b, T)) / b**2 if b > 1e-12 else T**3 / 3
    cab = (T - Ba - Bb + Bf(a + b, T)) / (a * b) if a > 1e-12 and b > 1e-12 else T**3 / 3
    return s1**2 * ca + s2**2 * cb + 2 * rho * s1 * s2 * cab


def _g2pp_zcb(
    a: float, b: float, s1: float, s2: float, rho: float,
    curve: DiscountCurve, x: float, y: float, T: float,
) -> float:
    """P(x, y; T) under G2++ (Brigo-Mercurio eq. 4.14)."""
    ref = curve.reference_date
    P_mkt = curve.df(date_from_year_fraction(ref, T))
    Bx = (1.0 - math.exp(-a * T)) / a if a > 0 else T
    By = (1.0 - math.exp(-b * T)) / b if b > 0 else T
    V = _g2pp_V(a, b, s1, s2, rho, T)
    return P_mkt * math.exp(-Bx * x - By * y + 0.5 * V)


# ---------------------------------------------------------------------------
# G2++ bond option: closed-form (Brigo-Mercurio 4.2.1)
# ---------------------------------------------------------------------------

def _g2pp_zcb_option(
    a: float, b: float, s1: float, s2: float, rho: float,
    curve: DiscountCurve, t: float, S: float, T: float,
    K: float, is_call: bool = True,
) -> float:
    """
    Price a ZCB option under G2++.

    Option expiry t, bond maturity T > t, bond maturity for the swap leg S
    (here S == T), strike K (on the ZCB).

    Uses Brigo-Mercurio Ch. 4.2 eq. 4.20:
        Price = omega * [P(0,T)*N(omega*h1) - K*P(0,t)*N(omega*h2)]
    where omega = +1 (call) or -1 (put).
    """
    ref = curve.reference_date
    P_t = curve.df(date_from_year_fraction(ref, t))
    P_T = curve.df(date_from_year_fraction(ref, T))

    def Bf(k: float, tau: float) -> float:
        return (1.0 - math.exp(-k * tau)) / k if k > 0 else tau

    tau = T - t
    Ba_tT = Bf(a, tau)
    Bb_tT = Bf(b, tau)

    # sigma_p^2: variance of ln(P(t,T)) as seen from time 0
    sigma_p2 = (
        (s1 * Ba_tT) ** 2 * (1.0 - math.exp(-2 * a * t)) / (2 * a)
        + (s2 * Bb_tT) ** 2 * (1.0 - math.exp(-2 * b * t)) / (2 * b)
        + 2 * rho * s1 * s2 * Ba_tT * Bb_tT
          * (1.0 - math.exp(-(a + b) * t)) / (a + b)
    )
    sigma_p = math.sqrt(max(sigma_p2, 0.0))

    if sigma_p < 1e-12 or P_t <= 0 or P_T <= 0 or K <= 0:
        intrinsic = P_T - K * P_t
        return max(intrinsic, 0.0) if is_call else max(-intrinsic, 0.0)

    h1 = math.log(P_T / (P_t * K)) / sigma_p + 0.5 * sigma_p
    h2 = h1 - sigma_p
    w = 1.0 if is_call else -1.0
    return w * (P_T * _norm.cdf(w * h1) - K * P_t * _norm.cdf(w * h2))


# ---------------------------------------------------------------------------
# G2++ swaption price via Jamshidian decomposition
# ---------------------------------------------------------------------------

def g2pp_swaption_price(
    a: float,
    b: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    curve: DiscountCurve,
    expiry_years: float,
    tenor_years: float,
    strike: float,
    is_payer: bool = True,
    n_quad: int = 32,
) -> float:
    """Price a European swaption under G2++ analytically.

    Implements Brigo-Mercurio Ch. 4.2 swaption formula via a 1-D numerical
    integration: for each value of the first factor x (Gauss-Hermite nodes),
    apply Jamshidian's trick on the second factor y to find the bond strike
    and price a sum of ZCB options analytically, then integrate over x.

    Args:
        a, b: mean reversion speeds for the two factors.
        sigma1, sigma2: factor volatilities.
        rho: factor correlation.
        curve: initial discount curve.
        expiry_years: swaption expiry in years.
        tenor_years: underlying swap tenor in years.
        strike: fixed rate of the underlying swap.
        is_payer: True for payer swaption (long floating, pay fixed).
        n_quad: number of Gauss-Hermite quadrature nodes over first factor.

    Returns:
        Swaption price as a fraction of notional.
    """
    # Fix T2.11: pre-fix this routine wrapped the ENTIRE body in
    # `try: ... except Exception: return 0.0`, masking every error mode
    # (bracketing failure, brentq divergence, numerical-overflow in the
    # ZCB formula, calibration bugs) as a silent zero price.  Callers had
    # no way to distinguish "swaption worth ≈ 0" from "pricer crashed".
    # Post-fix: let real exceptions propagate.  Specific recoverable cases
    # (no payment dates, identically-zero variance) still return 0.0
    # explicitly.
    t = expiry_years
    ref = curve.reference_date

    # Payment dates: annual, could be extended for semi-annual
    n_pay = max(1, int(round(tenor_years)))
    pay_times = [t + k for k in range(1, n_pay + 1)]
    pay_times = [s for s in pay_times if s <= t + tenor_years + 1e-9]
    if not pay_times:
        return 0.0

    # Coupon weights c_k for each payment date
    c = [strike] * len(pay_times)
    c[-1] += 1.0  # principal repayment on final leg

    # omega: receiver = call on ZCBs, payer = put on ZCBs
    # Payer swaption = P(fixed) - pay float = short bond portfolio
    # => equivalent to a put on bond portfolio for payer
    # Brigo-Mercurio: payer swaption = sum of put ZCB options
    is_zcb_call = not is_payer  # payer swaption => put ZCBs

    # First factor conditional distribution: x | curve
    # x ~ N(mu_x, var_x) at time t (starting from x0=0)
    var_x = sigma1 ** 2 * (1.0 - math.exp(-2 * a * t)) / (2 * a) if a > 0 else sigma1 ** 2 * t
    std_x = math.sqrt(max(var_x, 0.0))

    # Gauss-Hermite nodes and weights for integration over x
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n_quad)

    total_price = 0.0

    for xi_raw, w_gh in zip(gh_nodes, gh_weights):
        # Map GH node to x value: integral over N(0, var_x)
        x_val = math.sqrt(2.0) * std_x * xi_raw

        # Given x_val, y* such that sum(c_k * P(x,y*,s_k)) = 1
        # Solve for y* numerically (1-D root finding)
        def bond_portfolio(y_val: float) -> float:
            total = 0.0
            for ck, sk in zip(c, pay_times):
                total += ck * _g2pp_zcb(a, b, sigma1, sigma2, rho,
                                        curve, x_val, y_val, sk)
            return total - 1.0

        # Bracket root for Jamshidian's y*.
        # Fix T2.12: pre-fix this fell back to `y_star = 0.0` if either
        # the bracket expansion failed (no sign change after 10 rounds) or
        # any internal exception raised.  `y_star = 0` is just wrong —
        # the K_k strikes derived from it are unrelated to the actual
        # swaption, producing wildly wrong prices that the outer
        # `except Exception: return 0.0` then masked further.
        # Post-fix: raise instead of silent fallback.
        y_lo, y_hi = -0.5, 0.5
        bp_lo = bond_portfolio(y_lo)
        bp_hi = bond_portfolio(y_hi)
        # Expand bracket up to 10 doublings.
        for _ in range(10):
            if bp_lo * bp_hi < 0:
                break
            y_lo -= 0.5
            y_hi += 0.5
            bp_lo = bond_portfolio(y_lo)
            bp_hi = bond_portfolio(y_hi)

        if bp_lo * bp_hi >= 0:
            raise RuntimeError(
                f"g2pp_swaption_price: failed to bracket Jamshidian y* "
                f"for x={x_val:.4f}; bond_portfolio({y_lo:.2f})={bp_lo:.4e}, "
                f"bond_portfolio({y_hi:.2f})={bp_hi:.4e} have the same "
                f"sign after 10 expansions.  Likely model parameters "
                f"are out of regime (a, b, sigma1, sigma2, rho)."
            )
        from scipy.optimize import brentq
        y_star = brentq(bond_portfolio, y_lo, y_hi, xtol=1e-10, maxiter=50)

        # Strike prices for individual ZCB options
        # K_k = P(x_val, y_star; s_k) (the ZCB value at y*)
        k_strikes = [
            _g2pp_zcb(a, b, sigma1, sigma2, rho, curve, x_val, y_star, sk)
            for sk in pay_times
        ]

        # Price each ZCB option (analytically in y given x)
        # Conditional vol of y: N(rho*sigma2/sigma1 * (1-e^{-bt})/b * x, var_y|x)
        # For the G2++ bond option formula, we use the unconditional
        # formula (which already integrates over both factors).
        # Here we weight by the x density and price analytically.
        swaption_contrib = 0.0
        for ck, sk, kk in zip(c, pay_times, k_strikes):
            opt_price = _g2pp_zcb_option(
                a, b, sigma1, sigma2, rho, curve,
                t, sk, sk, kk, is_zcb_call,
            )
            swaption_contrib += ck * opt_price

        # GH weight includes 1/sqrt(pi) normalisation
        total_price += w_gh * swaption_contrib

    # Normalise: GH integral of f(x)*exp(-x^2) dx, we want E[...] over N(0,1)
    # gh_weights already normalised for exp(-x^2); divide by sqrt(pi)
    total_price /= math.sqrt(math.pi)

    return max(total_price, 0.0)



# ---------------------------------------------------------------------------
# G2++ implied Black vol inversion
# ---------------------------------------------------------------------------

def g2pp_implied_vol(
    a: float,
    b: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    curve: DiscountCurve,
    expiry_years: float,
    tenor_years: float,
    strike: float,
) -> float:
    """Compute G2++-implied Black-76 swaption vol.

    Prices the swaption analytically then inverts Black-76 for the implied vol.

    Args:
        a, b, sigma1, sigma2, rho: G2++ parameters.
        curve: discount curve.
        expiry_years: swaption expiry in years.
        tenor_years: underlying swap tenor in years.
        strike: swap fixed rate.

    Returns:
        Implied Black-76 vol, or 0.0 if inversion fails.
    """
    from pricebook.options.implied_vol import implied_vol_black76

    ref = curve.reference_date
    price = g2pp_swaption_price(
        a, b, sigma1, sigma2, rho, curve,
        expiry_years, tenor_years, strike, is_payer=True,
    )
    if price <= 0:
        return 0.0

    # Annuity and forward swap rate
    swap_end = expiry_years + tenor_years
    n_pay = max(1, int(round(tenor_years)))
    annuity = 0.0
    for k in range(1, n_pay + 1):
        t_pay = expiry_years + k
        if t_pay <= swap_end + 1e-9:
            annuity += curve.df(date_from_year_fraction(ref, t_pay))

    if annuity <= 0:
        return 0.0

    df_exp = curve.df(date_from_year_fraction(ref, expiry_years))
    df_end = curve.df(date_from_year_fraction(ref, swap_end))
    fwd_swap = (df_exp - df_end) / annuity if annuity > 0 else strike
    if fwd_swap <= 0:
        fwd_swap = strike

    try:
        iv = implied_vol_black76(
            price / annuity,
            fwd_swap, strike, expiry_years, 1.0, OptionType.CALL,
        )
        return iv
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_g2pp(
    curve: DiscountCurve,
    swaption_vols: dict[tuple[float, float], float],
    strike: float | None = None,
    method: str = "differential_evolution",
    a_bounds: tuple[float, float] = (0.001, 0.5),
    b_bounds: tuple[float, float] = (0.001, 0.5),
    sigma1_bounds: tuple[float, float] = (0.001, 0.05),
    sigma2_bounds: tuple[float, float] = (0.001, 0.05),
    rho_bounds: tuple[float, float] = (-0.99, 0.99),
    *,
    market_snapshot: MarketSnapshot | None = None,
) -> G2PPCalibrationResult:
    """Calibrate G2++ (a, b, sigma1, sigma2, rho) to swaption vols.

    Minimises RMSE of Black-implied vol errors. The 5-parameter space uses
    Differential Evolution as the global optimizer, followed by an L-BFGS-B
    local polish step.

    Args:
        curve: initial discount curve.
        swaption_vols: {(expiry_years, tenor_years): black_vol}.
            Example: {(1, 5): 0.0065, (5, 5): 0.0055, (10, 10): 0.0045}
        strike: fixed ATM strike. If None, computed from forward swap rates.
        method: "differential_evolution" (default) or "minimize".
        a_bounds: bounds for first factor mean reversion.
        b_bounds: bounds for second factor mean reversion.
        sigma1_bounds: bounds for first factor vol.
        sigma2_bounds: bounds for second factor vol.
        rho_bounds: bounds for factor correlation.

    Returns:
        G2PPCalibrationResult with calibrated G2PlusPlus model.
    """
    from pricebook.models.black76 import black76_price as _b76, OptionType as _OT

    ref = curve.reference_date
    keys = list(swaption_vols.keys())
    market_vols = [swaption_vols[k] for k in keys]

    # Compute ATM strikes + annuities + forwards once.
    strikes: dict[tuple[float, float], float] = {}
    annuities: dict[tuple[float, float], float] = {}
    forwards: dict[tuple[float, float], float] = {}
    for (exp_y, tenor_y) in keys:
        swap_end = exp_y + tenor_y
        n_pay = max(1, int(round(tenor_y)))
        ann = sum(
            curve.df(date_from_year_fraction(ref, exp_y + k))
            for k in range(1, n_pay + 1)
            if exp_y + k <= swap_end + 1e-9
        )
        df_exp = curve.df(date_from_year_fraction(ref, exp_y))
        df_end = curve.df(date_from_year_fraction(ref, swap_end))
        fwd = (df_exp - df_end) / ann if ann > 0 else (strike if strike else 0.04)
        annuities[(exp_y, tenor_y)] = ann
        forwards[(exp_y, tenor_y)] = fwd
        strikes[(exp_y, tenor_y)] = strike if strike is not None else fwd

    # Precompute market Black-76 prices once — the objective then compares
    # G2++ prices to these directly, avoiding an implied-vol root-find per
    # objective evaluation (a ~2x speedup on the inner loop).
    market_prices: dict[tuple[float, float], float] = {}
    for i, (exp_y, tenor_y) in enumerate(keys):
        k = strikes[(exp_y, tenor_y)]
        market_prices[(exp_y, tenor_y)] = annuities[(exp_y, tenor_y)] * _b76(
            forwards[(exp_y, tenor_y)], k, market_vols[i], exp_y, 1.0, _OT.CALL,
        )

    def objective(params: np.ndarray) -> float:
        a_, b_, s1, s2, rho_ = params
        # Enforce strict positivity and correlation bounds
        if a_ <= 0 or b_ <= 0 or s1 <= 0 or s2 <= 0 or abs(rho_) >= 1.0:
            return 1e6
        total = 0.0
        for (exp_y, tenor_y) in keys:
            k = strikes[(exp_y, tenor_y)]
            ann = annuities[(exp_y, tenor_y)]
            model_p = g2pp_swaption_price(
                a_, b_, s1, s2, rho_, curve,
                exp_y, tenor_y, k, is_payer=True,
            )
            # Scale by annuity so the loss has annuity-free units and
            # matches the order of magnitude across swaptions.
            diff = (model_p - market_prices[(exp_y, tenor_y)]) / ann
            total += diff * diff
        return total

    bounds = [a_bounds, b_bounds, sigma1_bounds, sigma2_bounds, rho_bounds]
    x0 = [0.05, 0.10, 0.01, 0.008, -0.5]

    converged = False
    if method == "differential_evolution":
        # DE for the global search. Per-objective cost is dominated by 4 G2++
        # swaption prices (~30ms each via Jamshidian + brentq), so each DE
        # iteration costs ~popsize × N_params × 4 × 30ms. Keep popsize and
        # maxiter modest; L-BFGS-B polish refines locally.
        # Previous defaults (popsize=15, maxiter=300, tol=1e-9, vol-objective)
        # ran ~10 min on the default fixture without measurable improvement
        # in final RMSE.
        de_result = differential_evolution(
            objective, bounds,
            maxiter=30, popsize=6, tol=1e-4,
            seed=42, workers=1, polish=False, init="sobol",
        )
        # Polish with L-BFGS-B — given the budget, do the bulk of the
        # refinement here (cheap, deterministic) rather than in DE.
        local_result = minimize(
            objective, de_result.x, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 150, "ftol": 1e-9},
        )
        result_x = local_result.x if local_result.fun < de_result.fun else de_result.x
        converged = de_result.success or local_result.success
    else:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-12})
        result_x = res.x
        converged = res.success

    a_opt, b_opt, s1_opt, s2_opt, rho_opt = (
        max(result_x[0], 1e-4),
        max(result_x[1], 1e-4),
        max(result_x[2], 1e-6),
        max(result_x[3], 1e-6),
        float(np.clip(result_x[4], -0.999, 0.999)),
    )

    g2 = G2PlusPlus(a=a_opt, b=b_opt, sigma1=s1_opt, sigma2=s2_opt,
                    rho=rho_opt, curve=curve)

    errors = []
    for i, (exp_y, tenor_y) in enumerate(keys):
        k = strikes[(exp_y, tenor_y)]
        mv = g2pp_implied_vol(a_opt, b_opt, s1_opt, s2_opt, rho_opt,
                              curve, exp_y, tenor_y, k)
        err_bp = (mv - market_vols[i]) * 10_000
        errors.append({
            "expiry": exp_y,
            "tenor": tenor_y,
            "market_vol": market_vols[i],
            "model_vol": mv,
            "error_bp": err_bp,
        })

    rmse_vol = math.sqrt(
        sum((e["model_vol"] - e["market_vol"]) ** 2 for e in errors) / len(errors)
    ) if errors else 0.0

    # Build the canonical CalibrationResult. The objective the optimiser
    # actually minimised is in *price* space (annuity-normalised diffs);
    # we record residuals in *vol* space (bp of Black vol) because that's
    # the unit users read in audit logs and that matches the existing
    # per_swaption_errors[i]["error_bp"]. The mapping between the two
    # spaces is the per-swaption annuity, which is implicit in g2.
    if method == "differential_evolution":
        algo_name = "differential_evolution+L-BFGS-B"
        algo_tol = 1e-9   # the polish step's ftol
        algo_maxiter = 150  # the polish step's maxiter
        algo_seed = 42
    else:
        algo_name = "L-BFGS-B"
        algo_tol = 1e-12
        algo_maxiter = 500
        algo_seed = None

    cr = CalibrationResult.new(
        model_class="g2pp",
        parameters={
            "a": float(a_opt), "b": float(b_opt),
            "sigma1": float(s1_opt), "sigma2": float(s2_opt), "rho": float(rho_opt),
        },
        residuals=[e["error_bp"] for e in errors],  # bp of Black vol
        objective=ObjectiveKind.SSE,                  # price-space SSE; bp-residuals for readability
        optimiser=OptimiserSpec(
            algorithm=algo_name,
            tolerance=algo_tol,
            max_iterations=algo_maxiter,
            seed=algo_seed,
            extra={
                "a_bounds": a_bounds,
                "b_bounds": b_bounds,
                "sigma1_bounds": sigma1_bounds,
                "sigma2_bounds": sigma2_bounds,
                "rho_bounds": rho_bounds,
            },
        ),
        iterations=0,  # scipy's compound optimisers don't surface a single iteration count
        converged=bool(converged),
        quotes_fitted=[f"swaption_{exp_y}x{tenor_y}" for (exp_y, tenor_y) in keys],
        diagnostics=CalibrationDiagnostics(
            extra={"rmse_vol": float(rmse_vol)},
        ),
        market_snapshot_id=market_snapshot.id if market_snapshot is not None else None,
    )

    return G2PPCalibrationResult(
        model=g2,
        a=a_opt, b=b_opt, sigma1=s1_opt, sigma2=s2_opt, rho=rho_opt,
        rmse_vol=rmse_vol,
        per_swaption_errors=errors,
        n_swaptions=len(keys),
        converged=converged,
        calibration_result=cr,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def g2pp_vs_hw1f(
    curve: DiscountCurve,
    swaption_vols: dict[tuple[float, float], float],
) -> dict:
    """Calibrate both G2++ and Hull-White 1F; compare RMSE and per-swaption fits.

    Shows where the two-factor model adds explanatory power over the
    one-factor Hull-White model.

    Args:
        curve: initial discount curve.
        swaption_vols: {(expiry_years, tenor_years): black_vol}.

    Returns:
        dict with keys "g2pp", "hw1f", "improvement_bp", "per_swaption".
        "improvement_bp" is the reduction in RMSE from G2++ vs HW1F (in bp).
        "per_swaption" is a list of dicts comparing both models per swaption.
    """
    from pricebook.models.hw_calibration import calibrate_hull_white

    g2_result = calibrate_g2pp(curve, swaption_vols)
    hw_result = calibrate_hull_white(curve, swaption_vols)

    # Build per-swaption comparison table
    g2_by_key = {
        (e["expiry"], e["tenor"]): e for e in g2_result.per_swaption_errors
    }
    hw_by_key = {
        (e["expiry"], e["tenor"]): e for e in hw_result.per_swaption_errors
    }

    per_swaption = []
    for key in swaption_vols:
        exp_y, tenor_y = key
        g2e = g2_by_key.get(key, {})
        hwe = hw_by_key.get(key, {})
        per_swaption.append({
            "expiry": exp_y,
            "tenor": tenor_y,
            "market_vol": swaption_vols[key],
            "g2pp_vol": g2e.get("model_vol", float("nan")),
            "hw1f_vol": hwe.get("model_vol", float("nan")),
            "g2pp_error_bp": g2e.get("error_bp", float("nan")),
            "hw1f_error_bp": hwe.get("error_bp", float("nan")),
        })

    improvement_bp = (hw_result.rmse_vol - g2_result.rmse_vol) * 10_000

    return {
        "g2pp": {
            "a": g2_result.a, "b": g2_result.b,
            "sigma1": g2_result.sigma1, "sigma2": g2_result.sigma2,
            "rho": g2_result.rho,
            "rmse_vol": g2_result.rmse_vol,
            "rmse_bp": g2_result.rmse_vol * 10_000,
            "converged": g2_result.converged,
        },
        "hw1f": {
            "a": hw_result.a,
            "sigma": hw_result.sigma,
            "rmse_vol": hw_result.rmse_vol,
            "rmse_bp": hw_result.rmse_vol * 10_000,
            "converged": hw_result.converged,
        },
        "improvement_bp": improvement_bp,
        "g2pp_better": improvement_bp > 0,
        "per_swaption": per_swaption,
    }
