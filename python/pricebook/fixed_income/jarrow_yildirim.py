"""Jarrow-Yildirim inflation model: joint nominal rate, real rate, CPI.

Three-factor model for pricing inflation-linked derivatives:

* :class:`JarrowYildirim` — joint dynamics and simulation.
* :func:`jy_zc_inflation_swap` — analytical ZC inflation swap price.
* :func:`jy_yoy_caplet` — YoY inflation caplet under JY.
* :func:`jy_calibrate` — calibrate to ZC and YoY term structures.

References:
    Jarrow & Yildirim, *Pricing Treasury Inflation Protected Securities and
    Related Derivatives Using an HJM Model*, JFE, 2003.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 15.
    Mercurio, *Pricing Inflation-Indexed Derivatives*, QF, 2005.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    CanonicalCalibrationResult,
    SolveReport,
    model_calibration_record,
)


@dataclass
class JYParams:
    """Jarrow-Yildirim model parameters."""
    a_n: float      # nominal rate mean reversion
    sigma_n: float  # nominal rate vol
    a_r: float      # real rate mean reversion
    sigma_r: float  # real rate vol
    sigma_I: float  # CPI index vol
    rho_nr: float   # correlation: nominal × real
    rho_nI: float   # correlation: nominal × CPI
    rho_rI: float   # correlation: real × CPI



    def to_dict(self) -> dict:
        return dict(vars(self))
@dataclass
class JYSimulationResult:
    """JY simulation result."""
    nominal_rate_paths: np.ndarray   # (n_paths, n_steps+1)
    real_rate_paths: np.ndarray
    cpi_paths: np.ndarray
    times: np.ndarray
    mean_terminal_cpi: float
    mean_terminal_nominal: float
    mean_terminal_real: float



    def to_dict(self) -> dict:
        return dict(vars(self))
class JarrowYildirim:
    """Jarrow-Yildirim three-factor inflation model.

    Dynamics (under nominal risk-neutral measure Q_n):
        dr_n = [θ_n(t) − a_n r_n] dt + σ_n dW_n
        dr_r = [θ_r(t) − a_r r_r − ρ_{rI} σ_r σ_I] dt + σ_r dW_r
        dI/I = [r_n − r_r] dt + σ_I dW_I

    θ_n(t), θ_r(t) calibrated to initial nominal and real term structures.

    Args:
        params: JY model parameters.
        r_n0: initial nominal short rate.
        r_r0: initial real short rate.
        I0: initial CPI index level.
    """

    def __init__(
        self,
        params: JYParams,
        r_n0: float,
        r_r0: float,
        I0: float,
        nominal_forwards: np.ndarray | None = None,
        real_forwards: np.ndarray | None = None,
        forward_times: np.ndarray | None = None,
    ):
        self.params = params
        self.r_n0 = r_n0
        self.r_r0 = r_r0
        self.I0 = I0
        self._nominal_fwd = nominal_forwards
        self._real_fwd = real_forwards
        self._fwd_times = forward_times

    @staticmethod
    def _theta(
        t: float,
        a: float,
        sigma: float,
        r0: float,
        forwards: np.ndarray | None,
        fwd_times: np.ndarray | None,
    ) -> float:
        """Hull-White θ(t) calibrated to initial term structure.

        θ(t) = ∂f(0,t)/∂t + a×f(0,t) + σ²/(2a)(1 - e^{-2at})

        If no forward curve provided, falls back to constant θ = a×r0
        (mean-reversion to initial short rate).
        """
        if forwards is None or fwd_times is None or len(forwards) < 2:
            return a * r0

        # Interpolate f(0,t) and ∂f/∂t from the forward curve
        f_t = float(np.interp(t, fwd_times, forwards))
        dt = fwd_times[1] - fwd_times[0] if len(fwd_times) > 1 else 0.01
        idx = min(int(t / dt), len(forwards) - 2)
        dfdt = (forwards[min(idx + 1, len(forwards) - 1)] - forwards[idx]) / dt

        if a > 1e-10:
            return dfdt + a * f_t + sigma**2 / (2 * a) * (1 - math.exp(-2 * a * t))
        return dfdt + sigma**2 * t

    def simulate(
        self,
        T: float,
        n_paths: int = 5_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ) -> JYSimulationResult:
        """Simulate joint (r_n, r_r, I) paths."""
        p = self.params
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0, T, n_steps + 1)

        # Cholesky for 3 correlated Brownians
        corr = np.array([
            [1.0, p.rho_nr, p.rho_nI],
            [p.rho_nr, 1.0, p.rho_rI],
            [p.rho_nI, p.rho_rI, 1.0],
        ])
        # Ensure PD
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 0:
            corr += (-eigvals.min() + 1e-6) * np.eye(3)
        L = np.linalg.cholesky(corr)

        r_n = np.full((n_paths, n_steps + 1), self.r_n0)
        r_r = np.full((n_paths, n_steps + 1), self.r_r0)
        I = np.full((n_paths, n_steps + 1), float(self.I0))

        for step in range(n_steps):
            t = step * dt
            Z = rng.standard_normal((n_paths, 3)) @ L.T

            # θ(t) calibrated to initial term structure (if available)
            theta_n = self._theta(t, p.a_n, p.sigma_n, self.r_n0,
                                   self._nominal_fwd, self._fwd_times)
            theta_r = self._theta(t, p.a_r, p.sigma_r, self.r_r0,
                                   self._real_fwd, self._fwd_times)

            # Nominal rate (HW): dr_n = (θ_n(t) - a_n r_n) dt + σ_n dW
            r_n[:, step + 1] = (r_n[:, step]
                                 + (theta_n - p.a_n * r_n[:, step]) * dt
                                 + p.sigma_n * Z[:, 0] * sqrt_dt)

            # Real rate (HW with drift adjustment)
            drift_r = (theta_r - p.a_r * r_r[:, step]
                       - p.rho_rI * p.sigma_r * p.sigma_I) * dt
            r_r[:, step + 1] = r_r[:, step] + drift_r + p.sigma_r * Z[:, 1] * sqrt_dt

            # CPI index
            drift_I = (r_n[:, step] - r_r[:, step]) * dt
            I[:, step + 1] = I[:, step] * np.exp(
                drift_I - 0.5 * p.sigma_I**2 * dt + p.sigma_I * Z[:, 2] * sqrt_dt
            )

        return JYSimulationResult(
            nominal_rate_paths=r_n,
            real_rate_paths=r_r,
            cpi_paths=I,
            times=times,
            mean_terminal_cpi=float(I[:, -1].mean()),
            mean_terminal_nominal=float(r_n[:, -1].mean()),
            mean_terminal_real=float(r_r[:, -1].mean()),
        )


# ---- Analytical ZC inflation swap ----

@dataclass
class JYZCSwapResult:
    """JY ZC inflation swap result."""
    fair_rate: float
    nominal_zcb: float
    real_zcb: float
    convexity_adjustment: float



    def to_dict(self) -> dict:
        return dict(vars(self))
def jy_zc_inflation_swap(
    params: JYParams,
    r_n0: float,
    r_r0: float,
    T: float,
) -> JYZCSwapResult:
    """Analytical ZC inflation swap rate under JY.

    Fair ZC rate ≈ (P_n(0,T) / P_r(0,T)) × exp(convexity) − 1

    where P_n, P_r are nominal and real ZCB prices.

    Simplified: use Hull-White analytical ZCB for each rate.
    """
    p = params

    # Vasicek ZCB with constant mean-reversion target θ = r₀.
    # P(0, T) = exp(A(0, T) - B(T)·r₀), where for a > 0:
    #   B(T) = (1 - e^{-aT}) / a
    #   A(0, T) = (B - T)·(θ - σ²/(2a²)) - σ²·B²/(4a)   with θ = r₀
    # So the ZCB exponent simplifies to
    #   -T·r₀ + (T - B)·σ²/(2a²) - σ²·B²/(4a)
    # At σ → 0 this collapses to the flat-rate ZCB exp(-r₀·T).
    #
    # Fix T4-JY1: the pre-fix expression used
    #   exp(-B·r₀  −σ²(T-B)/(2a²)·... + σ²·B²/(4a))
    # which:
    #   (a) replaced ``-T·r₀`` with ``-B·r₀`` — missing the
    #       ``-(T-B)·r₀`` term entirely (the dominant rate-only piece),
    #   (b) flipped the sign of the ``(T-B)·σ²/(2a²)`` convexity term,
    #   (c) flipped the sign of the ``σ²·B²/(4a)`` term.
    # For a=0.05, σ=0.01, r₀=0.04, T=5 this produced ZCB ≈ 0.836 vs
    # the correct 0.820 — a ~2% bias propagating into both P_n and P_r
    # (with different magnitudes for σ_n vs σ_r so the ratio does not
    # cancel), corrupting fair_rate.
    def hw_zcb(a, sigma, r0, T):
        if a < 1e-10:
            # Limit: σ²·T³/6 convexity, no mean reversion.
            return math.exp(-r0 * T + sigma**2 * T**3 / 6.0)
        B = (1.0 - math.exp(-a * T)) / a
        exponent = (-r0 * T
                    + (T - B) * sigma**2 / (2.0 * a**2)
                    - sigma**2 * B**2 / (4.0 * a))
        return math.exp(exponent)

    P_n = hw_zcb(p.a_n, p.sigma_n, r_n0, T)
    P_r = hw_zcb(p.a_r, p.sigma_r, r_r0, T)

    # Convexity adjustment from correlation between nominal rate and CPI
    B_n = (1 - math.exp(-p.a_n * T)) / max(p.a_n, 1e-10)
    conv_adj = -p.rho_nI * p.sigma_n * p.sigma_I * B_n * T

    # JY ZC inflation forward (Jarrow-Yildirim 2003 eq. 16, Mercurio 2005):
    #   I_fwd(0, T) / I(0) = P_r(0, T) / P_n(0, T) · exp(conv_adj)
    # where P_r is the real-currency ZCB.  Pre-fix the code used the
    # inverted ratio P_n / P_r, giving fair_rate ≈ exp(-(r_n - r_r)·T) − 1
    # — negative for the typical r_n > r_r setup (instead of the expected
    # positive inflation-breakeven).  Fix T4-JY1: invert the ratio.
    ratio = P_r / max(P_n, 1e-10) * math.exp(conv_adj)
    fair_rate = ratio - 1

    return JYZCSwapResult(
        fair_rate=float(fair_rate),
        nominal_zcb=float(P_n),
        real_zcb=float(P_r),
        convexity_adjustment=float(conv_adj),
    )


# ---- YoY inflation caplet ----

@dataclass
class JYCapletResult:
    """JY YoY inflation caplet result."""
    price: float
    forward_yoy_rate: float
    effective_vol: float
    strike: float



    def to_dict(self) -> dict:
        return dict(vars(self))
def jy_yoy_caplet(
    params: JYParams,
    r_n0: float,
    r_r0: float,
    T_start: float,
    T_end: float,
    strike: float,
    notional: float = 1.0,
) -> JYCapletResult:
    """YoY inflation caplet under JY.

    Payoff: notional × max(I(T_end)/I(T_start) − 1 − K, 0).

    Priced via Black on the forward YoY rate with JY effective vol.
    """
    from pricebook.models.black76 import black76_price, OptionType

    p = params
    tau = T_end - T_start

    # Forward YoY rate ≈ ZC rate over [T_start, T_end]
    zc_start = jy_zc_inflation_swap(params, r_n0, r_r0, T_start)
    zc_end = jy_zc_inflation_swap(params, r_n0, r_r0, T_end)

    forward_yoy = (1 + zc_end.fair_rate) / (1 + zc_start.fair_rate) - 1

    # Effective vol: σ_I is the instantaneous CPI index vol.
    # Under JY, the YoY forward rate vol ≈ σ_I (simplified, ignoring
    # rate-CPI correlation and B(T) corrections). Black76 with T=T_start
    # then produces the correct √T_start scaling internally.
    eff_vol = p.sigma_I

    # Black on forward YoY rate
    df = math.exp(-r_n0 * T_end)
    fwd = max(forward_yoy, 1e-6)  # floor for Black

    price = notional * tau * black76_price(fwd, strike, eff_vol, T_start, df, OptionType.CALL)

    return JYCapletResult(
        price=float(max(price, 0.0)),
        forward_yoy_rate=float(forward_yoy),
        effective_vol=float(eff_vol),
        strike=strike,
    )


# ---- Calibration ----

@dataclass
class JYCalibrationResult(CanonicalCalibrationResult):
    """JY calibration result.

    `calibration_result` carries the canonical provenance artefact;
    `to_calibration_result()` returns it when populated by `jy_calibrate`,
    or builds one on-demand from the existing fields (back-compat path —
    which only has the aggregate `residual`).
    """

    params: JYParams
    residual: float
    n_instruments: int
    # Canonical calibration artefact (G1 P2 — widen producers).
    calibration_result: CalibrationResult | None = None

    def to_dict(self) -> dict:
        return {
            "params": dict(vars(self.params)),
            "residual": self.residual,
            "n_instruments": self.n_instruments,
            "calibration_id": self.calibration_id,
        }

    def _build_calibration_record(self) -> CalibrationResult:
        p = self.params
        # Lazy reconstruction (hand-built instance): no optimiser ran, so
        # convergence is not captured (None) — never guessed from a threshold.
        solve = SolveReport.external(algorithm="Nelder-Mead", converged=None, iterations=0)
        return model_calibration_record(
            model_class="jarrow_yildirim",
            parameters={
                "sigma_n": float(p.sigma_n),
                "sigma_r": float(p.sigma_r),
                "sigma_I": float(p.sigma_I),
            },
            residuals=[float(self.residual)],
            quotes_fitted=["aggregate_objective"],
            solve=solve,
            diagnostics=CalibrationDiagnostics(
                extra={"n_instruments": int(self.n_instruments)}, reconstructed=True),
        )


def jy_calibrate(
    zc_swap_rates: dict[float, float],      # {T: fair_ZC_rate}
    r_n0: float,
    r_r0: float,
    a_n: float = 0.05,
    a_r: float = 0.05,
    rho_nr: float = 0.3,
    rho_nI: float = -0.2,
    rho_rI: float = 0.1,
) -> JYCalibrationResult:
    """Calibrate JY (σ_n, σ_r, σ_I) to ZC inflation swap term structure.

    Fixed: a_n, a_r, correlations (typically from historical estimation).
    Free: σ_n, σ_r, σ_I (fit to ZC swap quotes).
    """
    tenors = sorted(zc_swap_rates.keys())
    targets = np.array([zc_swap_rates[t] for t in tenors])

    def objective(x):
        sigma_n, sigma_r, sigma_I = x
        if sigma_n <= 0 or sigma_r <= 0 or sigma_I <= 0:
            return 1e10
        params = JYParams(a_n, sigma_n, a_r, sigma_r, sigma_I,
                           rho_nr, rho_nI, rho_rI)
        err = 0.0
        for t, target_rate in zip(tenors, targets):
            model = jy_zc_inflation_swap(params, r_n0, r_r0, t)
            err += (model.fair_rate - target_rate) ** 2
        return err

    result = minimize(objective, [0.01, 0.005, 0.02], method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-10})

    sigma_n, sigma_r, sigma_I = result.x
    params = JYParams(a_n, max(sigma_n, 1e-6), a_r, max(sigma_r, 1e-6),
                       max(sigma_I, 1e-6), rho_nr, rho_nI, rho_rI)
    residual = math.sqrt(result.fun / len(tenors))

    residuals_per = [
        jy_zc_inflation_swap(params, r_n0, r_r0, t).fair_rate - zc_swap_rates[t]
        for t in tenors
    ]
    solve = SolveReport.external(
        algorithm="Nelder-Mead", converged=bool(getattr(result, "success", True)),
        iterations=int(getattr(result, "nit", 0)), tolerance=1e-10, max_iterations=3000,
    )
    cr = model_calibration_record(
        model_class="jarrow_yildirim",
        parameters={
            "sigma_n": float(params.sigma_n),
            "sigma_r": float(params.sigma_r),
            "sigma_I": float(params.sigma_I),
        },
        residuals=residuals_per,
        quotes_fitted=[f"zc_inflation_swap_{t}" for t in tenors],
        solve=solve,
    )

    return JYCalibrationResult(params, float(residual), len(tenors), calibration_result=cr)


# ---------------------------------------------------------------------------
# Unified MC Engine migration
# ---------------------------------------------------------------------------


def jy_simulate_via_engine(
    jy: JarrowYildirim,
    T: float,
    n_paths: int = 5_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> JYSimulationResult:
    """``JarrowYildirim.simulate`` with HW rate paths from unified MC engine.

    The nominal and real rates use the engine's HW path generator;
    CPI uses GBM driven by (r_n - r_r) drift.
    """
    from pricebook.models.mc_migrate import hw_paths  # noqa: lazy

    p = jy.params
    _seed = seed if seed is not None else 42

    # Nominal rate via HW engine
    r_n = hw_paths(
        r0=jy.r_n0, a=p.a_n, sigma=p.sigma_n, T=T,
        n_steps=n_steps, n_paths=n_paths, seed=_seed,
        theta_func=lambda t: jy._theta(t, p.a_n, p.sigma_n, jy.r_n0,
                                         jy._nominal_fwd, jy._fwd_times),
    )

    # Real rate via HW engine (with drift adjustment for rho_rI)
    r_r = hw_paths(
        r0=jy.r_r0, a=p.a_r, sigma=p.sigma_r, T=T,
        n_steps=n_steps, n_paths=n_paths, seed=_seed + 1,
        theta_func=lambda t: jy._theta(t, p.a_r, p.sigma_r, jy.r_r0,
                                         jy._real_fwd, jy._fwd_times)
                             - p.rho_rI * p.sigma_r * p.sigma_I,
    )

    # CPI via manual Euler (driven by r_n - r_r drift, not a standard process)
    rng = np.random.default_rng(_seed + 2)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    I = np.full((n_paths, n_steps + 1), float(jy.I0))
    for step in range(n_steps):
        Z = rng.standard_normal(n_paths)
        drift_I = (r_n[:, step] - r_r[:, step]) * dt
        I[:, step + 1] = I[:, step] * np.exp(
            drift_I - 0.5 * p.sigma_I**2 * dt + p.sigma_I * Z * sqrt_dt
        )

    times = np.linspace(0, T, n_steps + 1)
    return JYSimulationResult(
        nominal_rate_paths=r_n, real_rate_paths=r_r, cpi_paths=I,
        times=times,
        mean_terminal_cpi=float(I[:, -1].mean()),
        mean_terminal_nominal=float(r_n[:, -1].mean()),
        mean_terminal_real=float(r_r[:, -1].mean()),
    )
