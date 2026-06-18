"""Bootstrap hazard rates from bond prices when no CDS curve is available.

The problem: given N risky bonds of the same issuer at observed market prices,
extract a piecewise constant hazard rate term structure that reprices all bonds.

This is harder than CDS bootstrapping because:
1. Bonds have large intermediate cashflows (coupons), not just protection/premium legs
2. Coupon dates may not align across bonds
3. Recovery assumption has more impact (recovery of par vs recovery of market value)
4. Illiquid bonds have wide bid-ask → noisy input prices
5. Bonds may have embedded options (callable), accrued interest, settlement conventions
6. The problem is over-determined if N > number of hazard pillars (need fitting, not exact)

    from pricebook.credit.bond_hazard_bootstrap import (
        bootstrap_hazard_from_bonds, HazardBootstrapResult,
    )

References:
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives, Ch 6.
    Duffie & Singleton (1999). Modeling Term Structures of Defaultable Bonds.
    Hull, Predescu & White (2004). Bond Prices, Default Probabilities and Risk Premiums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize, brentq

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.schedule import Frequency, generate_schedule
from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)

if TYPE_CHECKING:
    from pricebook.market_data import MarketSnapshot


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class BondInput:
    """A single bond observation for hazard rate calibration."""
    maturity: date
    coupon: float          # annual coupon rate (e.g. 0.05 = 5%)
    market_price: float    # dirty price per 100 face
    frequency: int = 2     # coupons per year
    recovery: float = 0.40 # assumed recovery on default (fraction of par)
    weight: float = 1.0    # fitting weight (lower for illiquid bonds)
    liquidity_spread_bp: float = 0.0  # liquidity premium to strip before credit extraction

    def to_dict(self) -> dict:
        return {**vars(self), "maturity": self.maturity.isoformat()}


RECOVERY_PAR = "par"
RECOVERY_MARKET_VALUE = "market_value"


@dataclass
class HazardBootstrapResult:
    """Result of hazard rate calibration from bond prices.

    Carries both the bond-hazard-specific outputs (survival curve, per-bond
    fitted prices, residuals) AND the canonical `CalibrationResult` artefact
    (provenance, optimiser story, structured diagnostics). The two views
    overlap on `pillar_hazards` (as `calibration_result.parameters`) and
    on `rmse_bp` / `max_error_bp` (derivable from
    `calibration_result.residuals`). Callers that need the canonical form
    for audit / persistence read `calibration_result`; callers that need
    the bond-specific shape stay on the existing fields.
    """
    survival_curve: SurvivalCurve
    pillar_dates: list[date]
    pillar_hazards: list[float]   # piecewise constant h between pillars
    fitted_prices: list[float]    # model price per bond
    market_prices: list[float]
    residuals_bp: list[float]     # (model - market) in bp of par
    rmse_bp: float                # root mean square error in bp
    max_error_bp: float
    n_bonds: int
    method: str
    converged: bool
    lam: float = 0.0              # Tikhonov regularisation strength used (0 = unregularised)
    roughness: float = 0.0        # ||L h||² of the fitted hazard (second-difference penalty)
    # New in G1 P1 Slice 2 — canonical calibration artefact. `None` only
    # for back-compat with hand-constructed instances; entry points
    # `_bootstrap_sequential` / `_bootstrap_global` always populate it.
    calibration_result: CalibrationResult | None = None

    def to_dict(self) -> dict:
        return {
            "pillar_dates": [d.isoformat() for d in self.pillar_dates],
            "pillar_hazards": self.pillar_hazards,
            "fitted_prices": self.fitted_prices,
            "market_prices": self.market_prices,
            "residuals_bp": self.residuals_bp,
            "rmse_bp": self.rmse_bp,
            "max_error_bp": self.max_error_bp,
            "n_bonds": self.n_bonds,
            "method": self.method,
            "converged": self.converged,
            "lam": self.lam,
            "roughness": self.roughness,
            "calibration_id": (
                str(self.calibration_result.id) if self.calibration_result else None
            ),
        }

    def to_calibration_result(self) -> CalibrationResult:
        """Return the canonical CalibrationResult.

        If populated by the entry point (the normal path), returns that
        instance — preserving its `id`, `timestamp`, and full provenance.
        Otherwise constructs one on the fly from the available fields with
        minimal optimiser metadata (used only when the result was hand-built
        rather than produced by `bootstrap_hazard_from_bonds`).
        """
        if self.calibration_result is not None:
            return self.calibration_result
        return CalibrationResult.new(
            model_class="bond_hazard_pwc",
            parameters=_pillar_hazards_as_parameters(self.pillar_dates, self.pillar_hazards),
            residuals=self.residuals_bp,
            objective=ObjectiveKind.WEIGHTED_SSE,
            optimiser=OptimiserSpec(algorithm=self.method, tolerance=0.0, max_iterations=0),
            iterations=0,
            converged=self.converged,
            quotes_fitted=[f"bond_{i}" for i in range(self.n_bonds)],
        )


def _pillar_hazards_as_parameters(
    pillar_dates: list[date], pillar_hazards: list[float],
) -> dict[str, float]:
    """Encode piecewise-constant hazard rates as a named parameter dict.

    Key shape `h(<tenor>y)` where `<tenor>` is the pillar tenor in years
    (formatted to 3 decimals) — readable in audit logs and stable across
    refits at the same pillar set.
    """
    # We can't compute tenors without a reference date; the pillar_dates
    # are the absolute dates. Caller should pass dates relative to ref.
    # Here we just use a stable index-based key plus the date as a hint.
    return {
        f"h_pillar_{i}({d.isoformat()})": float(h)
        for i, (d, h) in enumerate(zip(pillar_dates, pillar_hazards))
    }


# ═══════════════════════════════════════════════════════════════
# Risky Bond Pricing (internal)
# ═══════════════════════════════════════════════════════════════

def _price_risky_bond(
    reference_date: date,
    maturity: date,
    coupon: float,
    frequency: int,
    recovery: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> float:
    """Price a risky bond given discount and survival curves.

    PV = Σ c × τ × df(t_i) × Q(t_i)                   [coupon leg]
       + 1.0 × df(T) × Q(T)                             [principal leg]
       + R × Σ df(t_mid) × (Q(t_{i-1}) - Q(t_i))       [recovery leg]

    where Q(t) = survival probability, df(t) = discount factor.
    """
    freq_map = {1: Frequency.ANNUAL, 2: Frequency.SEMI_ANNUAL, 4: Frequency.QUARTERLY}
    freq = freq_map.get(frequency, Frequency.SEMI_ANNUAL)

    schedule = generate_schedule(reference_date, maturity, freq)
    dc = DayCountConvention.ACT_365_FIXED

    pv = 0.0
    prev_surv = 1.0

    for i in range(1, len(schedule)):
        t_start = schedule[i - 1]
        t_end = schedule[i]
        yf = year_fraction(t_start, t_end, dc)
        df = discount_curve.df(t_end)
        surv = survival_curve.survival(t_end)

        # Coupon (survival-weighted)
        pv += coupon * yf * df * surv * 100.0

        # Recovery on default (mid-period)
        t_mid_yf = year_fraction(reference_date, t_start, dc) + 0.5 * yf
        t_mid = date.fromordinal(
            reference_date.toordinal() + int(t_mid_yf * 365)
        )
        df_mid = discount_curve.df(t_mid)
        default_prob = prev_surv - surv
        pv += recovery * 100.0 * df_mid * default_prob

        prev_surv = surv

    # Principal at maturity (survival-weighted)
    pv += 100.0 * discount_curve.df(maturity) * survival_curve.survival(maturity)

    return pv


def _price_risky_bond_rmv(
    reference_date: date,
    maturity: date,
    coupon: float,
    frequency: int,
    recovery: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> float:
    """Price a risky bond under recovery of market value (Duffie-Singleton 1999).

    Under RMV, recovery on default = R × V(t⁻) (fraction of pre-default value).
    This simplifies to discounting all cashflows at adjusted survival:

        Q̃(t) = Q(t)^(1-R)

    PV = Σ c × τ × df(t_i) × Q̃(t_i) + Face × df(T) × Q̃(T)

    No separate recovery leg — recovery is embedded in the discounting.

    Reference: Duffie & Singleton (1999), eq. (6).
    """
    freq_map = {1: Frequency.ANNUAL, 2: Frequency.SEMI_ANNUAL, 4: Frequency.QUARTERLY}
    freq = freq_map.get(frequency, Frequency.SEMI_ANNUAL)

    schedule = generate_schedule(reference_date, maturity, freq)
    dc = DayCountConvention.ACT_365_FIXED

    lgd = 1.0 - recovery  # loss given default

    pv = 0.0
    for i in range(1, len(schedule)):
        t_start = schedule[i - 1]
        t_end = schedule[i]
        yf = year_fraction(t_start, t_end, dc)
        df = discount_curve.df(t_end)
        q = survival_curve.survival(t_end)
        q_adj = q ** lgd  # Duffie-Singleton adjusted survival

        # Coupon
        pv += coupon * yf * df * q_adj * 100.0

    # Principal at maturity
    q_mat = survival_curve.survival(maturity)
    q_adj_mat = q_mat ** (1.0 - recovery)
    pv += 100.0 * discount_curve.df(maturity) * q_adj_mat

    return pv


def _price_bond(
    reference_date: date,
    maturity: date,
    coupon: float,
    frequency: int,
    recovery: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery_mode: str = RECOVERY_PAR,
) -> float:
    """Dispatch to the appropriate risky bond pricer."""
    if recovery_mode == RECOVERY_MARKET_VALUE:
        return _price_risky_bond_rmv(
            reference_date, maturity, coupon, frequency,
            recovery, discount_curve, survival_curve,
        )
    return _price_risky_bond(
        reference_date, maturity, coupon, frequency,
        recovery, discount_curve, survival_curve,
    )


def _adjust_curve_for_liquidity(
    discount_curve: DiscountCurve,
    liquidity_bp: float,
    reference_date: date,
) -> DiscountCurve:
    """Bump discount curve by liquidity spread to isolate credit component.

    If the bond spread = credit + liquidity, we shift the risk-free curve
    UP by the liquidity premium so the residual spread is credit-only.

    This is equivalent to treating the liquidity premium as a known component
    of the bond's yield and removing it before hazard extraction.
    """
    if liquidity_bp == 0.0:
        return discount_curve
    bump = liquidity_bp / 10_000
    return discount_curve.bumped(bump)


# ═══════════════════════════════════════════════════════════════
# Sequential Bootstrap (exact fit, N bonds → N hazard pillars)
# ═══════════════════════════════════════════════════════════════

def _bootstrap_sequential(
    reference_date: date,
    bonds: list[BondInput],
    discount_curve: DiscountCurve,
    recovery_mode: str = RECOVERY_PAR,
    market_snapshot_id=None,
) -> HazardBootstrapResult:
    """Sequential bootstrap: one bond per maturity, exact fit.

    Like CDS bootstrapping but from bond prices.
    Bonds MUST be sorted by maturity. Each bond adds one hazard pillar.

    Limitation: requires exactly one bond per target maturity.
    Works best with 3-8 bonds spanning the curve.
    """
    sorted_bonds = sorted(bonds, key=lambda b: b.maturity)
    n = len(sorted_bonds)

    pillar_dates = [reference_date]
    pillar_survivals = [1.0]
    fitted_prices = []

    for i, bond in enumerate(sorted_bonds):
        # Adjust discount curve for this bond's liquidity premium
        dc_adj = _adjust_curve_for_liquidity(
            discount_curve, bond.liquidity_spread_bp, reference_date,
        )

        # Build trial survival curve with prior pillars + new pillar guess
        def objective(q_new: float, _bond=bond, _dc=dc_adj) -> float:
            trial_dates = pillar_dates + [_bond.maturity]
            trial_survivals = pillar_survivals + [max(q_new, 1e-10)]
            trial_curve = SurvivalCurve(reference_date, trial_dates[1:], trial_survivals[1:])
            model_price = _price_bond(
                reference_date, _bond.maturity, _bond.coupon, _bond.frequency,
                _bond.recovery, _dc, trial_curve, recovery_mode,
            )
            return model_price - _bond.market_price

        # Solve for Q(T) that reprices this bond
        try:
            q_solved = brentq(objective, 1e-10, 1.0 - 1e-10)
        except ValueError:
            # Bracket failure — bond price inconsistent with model
            q_solved = max(pillar_survivals[-1] * 0.95, 1e-6)

        pillar_dates.append(bond.maturity)
        pillar_survivals.append(q_solved)

        # Compute fitted price (using adjusted curve)
        curve = SurvivalCurve(reference_date, pillar_dates[1:], pillar_survivals[1:])
        fitted = _price_bond(
            reference_date, bond.maturity, bond.coupon, bond.frequency,
            bond.recovery, dc_adj, curve, recovery_mode,
        )
        fitted_prices.append(fitted)

    # Build final survival curve
    final_curve = SurvivalCurve(reference_date, pillar_dates[1:], pillar_survivals[1:])

    # Extract piecewise hazards
    hazards = []
    dc = DayCountConvention.ACT_365_FIXED
    for i in range(1, len(pillar_dates)):
        dt = year_fraction(pillar_dates[i-1], pillar_dates[i], dc)
        if dt > 0 and pillar_survivals[i] > 0 and pillar_survivals[i-1] > 0:
            h = -math.log(pillar_survivals[i] / pillar_survivals[i-1]) / dt
        else:
            h = 0.0
        hazards.append(max(h, 0.0))

    # Compute residuals
    market_prices = [b.market_price for b in sorted_bonds]
    residuals = [(f - m) * 100 for f, m in zip(fitted_prices, market_prices)]
    rmse = math.sqrt(sum(r**2 for r in residuals) / n) if n > 0 else 0.0

    cr = CalibrationResult.new(
        model_class="bond_hazard_pwc",
        parameters=_pillar_hazards_as_parameters(pillar_dates[1:], hazards),
        residuals=residuals,
        objective=ObjectiveKind.SSE,
        optimiser=OptimiserSpec(
            algorithm="brentq-per-bond",
            tolerance=0.0,  # brentq XTOL defaults — not user-specified here
            max_iterations=0,
            extra={"recovery_mode": recovery_mode},
        ),
        iterations=n,  # one root-find per bond
        converged=True,
        quotes_fitted=[f"bond_{i}" for i in range(n)],
        weights=[b.weight for b in sorted_bonds],
        market_snapshot_id=market_snapshot_id,
    )

    return HazardBootstrapResult(
        survival_curve=final_curve,
        pillar_dates=pillar_dates[1:],
        pillar_hazards=hazards,
        fitted_prices=fitted_prices,
        market_prices=market_prices,
        residuals_bp=residuals,
        rmse_bp=rmse,
        max_error_bp=max(abs(r) for r in residuals) if residuals else 0.0,
        n_bonds=n,
        method="sequential",
        converged=True,
        calibration_result=cr,
    )


# ═══════════════════════════════════════════════════════════════
# Global Fit (least-squares, N bonds → M hazard pillars, M ≤ N)
# ═══════════════════════════════════════════════════════════════

def _bootstrap_global(
    reference_date: date,
    bonds: list[BondInput],
    discount_curve: DiscountCurve,
    n_pillars: int | None = None,
    recovery_mode: str = RECOVERY_PAR,
    lam: float = 0.0,
    pillar_times: list[float] | None = None,
    market_snapshot_id=None,
) -> HazardBootstrapResult:
    """Global least-squares fit: fit M hazard pillars to N bond prices.

    More robust than sequential when bonds are noisy or have gaps.
    Minimises weighted sum of squared pricing errors, optionally with a
    Tikhonov second-difference (curvature) penalty:

        objective(h) = Σ_j w_j (f_j(h) − P_j)² · 10⁴   +   lam · ‖L h‖²

    where L is the second-difference operator (discrete Laplacian).
    `lam=0` gives the pure unregularised least-squares fit (default,
    backward-compatible with the original behaviour). `lam>0` adds the
    smoothness penalty; pick `lam` either by hand or via
    `find_lcurve_lambda`.

    Args:
        n_pillars: number of hazard rate segments. Default = min(N, 5).
            Ignored if `pillar_times` is provided.
        lam: Tikhonov regularisation strength (>= 0). See the
             `notebooks/credit/hazard_from_bonds_when_maturities_are_close.ipynb`
             for theory + L-curve picker.
        pillar_times: optional explicit pillar locations in years from
            `reference_date`. Must be sorted and strictly positive. When
            provided, `n_pillars` is ignored and len(pillar_times) is the
            pillar count. Useful for placing pillars at bond maturities
            (exactly-determined fit) or at calendar benchmarks.
    """
    sorted_bonds = sorted(bonds, key=lambda b: b.maturity)
    n = len(sorted_bonds)

    # Pre-compute per-bond adjusted discount curves for liquidity
    dc_per_bond = [
        _adjust_curve_for_liquidity(discount_curve, b.liquidity_spread_bp, reference_date)
        for b in sorted_bonds
    ]

    dc = DayCountConvention.ACT_365_FIXED
    t_max = year_fraction(reference_date, sorted_bonds[-1].maturity, dc)

    if pillar_times is not None:
        if not pillar_times:
            raise ValueError("pillar_times, if provided, must be non-empty")
        if any(t <= 0 for t in pillar_times):
            raise ValueError(f"pillar_times must all be > 0, got {pillar_times}")
        if any(pillar_times[i + 1] <= pillar_times[i] for i in range(len(pillar_times) - 1)):
            raise ValueError(f"pillar_times must be strictly increasing, got {pillar_times}")
        pillar_times = [float(t) for t in pillar_times]
        n_pillars = len(pillar_times)
    else:
        if n_pillars is None:
            n_pillars = min(n, 5)
        # Even spacing across bond maturities
        pillar_times = [t_max * (i + 1) / n_pillars for i in range(n_pillars)]

    pillar_dates = [
        date.fromordinal(reference_date.toordinal() + int(t * 365))
        for t in pillar_times
    ]

    # Second-difference (Tikhonov) operator. Shape (n_pillars-2, n_pillars).
    # ‖L h‖² = Σ_i (h_{i-1} − 2 h_i + h_{i+1})²  — discrete curvature.
    # Build it whenever n_pillars >= 3 (independent of lam) so that the
    # `roughness` diagnostic in the result is always populated. The lam>0
    # gate is applied only in the objective sum below.
    if n_pillars >= 3:
        L_mat = np.zeros((n_pillars - 2, n_pillars))
        for i in range(n_pillars - 2):
            L_mat[i, i]   = -1.0
            L_mat[i, i+1] =  2.0
            L_mat[i, i+2] = -1.0
    else:
        L_mat = None  # fewer than 3 pillars — no curvature is defined

    def objective(hazard_rates: np.ndarray) -> float:
        """Weighted sum of squared pricing errors, optionally with Tikhonov."""
        # Build survival curve from hazard rates
        survivals = []
        cum_surv = 1.0
        prev_t = 0.0
        for i, t in enumerate(pillar_times):
            dt = t - prev_t
            cum_surv *= math.exp(-max(hazard_rates[i], 0.0) * dt)
            survivals.append(cum_surv)
            prev_t = t

        try:
            curve = SurvivalCurve(reference_date, pillar_dates, survivals)
        except Exception:
            return 1e10

        total_err = 0.0
        for j, bond in enumerate(sorted_bonds):
            model = _price_bond(
                reference_date, bond.maturity, bond.coupon, bond.frequency,
                bond.recovery, dc_per_bond[j], curve, recovery_mode,
            )
            err_bp = (model - bond.market_price) * 100  # in bp of par
            total_err += bond.weight * err_bp ** 2

        if L_mat is not None and lam > 0:
            Lh = L_mat @ hazard_rates
            total_err += lam * float(Lh @ Lh)
        return total_err

    # Initial guess: flat hazard from average Z-spread
    avg_spread = 0.0
    for b in sorted_bonds:
        t = year_fraction(reference_date, b.maturity, dc)
        if t > 0 and b.market_price > 0:
            implied_yield = b.coupon + (100 - b.market_price) / (t * 100)
            rf = -math.log(discount_curve.df(b.maturity)) / t if t > 0 else 0.0
            avg_spread += max(implied_yield - rf, 0.0) / (1 - b.recovery)
    avg_spread /= max(n, 1)
    x0 = np.full(n_pillars, max(avg_spread, 0.001))

    # Optimise with bounds (hazard >= 0)
    bounds = [(0.0, 2.0)] * n_pillars  # cap at 200% annual hazard
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-12})

    # Build final curve from optimised hazard rates
    hazard_rates = result.x
    survivals = []
    cum_surv = 1.0
    prev_t = 0.0
    for i, t in enumerate(pillar_times):
        dt = t - prev_t
        cum_surv *= math.exp(-max(hazard_rates[i], 0.0) * dt)
        survivals.append(cum_surv)
        prev_t = t

    final_curve = SurvivalCurve(reference_date, pillar_dates, survivals)

    # Compute fitted prices and residuals
    fitted_prices = []
    for j, bond in enumerate(sorted_bonds):
        fitted = _price_bond(
            reference_date, bond.maturity, bond.coupon, bond.frequency,
            bond.recovery, dc_per_bond[j], final_curve, recovery_mode,
        )
        fitted_prices.append(fitted)

    market_prices = [b.market_price for b in sorted_bonds]
    residuals = [(f - m) * 100 for f, m in zip(fitted_prices, market_prices)]
    rmse = math.sqrt(sum(r**2 for r in residuals) / n) if n > 0 else 0.0

    if L_mat is not None:
        Lh_final = L_mat @ hazard_rates
        roughness_val = float(Lh_final @ Lh_final)
    else:
        roughness_val = 0.0

    cr = CalibrationResult.new(
        model_class="bond_hazard_pwc",
        parameters=_pillar_hazards_as_parameters(
            pillar_dates, [float(h) for h in hazard_rates],
        ),
        residuals=residuals,
        objective=ObjectiveKind.WEIGHTED_SSE,
        optimiser=OptimiserSpec(
            algorithm="L-BFGS-B" + (
                f"+tikhonov(lam={float(lam):.3e})" if lam > 0 else ""
            ),
            tolerance=1e-12,  # see options ftol in minimize call
            max_iterations=500,
            extra={
                "recovery_mode": recovery_mode,
                "lam": float(lam),
                "n_pillars": n_pillars,
            },
        ),
        iterations=int(getattr(result, "nit", 0)),
        converged=bool(result.success),
        quotes_fitted=[f"bond_{i}" for i in range(n)],
        weights=[b.weight for b in sorted_bonds],
        diagnostics=CalibrationDiagnostics(
            extra={"roughness": roughness_val},
        ),
        market_snapshot_id=market_snapshot_id,
    )

    return HazardBootstrapResult(
        survival_curve=final_curve,
        pillar_dates=pillar_dates,
        pillar_hazards=[float(h) for h in hazard_rates],
        fitted_prices=fitted_prices,
        market_prices=market_prices,
        residuals_bp=residuals,
        rmse_bp=rmse,
        max_error_bp=max(abs(r) for r in residuals) if residuals else 0.0,
        n_bonds=n,
        method="global_ls_tikhonov" if lam > 0 else "global_ls",
        converged=result.success,
        lam=float(lam),
        roughness=roughness_val,
        calibration_result=cr,
    )


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def bootstrap_hazard_from_bonds(
    reference_date: date,
    bonds: list[BondInput],
    discount_curve: DiscountCurve,
    method: str = "auto",
    n_pillars: int | None = None,
    recovery_mode: str = RECOVERY_PAR,
    lam: float | str = 0.0,
    pillar_times: list[float] | None = None,
    *,
    market_snapshot: MarketSnapshot | None = None,
) -> HazardBootstrapResult:
    """Bootstrap a survival/hazard curve from risky bond prices.

    Args:
        reference_date: valuation date.
        bonds: list of BondInput with market prices.
        discount_curve: risk-free OIS discount curve.
        method: "sequential" (exact, 1 bond per pillar),
                "global" (least-squares, N bonds → M pillars),
                "auto" (sequential if N ≤ 8 with distinct maturities, else global).
        n_pillars: for global method, number of hazard segments.
        recovery_mode: "par" (ISDA standard, R × Face on default) or
                       "market_value" (Duffie-Singleton 1999, R × V(t⁻)).
        lam: Tikhonov regularisation strength for `method="global"`.
             - `0.0` (default): unregularised least-squares (original behaviour).
             - positive float: penalised LS with that fixed λ.
             - `"auto"`: pick λ via L-curve corner detection
               (see `find_lcurve_lambda`). Ignored when method="sequential".
        pillar_times: optional explicit pillar locations (years from
             reference_date). When provided, `n_pillars` is ignored.
             Common patterns: pillars at bond maturities (exactly-determined),
             or at calendar benchmarks (e.g. 1, 2, 5, 10, 30y for sovereign).

    Returns:
        HazardBootstrapResult with survival curve and diagnostics.

    Recovery modes:
        - "par": on default, investor receives R × 100 (Face).
          This is the ISDA CDS standard and most common for corporate bonds.
          Requires separate recovery leg in pricing.
        - "market_value": on default, investor receives R × V(t⁻) where V(t⁻) is
          the pre-default risky bond value. Duffie-Singleton (1999) shows this reduces
          to discounting at adjusted survival Q̃(t) = Q(t)^(1-R). No recovery leg.
          Produces lower hazard rates for the same market prices.

    Liquidity spread:
        Each BondInput may specify liquidity_spread_bp. The bootstrap bumps the
        discount curve by this amount before extracting the credit component,
        so the resulting hazard rates reflect pure credit risk, not liquidity.

    Edge cases:
        - 1 bond: flat hazard rate (single pillar)
        - Bonds with same maturity: global method required (over-determined)
        - Very high spreads (distressed): solver may struggle, use global
        - Negative hazard: clamped to 0 (bond trades above risk-free par)

    Tikhonov regularisation:
        For universes with close maturities (any pair within ~3 months) the
        sequential bootstrap is ill-conditioned and small price noise produces
        huge swings in the implied hazard. The global LS with `lam > 0` adds
        a second-difference (curvature) penalty on the hazard vector that
        absorbs the noise into smoothness rather than oscillation. See the
        notebook `notebooks/credit/hazard_from_bonds_when_maturities_are_close.ipynb`
        for the full derivation, L-curve picking, and a side-by-side
        comparison vs the unregularised fit.
    """
    if not bonds:
        raise ValueError("At least one bond required")
    if recovery_mode not in (RECOVERY_PAR, RECOVERY_MARKET_VALUE):
        raise ValueError(f"recovery_mode must be '{RECOVERY_PAR}' or '{RECOVERY_MARKET_VALUE}'")

    n = len(bonds)
    sorted_bonds = sorted(bonds, key=lambda b: b.maturity)

    # Check for duplicate maturities
    maturities = [b.maturity for b in sorted_bonds]
    unique_mats = len(set(maturities))

    if method == "auto":
        if unique_mats == n and n <= 8:
            method = "sequential"
        else:
            method = "global"

    snapshot_id = market_snapshot.id if market_snapshot is not None else None

    if method == "sequential":
        if unique_mats < n:
            raise ValueError(
                f"Sequential method requires distinct maturities, "
                f"got {n} bonds with {unique_mats} unique maturities. Use method='global'."
            )
        return _bootstrap_sequential(
            reference_date, sorted_bonds, discount_curve, recovery_mode,
            market_snapshot_id=snapshot_id,
        )

    # Global method — resolve lam
    if isinstance(lam, str):
        if lam == "auto":
            lam_value = find_lcurve_lambda(
                reference_date, sorted_bonds, discount_curve,
                n_pillars=n_pillars, recovery_mode=recovery_mode,
                pillar_times=pillar_times,
            )
        else:
            raise ValueError(f"lam string must be 'auto', got '{lam}'")
    else:
        lam_value = float(lam)
        if lam_value < 0:
            raise ValueError(f"lam must be >= 0, got {lam_value}")

    return _bootstrap_global(
        reference_date, sorted_bonds, discount_curve, n_pillars,
        recovery_mode, lam=lam_value, pillar_times=pillar_times,
        market_snapshot_id=snapshot_id,
    )


def find_lcurve_lambda(
    reference_date: date,
    bonds: list[BondInput],
    discount_curve: DiscountCurve,
    n_pillars: int | None = None,
    recovery_mode: str = RECOVERY_PAR,
    lam_min: float = 1e2,
    lam_max: float = 1e10,
    n_lam: int = 21,
    pillar_times: list[float] | None = None,
) -> float:
    """Pick the Tikhonov λ at the corner of the L-curve.

    Sweeps λ on a logarithmic grid, runs the regularised global LS at each,
    and locates the point of maximum curvature on the (log misfit, log
    roughness) trade-off curve. This is Hansen's L-curve criterion
    (Hansen 1992, *Analysis of Discrete Ill-Posed Problems by Means of the
    L-Curve*, SIAM Review 34(4)).

    Args:
        reference_date, bonds, discount_curve, n_pillars, recovery_mode:
            same as `bootstrap_hazard_from_bonds`.
        lam_min, lam_max: log-spaced sweep bounds. Defaults span 8 decades,
            which covers the realistic range for typical bond universes.
        n_lam: number of grid points.

    Returns:
        Optimal λ (float). If the L-curve is too noisy to identify a clean
        corner (rare on real data), returns the geometric mean of `lam_min`
        and `lam_max`.

    See the notebook
    `notebooks/credit/hazard_from_bonds_when_maturities_are_close.ipynb`
    §6 for the geometry and a worked example.
    """
    if not bonds:
        raise ValueError("At least one bond required")

    sorted_bonds = sorted(bonds, key=lambda b: b.maturity)

    lam_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lam)
    misfit = np.zeros(n_lam)
    rough = np.zeros(n_lam)

    for i, lam in enumerate(lam_grid):
        result = _bootstrap_global(
            reference_date, sorted_bonds, discount_curve,
            n_pillars=n_pillars, recovery_mode=recovery_mode, lam=float(lam),
            pillar_times=pillar_times,
        )
        # misfit = Σ w_j (residual_bp_j)² across all bonds
        misfit[i] = float(np.sum(np.array(result.residuals_bp) ** 2))
        rough[i] = result.roughness

    # Signed curvature of (log misfit, log roughness) parameterised by log λ.
    # The L-curve elbow is a concave-from-above corner in (log m, log r)-space
    # (misfit grows, roughness shrinks); its signed curvature is the *most
    # negative* value, not the largest absolute value. Taking max|κ| would
    # confuse boundary finite-difference artifacts (where the curve is
    # essentially flat in one direction) with the real corner.
    log_l = np.log10(lam_grid)
    log_m = np.log10(np.maximum(misfit, 1e-30))
    log_r = np.log10(np.maximum(rough, 1e-30))

    dx = np.gradient(log_m, log_l)
    dy = np.gradient(log_r, log_l)
    ddx = np.gradient(dx, log_l)
    ddy = np.gradient(dy, log_l)
    denom = np.power(dx ** 2 + dy ** 2, 1.5)
    denom = np.where(denom < 1e-20, 1e-20, denom)
    signed_kappa = (dx * ddy - ddx * dy) / denom

    # Drop endpoints — finite-difference derivatives are unreliable there.
    if n_lam >= 5:
        idx = int(np.argmin(signed_kappa[2:-2]) + 2)
    else:
        idx = int(np.argmin(signed_kappa))

    return float(lam_grid[idx])


def implied_hazard_from_spread(
    spread_bp: float,
    recovery: float = 0.40,
) -> float:
    """Quick approximation: constant hazard from spread.

    h ≈ spread / (1 - R)

    This is the starting point, not a calibration.
    """
    return spread_bp / 10_000 / (1 - recovery)


def minimum_bonds_for_calibration(
    target_maturities: list[float],
) -> dict:
    """Guidance on minimum bond requirements for robust calibration.

    Returns recommended number and maturity distribution.
    """
    max_t = max(target_maturities) if target_maturities else 10
    return {
        "minimum": 1,
        "recommended": max(3, min(len(target_maturities), 8)),
        "ideal_spread": f"2Y, 5Y, 10Y (or evenly across 0-{max_t:.0f}Y)",
        "notes": [
            "1 bond: flat hazard only — no term structure",
            "2 bonds: linear hazard — captures slope but not curvature",
            "3-5 bonds: good term structure — covers short/medium/long",
            "6-8 bonds: rich structure — can capture humps/inversions",
            "8+ bonds: use global fit (over-determined, least-squares)",
            "Bonds at similar maturities: use global fit (sequential fails)",
            "Wide bid-ask: lower weight in global fit for noisy prices",
            "Distressed (spread > 1000bp): use global fit with tight bounds",
        ],
    }


# ═══════════════════════════════════════════════════════════════
# FRN (Floating-Rate Note) Input
# ═══════════════════════════════════════════════════════════════


@dataclass
class FRNInput:
    """A floating-rate note observation for hazard rate calibration.

    FRNs pay benchmark + spread. The credit component is in the discount
    margin (DM) — the spread that reprices the FRN to market.

    Args:
        maturity: maturity date.
        spread: contractual spread over benchmark (e.g. 0.015 = 150bp).
        market_price: dirty price per 100 face.
        benchmark_rate: current benchmark fixing (e.g. SOFR = 0.053).
        frequency: coupons per year (4=quarterly, typical for FRNs).
        recovery: assumed recovery on default.
        weight: fitting weight (lower for illiquid).
        liquidity_spread_bp: liquidity premium to strip.
    """
    maturity: date
    spread: float              # contractual spread over benchmark
    market_price: float        # dirty price per 100 face
    benchmark_rate: float      # current benchmark rate
    frequency: int = 4         # quarterly (standard for FRNs)
    recovery: float = 0.40
    weight: float = 1.0
    liquidity_spread_bp: float = 0.0

    def to_dict(self) -> dict:
        return {**vars(self), "maturity": self.maturity.isoformat()}


def _price_risky_frn(
    reference_date: date,
    frn: FRNInput,
    discount_curve: DiscountCurve,
    survival_probs: dict[date, float],
    recovery: float,
) -> float:
    """Price a risky FRN given survival probabilities.

    PV = Σ (benchmark + spread) × τ × df × Q + 100 × df(T) × Q(T)
         + R × 100 × Σ df_mid × (Q_{i-1} - Q_i)

    Assumes flat forward rates at current benchmark (no projection curve).
    """
    # Generate coupon schedule
    freq = {1: Frequency.ANNUAL, 2: Frequency.SEMI_ANNUAL,
            4: Frequency.QUARTERLY}.get(frn.frequency, Frequency.QUARTERLY)
    schedule = generate_schedule(reference_date, frn.maturity, freq)

    def _get_q(d):
        if d in survival_probs:
            return survival_probs[d]
        # Interpolate from nearest
        dates_sorted = sorted(survival_probs.keys())
        if d <= dates_sorted[0]:
            return survival_probs[dates_sorted[0]]
        if d >= dates_sorted[-1]:
            return survival_probs[dates_sorted[-1]]
        for k in range(len(dates_sorted) - 1):
            if dates_sorted[k] <= d <= dates_sorted[k + 1]:
                t1 = year_fraction(reference_date, dates_sorted[k], DayCountConvention.ACT_365_FIXED)
                t2 = year_fraction(reference_date, dates_sorted[k + 1], DayCountConvention.ACT_365_FIXED)
                t = year_fraction(reference_date, d, DayCountConvention.ACT_365_FIXED)
                q1 = survival_probs[dates_sorted[k]]
                q2 = survival_probs[dates_sorted[k + 1]]
                if t2 > t1:
                    w = (t - t1) / (t2 - t1)
                    return q1 * (q2 / q1) ** w  # log-linear
                return q1
        return 1.0

    pv = 0.0
    prev_q = 1.0

    # Apply liquidity spread to discount curve
    liq_bump = frn.liquidity_spread_bp / 10_000

    for i in range(1, len(schedule)):
        t_prev, t_now = schedule[i - 1], schedule[i]
        tau = year_fraction(t_prev, t_now, DayCountConvention.ACT_365_FIXED)

        df = discount_curve.df(t_now) * math.exp(-liq_bump * year_fraction(
            reference_date, t_now, DayCountConvention.ACT_365_FIXED))
        q = _get_q(t_now)

        # Coupon: (benchmark + spread) × τ × df × Q
        coupon = (frn.benchmark_rate + frn.spread) * tau * 100
        pv += coupon * df * q

        # Recovery: R × 100 × df_mid × ΔPD
        t_mid_years = 0.5 * (
            year_fraction(reference_date, t_prev, DayCountConvention.ACT_365_FIXED)
            + year_fraction(reference_date, t_now, DayCountConvention.ACT_365_FIXED)
        )
        df_mid = discount_curve.df(t_prev) * math.exp(-liq_bump * t_mid_years)
        default_prob = prev_q - q
        pv += recovery * 100 * df_mid * max(default_prob, 0)

        prev_q = q

    # Principal at maturity
    df_T = discount_curve.df(frn.maturity) * math.exp(-liq_bump * year_fraction(
        reference_date, frn.maturity, DayCountConvention.ACT_365_FIXED))
    q_T = _get_q(frn.maturity)
    pv += 100 * df_T * q_T

    return pv


# ═══════════════════════════════════════════════════════════════
# Mixed Fixed + Float Bootstrap
# ═══════════════════════════════════════════════════════════════


def bootstrap_hazard_mixed(
    reference_date: date,
    fixed_bonds: list[BondInput] | None = None,
    floaters: list[FRNInput] | None = None,
    discount_curve: DiscountCurve = None,
    method: str = "global",
    n_pillars: int | None = None,
    recovery_mode: str = RECOVERY_PAR,
    *,
    market_snapshot: MarketSnapshot | None = None,
) -> HazardBootstrapResult:
    """Bootstrap hazard rates from a mix of fixed-rate bonds and FRNs.

    Handles the case where an issuer has both fixed and floating debt.
    Uses global (least-squares) fit by default since mixed instruments
    rarely have enough maturity diversity for sequential bootstrap.

    Args:
        fixed_bonds: list of BondInput (fixed-rate bonds).
        floaters: list of FRNInput (floating-rate notes).
        discount_curve: risk-free discount curve.
        method: "global" (default) or "sequential" (fixed only).
        n_pillars: number of hazard pillars (default: auto).
        recovery_mode: "par" or "market_value".

    Returns:
        HazardBootstrapResult with survival curve and diagnostics.
    """
    fixed_bonds = fixed_bonds or []
    floaters = floaters or []

    if not fixed_bonds and not floaters:
        raise ValueError("Need at least one bond or FRN")

    # Collect all maturities
    all_maturities = sorted(set(
        [b.maturity for b in fixed_bonds] + [f.maturity for f in floaters]
    ))

    if n_pillars is None:
        n_pillars = min(len(all_maturities), 8)

    # Pillar dates: evenly spaced across maturity range
    if n_pillars >= len(all_maturities):
        pillar_dates = all_maturities
    else:
        # Select n_pillars evenly spaced from all maturities
        indices = np.linspace(0, len(all_maturities) - 1, n_pillars, dtype=int)
        pillar_dates = [all_maturities[i] for i in indices]

    n_p = len(pillar_dates)

    def _objective(log_survivals):
        """Objective: weighted sum of squared pricing errors."""
        survivals = np.exp(-np.abs(log_survivals))  # ensure Q in (0, 1)
        surv_list = [float(survivals[k]) for k in range(n_p)]
        sc = SurvivalCurve(reference_date, pillar_dates, surv_list)
        surv_dict = {pillar_dates[k]: float(survivals[k]) for k in range(n_p)}

        total_err = 0.0

        # Fixed bonds — use existing _price_risky_bond with correct signature
        for b in fixed_bonds:
            liq_bump = b.liquidity_spread_bp / 10_000
            dc_adj = discount_curve  # TODO: bump for liquidity
            model_px = _price_risky_bond(
                reference_date, b.maturity, b.coupon, b.frequency,
                b.recovery, dc_adj, sc,
            )  # returns per 100 face
            err = (model_px - b.market_price) / 100
            total_err += b.weight * err**2

        # FRNs
        for f in floaters:
            model_px = _price_risky_frn(reference_date, f, discount_curve,
                                         surv_dict, f.recovery)
            err = (model_px - f.market_price) / 100
            total_err += f.weight * err**2

        return total_err

    # Initial guess: survival at each pillar from flat hazard
    avg_spread = 0.02  # 200bp initial guess
    x0 = []
    for pd in pillar_dates:
        t = year_fraction(reference_date, pd, DayCountConvention.ACT_365_FIXED)
        x0.append(avg_spread * t)  # -log(Q) = h*t

    result = minimize(_objective, x0, method="L-BFGS-B",
                       bounds=[(0.001, 5.0)] * n_p)

    survivals = np.exp(-np.abs(result.x))
    surv_dict = {pillar_dates[k]: float(survivals[k]) for k in range(n_p)}

    # Build survival curve
    sc = SurvivalCurve(reference_date, pillar_dates, [surv_dict[d] for d in pillar_dates])

    # Compute fitted prices and residuals
    fitted_prices = []
    market_prices = []
    residuals = []

    for b in fixed_bonds:
        mp = _price_risky_bond(reference_date, b.maturity, b.coupon, b.frequency,
                                b.recovery, discount_curve, sc)
        fitted_prices.append(mp)
        market_prices.append(b.market_price)
        residuals.append((mp - b.market_price) * 100)  # bp of par

    for f in floaters:
        mp = _price_risky_frn(reference_date, f, discount_curve, surv_dict, f.recovery)
        fitted_prices.append(mp)
        market_prices.append(f.market_price)
        residuals.append((mp - f.market_price) * 100)

    rmse = math.sqrt(np.mean(np.array(residuals)**2)) if residuals else 0.0

    # Extract piecewise hazard rates
    hazards = []
    prev_t = 0.0
    prev_q = 1.0
    for d in pillar_dates:
        t = year_fraction(reference_date, d, DayCountConvention.ACT_365_FIXED)
        q = surv_dict[d]
        if t > prev_t and q > 0 and prev_q > 0:
            h = -math.log(q / prev_q) / (t - prev_t)
        else:
            h = 0.0
        hazards.append(max(h, 0.0))
        prev_t = t
        prev_q = q

    n_total = len(fixed_bonds) + len(floaters)
    cr = CalibrationResult.new(
        model_class="bond_hazard_pwc",
        parameters=_pillar_hazards_as_parameters(pillar_dates, hazards),
        residuals=residuals,
        objective=ObjectiveKind.WEIGHTED_SSE,
        optimiser=OptimiserSpec(
            algorithm="L-BFGS-B",
            tolerance=1e-12,
            max_iterations=500,
            extra={
                "method": "global_mixed",
                "n_fixed": len(fixed_bonds),
                "n_floaters": len(floaters),
            },
        ),
        iterations=int(getattr(result, "nit", 0)),
        converged=bool(result.success),
        quotes_fitted=(
            [f"bond_{i}" for i in range(len(fixed_bonds))]
            + [f"frn_{i}" for i in range(len(floaters))]
        ),
        weights=(
            [b.weight for b in fixed_bonds]
            + [getattr(f, "weight", 1.0) for f in floaters]
        ),
        market_snapshot_id=market_snapshot.id if market_snapshot is not None else None,
    )

    return HazardBootstrapResult(
        survival_curve=sc,
        pillar_dates=pillar_dates,
        pillar_hazards=hazards,
        fitted_prices=fitted_prices,
        market_prices=market_prices,
        residuals_bp=residuals,
        rmse_bp=rmse,
        max_error_bp=max(abs(r) for r in residuals) if residuals else 0.0,
        n_bonds=n_total,
        method="global_mixed",
        converged=result.success,
        calibration_result=cr,
    )


# ═══════════════════════════════════════════════════════════════
# Liquid vs Illiquid Regime
# ═══════════════════════════════════════════════════════════════

@dataclass
class LiquidityAssessment:
    """Assessment of bond liquidity for hazard bootstrapping."""
    regime: str                 # "liquid", "semi_liquid", "illiquid"
    recommended_method: str     # "sequential", "global", "global_mixed"
    recommended_n_pillars: int
    bid_ask_bp: float           # estimated bid-ask in bp
    confidence: str             # "high", "medium", "low"
    notes: list[str]

    def to_dict(self) -> dict:
        return dict(vars(self))


def assess_liquidity(
    bonds: list[BondInput] | None = None,
    floaters: list[FRNInput] | None = None,
    bid_ask_widths_bp: list[float] | None = None,
) -> LiquidityAssessment:
    """Assess bond pool liquidity and recommend bootstrap strategy.

    Heuristics:
    - Liquid: ≥3 bonds, bid-ask < 50bp, well-spaced maturities.
    - Semi-liquid: 2-5 bonds, bid-ask 50-200bp, some maturity gaps.
    - Illiquid: 1 bond or bid-ask > 200bp.

    Args:
        bonds: fixed-rate bonds.
        floaters: floating-rate notes.
        bid_ask_widths_bp: per-bond bid-ask widths in bp (optional).
    """
    bonds = bonds or []
    floaters = floaters or []
    n_total = len(bonds) + len(floaters)

    if n_total == 0:
        return LiquidityAssessment("illiquid", "none", 0, 0, "low",
                                    ["No bonds provided"])

    # Bid-ask assessment
    avg_ba = 0.0
    if bid_ask_widths_bp:
        avg_ba = sum(bid_ask_widths_bp) / len(bid_ask_widths_bp)
    else:
        # Estimate from price levels (distressed bonds have wider spreads)
        prices = [b.market_price for b in bonds] + [f.market_price for f in floaters]
        avg_price = sum(prices) / len(prices) if prices else 100
        if avg_price < 70:
            avg_ba = 200  # distressed
        elif avg_price < 90:
            avg_ba = 100
        else:
            avg_ba = 30

    # Maturity coverage
    all_mats = sorted(set(
        [b.maturity for b in bonds] + [f.maturity for f in floaters]
    ))
    n_distinct = len(all_mats)

    notes = []

    # Determine regime
    if n_total >= 3 and avg_ba < 50 and n_distinct >= 3:
        regime = "liquid"
        method = "sequential" if n_total <= 8 and not floaters else "global_mixed"
        n_pillars = min(n_distinct, 8)
        confidence = "high"
        notes.append("Good liquidity — sequential or global fit appropriate")
    elif n_total >= 2 and avg_ba < 200:
        regime = "semi_liquid"
        method = "global_mixed" if floaters else "global"
        n_pillars = min(n_distinct, 5)
        confidence = "medium"
        notes.append("Moderate liquidity — use global fit with weights")
        if avg_ba > 100:
            notes.append("Wide bid-ask — reduce weight on widest bonds")
    else:
        regime = "illiquid"
        method = "global"
        n_pillars = min(n_distinct, 3)
        confidence = "low"
        notes.append("Low liquidity — flat or 2-pillar hazard only")
        if n_total == 1:
            notes.append("Single bond — can only extract flat hazard rate")
            n_pillars = 1
        if avg_ba > 200:
            notes.append("Very wide bid-ask — consider mid-market average")

    if floaters:
        notes.append(f"{len(floaters)} FRN(s) — benchmark rate drives coupon; DM gives credit info")

    return LiquidityAssessment(
        regime=regime,
        recommended_method=method,
        recommended_n_pillars=n_pillars,
        bid_ask_bp=avg_ba,
        confidence=confidence,
        notes=notes,
    )


def bootstrap_hazard_adaptive(
    reference_date: date,
    bonds: list[BondInput] | None = None,
    floaters: list[FRNInput] | None = None,
    discount_curve: DiscountCurve = None,
    bid_ask_widths_bp: list[float] | None = None,
    recovery_mode: str = RECOVERY_PAR,
    *,
    market_snapshot: MarketSnapshot | None = None,
) -> HazardBootstrapResult:
    """Adaptive hazard bootstrapping based on liquidity assessment.

    Automatically selects the best method and parameters based on
    the available data quality:
    - Liquid: sequential bootstrap (exact fit, N pillars)
    - Semi-liquid: global fit with reduced pillars and weights
    - Illiquid: global fit with 1-3 pillars, heavy regularisation

    Args:
        bonds: fixed-rate bonds.
        floaters: floating-rate notes.
        discount_curve: risk-free curve.
        bid_ask_widths_bp: per-instrument bid-ask widths.
        recovery_mode: "par" or "market_value".

    Returns:
        HazardBootstrapResult (method field indicates regime).
    """
    assessment = assess_liquidity(bonds, floaters, bid_ask_widths_bp)

    if assessment.regime == "liquid" and not floaters:
        # Pure fixed-rate, liquid — use the existing sequential/auto bootstrap
        return bootstrap_hazard_from_bonds(
            reference_date, bonds, discount_curve,
            method="auto", recovery_mode=recovery_mode,
            market_snapshot=market_snapshot,
        )
    else:
        # Mixed or illiquid — use global mixed bootstrap
        # Adjust weights for illiquid bonds
        if bid_ask_widths_bp and bonds:
            for i, ba in enumerate(bid_ask_widths_bp[:len(bonds)]):
                # Wider bid-ask → lower weight: w = 1 / (1 + ba/100)
                bonds[i].weight = 1.0 / (1 + ba / 100)

        if bid_ask_widths_bp and floaters:
            offset = len(bonds or [])
            for i, ba in enumerate(bid_ask_widths_bp[offset:offset + len(floaters)]):
                floaters[i].weight = 1.0 / (1 + ba / 100)

        return bootstrap_hazard_mixed(
            reference_date, bonds, floaters, discount_curve,
            method="global", n_pillars=assessment.recommended_n_pillars,
            recovery_mode=recovery_mode,
            market_snapshot=market_snapshot,
        )
