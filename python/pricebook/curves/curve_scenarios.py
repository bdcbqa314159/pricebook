"""Curve scenario engine — named curve shape scenarios.

Steepener, flattener, butterfly, inversion, PCA-based scenarios.

    from pricebook.curves.curve_scenarios import (
        parallel_shift, steepener, flattener, butterfly,
        pca_scenarios, standard_scenario_set,
    )

References:
    Litterman & Scheinkman (1991). Common Factors Affecting Bond Returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


@dataclass
class CurveScenario:
    """A named curve perturbation scenario."""
    name: str
    description: str
    apply: callable   # Callable[[DiscountCurve], DiscountCurve]

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description}


def parallel_shift(shift_bp: float, name: str | None = None) -> CurveScenario:
    """Parallel shift of the entire curve."""
    shift = shift_bp / 10_000
    return CurveScenario(
        name=name or f"parallel_{shift_bp:+.0f}bp",
        description=f"Parallel shift {shift_bp:+.0f}bp",
        apply=lambda c: c.bumped(shift),
    )


def steepener(
    short_shift_bp: float,
    long_shift_bp: float,
    pivot_years: float = 5.0,
    name: str | None = None,
) -> CurveScenario:
    """Steepener: short end moves differently from long end."""
    def apply(curve: DiscountCurve) -> DiscountCurve:
        return _apply_tilt(curve, short_shift_bp, long_shift_bp, pivot_years)
    label = name or f"steepener_{short_shift_bp:+.0f}_{long_shift_bp:+.0f}"
    return CurveScenario(label, f"Short {short_shift_bp:+.0f}bp, Long {long_shift_bp:+.0f}bp", apply)


def flattener(
    short_shift_bp: float,
    long_shift_bp: float,
    pivot_years: float = 5.0,
    name: str | None = None,
) -> CurveScenario:
    """Flattener: typically short up, long down (or less up)."""
    return steepener(short_shift_bp, long_shift_bp, pivot_years,
                     name or f"flattener_{short_shift_bp:+.0f}_{long_shift_bp:+.0f}")


def bear_steepener(magnitude_bp: float = 50.0) -> CurveScenario:
    """Bear steepener: rates up, long end more than short."""
    return steepener(magnitude_bp * 0.5, magnitude_bp, name="bear_steepener")


def bull_flattener(magnitude_bp: float = 50.0) -> CurveScenario:
    """Bull flattener: rates down, long end more than short."""
    return steepener(-magnitude_bp * 0.5, -magnitude_bp, name="bull_flattener")


def butterfly(
    wing_shift_bp: float,
    belly_shift_bp: float,
    belly_years: float = 5.0,
    wing_years: tuple[float, float] = (2.0, 10.0),
    name: str | None = None,
) -> CurveScenario:
    """Butterfly: wings move differently from belly."""
    def apply(curve: DiscountCurve) -> DiscountCurve:
        ref = curve.reference_date
        new_dfs = []
        for d in curve.pillar_dates:
            t = (d - ref).days / 365.0
            df = curve.df(d)
            if t <= wing_years[0]:
                shift = wing_shift_bp / 10_000
            elif t >= wing_years[1]:
                shift = wing_shift_bp / 10_000
            else:
                # Linear interpolation to belly
                w = (t - wing_years[0]) / (belly_years - wing_years[0]) \
                    if t <= belly_years else \
                    (wing_years[1] - t) / (wing_years[1] - belly_years)
                shift = (wing_shift_bp + w * (belly_shift_bp - wing_shift_bp)) / 10_000
            new_dfs.append(df * math.exp(-shift * t) if t > 0 else df)
        return DiscountCurve(ref, curve.pillar_dates, new_dfs)

    label = name or f"butterfly_w{wing_shift_bp:+.0f}_b{belly_shift_bp:+.0f}"
    return CurveScenario(label, f"Wings {wing_shift_bp:+.0f}bp, Belly {belly_shift_bp:+.0f}bp", apply)


def inversion(magnitude_bp: float = 100.0) -> CurveScenario:
    """Curve inversion: short rates above long rates."""
    return steepener(magnitude_bp, -magnitude_bp * 0.3, name="inversion")


def historical_scenario(
    curve_before: DiscountCurve,
    curve_after: DiscountCurve,
    scale: float = 1.0,
    name: str = "historical",
) -> CurveScenario:
    """Replay a historical curve move, optionally scaled."""
    def apply(curve: DiscountCurve) -> DiscountCurve:
        ref = curve.reference_date
        new_dfs = []
        for d in curve.pillar_dates:
            df = curve.df(d)
            t = (d - ref).days / 365.0
            if t <= 0:
                new_dfs.append(df)
                continue
            z_before = -math.log(max(curve_before.df(d), 1e-15)) / t
            z_after = -math.log(max(curve_after.df(d), 1e-15)) / t
            dz = (z_after - z_before) * scale
            new_dfs.append(df * math.exp(-dz * t))
        return DiscountCurve(ref, curve.pillar_dates, new_dfs)
    return CurveScenario(name, f"Historical move (scale={scale})", apply)


def pca_scenarios(
    zero_rate_history: np.ndarray,
    tenor_grid: list[float],
    n_components: int = 3,
    shock_sigma: float = 2.0,
) -> list[CurveScenario]:
    """PCA-based scenarios: level, slope, curvature shocks.

    Args:
        zero_rate_history: (n_dates, n_tenors) matrix of historical zero rates.
        tenor_grid: tenors corresponding to columns.
        n_components: number of PCA components (default: 3).
        shock_sigma: magnitude in standard deviations.

    Returns:
        List of CurveScenario, one per component (±shock_sigma).
    """
    changes = np.diff(zero_rate_history, axis=0)
    cov = np.cov(changes, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    names = ["level", "slope", "curvature", "twist", "hump"]
    scenarios = []

    for i in range(min(n_components, len(eigenvalues))):
        pc = eigenvectors[:, i]
        std = math.sqrt(max(eigenvalues[i], 0))
        shock = pc * std * shock_sigma

        component_name = names[i] if i < len(names) else f"pc{i+1}"

        for direction, sign in [("up", 1.0), ("down", -1.0)]:
            def make_apply(s=shock * sign, tg=tenor_grid):
                def apply(curve: DiscountCurve) -> DiscountCurve:
                    ref = curve.reference_date
                    new_dfs = []
                    for d in curve.pillar_dates:
                        t = (d - ref).days / 365.0
                        df = curve.df(d)
                        if t <= 0:
                            new_dfs.append(df)
                            continue
                        dz = float(np.interp(t, tg, s))
                        new_dfs.append(df * math.exp(-dz * t))
                    return DiscountCurve(ref, curve.pillar_dates, new_dfs)
                return apply

            scenarios.append(CurveScenario(
                f"pca_{component_name}_{direction}",
                f"PCA {component_name} {direction} {shock_sigma}σ",
                make_apply(),
            ))

    return scenarios


def standard_scenario_set(currency: str = "USD") -> list[CurveScenario]:
    """Standard set of curve scenarios for a currency."""
    return [
        parallel_shift(+100, "parallel_up_100"),
        parallel_shift(-100, "parallel_dn_100"),
        parallel_shift(+25, "parallel_up_25"),
        parallel_shift(-25, "parallel_dn_25"),
        bear_steepener(50),
        bull_flattener(50),
        steepener(+50, -25, name="twist_steepener"),
        flattener(-25, +50, name="twist_flattener"),
        butterfly(+25, -25, name="butterfly_positive"),
        butterfly(-25, +25, name="butterfly_negative"),
        inversion(100),
    ]


def run_scenarios(
    curve: DiscountCurve,
    pricer: callable,
    scenarios: list[CurveScenario],
) -> list[dict]:
    """Run multiple scenarios and return results."""
    base_pv = pricer(curve)
    results = []
    for scenario in scenarios:
        shocked_curve = scenario.apply(curve)
        shocked_pv = pricer(shocked_curve)
        results.append({
            "name": scenario.name,
            "description": scenario.description,
            "base_pv": base_pv,
            "shocked_pv": shocked_pv,
            "pnl": shocked_pv - base_pv,
        })
    return results


# ═══════════════════════════════════════════════════════════════
# Internal
# ═══════════════════════════════════════════════════════════════


def _apply_tilt(curve, short_bp, long_bp, pivot_years):
    """Apply a linear tilt: short_bp at t=0, long_bp at t=max."""
    ref = curve.reference_date
    new_dfs = []
    max_t = max((d - ref).days / 365.0 for d in curve.pillar_dates)
    if max_t <= 0:
        max_t = 1.0
    for d in curve.pillar_dates:
        t = (d - ref).days / 365.0
        df = curve.df(d)
        if t <= 0:
            new_dfs.append(df)
            continue
        w = t / max_t
        shift = ((1 - w) * short_bp + w * long_bp) / 10_000
        new_dfs.append(df * math.exp(-shift * t))
    return DiscountCurve(ref, curve.pillar_dates, new_dfs)
