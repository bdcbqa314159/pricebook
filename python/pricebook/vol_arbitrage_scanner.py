"""Vol arbitrage scanner: detect and fix calendar + butterfly violations.

Practical tool wrapping vol_arb.py for trader use: scan any surface,
report violations, optionally enforce no-arb constraints.

    from pricebook.vol_arbitrage_scanner import (
        scan_surface, scan_all_surfaces, enforce_no_arb,
        ArbitrageScanResult,
    )

    result = scan_surface("EURUSD", fx_surface, strikes, expiries)
    if not result.is_clean:
        cleaned, n_fixed = enforce_no_arb(fx_surface, strikes, expiries)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction


@dataclass
class ArbViolation:
    """One arbitrage violation."""
    arb_type: str              # "calendar" or "butterfly"
    expiry: date | None
    strike: float | None
    severity: float            # magnitude of violation
    description: str

    def to_dict(self) -> dict:
        return {
            "type": self.arb_type,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "strike": self.strike, "severity": self.severity,
            "description": self.description,
        }


@dataclass
class ArbitrageScanResult:
    """Result of scanning one surface for arbitrage."""
    surface_name: str
    asset_class: str
    n_calendar_violations: int
    n_butterfly_violations: int
    violations: list[ArbViolation]
    is_clean: bool
    max_severity: float

    def to_dict(self) -> dict:
        return {
            "name": self.surface_name, "asset_class": self.asset_class,
            "calendar": self.n_calendar_violations,
            "butterfly": self.n_butterfly_violations,
            "is_clean": self.is_clean, "max_severity": self.max_severity,
            "violations": [v.to_dict() for v in self.violations],
        }


def scan_surface(
    name: str,
    surface,
    strikes: list[float],
    expiries: list[date],
    asset_class: str = "",
    ref: date | None = None,
) -> ArbitrageScanResult:
    """Scan a vol surface for calendar and butterfly arbitrage.

    Calendar: total variance σ²T must be non-decreasing in T.
    Butterfly: call prices must be convex in strike (no negative butterfly).

    Args:
        surface: object with vol(expiry, strike) method.
        strikes: strike grid to check.
        expiries: expiry dates to check (must be sorted).
    """
    if ref is None:
        ref = expiries[0] - __import__('datetime').timedelta(days=30)

    violations = []

    # Calendar arbitrage check
    for strike in strikes:
        prev_total_var = 0.0
        for i, exp in enumerate(expiries):
            T = year_fraction(ref, exp, DayCountConvention.ACT_365_FIXED)
            if T <= 0:
                continue
            try:
                vol = surface.vol(exp, strike)
            except (TypeError, AttributeError):
                try:
                    vol = surface.vol(exp)
                except Exception:
                    continue

            total_var = vol ** 2 * T
            if i > 0 and total_var < prev_total_var - 1e-10:
                severity = prev_total_var - total_var
                violations.append(ArbViolation(
                    "calendar", exp, strike, severity,
                    f"Total variance decreases at K={strike:.4f}, T={T:.2f}Y: "
                    f"{total_var:.6f} < {prev_total_var:.6f}",
                ))
            prev_total_var = total_var

    # Butterfly arbitrage check
    for exp in expiries:
        for i in range(1, len(strikes) - 1):
            k_low, k_mid, k_high = strikes[i-1], strikes[i], strikes[i+1]
            try:
                v_low = surface.vol(exp, k_low)
                v_mid = surface.vol(exp, k_mid)
                v_high = surface.vol(exp, k_high)
            except (TypeError, AttributeError):
                continue

            # Butterfly spread in vol space: should be non-negative
            # (convexity of call prices ↔ positive butterfly)
            butterfly = 0.5 * (v_low + v_high) - v_mid
            if butterfly < -0.001:  # tolerance for numerical noise
                violations.append(ArbViolation(
                    "butterfly", exp, k_mid, abs(butterfly),
                    f"Negative butterfly at K={k_mid:.4f}: "
                    f"0.5×({v_low:.4f}+{v_high:.4f})-{v_mid:.4f} = {butterfly:.4f}",
                ))

    n_cal = sum(1 for v in violations if v.arb_type == "calendar")
    n_bf = sum(1 for v in violations if v.arb_type == "butterfly")
    max_sev = max((v.severity for v in violations), default=0.0)

    return ArbitrageScanResult(
        surface_name=name, asset_class=asset_class,
        n_calendar_violations=n_cal, n_butterfly_violations=n_bf,
        violations=violations, is_clean=len(violations) == 0,
        max_severity=max_sev,
    )


def scan_all_surfaces(
    surfaces: dict[str, tuple[object, str, list[float], list[date]]],
    ref: date | None = None,
) -> list[ArbitrageScanResult]:
    """Scan multiple surfaces for arbitrage.

    Args:
        surfaces: {name: (surface, asset_class, strikes, expiries)}.
    """
    results = []
    for name, (surface, asset_class, strikes, expiries) in surfaces.items():
        results.append(scan_surface(name, surface, strikes, expiries, asset_class, ref))
    return results


def enforce_no_arb(
    surface,
    strikes: list[float],
    expiries: list[date],
    ref: date | None = None,
) -> tuple[dict, int]:
    """Enforce no-arbitrage on a surface by adjusting vols.

    Returns (adjusted_vols: dict[(expiry, strike), vol], n_fixes).
    Calendar: floor total variance to be non-decreasing.
    Butterfly: floor mid-strike vol to maintain convexity.
    """
    if ref is None:
        ref = expiries[0] - __import__('datetime').timedelta(days=30)

    adjusted = {}
    n_fixes = 0

    # First pass: calendar (enforce σ²T non-decreasing per strike)
    for strike in strikes:
        prev_total_var = 0.0
        for exp in expiries:
            T = year_fraction(ref, exp, DayCountConvention.ACT_365_FIXED)
            if T <= 0:
                continue
            try:
                vol = surface.vol(exp, strike)
            except (TypeError, AttributeError):
                try:
                    vol = surface.vol(exp)
                except Exception:
                    vol = 0.20

            total_var = vol ** 2 * T
            if total_var < prev_total_var:
                # Floor: use previous total variance
                total_var = prev_total_var
                vol = math.sqrt(total_var / T) if T > 0 else vol
                n_fixes += 1

            adjusted[(exp, strike)] = vol
            prev_total_var = total_var

    # Second pass: butterfly (enforce convexity per expiry)
    for exp in expiries:
        for i in range(1, len(strikes) - 1):
            k_low, k_mid, k_high = strikes[i-1], strikes[i], strikes[i+1]
            v_low = adjusted.get((exp, k_low), 0.20)
            v_mid = adjusted.get((exp, k_mid), 0.20)
            v_high = adjusted.get((exp, k_high), 0.20)

            butterfly = 0.5 * (v_low + v_high) - v_mid
            if butterfly < 0:
                # Floor mid vol down to enforce convexity
                adjusted[(exp, k_mid)] = 0.5 * (v_low + v_high)
                n_fixes += 1

    return adjusted, n_fixes
