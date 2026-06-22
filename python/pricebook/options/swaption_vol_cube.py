"""Swaption volatility cube: 3D (expiry × tenor × strike).

ATM surface backbone + per-node SABR smile layer for strike interpolation.

    from pricebook.options.swaption_vol_cube import (
        SwaptionVolCube, build_swaption_vol_cube,
    )

    cube = build_swaption_vol_cube(ref, atm_surface, smile_data)
    vol = cube.vol(expiry_date, tenor=5.0, strike=0.05)

References:
    Hagan et al. (2002). Managing Smile Risk.
    Rebonato (2004). Volatility and Correlation, Ch. 8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.options.sabr import sabr_implied_vol, sabr_calibrate


@dataclass
class SABRNode:
    """SABR parameters at a single (expiry, tenor) point."""
    expiry_years: float
    tenor_years: float
    forward: float
    alpha: float
    beta: float
    rho: float
    nu: float
    atm_vol: float

    def vol(self, strike: float) -> float:
        """Implied vol at a given strike via SABR Hagan formula."""
        if abs(strike - self.forward) < 1e-8:
            return self.atm_vol
        try:
            return sabr_implied_vol(self.forward, strike, self.expiry_years,
                                     self.alpha, self.beta, self.rho, self.nu)
        except (ValueError, ZeroDivisionError):
            return self.atm_vol

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class SwaptionVolCubeResult:
    """Result of vol cube construction."""
    n_atm_nodes: int
    n_sabr_nodes: int
    expiry_range: tuple[float, float]
    tenor_range: tuple[float, float]

    def to_dict(self) -> dict:
        return dict(vars(self))


class SwaptionVolCube:
    """3D swaption vol cube: expiry × tenor × strike.

    Structure:
    - **ATM backbone**: bilinear interpolation on (expiry, tenor) grid.
    - **Smile layer**: per-node SABR parameters for strike interpolation.
    - **Fallback**: if no SABR at a node, returns ATM vol.

    Usage:
        vol = cube.vol(expiry_date, tenor=5.0, strike=0.05)
        vol = cube.vol_by_years(expiry_years=5.0, tenor=5.0, strike=0.05)
    """

    def __init__(
        self,
        reference_date: date,
        atm_expiry_years: list[float],
        atm_tenor_years: list[float],
        atm_vols: list[list[float]],
        sabr_nodes: list[SABRNode] | None = None,
    ):
        self.reference_date = reference_date
        self._expiries = np.array(atm_expiry_years)
        self._tenors = np.array(atm_tenor_years)
        self._atm_grid = np.array(atm_vols)  # (n_expiries, n_tenors)
        self._sabr: dict[tuple[float, float], SABRNode] = {}

        if sabr_nodes:
            for node in sabr_nodes:
                self._sabr[(node.expiry_years, node.tenor_years)] = node

    def vol(self, expiry: date, tenor: float, strike: float | None = None) -> float:
        """Implied vol at (expiry, tenor, strike).

        If strike is None, returns ATM vol.
        """
        t_exp = year_fraction(self.reference_date, expiry,
                               DayCountConvention.ACT_365_FIXED)
        return self.vol_by_years(t_exp, tenor, strike)

    def vol_by_years(self, expiry_years: float, tenor: float,
                      strike: float | None = None) -> float:
        """Implied vol at (expiry_years, tenor, strike)."""
        atm = self._interp_atm(expiry_years, tenor)

        if strike is None:
            return atm

        # Look up nearest SABR node
        node = self._nearest_sabr(expiry_years, tenor)
        if node is not None:
            return node.vol(strike)

        return atm  # no smile data → flat

    def atm_vol(self, expiry_years: float, tenor: float) -> float:
        """ATM vol via bilinear interpolation."""
        return self._interp_atm(expiry_years, tenor)

    def smile(self, expiry_years: float, tenor: float,
              strikes: list[float]) -> list[float]:
        """Vol smile across strikes at a given (expiry, tenor)."""
        return [self.vol_by_years(expiry_years, tenor, k) for k in strikes]

    def bumped(self, shift: float) -> "SwaptionVolCube":
        """Parallel shift all vols by shift (additive).

        Shifts both the ATM grid and SABR alpha (vol level parameter)
        to ensure consistent smile across all strikes.
        """
        new_atm = (self._atm_grid + shift).tolist()
        new_nodes = []
        for node in self._sabr.values():
            new_nodes.append(SABRNode(
                node.expiry_years, node.tenor_years, node.forward,
                max(node.alpha + shift, 1e-6),  # shift alpha too for smile consistency
                node.beta, node.rho, node.nu,
                node.atm_vol + shift,
            ))
        return SwaptionVolCube(
            self.reference_date,
            self._expiries.tolist(), self._tenors.tolist(),
            new_atm, new_nodes,
        )

    def _interp_atm(self, t_exp: float, tenor: float) -> float:
        """Bilinear interpolation on ATM grid."""
        # Find bracketing indices
        i = np.searchsorted(self._expiries, t_exp) - 1
        j = np.searchsorted(self._tenors, tenor) - 1
        i = max(0, min(i, len(self._expiries) - 2))
        j = max(0, min(j, len(self._tenors) - 2))

        # Weights
        e0, e1 = self._expiries[i], self._expiries[i + 1]
        t0, t1 = self._tenors[j], self._tenors[j + 1]

        we = (t_exp - e0) / (e1 - e0) if e1 > e0 else 0.0
        wt = (tenor - t0) / (t1 - t0) if t1 > t0 else 0.0
        we = max(0, min(1, we))
        wt = max(0, min(1, wt))

        v00 = self._atm_grid[i, j]
        v01 = self._atm_grid[i, j + 1]
        v10 = self._atm_grid[i + 1, j]
        v11 = self._atm_grid[i + 1, j + 1]

        return (1 - we) * (1 - wt) * v00 + (1 - we) * wt * v01 + \
               we * (1 - wt) * v10 + we * wt * v11

    def _nearest_sabr(self, t_exp: float, tenor: float) -> SABRNode | None:
        """Find nearest SABR node within tolerance."""
        best = None
        best_dist = float("inf")
        for (e, t), node in self._sabr.items():
            dist = (e - t_exp)**2 + (t - tenor)**2
            if dist < best_dist:
                best_dist = dist
                best = node
        # Only use if reasonably close
        if best_dist < 4.0:  # within ~2Y on each axis
            return best
        return None

    def to_dict(self) -> dict:
        return {
            "expiries": self._expiries.tolist(),
            "tenors": self._tenors.tolist(),
            "n_sabr_nodes": len(self._sabr),
        }


def build_swaption_vol_cube(
    reference_date: date,
    atm_expiry_years: list[float],
    atm_tenor_years: list[float],
    atm_vols: list[list[float]],
    smile_data: dict | None = None,
    beta: float = 0.5,
) -> SwaptionVolCube:
    """Build a vol cube from ATM surface + optional smile data.

    Args:
        atm_expiry_years: option expiry points (e.g. [0.5, 1, 2, 5, 10]).
        atm_tenor_years: underlying swap tenors (e.g. [1, 2, 5, 10, 20, 30]).
        atm_vols: 2D grid of ATM Black vols, shape (n_expiries, n_tenors).
        smile_data: optional dict {(expiry_y, tenor_y): {"forward": f, "strikes": [...], "vols": [...]}}.
            If provided, SABR is calibrated per node.
        beta: SABR beta (default 0.5).

    Returns:
        SwaptionVolCube with ATM backbone + SABR smile.
    """
    sabr_nodes = []

    if smile_data:
        for (exp_y, tenor_y), data in smile_data.items():
            fwd = data["forward"]
            strikes = data["strikes"]
            vols = data["vols"]

            try:
                params = sabr_calibrate(fwd, strikes, vols, exp_y, beta=beta)
                # Find ATM vol from grid
                atm = _lookup_atm(atm_expiry_years, atm_tenor_years, atm_vols, exp_y, tenor_y)
                sabr_nodes.append(SABRNode(
                    exp_y, tenor_y, fwd,
                    params.alpha, params.beta, params.rho, params.nu,
                    atm,
                ))
            except Exception:
                pass

    return SwaptionVolCube(reference_date, atm_expiry_years, atm_tenor_years,
                            atm_vols, sabr_nodes)


def _lookup_atm(expiries, tenors, grid, exp_y, tenor_y):
    """Bilinear interpolation for ATM vol on the (expiry, tenor) grid.

    Fix T4-SVC1: pre-fix this function used ``np.searchsorted`` to pick
    an index and returned ``grid[i, j]`` — i.e. lookup-with-round-up
    rather than interpolation.  For any query strictly between two
    pillars on either axis, the function returned the upper-right
    cell's value instead of the interpolated one.  Used to populate
    each SABR node's ``atm_vol``, so off-pillar nodes carried the
    wrong ATM (showing up at the exactly-strike-equals-forward
    fast-path in ``SABRNode.vol``).
    """
    exp_arr = np.array(expiries, dtype=float)
    ten_arr = np.array(tenors, dtype=float)
    grid_arr = np.array(grid, dtype=float)

    if len(exp_arr) == 0 or len(ten_arr) == 0:
        return 0.0
    if len(exp_arr) == 1:
        # Degenerate expiry axis — fall through to tenor-only.
        if len(ten_arr) == 1:
            return float(grid_arr[0, 0])
        j = int(np.searchsorted(ten_arr, tenor_y)) - 1
        j = max(0, min(j, len(ten_arr) - 2))
        wt = (tenor_y - ten_arr[j]) / (ten_arr[j + 1] - ten_arr[j])
        wt = max(0.0, min(1.0, wt))
        return float((1 - wt) * grid_arr[0, j] + wt * grid_arr[0, j + 1])
    if len(ten_arr) == 1:
        i = int(np.searchsorted(exp_arr, exp_y)) - 1
        i = max(0, min(i, len(exp_arr) - 2))
        we = (exp_y - exp_arr[i]) / (exp_arr[i + 1] - exp_arr[i])
        we = max(0.0, min(1.0, we))
        return float((1 - we) * grid_arr[i, 0] + we * grid_arr[i + 1, 0])

    i = int(np.searchsorted(exp_arr, exp_y)) - 1
    j = int(np.searchsorted(ten_arr, tenor_y)) - 1
    i = max(0, min(i, len(exp_arr) - 2))
    j = max(0, min(j, len(ten_arr) - 2))

    we = (exp_y - exp_arr[i]) / (exp_arr[i + 1] - exp_arr[i])
    wt = (tenor_y - ten_arr[j]) / (ten_arr[j + 1] - ten_arr[j])
    we = max(0.0, min(1.0, we))
    wt = max(0.0, min(1.0, wt))

    v00 = grid_arr[i, j]
    v01 = grid_arr[i, j + 1]
    v10 = grid_arr[i + 1, j]
    v11 = grid_arr[i + 1, j + 1]
    return float(
        (1 - we) * (1 - wt) * v00
        + (1 - we) * wt * v01
        + we * (1 - wt) * v10
        + we * wt * v11
    )
