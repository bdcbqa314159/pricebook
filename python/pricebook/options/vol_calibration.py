"""Vol surface calibration pipeline: market quotes → SABR → surface → arb check.

One-call calibration per asset class. Each returns a calibrated surface
with vol(), arb_report(), and bumped() methods.

    from pricebook.options.vol_calibration import (
        calibrate_fx_surface, calibrate_equity_surface,
        calibrate_ir_surface, calibrate_commodity_surface,
    )

    fx_cube = calibrate_fx_surface(ref, quotes, spot, base_curve, quote_curve)
    vol = fx_cube.vol(expiry, strike)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.options.sabr import sabr_implied_vol
from pricebook.options.vol_surface import FlatVol


@dataclass
class CalibratedSABRNode:
    """SABR parameters at one expiry/tenor point."""
    expiry: date
    forward: float
    alpha: float
    beta: float
    rho: float
    nu: float
    atm_vol: float
    calibration_error: float = 0.0
    # Time-to-expiry at calibration, in years.  Required for SABR vol
    # evaluation: the Hagan correction terms scale with ``T``
    # (correction ~ 1 + (B1 + B2 + B3)·T), so using ``T = 1.0`` for a
    # 10y tenor adds 10× the correction.  Set during calibration.
    T_to_expiry: float = 1.0

    def vol(self, strike: float, T: float | None = None) -> float:
        """Implied vol at any strike via SABR.

        ``T`` defaults to the calibration ``T_to_expiry`` of this node.
        Pre-fix the default was a static ``1.0`` regardless of actual
        tenor — the SABR Hagan correction terms scale with T, so the
        smile was systematically distorted for any tenor ≠ 1y.
        """
        if T is None:
            T = self.T_to_expiry
        return sabr_implied_vol(self.forward, strike, T,
                               self.alpha, self.beta, self.rho, self.nu)



    def to_dict(self) -> dict:
        return dict(vars(self))
class CalibratedVolSurface:
    """Universal calibrated vol surface with arbitrage reporting."""

    def __init__(self, nodes: list[CalibratedSABRNode], asset_class: str = ""):
        self._nodes = sorted(nodes, key=lambda n: n.expiry)
        self.asset_class = asset_class

    @property
    def expiries(self) -> list[date]:
        return [n.expiry for n in self._nodes]

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """Vol at (expiry, strike). Interpolates between tenors."""
        if not self._nodes:
            raise ValueError("CalibratedVolSurface has no nodes — cannot compute vol")

        # Find surrounding nodes
        if expiry <= self._nodes[0].expiry:
            node = self._nodes[0]
        elif expiry >= self._nodes[-1].expiry:
            node = self._nodes[-1]
        else:
            # Linear interpolation between bracketing nodes
            for i in range(len(self._nodes) - 1):
                if self._nodes[i].expiry <= expiry <= self._nodes[i+1].expiry:
                    node = self._nodes[i]  # use lower node's SABR
                    break
            else:
                node = self._nodes[-1]

        if strike is None:
            return node.atm_vol
        return node.vol(strike)

    def bumped(self, shift: float) -> "CalibratedVolSurface":
        """Parallel vol bump."""
        bumped_nodes = []
        for n in self._nodes:
            bumped_nodes.append(CalibratedSABRNode(
                n.expiry, n.forward,
                n.alpha * (1 + shift / n.atm_vol) if n.atm_vol > 0 else n.alpha,
                n.beta, n.rho, n.nu, n.atm_vol + shift, n.calibration_error,
            ))
        return CalibratedVolSurface(bumped_nodes, self.asset_class)

    def smile_at(self, expiry: date, strikes: list[float]) -> list[float]:
        """Vol smile at one expiry across strikes."""
        return [self.vol(expiry, k) for k in strikes]

    def arb_report(self, strikes: list[float] | None = None):
        """Run arbitrage scan on this surface."""
        from pricebook.options.vol_arbitrage_scanner import scan_surface
        if strikes is None:
            # Default strike grid around ATM
            atm = self._nodes[0].forward if self._nodes else 100.0
            strikes = [atm * (0.8 + 0.05 * i) for i in range(9)]
        return scan_surface(
            self.asset_class, self, strikes, self.expiries,
            self.asset_class,
        )

    def to_dict(self) -> dict:
        return {
            "asset_class": self.asset_class,
            "n_tenors": len(self._nodes),
            "expiries": [n.expiry.isoformat() for n in self._nodes],
            "atm_vols": [n.atm_vol for n in self._nodes],
        }


# ---------------------------------------------------------------------------
# Per-asset calibration functions
# ---------------------------------------------------------------------------

def calibrate_fx_surface(
    reference_date: date,
    quotes: list[dict],
    spot: float,
    beta: float = 0.5,
) -> CalibratedVolSurface:
    """Calibrate FX vol surface from ATM/RR/BF quotes.

    Args:
        quotes: list of {"expiry": date, "atm": float, "rr25": float, "bf25": float}.
        spot: FX spot rate.
        beta: SABR beta (0.5 typical for FX).
    """
    from pricebook.options.sabr import sabr_calibrate

    nodes = []
    for q in sorted(quotes, key=lambda x: x["expiry"]):
        expiry = q["expiry"]
        atm = q["atm"]
        rr25 = q.get("rr25", 0.0)
        bf25 = q.get("bf25", 0.0)

        # 3-point smile: put_25d, atm, call_25d
        vol_put = atm - rr25 / 2 + bf25
        vol_call = atm + rr25 / 2 + bf25

        T = max((expiry - reference_date).days / 365.0, 0.001)
        forward = spot  # simplified: forward ≈ spot for short tenors

        # Strikes from 25-delta (approximate)
        from pricebook.models.black76 import black76_price
        k_put = forward * math.exp(-0.674 * vol_put * math.sqrt(T))  # ~25D put
        k_call = forward * math.exp(0.674 * vol_call * math.sqrt(T))  # ~25D call

        strikes = [k_put, forward, k_call]
        vols = [vol_put, atm, vol_call]

        # Calibrate SABR
        try:
            result = sabr_calibrate(forward, strikes, vols, T, beta=beta)
            nodes.append(CalibratedSABRNode(
                expiry=expiry, forward=forward,
                alpha=result.alpha, beta=beta,
                rho=result.rho, nu=result.nu,
                atm_vol=atm, calibration_error=result.rmse,
                T_to_expiry=T,
            ))
        except (ValueError, RuntimeError):
            # Fallback: alpha ≈ atm · F^(1-β) so SABR(K=F) ≈ atm.
            fallback_alpha = atm * forward ** (1.0 - beta)
            nodes.append(CalibratedSABRNode(
                expiry=expiry, forward=forward,
                alpha=fallback_alpha, beta=beta,
                rho=0.0, nu=0.0,
                atm_vol=atm, T_to_expiry=T,
            ))

    return CalibratedVolSurface(nodes, "fx")


def calibrate_equity_surface(
    reference_date: date,
    quotes: list[dict],
    spot: float,
    beta: float = 0.5,
) -> CalibratedVolSurface:
    """Calibrate equity vol surface from strike×vol grid.

    Args:
        quotes: list of {"expiry": date, "strikes": [K], "vols": [σ]}.
        spot: current spot price.
    """
    from pricebook.options.sabr import sabr_calibrate

    nodes = []
    for q in sorted(quotes, key=lambda x: x["expiry"]):
        expiry = q["expiry"]
        strikes = q["strikes"]
        vols = q["vols"]
        T = max((expiry - reference_date).days / 365.0, 0.001)
        forward = spot  # simplified

        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - forward))
        atm = vols[atm_idx]

        try:
            result = sabr_calibrate(forward, strikes, vols, T, beta=beta)
            nodes.append(CalibratedSABRNode(
                expiry=expiry, forward=forward,
                alpha=result.alpha, beta=beta,
                rho=result.rho, nu=result.nu,
                atm_vol=atm, calibration_error=result.rmse,
                T_to_expiry=T,
            ))
        except (ValueError, RuntimeError):
            fallback_alpha = atm * forward ** (1.0 - beta)
            nodes.append(CalibratedSABRNode(
                expiry=expiry, forward=forward,
                alpha=fallback_alpha, beta=beta,
                rho=-0.3, nu=0.3,
                atm_vol=atm, T_to_expiry=T,
            ))

    return CalibratedVolSurface(nodes, "equity")


def calibrate_ir_surface(
    reference_date: date,
    quotes: list[dict],
    beta: float = 0.5,
) -> CalibratedVolSurface:
    """Calibrate swaption vol surface from ATM vols + optional smile.

    Args:
        quotes: list of {"expiry": date, "tenor": str, "atm": float,
                         "strikes": [K], "vols": [σ]} (strikes/vols optional).
    """
    from pricebook.options.sabr import sabr_calibrate

    nodes = []
    for q in sorted(quotes, key=lambda x: x["expiry"]):
        expiry = q["expiry"]
        atm = q["atm"]
        forward = atm  # forward swap rate ≈ ATM for normal quoting
        T = max((expiry - reference_date).days / 365.0, 0.001)

        strikes = q.get("strikes", [])
        vols = q.get("vols", [])

        if strikes and vols and len(strikes) >= 3:
            try:
                result = sabr_calibrate(forward, strikes, vols, T, beta=beta)
                nodes.append(CalibratedSABRNode(
                    expiry=expiry, forward=forward,
                    alpha=result.alpha, beta=beta,
                    rho=result.rho, nu=result.nu,
                    atm_vol=atm, calibration_error=result.rmse,
                    T_to_expiry=T,
                ))
                continue
            except (ValueError, RuntimeError):
                pass

        # ATM-only: flat smile.  Alpha must convert atm_vol back via F^(1-β).
        fallback_alpha = atm * forward ** (1.0 - beta) if forward > 0 else atm
        nodes.append(CalibratedSABRNode(
            expiry=expiry, forward=forward,
            alpha=fallback_alpha, beta=beta,
            rho=0.0, nu=0.0,
            atm_vol=atm, T_to_expiry=T,
        ))

    return CalibratedVolSurface(nodes, "ir")


def calibrate_commodity_surface(
    reference_date: date,
    quotes: list[dict],
    spot: float,
    commodity_type: str = "crude",
) -> CalibratedVolSurface:
    """Calibrate commodity vol surface.

    Args:
        quotes: list of {"expiry": date, "atm": float, "strikes": [K], "vols": [σ]}.
        commodity_type: "crude" (β=1), "metals" (β=0.5), "power" (β=0).
    """
    beta_map = {"crude": 1.0, "metals": 0.5, "power": 0.0, "gas": 0.5}
    beta = beta_map.get(commodity_type, 0.5)

    return calibrate_equity_surface(reference_date, quotes, spot, beta=beta)
