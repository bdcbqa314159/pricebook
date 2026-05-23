"""Regime-dependent market data — vol surfaces and curves blended by regime.

    from pricebook.models.regime_surfaces import (
        RegimeVolSurface, RegimeCurve, regime_blend,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


@dataclass
class RegimeBlendResult:
    """Result of regime blending."""
    blended_value: float
    regime_values: list[float]
    regime_weights: list[float]
    n_regimes: int

    def to_dict(self) -> dict:
        return vars(self)


class RegimeVolSurface:
    """N vol surfaces, one per regime, blended by posterior probabilities.

    Given regime probabilities π = (π₁, ..., π_K):
        σ_blend(T, K) = Σ πᵢ × σᵢ(T, K)    [linear blend]
    or:
        σ²_blend(T, K) = Σ πᵢ × σᵢ²(T, K)  [variance blend, more theoretically correct]
    """

    def __init__(
        self,
        vol_surfaces: list,
        regime_probs: np.ndarray,
        blend_variance: bool = True,
    ):
        """
        Args:
            vol_surfaces: list of vol surface objects (must have .vol(T, K) method or be float).
            regime_probs: (K,) probability weights.
            blend_variance: if True, blend in variance space; if False, linear.
        """
        if len(vol_surfaces) != len(regime_probs):
            raise ValueError("Number of surfaces must match number of regime probabilities")
        self.surfaces = vol_surfaces
        self.probs = np.array(regime_probs)
        self.probs /= self.probs.sum()  # normalise
        self.blend_variance = blend_variance

    def vol(self, expiry: float = 1.0, strike: float | None = None) -> float:
        """Blended vol at given expiry/strike."""
        vols = []
        for s in self.surfaces:
            if callable(getattr(s, "vol", None)):
                v = s.vol(expiry, strike) if strike is not None else s.vol(expiry)
            elif isinstance(s, (int, float)):
                v = float(s)
            else:
                v = float(s)
            vols.append(v)

        if self.blend_variance:
            var = sum(p * v**2 for p, v in zip(self.probs, vols))
            return math.sqrt(max(var, 0))
        else:
            return sum(p * v for p, v in zip(self.probs, vols))

    def regime_vols(self, expiry: float = 1.0) -> RegimeBlendResult:
        """Return per-regime vols and blended result."""
        vols = []
        for s in self.surfaces:
            if callable(getattr(s, "vol", None)):
                vols.append(s.vol(expiry))
            else:
                vols.append(float(s))
        return RegimeBlendResult(
            blended_value=self.vol(expiry),
            regime_values=vols,
            regime_weights=self.probs.tolist(),
            n_regimes=len(self.surfaces),
        )

    def to_dict(self) -> dict:
        return {
            "n_regimes": len(self.surfaces),
            "regime_probs": self.probs.tolist(),
            "blend_variance": self.blend_variance,
        }


class RegimeCurve:
    """N discount/survival curves blended by regime probabilities.

    df_blend(T) = Σ πᵢ × dfᵢ(T)    [linear blend of discount factors]

    This is equivalent to pricing under each regime and averaging.
    """

    def __init__(
        self,
        curves: list[DiscountCurve],
        regime_probs: np.ndarray,
    ):
        if len(curves) != len(regime_probs):
            raise ValueError("Number of curves must match regime probabilities")
        self.curves = curves
        self.probs = np.array(regime_probs)
        self.probs /= self.probs.sum()
        self.reference_date = curves[0].reference_date

    def df(self, d: date) -> float:
        """Blended discount factor."""
        return sum(p * c.df(d) for p, c in zip(self.probs, self.curves))

    def zero_rate(self, d: date) -> float:
        """Implied zero rate from blended DF."""
        from pricebook.core.day_count import DayCountConvention, year_fraction
        t = year_fraction(self.reference_date, d, DayCountConvention.ACT_365_FIXED)
        df = self.df(d)
        if t <= 0 or df <= 0:
            return 0.0
        return -math.log(df) / t

    def regime_dfs(self, d: date) -> RegimeBlendResult:
        """Per-regime DFs and blended result."""
        dfs = [c.df(d) for c in self.curves]
        return RegimeBlendResult(
            blended_value=self.df(d),
            regime_values=dfs,
            regime_weights=self.probs.tolist(),
            n_regimes=len(self.curves),
        )

    def to_dict(self) -> dict:
        return {
            "n_regimes": len(self.curves),
            "regime_probs": self.probs.tolist(),
            "reference_date": self.reference_date.isoformat(),
        }


def regime_blend(values: list[float], weights: np.ndarray) -> float:
    """Generic regime-weighted blend of scalar values."""
    w = np.array(weights)
    w /= w.sum()
    return float(np.dot(w, values))


def regime_price(
    pricers: list[callable],
    regime_probs: np.ndarray,
) -> dict:
    """Price under each regime and blend.

    Args:
        pricers: list of callables, each returning a float price.
        regime_probs: (K,) weights.

    Returns dict with blended price, per-regime prices, weights.
    """
    w = np.array(regime_probs)
    w /= w.sum()
    prices = [p() for p in pricers]
    blended = float(np.dot(w, prices))
    return {
        "blended_price": blended,
        "regime_prices": prices,
        "regime_weights": w.tolist(),
        "regime_spread": max(prices) - min(prices),
    }
