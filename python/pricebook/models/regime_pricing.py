"""Regime-aware pricing engine — HMM-driven option pricing.

Connects regime detection (HMM) with regime-dependent pricing:
1. Fit HMM to historical data to detect market regimes
2. Extract regime-conditional vol, drift, and jump parameters
3. Price under each regime and blend by posterior probabilities
4. Compute regime-conditional Greeks and risk decomposition

    from pricebook.models.regime_pricing import (
        RegimePricingEngine, RegimePricingResult,
        regime_option_price, regime_greeks,
    )

References:
    Hamilton (1989). A New Approach to Nonstationary Time Series.
    Bollen (1998). Valuing Options in Regime-Switching Models.
    Hardy (2001). A Regime-Switching Model of Long-Term Stock Returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pricebook.models.black76 import OptionType


@dataclass
class RegimePricingResult:
    """Result of regime-aware option pricing."""
    blended_price: float
    regime_prices: list[float]
    regime_weights: list[float]
    regime_vols: list[float]
    n_regimes: int
    blended_vol: float
    regime_spread: float
    current_regime: int

    def to_dict(self) -> dict:
        return {
            "blended_price": self.blended_price,
            "regime_prices": self.regime_prices,
            "regime_weights": self.regime_weights,
            "regime_vols": self.regime_vols,
            "n_regimes": self.n_regimes,
            "blended_vol": self.blended_vol,
            "regime_spread": self.regime_spread,
            "current_regime": self.current_regime,
        }


@dataclass
class RegimeGreeksResult:
    """Regime-conditional Greeks."""
    blended_delta: float
    blended_gamma: float
    blended_vega: float
    regime_deltas: list[float]
    regime_gammas: list[float]
    regime_vegas: list[float]
    regime_weights: list[float]

    def to_dict(self) -> dict:
        return vars(self)


class RegimePricingEngine:
    """HMM-driven option pricing under regime switching.

    Fits an HMM to return data, extracts regime-conditional vols,
    then prices under each regime and blends by filtered probabilities.

    Usage:
        engine = RegimePricingEngine(n_regimes=3)
        engine.fit(historical_returns)
        result = engine.price(spot=100, strike=105, rate=0.05, T=1.0)
    """

    def __init__(self, n_regimes: int = 2, min_vol: float = 0.05):
        self.n_regimes = n_regimes
        self.min_vol = min_vol
        self._hmm = None
        self._regime_vols = None
        self._regime_drifts = None
        self._filtered_probs = None
        self._transition_matrix = None

    def fit(self, returns: np.ndarray, annualisation: float = 252.0) -> dict:
        """Fit HMM to return data and extract regime parameters.

        Args:
            returns: (T,) array of log returns.
            annualisation: trading days per year for vol scaling.

        Returns:
            dict with fit diagnostics.
        """
        from pricebook.statistics.hmm import HMM, GaussianEmission

        hmm = HMM(n_states=self.n_regimes, emission=GaussianEmission())
        fit_result = hmm.fit(returns, max_iter=100)

        self._hmm = hmm
        self._transition_matrix = fit_result.transition_matrix.copy()

        # Extract regime-conditional vols (annualised)
        means = np.array([p["mean"] for p in fit_result.emission_params])
        stds = np.array([p["std"] for p in fit_result.emission_params])

        self._regime_drifts = means * annualisation
        self._regime_vols = np.maximum(stds * math.sqrt(annualisation), self.min_vol)

        # Use last row of filtered probabilities as current regime weights
        self._filtered_probs = fit_result.filtered_probs[-1]

        # Sort regimes by vol (low → high)
        order = np.argsort(self._regime_vols)
        self._regime_vols = self._regime_vols[order]
        self._regime_drifts = self._regime_drifts[order]
        self._filtered_probs = self._filtered_probs[order]

        return {
            "log_likelihood": fit_result.log_likelihood,
            "n_iterations": fit_result.n_iterations,
            "converged": fit_result.converged,
            "regime_vols": self._regime_vols.tolist(),
            "regime_drifts": self._regime_drifts.tolist(),
            "filtered_probs": self._filtered_probs.tolist(),
        }

    def price(
        self,
        spot: float,
        strike: float,
        rate: float,
        T: float,
        option_type: OptionType = OptionType.CALL,
        div_yield: float = 0.0,
        regime_probs: np.ndarray | None = None,
    ) -> RegimePricingResult:
        """Price option under regime switching.

        Args:
            spot, strike, rate, T: standard option params.
            option_type: CALL or PUT.
            div_yield: continuous dividend yield.
            regime_probs: override HMM-filtered probabilities.
        """
        from pricebook.options.equity_option import equity_option_price

        if self._regime_vols is None:
            raise ValueError("Must call fit() before price()")

        probs = np.array(regime_probs) if regime_probs is not None else self._filtered_probs
        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("Regime probabilities sum to zero or negative")
        probs = probs / prob_sum

        # Price under each regime
        prices = []
        for i in range(self.n_regimes):
            p = equity_option_price(spot, strike, rate, self._regime_vols[i],
                                    T, option_type, div_yield)
            prices.append(p)

        blended = float(np.dot(probs, prices))
        current = int(np.argmax(probs))

        # Blended vol (variance blend)
        blended_var = float(np.dot(probs, self._regime_vols ** 2))
        blended_vol = math.sqrt(max(blended_var, 0))

        return RegimePricingResult(
            blended_price=blended,
            regime_prices=prices,
            regime_weights=probs.tolist(),
            regime_vols=self._regime_vols.tolist(),
            n_regimes=self.n_regimes,
            blended_vol=blended_vol,
            regime_spread=max(prices) - min(prices),
            current_regime=current,
        )

    def greeks(
        self,
        spot: float,
        strike: float,
        rate: float,
        T: float,
        option_type: OptionType = OptionType.CALL,
        div_yield: float = 0.0,
        regime_probs: np.ndarray | None = None,
    ) -> RegimeGreeksResult:
        """Regime-blended Greeks."""
        from pricebook.options.equity_option import equity_delta, equity_gamma, equity_vega

        if self._regime_vols is None:
            raise ValueError("Must call fit() before greeks()")

        probs = np.array(regime_probs) if regime_probs is not None else self._filtered_probs
        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("Regime probabilities sum to zero or negative")
        probs = probs / prob_sum

        deltas, gammas, vegas = [], [], []
        for i in range(self.n_regimes):
            d = equity_delta(spot, strike, rate, self._regime_vols[i], T, option_type, div_yield)
            g = equity_gamma(spot, strike, rate, self._regime_vols[i], T, div_yield)
            v = equity_vega(spot, strike, rate, self._regime_vols[i], T, div_yield)
            deltas.append(d)
            gammas.append(g)
            vegas.append(v)

        return RegimeGreeksResult(
            blended_delta=float(np.dot(probs, deltas)),
            blended_gamma=float(np.dot(probs, gammas)),
            blended_vega=float(np.dot(probs, vegas)),
            regime_deltas=deltas,
            regime_gammas=gammas,
            regime_vegas=vegas,
            regime_weights=probs.tolist(),
        )

    def regime_risk_decomposition(
        self,
        spot: float,
        strike: float,
        rate: float,
        T: float,
        option_type: OptionType = OptionType.CALL,
    ) -> dict:
        """Decompose option risk by regime.

        Shows how much of the price/risk comes from each regime.
        """
        result = self.price(spot, strike, rate, T, option_type)
        greeks = self.greeks(spot, strike, rate, T, option_type)

        contributions = []
        for i in range(self.n_regimes):
            w = result.regime_weights[i]
            contributions.append({
                "regime": i,
                "weight": w,
                "vol": result.regime_vols[i],
                "price_contribution": w * result.regime_prices[i],
                "price_pct": w * result.regime_prices[i] / max(result.blended_price, 1e-10) * 100,
                "delta": greeks.regime_deltas[i],
                "vega": greeks.regime_vegas[i],
            })

        return {
            "blended_price": result.blended_price,
            "blended_vol": result.blended_vol,
            "contributions": contributions,
            "transition_matrix": self._transition_matrix.tolist() if self._transition_matrix is not None else None,
        }

    def expected_regime_path(self, n_steps: int = 10, dt: float = 1/252) -> np.ndarray:
        """Forecast regime probabilities forward using transition matrix.

        Returns (n_steps+1, n_regimes) array of probability paths.
        """
        if self._transition_matrix is None:
            raise ValueError("Must call fit() first")

        from pricebook.numerical._linalg import expm

        probs = self._filtered_probs.copy()
        path = [probs.copy()]

        # Convert daily transition to dt-transition via matrix exponential
        # Q = log(P_daily) / (1/252), P_dt = expm(Q * dt)
        # Simpler: P_dt = P_daily^(dt * 252) ≈ expm(log(P_daily) * dt * 252)
        # Use direct matrix power for small dt
        P = self._transition_matrix
        for _ in range(n_steps):
            probs = probs @ P
            path.append(probs.copy())

        return np.array(path)

    def to_dict(self) -> dict:
        return {
            "n_regimes": self.n_regimes,
            "regime_vols": self._regime_vols.tolist() if self._regime_vols is not None else None,
            "filtered_probs": self._filtered_probs.tolist() if self._filtered_probs is not None else None,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def regime_option_price(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    regime_vols: list[float],
    regime_probs: list[float],
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> RegimePricingResult:
    """Price an option given explicit regime vols and probabilities.

    No HMM fitting required — for when regime parameters are known.
    """
    from pricebook.options.equity_option import equity_option_price

    probs = np.array(regime_probs)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        raise ValueError("Regime probabilities sum to zero or negative")
    probs = probs / prob_sum
    vols = np.array(regime_vols)

    prices = [equity_option_price(spot, strike, rate, v, T, option_type, div_yield)
              for v in vols]
    blended = float(np.dot(probs, prices))
    blended_vol = math.sqrt(float(np.dot(probs, vols ** 2)))

    return RegimePricingResult(
        blended_price=blended,
        regime_prices=prices,
        regime_weights=probs.tolist(),
        regime_vols=vols.tolist(),
        n_regimes=len(vols),
        blended_vol=blended_vol,
        regime_spread=max(prices) - min(prices),
        current_regime=int(np.argmax(probs)),
    )


def regime_greeks(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    regime_vols: list[float],
    regime_probs: list[float],
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> RegimeGreeksResult:
    """Compute regime-blended Greeks from explicit parameters."""
    from pricebook.options.equity_option import equity_delta, equity_gamma, equity_vega

    probs = np.array(regime_probs)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        raise ValueError("Regime probabilities sum to zero or negative")
    probs = probs / prob_sum

    deltas, gammas, vegas = [], [], []
    for v in regime_vols:
        deltas.append(equity_delta(spot, strike, rate, v, T, option_type, div_yield))
        gammas.append(equity_gamma(spot, strike, rate, v, T, div_yield))
        vegas.append(equity_vega(spot, strike, rate, v, T, div_yield))

    return RegimeGreeksResult(
        blended_delta=float(np.dot(probs, deltas)),
        blended_gamma=float(np.dot(probs, gammas)),
        blended_vega=float(np.dot(probs, vegas)),
        regime_deltas=deltas,
        regime_gammas=gammas,
        regime_vegas=vegas,
        regime_weights=probs.tolist(),
    )
