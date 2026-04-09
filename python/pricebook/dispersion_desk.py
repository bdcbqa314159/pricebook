"""Dispersion Trading: implied correlation, dispersion trades, correlation risk.

A dispersion trader buys variance on the basket of single names and sells
variance on the index (or vice versa). The trade isolates the difference
between the average single-name variance and the index variance — i.e.
the average pairwise correlation embedded in the index.

References:
    Bossu, *Implied Correlation Index*, JPMorgan, 2007.
    Marabel Romo, *Dispersion Trading and Implied Correlation*, 2012.
    Driessen, Maenhout & Vilkov, *The Price of Correlation Risk*, JF, 2009.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


# ---- Implied correlation ----

def index_variance(
    weights: list[float],
    single_vols: list[float],
    correlation: float,
) -> float:
    """Index variance assuming a uniform pairwise correlation ρ.

        σ²_idx = Σ w_i² σ_i² + 2 ρ Σ_{i<j} w_i w_j σ_i σ_j
              = Σ w_i² σ_i² + ρ · ((Σ w_i σ_i)² − Σ w_i² σ_i²)
    """
    sum_w_sigma = sum(w * s for w, s in zip(weights, single_vols))
    sum_w2_sigma2 = sum(w * w * s * s for w, s in zip(weights, single_vols))
    return sum_w2_sigma2 + correlation * (
        sum_w_sigma * sum_w_sigma - sum_w2_sigma2
    )


def index_vol(
    weights: list[float],
    single_vols: list[float],
    correlation: float,
) -> float:
    """Index volatility = sqrt(index_variance)."""
    var = index_variance(weights, single_vols, correlation)
    return math.sqrt(max(var, 0.0))


def implied_correlation(
    weights: list[float],
    single_vols: list[float],
    index_vol: float,
) -> float:
    """Average pairwise correlation implied by an index/basket vol.

        ρ_impl = (σ²_idx − Σ w_i² σ_i²) / ((Σ w_i σ_i)² − Σ w_i² σ_i²)

    Returns 0.0 if the denominator is degenerate (all weight on a single name).
    """
    sum_w_sigma = sum(w * s for w, s in zip(weights, single_vols))
    sum_w2_sigma2 = sum(w * w * s * s for w, s in zip(weights, single_vols))
    denom = sum_w_sigma * sum_w_sigma - sum_w2_sigma2
    if abs(denom) < 1e-12:
        return 0.0
    return (index_vol * index_vol - sum_w2_sigma2) / denom


def historical_correlation(returns: list[list[float]]) -> float:
    """Mean of the upper-triangle pairwise correlation matrix.

    Args:
        returns: ``N`` return series, each of equal length ``T``.

    Returns:
        Average pairwise correlation. 0.0 if N < 2.
    """
    n = len(returns)
    if n < 2:
        return 0.0

    means = [sum(r) / len(r) for r in returns]
    devs = [
        [r[t] - means[i] for t in range(len(r))]
        for i, r in enumerate(returns)
    ]
    var = [sum(d * d for d in devs[i]) / len(devs[i]) for i in range(n)]

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            cov = sum(
                devs[i][t] * devs[j][t] for t in range(len(devs[i]))
            ) / len(devs[i])
            denom = math.sqrt(var[i] * var[j])
            if denom > 0:
                total += cov / denom
                count += 1
    return total / count if count > 0 else 0.0


# ---- Dispersion Trade ----

@dataclass
class DispersionTrade:
    """Long basket of single-name variance vs short index variance.

    The trade pays:
        single leg:  Σ w_i N_i (σ²_i − K_i)
        index leg:  −N_idx (σ²_idx − K_idx)

    ``direction = +1`` is *long dispersion / short correlation*: the
    trader receives single-name variance and pays index variance.
    """
    tickers: list[str]
    weights: list[float]
    single_strikes: list[float]
    single_notionals: list[float]
    index_strike: float
    index_notional: float
    direction: int = 1

    def __post_init__(self) -> None:
        n = len(self.tickers)
        if not (
            len(self.weights)
            == len(self.single_strikes)
            == len(self.single_notionals)
            == n
        ):
            raise ValueError(
                "tickers / weights / single_strikes / single_notionals "
                "must all have the same length"
            )

    @property
    def n_names(self) -> int:
        return len(self.tickers)

    def pv(
        self,
        single_vols: list[float],
        correlation: float,
    ) -> float:
        """Model PV given current implied vols and a uniform correlation."""
        if len(single_vols) != self.n_names:
            raise ValueError("single_vols length must match number of names")

        single_pnl = sum(
            w * n * (s * s - k)
            for w, n, s, k in zip(
                self.weights,
                self.single_notionals,
                single_vols,
                self.single_strikes,
            )
        )
        var_idx = index_variance(self.weights, single_vols, correlation)
        idx_pnl = self.index_notional * (var_idx - self.index_strike)
        return self.direction * (single_pnl - idx_pnl)

    def correlation_sensitivity(
        self,
        single_vols: list[float],
    ) -> float:
        """Analytic ∂PV/∂ρ.

        Only the index leg depends on correlation:
            ∂σ²_idx/∂ρ = (Σ w_i σ_i)² − Σ w_i² σ_i²
            ∂PV/∂ρ     = −direction · N_idx · ∂σ²_idx/∂ρ
        """
        sum_w_sigma = sum(w * s for w, s in zip(self.weights, single_vols))
        sum_w2_sigma2 = sum(
            w * w * s * s for w, s in zip(self.weights, single_vols)
        )
        return -self.direction * self.index_notional * (
            sum_w_sigma * sum_w_sigma - sum_w2_sigma2
        )

    def dispersion_value(
        self,
        single_vols: list[float],
        correlation: float,
    ) -> float:
        """Pure dispersion = basket variance − index variance, no strikes.

        This is the model "fair value" expression that drives P&L: it
        is monotonically decreasing in correlation and reaches its
        maximum at ρ = 0.
        """
        basket_var = sum(
            w * s * s for w, s in zip(self.weights, single_vols)
        )
        idx_var = index_variance(self.weights, single_vols, correlation)
        return self.direction * (basket_var - idx_var)


# ---- Correlation term structure ----

@dataclass
class CorrelationTermStructure:
    """Implied correlation as a function of expiry, with linear interpolation
    between pillars and flat extrapolation at the wings.
    """
    reference_date: date
    expiries: list[date]
    correlations: list[float]

    def __post_init__(self) -> None:
        if len(self.expiries) != len(self.correlations):
            raise ValueError("expiries and correlations must have the same length")
        if len(self.expiries) < 1:
            raise ValueError("need at least 1 pillar")
        order = sorted(range(len(self.expiries)), key=lambda i: self.expiries[i])
        self.expiries = [self.expiries[i] for i in order]
        self.correlations = [self.correlations[i] for i in order]

    def correlation(self, expiry: date) -> float:
        if expiry <= self.expiries[0]:
            return self.correlations[0]
        if expiry >= self.expiries[-1]:
            return self.correlations[-1]
        for i in range(len(self.expiries) - 1):
            if self.expiries[i] <= expiry <= self.expiries[i + 1]:
                d_total = (self.expiries[i + 1] - self.expiries[i]).days
                if d_total == 0:
                    return self.correlations[i]
                d_part = (expiry - self.expiries[i]).days
                w = d_part / d_total
                return self.correlations[i] * (1 - w) + self.correlations[i + 1] * w
        return self.correlations[-1]

    @property
    def n_pillars(self) -> int:
        return len(self.expiries)
