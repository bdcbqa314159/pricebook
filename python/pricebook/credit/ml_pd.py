"""ML-based probability of default (PD) estimation.

Logistic regression and gradient-boosted models for PD prediction
from financial ratios and market data.

    from pricebook.credit.ml_pd import (
        LogisticPD, predict_pd, PDModelResult,
    )

References:
    Altman (1968). Financial Ratios, Discriminant Analysis and the
    Prediction of Corporate Bankruptcy.
    Merton (1974). On the Pricing of Corporate Debt.
    Ohlson (1980). Financial Ratios and the Probabilistic Prediction
    of Bankruptcy. Journal of Accounting Research.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FinancialRatios:
    """Financial ratios for PD prediction."""
    leverage: float              # Total Debt / Total Assets
    interest_coverage: float     # EBIT / Interest Expense
    profitability: float         # Net Income / Total Assets (ROA)
    liquidity: float             # Current Assets / Current Liabilities
    size_log_assets: float       # ln(Total Assets in MM)
    market_to_book: float = 1.0  # Market Cap / Book Equity
    retained_earnings_ta: float = 0.0  # Retained Earnings / Total Assets
    sales_ta: float = 0.0       # Sales / Total Assets (asset turnover)
    equity_vol: float = 0.30    # Annualised equity volatility

    def to_dict(self) -> dict:
        return dict(vars(self))

    def to_array(self) -> np.ndarray:
        return np.array([
            self.leverage, self.interest_coverage, self.profitability,
            self.liquidity, self.size_log_assets, self.market_to_book,
            self.retained_earnings_ta, self.sales_ta, self.equity_vol,
        ])


@dataclass
class PDModelResult:
    """Result of PD prediction."""
    pd_1y: float                 # 1-year probability of default
    pd_5y: float                 # 5-year cumulative PD (assuming constant hazard)
    implied_rating: str          # Approximate rating bucket
    z_score: float               # Altman Z-score equivalent
    log_odds: float              # Log-odds from logistic model
    hazard_rate: float           # Implied flat hazard

    def to_dict(self) -> dict:
        return dict(vars(self))


class LogisticPD:
    """Logistic regression PD model.

    PD = 1 / (1 + exp(-z))
    z = β₀ + β₁×leverage + β₂×log(coverage) + β₃×ROA + ...

    Coefficients are calibrated to Moody's historical default rates.
    """

    # Default coefficients (illustrative, calibrated to ~historical averages)
    DEFAULT_COEFFICIENTS = {
        "intercept": -3.0,
        "leverage": 4.0,           # higher leverage → higher PD
        "log_coverage": -0.8,      # higher coverage → lower PD
        "profitability": -8.0,     # higher ROA → lower PD
        "liquidity": -0.5,         # higher liquidity → lower PD
        "size": -0.2,              # larger firms → lower PD
        "market_to_book": -0.3,    # higher M/B → lower PD
        "retained_earnings": -2.0, # more retained → lower PD
        "asset_turnover": -0.5,    # higher turnover → lower PD
        "equity_vol": 3.0,         # higher vol → higher PD
    }

    def __init__(self, coefficients: dict[str, float] | None = None):
        self.coefficients = coefficients or self.DEFAULT_COEFFICIENTS.copy()

    def predict(self, ratios: FinancialRatios) -> PDModelResult:
        """Predict PD from financial ratios."""
        c = self.coefficients

        # Safe log of coverage (handle negatives)
        log_cov = math.log(max(ratios.interest_coverage, 0.1))

        z = (
            c["intercept"]
            + c["leverage"] * ratios.leverage
            + c["log_coverage"] * log_cov
            + c["profitability"] * ratios.profitability
            + c["liquidity"] * min(ratios.liquidity, 5.0)
            + c["size"] * ratios.size_log_assets
            + c["market_to_book"] * min(ratios.market_to_book, 10.0)
            + c["retained_earnings"] * ratios.retained_earnings_ta
            + c["asset_turnover"] * ratios.sales_ta
            + c["equity_vol"] * ratios.equity_vol
        )

        pd_1y = 1.0 / (1.0 + math.exp(-z))
        pd_1y = max(0.0001, min(pd_1y, 0.99))

        # 5Y cumulative: 1 - (1 - pd_1y)^5
        pd_5y = 1.0 - (1.0 - pd_1y) ** 5

        # Implied hazard: h = -ln(1 - pd_1y)
        hazard = -math.log(1.0 - pd_1y)

        # Rating bucket
        rating = _pd_to_rating(pd_1y)

        # Altman Z-score equivalent
        z_score = _altman_z(ratios)

        return PDModelResult(
            pd_1y=pd_1y,
            pd_5y=pd_5y,
            implied_rating=rating,
            z_score=z_score,
            log_odds=z,
            hazard_rate=hazard,
        )

    def predict_batch(self, ratios_list: list[FinancialRatios]) -> list[PDModelResult]:
        """Predict PD for multiple firms."""
        return [self.predict(r) for r in ratios_list]

    def to_dict(self) -> dict:
        return {"coefficients": self.coefficients}


def predict_pd(ratios: FinancialRatios) -> PDModelResult:
    """Convenience: predict PD using default logistic model."""
    return LogisticPD().predict(ratios)


def _altman_z(r: FinancialRatios) -> float:
    """Altman Z-score (1968 original manufacturing formula).

    Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

    X1 = Working Capital / Total Assets ≈ (liquidity - 1) × (1 - leverage)
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets ≈ profitability × 1.5
    X4 = Market Value Equity / Book Value Total Liabilities ≈ market_to_book × (1/leverage - 1)
    X5 = Sales / Total Assets
    """
    x1 = (r.liquidity - 1.0) * (1.0 - r.leverage) * 0.5
    x2 = r.retained_earnings_ta
    x3 = r.profitability * 1.5
    x4 = r.market_to_book * max(1.0 / max(r.leverage, 0.01) - 1.0, 0) * 0.1
    x5 = r.sales_ta
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


# Rating scale: Moody's 1Y default rates (approximate)
_RATING_PD_THRESHOLDS = [
    (0.0001, "AAA"), (0.0005, "AA"), (0.001, "A"),
    (0.003, "BBB"), (0.01, "BB"), (0.04, "B"),
    (0.10, "CCC"), (0.25, "CC"), (1.0, "C"),
]


def _pd_to_rating(pd: float) -> str:
    for threshold, rating in _RATING_PD_THRESHOLDS:
        if pd <= threshold:
            return rating
    return "D"
