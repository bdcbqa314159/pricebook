"""Per-issuer credit spread curve (C9).

Nelson-Siegel parameterisation of credit spreads (not yields).
Fits a smooth curve to observed bond spreads for a single issuer.

    from pricebook.credit.issuer_curve import (
        IssuerSpreadCurve, fit_issuer_curve,
    )

References:
    Nelson & Siegel (1987). Parsimonious Modeling of Yield Curves.
    Diebold & Li (2006). Forecasting the Term Structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class IssuerSpreadCurve:
    """Nelson-Siegel spread curve for a single issuer."""
    beta0: float    # level (long-run spread)
    beta1: float    # slope (short-term component)
    beta2: float    # curvature (medium-term hump)
    tau: float      # decay parameter
    issuer: str = ""

    def spread(self, t: float) -> float:
        """Spread at maturity t (years) in decimal (e.g. 0.015 = 150bp)."""
        if t <= 0:
            return self.beta0 + self.beta1
        x = t / self.tau
        exp_x = math.exp(-x)
        factor1 = (1 - exp_x) / x if x > 1e-10 else 1.0
        factor2 = factor1 - exp_x
        return self.beta0 + self.beta1 * factor1 + self.beta2 * factor2

    def spread_bp(self, t: float) -> float:
        """Spread at maturity t in basis points."""
        return self.spread(t) * 10_000

    def term_structure(self, tenors: list[float]) -> list[float]:
        """Spread term structure in bp."""
        return [self.spread_bp(t) for t in tenors]

    def to_dict(self) -> dict:
        return {
            "beta0": self.beta0, "beta1": self.beta1,
            "beta2": self.beta2, "tau": self.tau,
            "issuer": self.issuer,
        }


def fit_issuer_curve(
    tenors: list[float],
    spreads_bp: list[float],
    issuer: str = "",
) -> IssuerSpreadCurve:
    """Fit Nelson-Siegel spread curve to observed bond spreads.

    Args:
        tenors: maturity in years for each observation.
        spreads_bp: observed spread in bp for each bond.
        issuer: issuer name (metadata).
    """
    if len(tenors) < 2:
        raise ValueError("Need at least 2 observations to fit")

    spreads = np.array(spreads_bp) / 10_000
    t = np.array(tenors)

    def objective(x):
        b0, b1, b2, tau = x[0], x[1], x[2], max(x[3], 0.1)
        curve = IssuerSpreadCurve(b0, b1, b2, tau)
        fitted = np.array([curve.spread(ti) for ti in t])
        return np.sum((fitted - spreads) ** 2)

    # Initial guess
    b0_guess = spreads[-1]
    b1_guess = spreads[0] - spreads[-1]
    x0 = [b0_guess, b1_guess, 0.0, 2.0]

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-8})

    return IssuerSpreadCurve(
        result.x[0], result.x[1], result.x[2], max(result.x[3], 0.1),
        issuer=issuer,
    )
