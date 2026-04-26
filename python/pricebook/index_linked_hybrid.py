"""Index-linked hybrid: cash-settled swaption with equity index strike.

Payoff = Â(S_T) * (θ(S_T - U_T))^+  where Â is the cash annuity,
S_T the swap rate, U_T the index level at expiry.

Priced under Q^T (T-forward measure) via 2D local-vol MC.

* :func:`index_linked_hybrid_price` — main pricing function.
* :func:`index_linked_hybrid_payoff` — payoff function for MC.

References:
    Pucci, M. (2012b). Pricing Index-Linked Hybrids. SSRN 2056277.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.cms import cash_annuity
from pricebook.hybrid_mc import simulate_2d_local_vol, LocalVolHybridResult


@dataclass
class IndexLinkedHybridResult:
    """Index-linked hybrid pricing result."""
    price: float
    std_error: float
    n_paths: int
    mean_swap_rate: float
    mean_index: float
    mean_cash_annuity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.price, "std_error": self.std_error,
            "mean_swap_rate": self.mean_swap_rate, "mean_index": self.mean_index,
        }


def _make_cash_annuity_fn(
    year_fractions: list[float],
    times_to_payment: list[float],
):
    """Create a vectorised cash annuity function."""
    yfs = year_fractions
    taus = times_to_payment

    def fn(S: np.ndarray) -> np.ndarray:
        result = np.zeros_like(S)
        for yi, tau_i in zip(yfs, taus):
            denom = 1 + yi * S
            denom = np.maximum(denom, 1e-10)
            result += yi / denom ** tau_i
        return result

    return fn


def index_linked_hybrid_price(
    F0: float,
    U0: float,
    discount_factor: float,
    year_fractions: list[float],
    times_to_payment: list[float],
    sigma_F,
    sigma_U,
    rho: float,
    T: float,
    theta: int = 1,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> IndexLinkedHybridResult:
    """Price an index-linked cash-settled swaption (Pucci 2012b, Eq 7).

    v = D_{0,T} * E^T[ Â(F_T) * (θ(F_T - U_T))^+ ]

    Args:
        F0: convexity-adjusted forward swap rate (Q^T martingale).
        U0: index T-forward.
        discount_factor: D_{0,T}.
        year_fractions: y_i for the swap schedule.
        times_to_payment: yf(T, T_i) for each coupon date.
        sigma_F: local vol for rate (callable(t, F) or flat float).
        sigma_U: local vol for index (callable(t, U) or flat float).
        rho: correlation between rate and index Brownians.
        theta: +1 payer, -1 receiver.
    """
    F_T, U_T = simulate_2d_local_vol(
        F0, U0, sigma_F, sigma_U, rho, T, n_paths, n_steps, seed)

    # Cash annuity (vectorised)
    annuity_fn = _make_cash_annuity_fn(year_fractions, times_to_payment)
    A_hat = annuity_fn(F_T)

    # Payoff
    intrinsic = np.maximum(theta * (F_T - U_T), 0.0)
    payoffs = A_hat * intrinsic

    price = discount_factor * float(payoffs.mean())
    std_err = discount_factor * float(payoffs.std()) / math.sqrt(n_paths)

    return IndexLinkedHybridResult(
        price=price,
        std_error=std_err,
        n_paths=n_paths,
        mean_swap_rate=float(F_T.mean()),
        mean_index=float(U_T.mean()),
        mean_cash_annuity=float(A_hat.mean()),
    )


# ---- Instrument class ----

class IndexLinkedHybridInstrument:
    """Index-linked cash-settled swaption for Trade/Portfolio.

        hybrid = IndexLinkedHybridInstrument(
            expiry=date(2031, 1, 15), swap_tenor=5,
            index_forward=0.04, sigma_F=0.30, sigma_U=0.25, rho=0.3)
        result = hybrid.price(discount_curve)
        portfolio.add(Trade(hybrid))
    """

    def __init__(
        self,
        expiry,
        swap_tenor: int = 5,
        index_forward: float = 0.04,
        notional: float = 1_000_000.0,
        theta: int = 1,
        sigma_F: float = 0.30,
        sigma_U: float = 0.25,
        rho: float = 0.0,
        frequency: int = 2,
        n_paths: int = 50_000,
        n_steps: int = 100,
        seed: int | None = 42,
    ):
        self.expiry = expiry
        self.swap_tenor = swap_tenor
        self.index_forward = index_forward
        self.notional = notional
        self.theta = theta
        self.sigma_F = sigma_F
        self.sigma_U = sigma_U
        self.rho = rho
        self.frequency = frequency
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def price(self, curve) -> IndexLinkedHybridResult:
        """Price the index-linked hybrid using a discount curve."""
        from pricebook.day_count import year_fraction, DayCountConvention
        from datetime import timedelta

        T = year_fraction(curve.reference_date, self.expiry,
                          DayCountConvention.ACT_365_FIXED)
        df = curve.df(self.expiry)

        n = self.swap_tenor * self.frequency
        dt_period = 1.0 / self.frequency
        yfs = [dt_period] * n
        taus = [dt_period * (i + 1) for i in range(n)]

        # Convexity-adjusted forward swap rate (approximate: par rate from curve)
        dates = [self.expiry + timedelta(days=int(dt_period * 365 * (i + 1)))
                 for i in range(n)]
        dfs = [curve.df(d) for d in dates]
        annuity = sum(y * d for y, d in zip(yfs, dfs))
        F0 = (curve.df(self.expiry) - dfs[-1]) / annuity if annuity > 0 else 0.0

        return index_linked_hybrid_price(
            F0, self.index_forward, df, yfs, taus,
            self.sigma_F, self.sigma_U, self.rho, T,
            self.theta, self.n_paths, self.n_steps, self.seed)

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — compatible with Trade.pv()."""
        result = self.price(ctx.discount_curve)
        return self.notional * result.price
