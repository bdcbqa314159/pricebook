"""Hull-White one-factor short-rate model — analytic core (L3).

dr(t) = (theta(t) - a r(t)) dt + sigma dW(t)

theta(t) is chosen so the model reprices the initial discount curve exactly, so
the model carries the market curve and reads P(0,T) and the instantaneous forward
f(0,t) from it (no separate fit to store). Provides the analytic quantities: the
B(t,T) factor, the r(t) reconstitution P(t,S), and the closed-form European option
on a zero-coupon bond (the Jamshidian swaption building block).

General-curve (CP-2b): the model works on ANY curve exposing `df` + the market
instantaneous forward `instantaneous_forward` — flat OR bootstrapped. The flat
curve is the degenerate case f(0,t)=r0, so its results are unchanged. Constant
`a`, `sigma` (a term-structure of vol is a later slice).

Provenance:
  quarry: python/pricebook/models/ (hull_white)
  source: Brigo & Mercurio, Interest Rate Models, s.3.3 (HW 1F; eqs. 3.39-3.41)
  oracle: curve refit P(0,S) + ZCB-option put-call parity + sigma->0 intrinsic; on a
          bootstrapped curve, analytic swaption == MC
  slice:  S07; S08 (zero_bond reconstitution); general-curve-hw (CP-2b #2 — f(0,t) curve)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.distributions import norm_cdf
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.snapshot import CurveHandle, MarketSnapshot

_CURVE_DC = DayCountConvention.ACT_365_FIXED  # the model's time axis, from the valuation date


@dataclass(frozen=True)
class HullWhite:
    """Hull-White 1F with constant mean reversion `a` and vol `sigma`, calibrated to
    (and carrying) a `MarketSnapshot` (Amendment A1). Reads the market curve through the
    snapshot — `df` for P(0,T) and `instantaneous_forward` for f(0,t)."""

    a: float
    sigma: float
    market: MarketSnapshot

    @property
    def curve(self) -> CurveHandle:
        """The market discount curve (flat or bootstrapped); reached via its `df` +
        `instantaneous_forward` capabilities, no concrete type assumed."""
        return self.market.discount_curve

    def _t(self, d: date) -> float:
        return year_fraction(self.market.valuation_date, d, _CURVE_DC)

    def _forward(self, d: date) -> float:
        """Market instantaneous forward f(0,t) at date `d` (r0 for a flat curve)."""
        return self.curve.instantaneous_forward(d)

    def b(self, t: float, maturity: float) -> float:
        """B(t,T) = (1 - exp(-a (T-t))) / a, the ZCB-price sensitivity factor."""
        gap = maturity - t
        if abs(self.a) < 1e-12:
            return gap  # a -> 0 limit
        return (1.0 - math.exp(-self.a * gap)) / self.a

    def discount_factor(self, d: date) -> float:
        """P(0, d) — equals the market curve (the model refits it exactly)."""
        return self.curve.df(d)

    def forward_short_rate(self, at: date, z: float) -> float:
        """Short rate `r(t)` at date `at` under the t-forward measure from a standard-normal
        draw `z` (exact Gaussian — no discretisation error). Shared by MC swaption pricing and
        exposure simulation. `alpha(t) = f(0,t) + (sigma^2/2a^2)(1-e^{-at})^2` uses the market
        forward, so it is general-curve. Assumes `a > 0` (the MC path)."""
        a, sigma = self.a, self.sigma
        t = self._t(at)
        variance = sigma**2 * (1.0 - math.exp(-2.0 * a * t)) / (2.0 * a)
        fwd_mean = -(sigma**2 / a**2) * (
            (1.0 - math.exp(-a * t)) - 0.5 * (1.0 - math.exp(-2.0 * a * t))
        )
        alpha = self._forward(at) + (sigma**2 / (2.0 * a**2)) * (1.0 - math.exp(-a * t)) ** 2
        return alpha + fwd_mean + math.sqrt(variance) * z

    def zero_bond(self, expiry: date, bond_maturity: date, short_rate: float) -> float:
        """P(T, S) as a function of the short rate r(T) at `expiry` = T (reconstitution,
        Brigo & Mercurio 3.39): A(T,S) exp(-B(T,S) r), with
        `ln A = ln(P(0,S)/P(0,T)) + B f(0,T) - (sigma^2/4a)(1-e^{-2aT}) B^2` — the market
        forward `f(0,T)` generalises the flat `r0`."""
        t, tm = self._t(expiry), self._t(bond_maturity)
        b = self.b(t, tm)
        forward = self._forward(expiry)
        variance = (1.0 - math.exp(-2.0 * self.a * t)) / (2.0 * self.a) \
            if abs(self.a) > 1e-12 else t
        ln_a = (
            math.log(self.curve.df(bond_maturity) / self.curve.df(expiry))
            + b * forward
            - 0.5 * self.sigma**2 * variance * b**2
        )
        return math.exp(ln_a - b * short_rate)

    def zero_bond_option(
        self,
        expiry: date,
        bond_maturity: date,
        strike: float,
        is_call: bool,
    ) -> float:
        """Value at t=0 of a European option (expiry T) on the zero-coupon bond
        P(T, S), struck at `strike` (Brigo & Mercurio 3.40-3.41). Uses only P(0,·) and
        the bond-option vol, so it is already general-curve."""
        p_t = self.curve.df(expiry)
        p_s = self.curve.df(bond_maturity)
        t = self._t(expiry)
        sigma_p = self._bond_option_vol(t, self._t(bond_maturity))
        if sigma_p < 1e-15:  # sigma=0, T=0, or S=T -> deterministic intrinsic
            intrinsic = p_s - strike * p_t
            return max(intrinsic, 0.0) if is_call else max(-intrinsic, 0.0)
        h = math.log(p_s / (strike * p_t)) / sigma_p + sigma_p / 2.0
        if is_call:
            return p_s * norm_cdf(h) - strike * p_t * norm_cdf(h - sigma_p)
        return strike * p_t * norm_cdf(-h + sigma_p) - p_s * norm_cdf(-h)

    def _bond_option_vol(self, expiry_t: float, bond_maturity_t: float) -> float:
        """sigma_P: the lognormal vol of P(T,S) seen at time 0."""
        var_x = (1.0 - math.exp(-2.0 * self.a * expiry_t)) / (2.0 * self.a) \
            if abs(self.a) > 1e-12 else expiry_t
        return self.sigma * math.sqrt(var_x) * self.b(expiry_t, bond_maturity_t)
