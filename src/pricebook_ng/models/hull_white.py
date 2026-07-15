"""Hull-White one-factor short-rate model — analytic core (L3).

dr(t) = (theta(t) - a r(t)) dt + sigma dW(t)

theta(t) is chosen so the model reprices the initial discount curve exactly, so
the model carries the market curve and reads P(0,T) from it (no separate fit to
store). This slice provides the model's analytic quantities: the B(t,T) factor
and the closed-form European option on a zero-coupon bond (the building block for
the Jamshidian swaption in the next slice).

Constant `a`, `sigma`, fitted to a flat curve. A general (bootstrapped) curve and
the swaption engine land with later slices (CLAUDE.md 6b).

Note (design drift, for the L3 report): the analytic ZCB-option *formula* is a
model capability (a closed-form property of the dynamics, like B(t,T)), not
instrument pricing — binding a Swaption instrument to this model is the L4 engine
in the next slice. This mirrors the QuantLib `discountBondOption` placement.

Provenance:
  quarry: python/pricebook/models/ (hull_white)
  source: Brigo & Mercurio, Interest Rate Models, s.3.3 (HW 1F; eqs. 3.40-3.41)
  oracle: curve refit + ZCB-option put-call parity + sigma->0 intrinsic, exact
  slice:  S07; S08 (zero_bond reconstitution for the Jamshidian swaption)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.distributions import norm_cdf
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot


@dataclass(frozen=True)
class HullWhite:
    """Hull-White 1F with constant mean reversion `a` and vol `sigma`, calibrated
    to (and carrying) a `MarketSnapshot` with a flat curve (Amendment A1: the
    model carries its market; the engine reaches the curve through it)."""

    a: float
    sigma: float
    market: MarketSnapshot

    @property
    def curve(self) -> FlatDiscountCurve:
        # flat-curve HW (S07 scope): the snapshot's curve is a FlatDiscountCurve;
        # a general-curve fit is a later slice.
        curve = self.market.discount_curve
        assert isinstance(curve, FlatDiscountCurve)
        return curve

    def _t(self, d: date) -> float:
        return year_fraction(self.curve.anchor, d, self.curve.day_count)

    def b(self, t: float, maturity: float) -> float:
        """B(t,T) = (1 - exp(-a (T-t))) / a, the ZCB-price sensitivity factor."""
        gap = maturity - t
        if abs(self.a) < 1e-12:
            return gap  # a -> 0 limit
        return (1.0 - math.exp(-self.a * gap)) / self.a

    def discount_factor(self, d: date) -> float:
        """P(0, d) — equals the market curve (the model refits it exactly)."""
        return self.curve.df(d)

    def forward_short_rate(self, t: float, z: float) -> float:
        """Short rate `r(t)` under the t-forward measure from a standard-normal draw
        `z` (exact Gaussian — no discretisation error). Shared by MC swaption pricing
        and MC exposure simulation; `t` is a year fraction. Assumes `a > 0` (the MC
        path; the analytic a->0 limits live on the other methods)."""
        a, sigma = self.a, self.sigma
        variance = sigma**2 * (1.0 - math.exp(-2.0 * a * t)) / (2.0 * a)
        fwd_mean = -(sigma**2 / a**2) * (
            (1.0 - math.exp(-a * t)) - 0.5 * (1.0 - math.exp(-2.0 * a * t))
        )
        alpha = self.curve.rate + (sigma**2 / (2.0 * a**2)) * (1.0 - math.exp(-a * t)) ** 2
        return alpha + fwd_mean + math.sqrt(variance) * z

    def zero_bond(self, expiry: date, bond_maturity: date, short_rate: float) -> float:
        """P(T, S) as a function of the short rate r(T) at `expiry` = T
        (reconstitution, Brigo & Mercurio 3.39): A(T,S) exp(-B(T,S) r).

        Used by the Jamshidian swaption to solve for the critical rate and the
        per-bond strikes. Flat-curve form: f(0,t)=r0, P(0,t)=exp(-r0 t)."""
        t, tm = self._t(expiry), self._t(bond_maturity)
        b = self.b(t, tm)
        r0 = self.curve.rate
        variance = (1.0 - math.exp(-2.0 * self.a * t)) / (2.0 * self.a) \
            if abs(self.a) > 1e-12 else t
        ln_a = (
            math.log(self.curve.df(bond_maturity) / self.curve.df(expiry))
            + b * r0
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
        P(T, S), struck at `strike` (Brigo & Mercurio 3.40-3.41)."""
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
