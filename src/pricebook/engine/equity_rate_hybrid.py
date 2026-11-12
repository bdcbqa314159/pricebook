"""The equity vanilla under stochastic (Hull-White) rates (L4) - Brockhaus eq 11.1.

With deterministic rates the equity vanilla is Black on the forward. With stochastic rates it is
**still Black on the forward**, with two changes, and both are the tranche's content:

* `d+/-` use the **total** volatility of `log S`, which includes the cash bond's contribution and
  their cross term (eq 11.2), not the quoted equity vol `sigma_{S/B}`;
* both `d+/-` are **shifted** by `rho_{B,S} sigma_B sqrt(t)` (eq 11.1/11.3).

        E[B_t^-1 (S_t - K)^+] = P(t) [ F(t) N(d+ + rho_BS sigma_B sqrt t)
                                       - K    N(d- + rho_BS sigma_B sqrt t) ]

Switch the rate vol off and every one of those terms vanishes: the shift goes to zero, the cross
term goes to zero, the total vol collapses to the quoted one, and the formula is tranche 30's
exactly. That reduction is the anchor - and it is also what identifies the cumulative-normal glyph,
which `pdftotext` drops in **both** renderings of (11.1).

**Two findings from reading the source, recorded because both changed the code.**

1. **The printed §11.1.2 bond integrand cannot be read naively.** It gives
   `(sigma^B_t)^2 t = integral_0^t (v_s(T) - v_s(t))^2 ds`. For a European option the payoff date
   *is* `t`, so a naive `T = t` makes that **identically zero** and with it every hybrid term - the
   formula would silently collapse to Black and the whole tranche would be a no-op that still
   passed its reduction oracle. §11.1.1 settles it: `B` is the **cash bond**, defined there by
   `E[B_t^-1] = P(t)`, so `sigma_B` is the volatility of `log B_t`, which for Hull-White is
   `sigma_r^2 integral_0^t beta(s,t)^2 ds`. That is what this module computes, and the test checks
   it against its own quadrature of that integrand.

2. **Eq (11.3)'s numerator is not `sigma_B + rho*sigma_{S/B}`.** Read that way - taking the printed
   `rho_{B,S/B}` for the model's instantaneous `rho` - it disagrees with the book's **own
   derivation** three lines above, which computes
   `rho_BS sigma_B sigma_S t = E[sigma_B W^B * sigma_S W^S] = sigma_B^2 t + rho_{B,S/B} sigma_B sigma_{S/B} t`.
   The two differ because `rho_{B,S/B}` is a *time-averaged* correlation: the cross covariance is
   `rho*sigma*X` with `X` the **mean** bond vol, and `X != sigma_B` unless the bond vol is constant
   in time, which under mean reversion it is not. The derivation step is the authority (§4 rates an
   earlier step of a source's own derivation above its final line), so this uses
   `rho_BS = (sigma_B^2 + cross) / (sigma_B sigma_S)`.

   That form is also the numerically sound one: what (11.1) actually needs is the **product**
   `rho_BS * sigma_B = (sigma_B^2 + cross)/sigma_S`, which has no division by `sigma_B` at all and
   is therefore exactly zero at zero rate vol rather than `0/0`.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 11.1.1 eq (11.1)/(11.2)/(11.3) and ch 11.1.2 (the
          Hull-White specialisation), read in `-raw` and `-layout` and reconciled; `v_s(T) =
          sigma_r beta(s,T)` is tranche 4/10a's verified `hw_beta`
  oracle: tests/L4/test_equity_rate_hybrid.py (PRE-REGISTERED) - the (11.2) and (11.3) derivation-
          step identities over many parameter sets; the `sigma_r -> 0` reduction **bit-identical**
          to tranche 30, with the mandatory material companion at `sigma_r > 0, rho != 0`; put-call
          parity and the no-arbitrage bounds; the integrated variances against the test's own
          quadrature; and `|rho_BS| <= 1` for every admissible parameter set
  crosscheck: quantlib AnalyticBSMHullWhiteEngine - deferred to the optional QuantLib tier
  slice:  tranche 34 - the equity-rate hybrid, the arc's first
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.models.black import black_price
from pricebook.models.equity_hull_white import EquityHullWhiteModel
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption


def integrated_bond_variance(kappa: float, sigma_r: float, years: float) -> float:
    """`Var[log B_t] = sigma_r^2 * integral_0^t beta(s,t)^2 ds` - the cash bond's log-variance.

    Closed form with constant parameters, from `beta(s,t) = (1 - e^-k(t-s))/k`:

        sigma_r^2/k^2 * [ t - 2(1 - e^-kt)/k + (1 - e^-2kt)/(2k) ]

    Returned as the **integral**, not the annualised variance, because that is what §11.1.2 writes
    and what the cross term below must be consistent with; the caller divides by `t` once, in one
    place, rather than each piece doing it differently.
    """
    if years <= 0.0:
        raise ValueError(f"the integrated variance needs a positive horizon, got {years}")
    if kappa <= 0.0:
        raise ValueError(f"the mean reversion must be positive, got {kappa}")
    decay = math.exp(-kappa * years)
    return (
        sigma_r
        * sigma_r
        / (kappa * kappa)
        * (
            years
            - 2.0 * (1.0 - decay) / kappa
            + (1.0 - decay * decay) / (2.0 * kappa)
        )
    )


def integrated_cross_covariance(
    kappa: float, sigma_r: float, equity_vol: float, correlation: float, years: float
) -> float:
    """`integral_0^t rho*sigma*v_s(t) ds` - §11.1.2's third line, the equity/bond cross term.

    `= rho*sigma*sigma_r/k * [ t - (1 - e^-kt)/k ]`. It carries the sign of `rho`, which is the
    whole reason a hybrid price is not symmetric in the correlation.
    """
    if years <= 0.0:
        raise ValueError(f"the cross covariance needs a positive horizon, got {years}")
    if kappa <= 0.0:
        raise ValueError(f"the mean reversion must be positive, got {kappa}")
    return (
        correlation
        * equity_vol
        * sigma_r
        / kappa
        * (years - (1.0 - math.exp(-kappa * years)) / kappa)
    )


@dataclass(frozen=True)
class HybridBlackInputs:
    """What (11.1) feeds to Black: the total log-spot vol and the `d+/-` shift.

    Named surface rather than locals, because these are the two things the tranche's oracles must
    reach: the (11.2) identity is a statement about `total_vol`, and the shift is the term every
    `rho` mutation moves (§4 - extract the buried step so a test can see it).
    """

    total_vol: float
    d_shift: float


def hybrid_black_inputs(
    model: EquityHullWhiteModel,
    underlying: EquityUnderlying,
    expiry: date,
    discount: CurveHandle,
) -> HybridBlackInputs:
    """Assemble (11.2)'s total volatility and (11.1)'s `d+/-` shift."""
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the hybrid inputs need a positive horizon, got {years}")
    kappa, sigma_r = model.rate_params.kappa, model.rate_params.sigma
    equity_vol = model.equity_vol

    bond_variance = integrated_bond_variance(kappa, sigma_r, years) / years
    cross = integrated_cross_covariance(kappa, sigma_r, equity_vol, model.correlation, years) / years
    total_variance = bond_variance + equity_vol * equity_vol + 2.0 * cross
    total_vol = math.sqrt(max(total_variance, 0.0))

    # `rho_BS * sigma_B` as the single product (sigma_B^2 + cross)/sigma_S, which is exactly zero at
    # zero rate vol instead of 0/0 - see the module docstring, finding 2.
    shift = 0.0 if total_vol == 0.0 else (bond_variance + cross) / total_vol * math.sqrt(years)
    return HybridBlackInputs(total_vol=total_vol, d_shift=shift)


def hybrid_correlation(
    model: EquityHullWhiteModel,
    underlying: EquityUnderlying,
    expiry: date,
    discount: CurveHandle,
) -> float:
    """`rho_{B,S}` itself - eq (11.3), as the book's own derivation gives it.

    Exposed separately from the shift because it is a **correlation** and the oracle asserts it
    stays in `[-1, 1]` for every admissible parameter set, which is the cheapest check that the
    denominator is `sigma_S` and not something that merely agrees at one point.
    """
    years = discount.time_measure.year_fraction(expiry)
    kappa, sigma_r = model.rate_params.kappa, model.rate_params.sigma
    bond_variance = integrated_bond_variance(kappa, sigma_r, years) / years
    bond_vol = math.sqrt(bond_variance)
    cross = (
        integrated_cross_covariance(kappa, sigma_r, model.equity_vol, model.correlation, years)
        / years
    )
    total_vol = hybrid_black_inputs(model, underlying, expiry, discount).total_vol
    if bond_vol == 0.0 or total_vol == 0.0:
        return 0.0
    return (bond_variance + cross) / (bond_vol * total_vol)


@register(EuropeanEquityOption, EquityHullWhiteModel)
def price_european_equity_option_hull_white(
    product: EuropeanEquityOption, model: EquityHullWhiteModel
) -> PricingResult | PricingFailure:
    """Eq (11.1) - Black on the forward, with the total vol and the shifted `d+/-`."""
    market = model.market
    currency = product.notional.currency
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    underlying = market.equities.get(product.asset)
    if underlying is None:
        return PricingFailure(f"no equity data for {product.asset.ticker} in the snapshot")
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the equity option under Hull-White: {exc}")

    years = discount.time_measure.year_fraction(product.expiry)
    forward = dividend_forward(underlying, product.expiry, discount)
    inputs = hybrid_black_inputs(model, underlying, product.expiry, discount)

    if inputs.d_shift == 0.0:
        # No rate dynamics: this IS tranche 30's call, computed by the same operations in the same
        # order, so the reduction is bit-identical rather than merely close (§4's stronger proof).
        undiscounted = black_price(
            forward, product.strike, inputs.total_vol * inputs.total_vol * years, product.kind
        )
    else:
        undiscounted = _shifted_black(
            forward, product.strike, inputs, years, product.kind
        )
    return PricingResult(
        pv=Money(discount.df(product.expiry) * product.notional.amount * undiscounted, currency)
    )


def _shifted_black(
    forward: float,
    strike: float,
    inputs: HybridBlackInputs,
    years: float,
    kind: EquityOptionKind,
) -> float:
    """`omega[F N(omega(d+ + shift)) - K N(omega(d- + shift))]`, undiscounted.

    The put comes from the same two shifted arguments with `omega = -1` rather than from put-call
    parity applied afterwards, so parity becomes a genuine *check* on this expression instead of
    the way the put was produced.
    """
    from pricebook.foundation.distributions import norm_cdf

    omega = kind.omega
    width = inputs.total_vol * math.sqrt(years)
    d_plus = (math.log(forward / strike) + 0.5 * width * width) / width
    d_minus = d_plus - width
    return omega * (
        forward * norm_cdf(omega * (d_plus + inputs.d_shift))
        - strike * norm_cdf(omega * (d_minus + inputs.d_shift))
    )
