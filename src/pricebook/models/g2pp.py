"""The two-factor Hull-White model (G2++) (L3). `MCINERNEY-ZASTAWNIAK` §3.7.

Two **correlated** Gaussian factors instead of one - the first model in this tree carrying a factor
correlation in its own dynamics, which is why the T17/T43 correlation discipline from the spread arc
returns here, and why §4's newest clause gets its first real use.

Why two factors at all, in the source's own terms: vanillas and even Bermudans are only weakly
affected by how rates of different maturities co-move, but exotics that depend non-linearly on the
*difference* between two rates are "particularly sensitive" to it, and a one-factor model makes all
rates perfectly correlated by construction.

**The source's notation, followed rather than a familiar one (§5, and the tranche-66 lesson).** The
book uses `alpha, beta` for the two mean-reversion speeds and `sigma, eta` for the two vols; the
tranche spec wrote `(a,b)` and `(sigma_x, sigma_y)`. Following the print keeps this arc's chapters
readable against one vocabulary - which matters more than usual here, since §3.7 also carries a
*second* correlation (`varrho`, between the original `W` and `Z`) that is **not** `rho`, and the book
says so explicitly.

    r(t)   = phi(t) + x(t) + y(t)                                                      (3.34)
    dx     = -alpha x dt + sigma dU,   dy = -beta y dt + eta dV,   dU dV = rho dt      (3.35-3.37)

    V(t,T) = (sigma^2/alpha^2) int (1-e^{-alpha(T-u)})^2 du
           + (eta^2/beta^2)    int (1-e^{-beta(T-u)})^2 du
           + 2 rho (sigma eta/(alpha beta)) int (1-e^{-alpha(T-u)})(1-e^{-beta(T-u)}) du   (3.40)

    B(t,T) = (B(0,T)/B(0,t)) exp( -x B_alpha - y B_beta
                                  + 1/2 (V(0,t) - V(0,T) + V(t,T)) )                   (3.42)

    nu(0,S) = sigma^2/(2 alpha^3) (1-e^{-alpha(T-S)})^2 (1-e^{-2 alpha S})
            + eta^2/(2 beta^3)    (1-e^{-beta(T-S)})^2  (1-e^{-2 beta S})
            + 2 rho sigma eta/(alpha beta (alpha+beta))
              (1-e^{-alpha(T-S)})(1-e^{-beta(T-S)})(1-e^{-(alpha+beta)S})              (3.43)

**Both affine coefficients ARE `hw_beta`.** `(1-e^{-alpha tau})/alpha` is the function tranche 10
defined for one-factor Hull-White, called once per factor rather than rewritten - a consistency
guard where they are compared, not cross-source independence (§4's same-formula caveat).

**And `nu(0,S)` at `eta = 0` IS tranche 10's `bond_option_variance`** - the first term of (3.43) is
eq 3.20 verbatim. So the one-factor reduction is a cross-check against **already-verified code**,
which is the strongest form a reduction anchor takes in this tree: not a second transcription, not a
numerical limit, but the same expression evaluated by code that has had its own oracles since T10.

**The source prints `V` twice - as integrals (3.40) and integrated - and that is better than a dual
rendering.** A dual rendering gives two typesettings of one expression; this gives two *different
computations* the source itself supplies, so quadrature of the printed integrand is an independent
check on the printed closed form. (Tranche 66 had the other half of this: there the book declined the
algebra and left it an exercise, so quadrature checked the reader's working.)

**`alpha != beta` is the source's own assumption, so it is refused rather than approximated.** §3.7
states "we assume that alpha != beta" after eq (3.33), and the structure depends on it twice over:
eq (3.43)'s cross term divides by `alpha + beta`, while Exercise 3.10's `y = u/(alpha - beta)`
divides by the *difference*. Equal speeds are outside the model as published, and producing a number
there would be inventing a result the source does not stand behind.

**§4's newest clause applies here, and the regime is NARROWER than tranche 66's.** `V`'s
single-factor terms divide by `alpha^3` after a bracket that is itself `O(alpha^3)` - a `0/0` three
times over, where Vasicek's `(tau - D)/alpha` was once. Measured against quadrature: 3.3e-14 at
`alpha = 0.1`, 2.6e-11 at `0.01`, 1.7e-8 at `1e-3` and **1.9e-6 at `3e-4`**. So the sound regime is
roughly `alpha >= 0.01`, an order narrower than tranche 66's `1e-3`, and the speed -> 0 limits are
asserted as convergence **rates inside that regime** with a companion stating where it degrades -
never banded, because a band at any convenient speed passes and says nothing about the limit.

Provenance:
  stock:  none - the archived predecessor had no two-factor short-rate model
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §3.7 - eqs (3.34)-(3.37) the Gaussian two-factor
          representation, (3.39)/(3.40) the bond and its variance in both printed forms, (3.42) the
          term-structure-fitted bond, (3.43) the bond-option variance; transcribed in both
          renderings and reconciled
  oracle: tests/L3/test_g2pp.py - `V` checked against QUADRATURE of the source's printed integrals
          across the whole correlation range (two printed forms, two computation paths); `nu` at
          `eta = 0` asserted EQUAL to tranche 10's `bond_option_variance` (a reduction to verified
          code) with a T39 companion measuring the genuine two-factor case 26% away; the cross
          term's VALUE transcribed pointwise at four correlations (T43 - presence and sign are not
          magnitude); both affine coefficients cross-checked against `hw_beta` and labelled a
          consistency guard; `B(T,T) = 1` exactly and `B(0,T) = ` the curve exactly (the
          term-structure fit, which is this arc's thesis); the realised correlation of the simulated
          factors pinned against `rho` BEFORE any price band (T17/T19) with the error improving as
          sqrt(N) over six seeds (T21); the `rho = 1` collapse to a single combined factor as a
          requirement oracle; and the speed -> 0 limits asserted as O(alpha) RATES in a stated
          regime with the degradation companion (§4's newest clause, first use)
  crosscheck: quantlib G2 / G2SwaptionEngine - deferred to the optional QuantLib tier
  slice:  tranche 67 - the two-factor Hull-White model (G2++)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.curve import CurveHandle
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.hull_white import hw_beta


@dataclass(frozen=True)
class G2ppParams:
    """The two factors and the correlation between them - five cohesive fields at the §7 ceiling.

    **One object because the correlation is what makes it one.** Two `HullWhiteParams` plus a loose
    `rho` would be three arguments that mean nothing apart: `rho` is the correlation *between these
    two factors*, so pairing one model's factors with another's correlation would be expressible.
    That is §6's compelled-bundling case - vocabulary, not abstraction - kept concrete and frozen.
    """

    alpha: float
    beta: float
    sigma: float
    eta: float
    rho: float

    def __post_init__(self) -> None:
        # Named with the domain word where there is one, so a refusal says what the reader
        # calls the quantity: a `nan` correlation should mention the correlation (§4 - a guard's
        # message is what distinguishes it from every other module's ValueError).
        for name, value in (
            ("alpha", self.alpha), ("beta", self.beta),
            ("sigma", self.sigma), ("eta", self.eta),
            ("the correlation rho", self.rho),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        for name, speed in (("alpha", self.alpha), ("beta", self.beta)):
            if speed <= 0.0:
                raise ValueError(f"mean reversion {name} must be positive, got {speed}")
        for name, vol in (("sigma", self.sigma), ("eta", self.eta)):
            if vol < 0.0:
                raise ValueError(f"volatility {name} cannot be negative, got {vol}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"the correlation must be within [-1, 1], got {self.rho}")
        # The source's own assumption (§3.7, after eq 3.33), and the structure needs it twice: eq
        # (3.43)'s cross term divides by `alpha + beta` while Exercise 3.10's `y = u/(alpha - beta)`
        # divides by the difference. Equal speeds are outside the published model.
        if self.alpha == self.beta:
            raise ValueError(
                f"the two mean reversions must be distinct (the source assumes alpha != beta), "
                f"got {self.alpha} twice"
            )


@dataclass(frozen=True)
class G2ppState:
    """The two factor levels at one time - the model's state vector.

    Compelled bundling (§7): `g2pp_bond` otherwise takes six arguments, and the rule's answer is
    always to group the cohesive ones rather than raise the limit. These are cohesive for the same
    reason the parameters are: `x` and `y` are the two *correlated* factors of one model, the
    simulation produces them as a pair, and a signature taking them loose would let one path's `x`
    be paired with another's `y`. Concrete and frozen, no polymorphism.

    `x(0) = y(0) = 0` by eq (3.35)/(3.36), which is why the term-structure fit is exact at `t = 0`.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y)):
            if not math.isfinite(value):
                raise ValueError(f"the factor {name} must be finite, got {value}")

    @classmethod
    def at_inception(cls) -> G2ppState:
        """`x(0) = y(0) = 0` - the source's initial condition, named so a caller does not restate
        it as a magic pair of zeroes."""
        return cls(x=0.0, y=0.0)


@dataclass(frozen=True)
class G2ppModel:
    """The G2++ parameters, carrying the snapshot they belong to (§3).

    The deterministic `phi` is **not** carried: eq (3.42) needs only the *integral* of `phi`, which
    the source shows is `ln(B(0,t)/B(0,T)) + (V(0,T) - V(0,t))/2` - so the initial curve supplies it
    and a separate `phi` field would be a second spelling of the same fact. That is the term-structure
    fit made structural rather than fitted: the model reproduces today's curve by construction.
    """

    market: MarketSnapshot
    params: G2ppParams


def g2pp_bond_variance(params: G2ppParams, t: float, maturity: float) -> float:
    """`V(t,T)` - eq (3.40), in the integrated form the source also prints.

    Three terms: one per factor, plus the `rho`-weighted cross term. Named surface rather than a
    local inside the bond, so a test can check it against quadrature of the printed integrand and
    sweep the correlation (the F-6.1 rule).

    Numerically the single-factor terms are the fragile part: each divides by its speed **cubed**
    after a bracket that is itself `O(speed^3)`, so precision falls away as a speed tends to zero.
    The module docstring states the regime and the oracle measures it.
    """
    if maturity < t:
        raise ValueError(f"a variance cannot end ({maturity}) before it starts ({t})")
    tau = maturity - t
    if tau == 0.0:
        return 0.0
    alpha, beta, sigma, eta, rho = (
        params.alpha, params.beta, params.sigma, params.eta, params.rho
    )
    first = (
        sigma * sigma / (2.0 * alpha**3)
        * (-3.0 - math.exp(-2.0 * alpha * tau) + 4.0 * math.exp(-alpha * tau) + 2.0 * alpha * tau)
    )
    second = (
        eta * eta / (2.0 * beta**3)
        * (-3.0 - math.exp(-2.0 * beta * tau) + 4.0 * math.exp(-beta * tau) + 2.0 * beta * tau)
    )
    cross = (
        2.0 * rho * sigma * eta / (alpha * beta)
        * (
            tau
            - hw_beta(alpha, t, maturity)
            - hw_beta(beta, t, maturity)
            + hw_beta(alpha + beta, t, maturity)
        )
    )
    return first + second + cross


def g2pp_bond_option_variance(
    params: G2ppParams, option_expiry: float, bond_maturity: float
) -> float:
    """`nu(0,S)` - eq (3.43), the variance of `ln B(S,T)`.

    At `eta = 0` this is **exactly** tranche 10's `bond_option_variance`: the first term is eq 3.20
    verbatim, so the one-factor reduction is a cross-check against already-verified code rather than
    a second transcription. The cross term carries `rho` as the product `rho sigma eta`, which is
    why the oracle pins its **value** and not merely its presence and sign (T43).
    """
    if bond_maturity < option_expiry:
        raise ValueError(
            f"a bond cannot mature ({bond_maturity}) before the option expires ({option_expiry})"
        )
    alpha, beta, sigma, eta, rho = (
        params.alpha, params.beta, params.sigma, params.eta, params.rho
    )
    span = bond_maturity - option_expiry
    first = (
        sigma * sigma / (2.0 * alpha**3)
        * (1.0 - math.exp(-alpha * span)) ** 2
        * (1.0 - math.exp(-2.0 * alpha * option_expiry))
    )
    second = (
        eta * eta / (2.0 * beta**3)
        * (1.0 - math.exp(-beta * span)) ** 2
        * (1.0 - math.exp(-2.0 * beta * option_expiry))
    )
    cross = (
        2.0 * rho * sigma * eta / (alpha * beta * (alpha + beta))
        * (1.0 - math.exp(-alpha * span))
        * (1.0 - math.exp(-beta * span))
        * (1.0 - math.exp(-(alpha + beta) * option_expiry))
    )
    return first + second + cross


def g2pp_bond(
    params: G2ppParams,
    state: G2ppState,
    t: float,
    maturity: float,
    discount: CurveHandle,
) -> float:
    """`B(t,T)` - eq (3.42), the term-structure-fitted two-factor bond.

    The curve ratio `B(0,T)/B(0,t)` carries the integral of `phi`, so the model **reproduces today's
    curve exactly** at `t = 0` with `x = y = 0` rather than approximating it - which is the whole
    content of the deterministic shift, and this arc's thesis: a short-rate model *makes* the curve.

    Exactly `1.0` at `t = T`: the curve ratio is one, both affine coefficients vanish and the three
    variance terms cancel identically, so the oracle can assert equality rather than a tolerance.

    The state is bundled rather than passed loose because the signature would otherwise be six
    arguments (§7) - and the pair is cohesive, not merely co-located: see `G2ppState`.
    """
    if maturity < t:
        raise ValueError(f"a bond cannot mature ({maturity}) before it is observed ({t})")
    if maturity == t:
        return 1.0
    exponent = (
        -state.x * hw_beta(params.alpha, t, maturity)
        - state.y * hw_beta(params.beta, t, maturity)
        + 0.5
        * (
            g2pp_bond_variance(params, 0.0, t)
            - g2pp_bond_variance(params, 0.0, maturity)
            + g2pp_bond_variance(params, t, maturity)
        )
    )
    return discount.df_at(maturity) / discount.df_at(t) * math.exp(exponent)
