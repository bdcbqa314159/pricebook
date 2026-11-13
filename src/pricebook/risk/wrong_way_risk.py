"""Wrong-way risk: equity CVA with correlated counterparty default (L5) - Brockhaus 14.5.1.

T36a's CVA assumed the counterparty's default independent of the option's exposure. That is the
assumption the book calls *questionable*, "due to typical negative correlation between CDS spreads
and stocks" - and dropping it is what this module does.

**The setup.** The stock does **not** default here (that was tranche 35); the *counterparty* does.
`S_T = F e^{s_S X - s_S^2/2}`, and the risky value of a positive payoff is
`E[f(S_T) 1{tau>T}] = E[f(S_T) e^{-H_T}]`. The integrated hazard is approximated by a **truncated
Gaussian** `H_T ~ (a_T + s_H Y)^+`, with `Y` correlated to the equity's `X` at `rho`, and `a_T`
solved from the survival probability by eq 13.4. The call then has four terms, each a bivariate
normal (eq 14.5.1), and `CVA = V_hat - V`.

**The `rho = 0` reduction is exact, and exact for a reason worth stating.** At zero correlation the
first two terms sum to `F Phi(d+) [Phi(-a/s_H) + e^{-a+s_H^2/2} Phi(a/s_H - s_H)]`, and that bracket
**is eq 13.4** - it is `p(T)`. So `V = p(T) V_hat` and `CVA = (1-p(T)) V_hat`, which is tranche 36a
exactly. Two independent things make it bit-identical rather than close: `Phi2` is exact at
`rho = 0` by construction (its correlation integral is empty, which is *why* that form was chosen),
and the `a_T` solve is the inverse of the very identity the reduction collapses through - so a
solver error and a formula error cannot cancel each other.

**The wrong-way DIRECTION is a property of the model, not of the phrase.** `H_T` is increasing in
`Y`, and `Y` correlates with the equity's `X` at `rho`. So `rho > 0` makes default more likely
exactly when the stock is high - which is when a long call's exposure is largest, hence *wrong-way*
for that exposure and CVA-increasing. For a long put the same `rho` would be right-way. The
convention is stated here and asserted in the oracle so no later reader re-imports "wrong-way means
bigger" as a fact about the sign of `rho` (tranche 35 shipped a backwards economics test from
exactly that kind of intuition).

**The truncated Gaussian is an approximation, and it is named as one.** `H_T ~ (a_T + s_H Y)^+` is
exact only if the integrated hazard really is a truncated Gaussian; for a general positive hazard it
is a modelling choice whose error the correlated Monte-Carlo measures rather than assumes.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 14.5.1 (the counterparty-default application) on
          ch 13.1.2's apparatus (the truncated-Gaussian hazard and eq 13.4), read in `-raw` and
          `-layout` and reconciled. The cumulative-normal glyph is dropped by `pdftotext` in BOTH
          renderings of 13.4 and 14.5.1, as it was for eq 11.1 at tranche 34; the identification is
          settled by the `rho -> 0` reduction, which fails against any other function
  oracle: tests/L5/test_wrong_way_risk.py (PRE-REGISTERED) - the eq-13.4 solve inverting its own
          forward formula and matching the book's own small-`a_T` approximation (a derivation-step
          check); the **exact** `rho = 0` reduction to tranche 36a; the wrong-way direction DERIVED
          from the model and the CVA monotone in `rho`; the requirement oracles; and the consumer's
          assertion that every `Phi2` call lands inside the primitive's stated validity regime
  crosscheck: quantlib (wrong-way CVA) - none implements this; deferred/none
  slice:  tranche 36b - wrong-way risk, the arc capstone's correlated seam
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.bivariate_normal import bivariate_normal_cdf
from pricebook.foundation.distributions import norm_cdf
from pricebook.models.black import black_price
from pricebook.products.equity_option import EquityOptionKind

#: `Phi2`'s stated correlation regime. Refused rather than approached: between 0.99 and 1 the
#: primitive's fixed rule cannot resolve the peaked integrand, and `|rho| = 1` is exact but is a
#: different branch. A consumer that silently entered that gap would be trusting a number the
#: primitive's own docstring says not to.
_MAX_CORRELATION = 0.99


def survival_from_truncation(level: float, hazard_vol: float) -> float:
    """Eq 13.4 forward: `p = Phi(-a/s_H) + e^{-a + s_H^2/2} Phi(a/s_H - s_H)`.

    Named surface rather than a step inside the solver, because the oracle inverts it against the
    solver and checks it against the book's own small-`a` approximation - two uses that both need
    to reach it (§4).
    """
    if hazard_vol <= 0.0:
        raise ValueError(f"the hazard volatility must be positive, got {hazard_vol}")
    standardised = level / hazard_vol
    return norm_cdf(-standardised) + math.exp(
        -level + 0.5 * hazard_vol * hazard_vol
    ) * norm_cdf(standardised - hazard_vol)


def solve_truncation_level(survival: float, hazard_vol: float) -> float:
    """Invert eq 13.4 for `a_T` - bisection, because the forward formula is monotone decreasing.

    Bisection rather than Newton: the function is monotone and bracketed, so bisection cannot fail
    to converge, and the derivative would be a second transcription with its own chance of being
    wrong. The cost is irrelevant beside a single option price.
    """
    if not 0.0 < survival < 1.0:
        raise ValueError(f"the survival probability must lie in (0, 1), got {survival}")
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if survival_from_truncation(middle, hazard_vol) > survival:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True)
class GaussianHazard:
    """The 13.1.2 hazard: a survival probability, a hazard volatility, and the equity correlation.

    Three fields, and the truncation level is **derived** rather than stored - it is a function of
    the other two, and carrying it would let a caller construct a pair that disagrees.
    """

    survival: float
    hazard_vol: float
    correlation: float

    def __post_init__(self) -> None:
        if not 0.0 < self.survival <= 1.0:
            raise ValueError(f"the survival probability must lie in (0, 1], got {self.survival}")
        if self.hazard_vol <= 0.0:
            raise ValueError(f"the hazard volatility must be positive, got {self.hazard_vol}")
        if abs(self.correlation) > _MAX_CORRELATION:
            raise ValueError(
                f"the correlation must satisfy |rho| <= {_MAX_CORRELATION}, got "
                f"{self.correlation}: the bivariate normal's fixed rule cannot resolve the peaked "
                f"integrand between there and 1"
            )

    @property
    def truncation_level(self) -> float:
        """`a_T`, solved from the survival probability by eq 13.4."""
        return solve_truncation_level(min(self.survival, 1.0 - 1e-15), self.hazard_vol)

    def phi2_arguments(
        self, forward: float, strike: float, equity_vol: float
    ) -> tuple[tuple[float, float, float], ...]:
        """The four `(h, k, rho)` triples this formula hands the primitive.

        Exposed so the oracle can assert every one of them lands inside `Phi2`'s stated validity
        regime, rather than trusting the primitive silently - the consumer obligation its docstring
        names (§4: a green is evidence only about what the band can resolve).
        """
        level, hazard_vol, rho = self.truncation_level, self.hazard_vol, self.correlation
        standardised = level / hazard_vol
        d_plus = (math.log(forward / strike) + 0.5 * equity_vol * equity_vol) / equity_vol
        d_minus = d_plus - equity_vol
        return (
            (d_plus, -standardised - rho * equity_vol, -rho),
            (d_plus - rho * hazard_vol, standardised + rho * equity_vol - hazard_vol, rho),
            (d_minus, -standardised, -rho),
            (d_minus - rho * hazard_vol, standardised - hazard_vol, rho),
        )


def risky_call_value(
    forward: float, strike: float, equity_vol: float, hazard: GaussianHazard
) -> float:
    """`E[(S_T - K)^+ 1{tau>T}]` - eq 14.5.1's four terms, undiscounted."""
    level, hazard_vol, rho = hazard.truncation_level, hazard.hazard_vol, hazard.correlation
    arguments = hazard.phi2_arguments(forward, strike, equity_vol)

    first = forward * bivariate_normal_cdf(*arguments[0])
    second = (
        forward
        * math.exp(-level - rho * equity_vol * hazard_vol + 0.5 * hazard_vol * hazard_vol)
        * bivariate_normal_cdf(*arguments[1])
    )
    third = strike * bivariate_normal_cdf(*arguments[2])
    fourth = (
        strike
        * math.exp(-level + 0.5 * hazard_vol * hazard_vol)
        * bivariate_normal_cdf(*arguments[3])
    )
    return first + second - third - fourth


def wrong_way_cva(
    forward: float, strike: float, equity_vol: float, hazard: GaussianHazard
) -> float:
    """`CVA = V_hat - V` - the default-free value less the risky one, undiscounted."""
    default_free = black_price(forward, strike, equity_vol * equity_vol, EquityOptionKind.CALL)
    return default_free - risky_call_value(forward, strike, equity_vol, hazard)
