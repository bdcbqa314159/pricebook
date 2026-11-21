"""Multi-factor Gaussian HJM (L3). `MCINERNEY-ZASTAWNIAK` §4.4.

Tranche 69 with `n` driving factors. Transcribed from both renderings and reconciled (§4):

    df(t,T) = alpha(t,T) dt + sum_i sigma_i(t,T) dW_i(t)                           (4.10)
    alpha(t,T) = sum_i sigma_i(t,T) integral_t^T sigma_i(t,u) du        (Theorem 4.6, eq 4.11)

**Two things the source settles that the tranche spec had differently, and the second removes an
oracle rather than adding one.** The spec wrote per-factor drifts `alpha_k` and an explicit factor
correlation `rho_kj`. §4.4 has neither:

1. There is **one** drift. The forward rate has a single drift and `n` diffusion terms, not `n`
   drifts - which (4.10) makes plain and (4.11) confirms.
2. The Brownian motions are **independent** - the section opens "the forward-rate process is driven
   by `n` *independent* Brownian motions" - so no `rho_kj` exists to carry.

The second is worth more than a notational note: the spec expected the T17/T43 **correlation
discipline** to return here, as it did at G2++, and it does not, because **there is no correlation
parameter to pin.** Correlation between forward rates of *different maturities* is an **output** of
the volatility vector rather than an input to it, which is the framework's actual content. Where
G2++ had a `rho` and needed two independent instruments to pin it, this model has a structure to
check instead - and the structure is checkable exactly.

**With independent factors every quantity is a plain SUM of tranche 69's**, which is why this module
is short rather than thin:

    alpha(t,T) = sum_i (T69's drift for sigma_i)
    nu(0,S)    = sum_i (T69's variance for sigma_i)     -- independence removes the cross terms

So it **calls tranche 69's functions and sums them**, and the `N = 1` reduction is **bit-identical by
construction**: a one-element sum *is* its summand, so the one-factor model returns the floats
tranche 69 returns, bit for bit. That is the arc's reduce-to-already-verified-code anchor in its
strongest possible form - identity rather than agreement - and it is why there is no second
transcription of the per-factor integrals here. Writing them again to "generalise" them would have
created the one thing the reduction could not then detect.

**There is no `MultiFactorHjmModel`, and that was the tranche's §6 ruling.** This module holds the
summing *functions*; the model is tranche 69's `GaussianHjmModel`, generalised to a tuple of vols.
A one-factor HJM is not a model that resembles the `n`-factor one - it **is** the `n`-factor one at
`n = 1`. What would normally argue for two types is that each one's reduction to the other is a
self-verifying oracle, and here that reduction is **bit-identical by construction**, so it is a
tautology rather than a cross-check: two types would have been separated to preserve nothing.
*(Code raised it as a finding with the count - one construction site, one field read, functions
untouched; Cowork ruled generalise.)*

Provenance:
  stock:  none - the archived predecessor had no HJM
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §4.4 - eq (4.10) the n-factor forward-rate SDE
          with INDEPENDENT Brownian motions, and Theorem 4.6 eq (4.11) the no-arbitrage drift as a
          sum over factors; transcribed in both renderings and reconciled
  oracle: tests/L3/test_multifactor_hjm.py - the factor-sum drift asserted against the integrated
          condition per factor (two forms, two paths) and transcribed pointwise besides (T12(b)),
          with each of three factors measured to matter so a dropped term cannot hide; the N=1
          reduction asserted BIT-IDENTICAL to tranche 69 (identity, not tolerance) and chained
          through to tranche 10's `bond_option_variance` at the Hull-White vol - three tranches, one
          number; a MANDATORY three-factor companion measured 43% away (T39); the variance shown
          equal to a SINGLE integral of the summed squares (a different computation path) and
          ORDER-INDEPENDENT in the factors, which a spurious cross term would break; the
          empty-factor-set refusal; and the inverted-interval guards pinned by their own messages
          (the tranche-69 reachability lesson carried forward rather than rediscovered)
  crosscheck: quantlib's multi-factor Gaussian/HJM engines - deferred to the optional QuantLib tier
  slice:  tranche 70 - multi-factor HJM
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pricebook.models.gaussian_hjm import (
    ForwardVolatility,
    hjm_bond_option_variance,
    hjm_drift,
)


def _checked_factors(volatilities: Sequence[ForwardVolatility]) -> None:
    """Refuse an empty factor set.

    A zero-factor HJM is a deterministic forward curve, not a model - and both the drift and the
    variance would come back as `0.0`, which is a plausible wrong number rather than a failure. That
    is the §3.4 distinction: a precondition surfaces as a refusal, never as a quiet zero.
    """
    if not volatilities:
        raise ValueError("a multi-factor HJM quantity needs at least one factor")


def multifactor_drift(
    volatilities: Sequence[ForwardVolatility], t: float, maturity: float
) -> float:
    """`alpha(t,T) = sum_i sigma_i(t,T) integral_t^T sigma_i(t,u) du` - eq (4.11).

    One drift, summed over factors - not one drift per factor. Forced by no-arbitrage, so the oracle
    asserts it as an identity holding for every set of vols rather than against a printed number.

    Each term is tranche 69's `hjm_drift` for that factor, called rather than rewritten: that is
    what makes the `N = 1` case bit-identical to tranche 69 and what a second transcription would
    have destroyed.
    """
    _checked_factors(volatilities)
    if maturity < t:
        raise ValueError(
            f"a multi-factor drift cannot be asked for a maturity ({maturity}) before the "
            f"observation time ({t})"
        )
    return math.fsum(hjm_drift(volatility, t, maturity) for volatility in volatilities)


def multifactor_bond_option_variance(
    volatilities: Sequence[ForwardVolatility],
    option_expiry: float,
    bond_maturity: float,
) -> float:
    """`nu(0,S) = sum_i integral_0^S ( integral_S^T sigma_i(s,u) du )^2 ds`.

    A sum of squares with **no cross terms**, and the independence of the Brownian motions is
    exactly why: a correlated model (G2++, tranche 67) carries `rho_ij` cross terms in the same
    place. So the variance is symmetric in the factors, which the oracle asserts by permuting them -
    a property a spurious asymmetric cross term would break.

    Each term is tranche 69's variance for that factor, so `N = 1` is bit-identical to it.

    The inverted-interval refusal names **this** operation, which is not cosmetic: tranche 69's
    per-factor function refuses the same interval with its own message, so a guard here phrased
    identically would be bypassable without any test noticing - the mutation sweep found exactly
    that, because the drift's message happened to differ and this one happened to coincide. A guard
    whose removal still raises is pinned only by what it says (the T44b shape).
    """
    _checked_factors(volatilities)
    if bond_maturity < option_expiry:
        raise ValueError(
            f"a multi-factor bond variance cannot be asked for a maturity ({bond_maturity}) "
            f"before the option expires ({option_expiry})"
        )
    return math.fsum(
        hjm_bond_option_variance(volatility, option_expiry, bond_maturity)
        for volatility in volatilities
    )
