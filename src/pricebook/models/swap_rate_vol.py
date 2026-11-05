"""The forward swap rate, its annuity, and Rebonato's swaption-vol approximation (L3).

MCINERNEY-ZASTAWNIAK §5.9-5.12. The swap rate and annuity are exact (eqs 5.31/5.32); the volatility
is an **approximation**, and the distinction is the whole point of this module.

    A_{0,n}(t) = Σ_k τ_k B(t,T_k)                                                       (5.32)
    S_{0,n}(t) = (B(t,T_0) − B(t,T_n)) / A_{0,n}(t)                                     (5.31)
    w_i(t)     = τ_i B(t,T_i) / A_{0,n}(t)
    v²_{0,n}   = 1/(T_0−t) Σ_{i,j} [w_i w_j F_i F_j / S²] ∫_t^{T_0} σ_i σ_j ρ_ij ds      (5.40)

**Eq 5.40 is Rebonato's formula, and it is an approximation for a reason worth naming.** §5.11 proves
that under the LMM the forward swap rate is *not* lognormal — `σ_{0,n}` is random — so a European
swaption has no exact LMM closed form and Black's swaption formula cannot be exactly right here. 5.40
recovers a deterministic volatility by freezing two sets of `P_A`-martingales (`w_iF_i` and `S`) at
their time-`t` values. Everything that consumes it owes a **bounded** check with the error measured,
never a tolerance tuned until it passes (the MM-I freezing lesson).

**The one-period case is exact, and that is what anchors the swaption engine.** For `count = 1` the
annuity is the single `τ_i B(t,T_{i+1})`, so `w = 1`; and `B(t,T_i) − B(t,T_{i+1}) = τ_i F_i
B(t,T_{i+1})` identically, so `S = F_i`. Then `Z = wF/S = 1` and `v² = σ_i² ρ_ii = σ_i²`. So a
one-period swaption **is** a caplet, with no freezing error at all — the exact reduction §4 prefers,
and its mandatory companion is the multi-period case where the correlation actually bites.

**Constant volatilities, which is why the time integral disappears.** Our `σ_i` do not vary with `s`,
so `∫_t^{T_0} σ_iσ_jρ_ij ds = σ_iσ_jρ_ij(T_0−t)` and the `1/(T_0−t)` prefactor cancels exactly.
Jäckel §12.4 eq 12.11 gives a time-dependent parametrisation and eq 12.13 its covariance integral in
closed form; that arrives with the unit that calibrates a term structure of volatility, not here.

**What the sources do NOT agree on, recorded because it changes how this is verified.** T17's spec
expected McInerney §5.12 and Jäckel §12.4 to render *one* formula, giving cross-source transcription
independence. They do not. Both write the swap-rate variance as the quadratic form `Z·C^FRA·Zᵗ`, but
Jäckel's `Z_k = (∂SR/∂f_k)(f_k/SR)` (eqs 12.18/12.21) is the **exact** derivative, while McInerney's
5.40 uses the **frozen** weight `w_kF_k/S` — which is 12.21's first term with the weight-derivative
terms dropped. McInerney's own Remark 5.13 says 5.40 can be improved by not freezing, so they are two
approximations of different order, not two readings of one. Measured here, they differ by ~0.5% in
vol. The oracle keeps Jäckel's form as an independent **companion** with that gap asserted to be
small *and non-zero* — a weaker claim than cross-source transcription, and the true one.

Provenance:
  stock:  — built fresh; the quarry's swaption analytics carry a conventions/calendar apparatus this
          tranche has no consumer for
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §5.9 eqs 5.31/5.32 (the forward swap rate and
          annuity), §5.10 (Black's swaption formula), §5.11 (why the LMM swap rate is not lognormal)
          and §5.12 eq 5.40 (Rebonato's approximation), read in `-raw` and `-layout`, which agree;
          JAECKEL-2002 §12.4 eqs 12.14-12.21 as the independent companion form, NOT as a second
          rendering of 5.40 — see the note above
  oracle: tests/L3/test_swap_rate_vol.py — the one-period exact reduction (`S = F_i`, `w = 1`,
          `v = σ_i` to 1e-15); the requirement identities `Σ w_iF_i = S` and `Σ w_iF_i/S = 1`; the
          pointwise transcription of 5.40 over six spans with a non-uniform `ρ`; the perfect-
          correlation upper bound; and Jäckel's exact-Z companion with the gap measured
  crosscheck: quantlib SwaptionVolatilityMatrix / LfmSwaptionEngine (the Rebonato approximation) —
          deferred to the optional QuantLib tier
  slice:  tranche 17 — the European swaption under the LMM
"""

from __future__ import annotations

import math

from pricebook.models.libor_market_model import LiborMarketModel, LmmTenor


def _check_span(tenor: LmmTenor, first: int, count: int) -> None:
    """A span is `count` consecutive forwards starting at `first`, and it must fit the tenor.

    Refused rather than clamped: a span running off the end would silently price a shorter swap,
    which is a plausible wrong number — the failure mode §3.4 exists to prevent."""
    if first < 0 or count < 1 or first + count > len(tenor):
        raise ValueError(
            f"the swap span [{first}, {first + count}) must lie within the tenor's "
            f"{len(tenor)} forwards"
        )


def tenor_discount_factors(tenor: LmmTenor) -> tuple[float, ...]:
    """`B(0,T_0) … B(0,T_n)`, with `B(0,T_0) = 1` — the tenor discounting itself.

    `n + 1` factors for `n` forwards, since `B(0,T_{i+1}) = B(0,T_i)/(1 + f_i τ_i)` is the definition
    of a spanning forward. The tenor starts at the valuation date, so `B(0,T_0) = 1` is not a
    normalisation choice but the first date being today; a model carrying its own `MarketSnapshot`
    would read the curve instead, and that arrives when a swaption is priced off a real curve."""
    factors = [1.0]
    for forward, accrual in zip(tenor.forwards, tenor.accruals, strict=True):
        factors.append(factors[-1] / (1.0 + forward * accrual))
    return tuple(factors)


def swap_annuity(tenor: LmmTenor, first: int, count: int) -> float:
    """`A = Σ_m τ_{first+m} B(0,T_{first+m+1})` — eq 5.32 over the span's payment dates."""
    _check_span(tenor, first, count)
    bonds = tenor_discount_factors(tenor)
    return math.fsum(tenor.accruals[first + m] * bonds[first + m + 1] for m in range(count))


def forward_swap_rate(tenor: LmmTenor, first: int, count: int) -> float:
    """`S = (B(0,T_first) − B(0,T_{first+count})) / A` — eq 5.31."""
    _check_span(tenor, first, count)
    bonds = tenor_discount_factors(tenor)
    return (bonds[first] - bonds[first + count]) / swap_annuity(tenor, first, count)


def swap_rate_weights(tenor: LmmTenor, first: int, count: int) -> tuple[float, ...]:
    """`w_m = τ_{first+m} B(0,T_{first+m+1}) / A` — the weights of eq 1.16 / §5.12.

    They satisfy `Σ_m w_m F_{first+m} = S` **exactly**, because `τ_i F_i B(T_{i+1}) = B(T_i) −
    B(T_{i+1})` telescopes to 5.31's numerator. That identity is the sharpest available check on the
    weights, and the weights are where the swaption's magnitude risk lives."""
    _check_span(tenor, first, count)
    bonds = tenor_discount_factors(tenor)
    annuity = swap_annuity(tenor, first, count)
    return tuple(
        tenor.accruals[first + m] * bonds[first + m + 1] / annuity for m in range(count)
    )


def rebonato_swaption_vol(model: LiborMarketModel, first: int, count: int) -> float:
    """`v_{0,n}` — eq 5.40 with constant volatilities, so the time integral cancels.

    `v² = Σ_{m,l} Z_m Z_l σ_m σ_l ρ_ml` with `Z_m = w_m F_m / S`, every index offset by the span's
    `first`. Returned as a volatility rather than a variance because that is what Black's swaption
    formula (§5.10) takes, and because `v ≥ 0` is then checked once here."""
    tenor = model.tenor
    _check_span(tenor, first, count)
    weights = swap_rate_weights(tenor, first, count)
    swap_rate = forward_swap_rate(tenor, first, count)
    sigma, rho = tenor.volatilities, model.correlation.matrix
    z = tuple(weights[m] * tenor.forwards[first + m] / swap_rate for m in range(count))

    variance = math.fsum(
        z[m] * z[l] * sigma[first + m] * sigma[first + l] * rho[first + m][first + l]
        for m in range(count)
        for l in range(count)
    )
    if variance < 0.0:
        # `v² ≥ 0` is a requirement, and the only way to break it is a correlation matrix that is
        # not positive semidefinite. Refused here rather than returned as a NaN from `sqrt`.
        raise ValueError(
            f"the swap-rate variance is negative ({variance}); the correlation matrix is not "
            f"positive semidefinite"
        )
    return math.sqrt(variance)
