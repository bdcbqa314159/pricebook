"""The non-central chi-squared distribution, CDF and survival (L0).

`Q(x; nu, lambda)` and its complement, for `nu > 0` degrees of freedom and noncentrality
`lambda >= 0`. Finance-free (§2): this is a statement about a distribution, and the finance is what
tranche 72's CEV caplet does with it.

**Two functions, not one, and the reason is measured rather than stylistic.** Cox's CEV call is
`F·[1 − Q(2w; ·)] − K·Q(2z; ·)`, so the *upper* tail is load-bearing. Forming it as `1 − cdf` repeats
`norm_cdf`'s OI-3 defect exactly (tranche 20c): measured against an independent high-precision
reference, `1 − cdf` carries **1.2% relative error at `x = 100`** and returns **exactly 0.0 beyond
`x ≈ 200`**, where the true survival is 8.7e-32 — not a small probability but a lost one. A deep
out-of-the-money caplet is precisely where a CEV pricer is asked for a number. `noncentral_chisquare_sf`
sums the **upper** incomplete gammas directly, so it is a sum of positive terms with no cancellation
anywhere, and holds **relative** accuracy to 1.1e-14 down to 3e-148.

**The representation.** A non-central chi-squared is a Poisson mixture of central ones — if
`X ~ chi2(nu, lambda)` then `X ~ chi2(nu + 2J)` with `J ~ Poisson(lambda/2)`. That is the
distribution's *definition* unrolled, not a cited algorithm, which matters for provenance: the
central chi-squared CDF is the regularised lower incomplete gamma `P(nu/2, x/2)`, so

    Q(x; nu, lambda) = sum_j  e^(-lambda/2) (lambda/2)^j / j!  ·  P(nu/2 + j, x/2)

and the survival is the same sum over `1 - P`, i.e. the upper regularised gamma. The sum is taken
**outward from the Poisson mode** rather than from `j = 0`: at `lambda = 1500` the weight at `j = 0`
underflows to zero while the mode's is O(1), so starting at zero would spend hundreds of terms
accumulating nothing and then lose the leading digits.

**The incomplete gamma is private, and stays private.** It has exactly one consumer (this module), so
§6's rule of two has not fired. Each branch is chosen so the quantity is computed **directly** rather
than as a complement: the series converges for `x < a + 1` and gives `P`, the continued fraction
converges for `x >= a + 1` and gives `Q`. Taking the complement only in the branch where the other
quantity is far from 1 is what keeps both tails accurate — the same structural choice as this
module's two public functions, one level down.

Provenance:
  stock:  none — the migration tree carried no non-central chi-squared
  source: BROCKHAUS-2016 (docs/references.md) §6.1 p.79 — the distribution's role in the CEV
          closed form, its `V^(delta,kappa)` notation and the `Q(.; delta, kappa)` convention. The
          Poisson-mixture representation and the incomplete-gamma identity for the central case are
          the definitions unrolled, not a transcription; the numerical branch choice (series below
          `a + 1`, continued fraction above) is the standard one and is justified here by the
          accuracy it buys, measured in the oracle
  oracle: tests/L0/test_noncentral_chisquare.py — both functions pointwise against
          `scipy.stats.ncx2` across the centre, the deep upper tail and HIGH noncentrality, asserted
          RELATIVELY (tranche 20a); the `lambda = 0` reduction to `scipy.stats.chi2` with a
          non-vacuity companion; `cdf + sf = 1`; monotonicity; and the measurement that `1 - cdf` is
          destroyed where `sf` is not, so the two-function design is proved necessary
  crosscheck: scipy.stats.ncx2 — already the oracle; scipy is a test dependency and deliberately
          NOT a library one (the tranche-20a arrangement)
  slice:  tranche 72 — the CEV smile model
"""

from __future__ import annotations

import math

#: Guard against a zero denominator in the continued fraction's Lentz recurrence. Far below any
#: representable intermediate the recursion legitimately produces, so reaching it means a term
#: underflowed rather than that the fraction is small.
_TINY = 1.0e-300

#: Relative size at which a term stops moving the sum. Both the gamma series and the Poisson sum
#: are sums of positive terms, so this is a genuine convergence test and not a hopeful one.
_EPSILON = 1.0e-17

#: Hard iteration ceilings. The series and the fraction converge in tens of terms over the range
#: this module is used on; these exist so a pathological argument fails loudly rather than hanging.
_MAX_TERMS = 100_000
_MAX_POISSON_TERMS = 1_000_000


def _log_gamma_factor(a: float, x: float) -> float:
    """`log(e^-x x^a / Gamma(a))` — the prefactor both branches share, kept in logs.

    Formed in logs because `x^a` overflows for the large `a` a high noncentrality produces while
    the whole factor stays representable."""
    return -x + a * math.log(x) - math.lgamma(a)


def _lower_series(a: float, x: float) -> float:
    """`P(a, x)` by its series, which converges for `x < a + 1`."""
    term = 1.0 / a
    total = term
    for n in range(1, _MAX_TERMS):
        term *= x / (a + n)
        total += term
        if term <= total * _EPSILON:
            return total * math.exp(_log_gamma_factor(a, x))
    raise ArithmeticError(f"the incomplete-gamma series did not converge at a={a}, x={x}")


def _upper_fraction(a: float, x: float) -> float:
    """`Q(a, x)` by the modified Lentz continued fraction, which converges for `x >= a + 1`."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_TERMS):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = _TINY if abs(d) < _TINY else d
        c = b + an / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= _EPSILON:
            return math.exp(_log_gamma_factor(a, x)) * h
    raise ArithmeticError(f"the incomplete-gamma fraction did not converge at a={a}, x={x}")


def _gamma_p(a: float, x: float) -> float:
    """The regularised LOWER incomplete gamma `P(a, x)`, computed directly where it is accurate."""
    if x <= 0.0:
        return 0.0
    return _lower_series(a, x) if x < a + 1.0 else 1.0 - _upper_fraction(a, x)


def _gamma_q(a: float, x: float) -> float:
    """The regularised UPPER incomplete gamma `Q(a, x)`, computed directly where it is accurate."""
    if x <= 0.0:
        return 1.0
    return 1.0 - _lower_series(a, x) if x < a + 1.0 else _upper_fraction(a, x)


def _check(x: float, dof: float, noncentrality: float) -> None:
    if not math.isfinite(x):
        raise ValueError(f"the argument must be finite, got {x}")
    if not math.isfinite(dof) or dof <= 0.0:
        raise ValueError(f"the degrees of freedom must be positive and finite, got {dof}")
    if not math.isfinite(noncentrality) or noncentrality < 0.0:
        raise ValueError(
            f"the noncentrality must be non-negative and finite, got {noncentrality}"
        )


def _poisson_mixture(x: float, dof: float, noncentrality: float, central) -> float:
    """`sum_j Poisson(j; lambda/2) * central(nu/2 + j, x/2)`, summed OUTWARD from the mode.

    `central` is one of the two private incomplete gammas, so one walk serves both public functions:
    the weights and the stopping rule are identical and only the central quantity differs. Summing
    from the mode keeps the largest weight first, which is what makes a `lambda` of 1500 tractable —
    from `j = 0` the leading weights underflow and the sum loses its significant digits."""
    half = 0.5 * noncentrality
    mode = int(half)
    log_weight = -half + mode * math.log(half) - math.lgamma(mode + 1.0)

    total = 0.0
    weight = math.exp(log_weight)
    index = mode
    for _ in range(_MAX_POISSON_TERMS):
        term = weight * central(0.5 * dof + index, 0.5 * x)
        total += term
        if index > mode and term <= total * 1.0e-18 and index > half:
            break
        index += 1
        weight *= half / index
    else:
        raise ArithmeticError(f"the Poisson mixture did not converge at nc={noncentrality}")

    weight = math.exp(log_weight)
    index = mode
    while index > 0:
        weight *= index / half
        index -= 1
        term = weight * central(0.5 * dof + index, 0.5 * x)
        total += term
        if term <= total * 1.0e-18:
            break
    return total


def noncentral_chisquare_cdf(x: float, dof: float, noncentrality: float) -> float:
    """`P(X <= x)` for `X ~ chi2(dof, noncentrality)`.

    Exact at and below zero rather than approached: the distribution has no mass there.

    **The `x <= 0` guard is a COST guard, not a correctness one, and that is proved rather than
    asserted.** Relaxing it to `x < 0` survives the whole suite, and it is a genuine equivalent
    mutant (§4) — not a hole: at `x = 0` every term of the mixture is `_gamma_p(a, 0.0)`, which is
    0.0, so the sum is 0.0 and the outputs are **bit-identical over 25 (dof, noncentrality) pairs**
    including a noncentrality of 1500. What the guard buys is skipping a Poisson walk that would
    add up a few thousand zeroes."""
    _check(x, dof, noncentrality)
    if x <= 0.0:
        return 0.0
    if noncentrality == 0.0:
        return _gamma_p(0.5 * dof, 0.5 * x)
    return min(_poisson_mixture(x, dof, noncentrality, _gamma_p), 1.0)


def noncentral_chisquare_sf(x: float, dof: float, noncentrality: float) -> float:
    """`P(X > x)` for `X ~ chi2(dof, noncentrality)` — the survival, NOT `1 - cdf`.

    See the module docstring: the complement loses the upper tail completely, and this is where the
    CEV call's first term lives."""
    _check(x, dof, noncentrality)
    if x <= 0.0:
        return 1.0
    if noncentrality == 0.0:
        return _gamma_q(0.5 * dof, 0.5 * x)
    return min(_poisson_mixture(x, dof, noncentrality, _gamma_q), 1.0)
