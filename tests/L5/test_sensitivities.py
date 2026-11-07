"""Greeks by simulation (L5) — the first module in the risk layer. JAECKEL-2002 ch 11.

Three methods, all on the same paths, all anchored on Black's analytic Greeks:

* **finite differencing with path recycling (§11.3)** — bump `F₀`, reprice on the *same* draws,
  difference. Sharing the draws is what makes the difference low-variance; it is common random
  numbers, and tranche 19 established that CRN is variance-*reduced*, not noise-free, so the oracle
  bands it by the paired difference's own error rather than treating it as exact.
* **pathwise differentiation (§11.5)** — differentiate the payoff along the path. Valid only for a
  Lipschitz payoff: *"for Lipschitz-continuous payoff functions ... the order of differentiation and
  expectation can be interchanged."*
* **the likelihood ratio method (§11.6)** — differentiate the *density* instead, giving a weight
  `ω` applied to the unchanged payoff. Needs no continuity at all, which is the whole reason it
  exists.

Jäckel's printed weights, for the single-horizon case this module covers:

    ω_Δ = z / (S₀ σ √Δt₁)                            (11.30)
    ω_Γ = (z² − z σ √Δt₁ − 1) / (S₀² σ² Δt₁)         (11.31)
    ω_V = (z² − 1)/σ − z √T                          (11.32, appendix §11.9.1)

**Adapted from spot to forward, which is a convention change and is stated.** Jäckel writes
`S_T = S₀·exp((r − d − ½σ²)T + σ√T z)`; our Black is a *forward* Black, undiscounted, so the forward
is driftless under its own measure: `F_T = F₀·exp(−½σ²T + σ√T z)`. The weights carry over with
`S₀ → F₀` and `r − d → 0`, and with a single horizon `Δt₁ = T`.

**The anchor did not exist before this tranche.** The spec says *"we have Black's analytic Greeks
(T9)"*; we did not — T9 built the implied-vol surface. They are built in `models/black.py` here and
oracled against finite differences of the price verified since tranche 5, so the chain is
`black_price` (verified) → its finite difference → the analytic Greek → the Monte Carlo Greek, each
link independently checkable.
"""

import math

import pytest

from pricebook.foundation.inflation import CapFloorKind
from pricebook.models.black import (
    black_delta,
    black_gamma,
    black_price,
    black_vega,
    digital_delta,
)
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.risk.sensitivities import (
    GreekMethod,
    black_mc_delta,
    black_mc_gamma,
    black_mc_vega,
    digital_mc_delta,
)

FORWARD, STRIKE = 0.04, 0.04
SIGMA, YEARS = 0.20, 1.0
VARIANCE = SIGMA**2 * YEARS
#: 200,000 paths: a Greek is a *derivative* of a price, so its standard error is larger than the
#: price's at the same count, and the anchor has to be resolvable above it. Stated because the count
#: is doing work rather than being a round number.
CONFIG = MonteCarloConfig(paths=200_000, seed=22_000)


def _analytic(greek: str) -> float:
    if greek == "delta":
        return black_delta(FORWARD, STRIKE, VARIANCE, CapFloorKind.CAP)
    if greek == "gamma":
        return black_gamma(FORWARD, STRIKE, VARIANCE, CapFloorKind.CAP)
    # Vega: our Black's natural vega is per unit of root-variance; the chain rule to per-unit-sigma
    # belongs here, where `YEARS` is known.
    return black_vega(FORWARD, STRIKE, VARIANCE, CapFloorKind.CAP) * math.sqrt(YEARS)


# ============================================================================================
# THE ANCHOR — every method converges to the closed form, within the GREEK's own standard error.
# ============================================================================================


@pytest.mark.parametrize(
    "method", [GreekMethod.FINITE_DIFFERENCE, GreekMethod.PATHWISE, GreekMethod.LIKELIHOOD_RATIO]
)
def test_T22_every_method_reproduces_BLACKS_analytic_DELTA(method):
    """The anchor, and invariant 7 extended to a Greek: a sensitivity is an estimate, so it carries
    its own standard error and the comparison is banded by *that* rather than by a chosen number."""
    estimate = black_mc_delta(FORWARD, STRIKE, VARIANCE, method, CONFIG)
    assert estimate.standard_error > 0.0
    assert abs(estimate.mean - _analytic("delta")) < 4.0 * estimate.standard_error


@pytest.mark.parametrize("method", [GreekMethod.FINITE_DIFFERENCE, GreekMethod.LIKELIHOOD_RATIO])
def test_T22_the_methods_that_can_do_GAMMA_reproduce_the_analytic_one(method):
    """Gamma is restricted, and the restriction is the source's.

    *"For the calculation of Gamma, we still have to implement a finite differencing scheme for two
    individually calculated Deltas"* — pathwise gives a delta per path, not a second derivative, so
    it is **not** offered for gamma. A method list that silently included it would be inventing a
    capability the method does not have."""
    estimate = black_mc_gamma(FORWARD, STRIKE, VARIANCE, method, CONFIG)
    assert abs(estimate.mean - _analytic("gamma")) < 4.0 * estimate.standard_error


@pytest.mark.parametrize(
    "method", [GreekMethod.FINITE_DIFFERENCE, GreekMethod.PATHWISE, GreekMethod.LIKELIHOOD_RATIO]
)
def test_T22_every_method_reproduces_BLACKS_analytic_VEGA(method):
    estimate = black_mc_vega(FORWARD, STRIKE, VARIANCE, method, CONFIG, YEARS)
    assert abs(estimate.mean - _analytic("vega")) < 4.0 * estimate.standard_error


def test_T22_the_three_methods_AGREE_with_each_other_on_a_smooth_payoff():
    """A cross-method oracle: three estimators, different code each side, one answer.

    Stronger than each matching the closed form separately, because a systematic error shared with
    the *analytic* formula — a wrong `d₊`, say — would shift all four together, while these three
    derive their answer by genuinely different routes: bumping the input, differentiating the payoff,
    and differentiating the density. Compared pairwise within the combined standard error."""
    estimates = {
        method: black_mc_delta(FORWARD, STRIKE, VARIANCE, method, CONFIG)
        for method in GreekMethod
    }
    methods = list(estimates)
    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            combined = math.hypot(
                estimates[left].standard_error, estimates[right].standard_error
            )
            assert abs(estimates[left].mean - estimates[right].mean) < 4.0 * combined


# ============================================================================================
# THE REDUCTION + COMPANION that justifies having two methods at all.
# ============================================================================================


def test_T22_PATHWISE_is_valid_on_a_CALL_and_BREAKS_on_a_DIGITAL():
    """The oracle that earns the likelihood-ratio method, and it is the point of the tranche.

    Pathwise matching on a call is **vacuous on its own** — finite differencing would do. What
    justifies a second method is a payoff where pathwise is not merely less accurate but *wrong*:
    a digital's payoff is flat almost everywhere, so its pathwise derivative is **exactly zero** on
    every path, while the true delta is strictly positive. Jäckel: the method *"cannot be readily
    applied since the payoff function is not even continuous, let alone Lipschitz-continuous."*

    So both halves are asserted — pathwise agreeing where it is valid, and returning exactly zero
    where it is not, against an analytic digital delta that is not zero. The failure is stated as a
    number rather than described."""
    true_delta = digital_delta(FORWARD, STRIKE, VARIANCE, CapFloorKind.CAP)
    assert true_delta > 0.0, "premise: the digital must have a real delta to be wrong about"

    broken = digital_mc_delta(FORWARD, STRIKE, VARIANCE, GreekMethod.PATHWISE, CONFIG)
    assert broken.mean == 0.0, (
        "a digital's pathwise derivative is zero on every path — if this is non-zero the payoff "
        "being differentiated is not a digital"
    )
    assert abs(broken.mean - true_delta) > 10.0 * max(broken.standard_error, 1e-12)

    # The likelihood ratio needs no continuity, and is the one that holds.
    sound = digital_mc_delta(FORWARD, STRIKE, VARIANCE, GreekMethod.LIKELIHOOD_RATIO, CONFIG)
    assert abs(sound.mean - true_delta) < 4.0 * sound.standard_error


def test_T22_PATHWISE_is_REFUSED_for_a_gamma_rather_than_quietly_approximated():
    """§11.5 says pathwise gives a delta per path, not a second derivative. Refused, not fudged."""
    with pytest.raises(ValueError, match="pathwise"):
        black_mc_gamma(FORWARD, STRIKE, VARIANCE, GreekMethod.PATHWISE, CONFIG)


# ============================================================================================
# CONTRACT — determinism, the −½ rate, and the refusals.
# ============================================================================================


def test_T22_the_estimates_are_DETERMINISTIC_and_carry_their_standard_error():
    for method in GreekMethod:
        first = black_mc_delta(FORWARD, STRIKE, VARIANCE, method, CONFIG)
        again = black_mc_delta(FORWARD, STRIKE, VARIANCE, method, CONFIG)
        assert first.mean == again.mean and first.standard_error == again.standard_error
        assert first.standard_error > 0.0


def test_T22_the_GREEKS_error_falls_at_the_MINUS_ONE_HALF_rate():
    """The requirement no single path count can fake — and it must hold for a *derivative* of the
    price, not only the price. Checked on the likelihood-ratio delta, whose weight makes it the
    noisiest of the three and therefore the one where a broken rate would show first."""
    errors = {}
    for paths in (5_000, 20_000, 80_000):
        estimate = black_mc_delta(
            FORWARD, STRIKE, VARIANCE, GreekMethod.LIKELIHOOD_RATIO,
            MonteCarloConfig(paths, CONFIG.seed),
        )
        errors[paths] = estimate.standard_error
    assert errors[20_000] / errors[80_000] == pytest.approx(2.0, rel=0.1)
    assert errors[5_000] / errors[20_000] == pytest.approx(2.0, rel=0.1)


def test_T22_a_DEGENERATE_input_is_refused_rather_than_returning_a_NaN():
    """§3.4 — failure is a value. Zero variance has no density to differentiate, so the likelihood
    ratio's `ω` would divide by zero; a non-positive forward is not a lognormal state at all."""
    with pytest.raises(ValueError, match="positive variance"):
        black_mc_delta(FORWARD, STRIKE, 0.0, GreekMethod.LIKELIHOOD_RATIO, CONFIG)
    with pytest.raises(ValueError, match="positive"):
        black_mc_delta(0.0, STRIKE, VARIANCE, GreekMethod.PATHWISE, CONFIG)
