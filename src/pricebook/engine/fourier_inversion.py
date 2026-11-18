"""The Gil-Pelaez inversion, as one function over any characteristic function (L4).

    P = 1/2 + (1/pi) * integral_0^inf Re[ exp(-iu*k) * phi(u) / (iu) ] du,   k = log(K/S_0)

`P2 = Prob(S_T > K)` under the T-forward measure and `P1` the same event under the **share**
measure, recovered as `phi(u - i)/phi(-i)` — the same characteristic function tilted, *not* a second
transcription. A single `phi` means a transcription error moves both probabilities together and
shows up in put-call parity, rather than hiding in one of two independently written formulas.

**Extracted at tranche 49, not written speculatively at tranche 33.** This lived inside
`heston_fourier.py` and was right there while Heston was its only consumer. SVJ (Bates) is the
second, and it differs from Heston in *exactly one thing* — its characteristic function is
`phi_Heston * phi_jump`. §6's rule-of-two triggers on a shared **shape**, and the shape here is the
entire inversion: the same integrand, the same share-measure tilt, the same quadrature. Doing it
this way is what makes "tranche 33's engine is reused whole" true in code rather than in intent;
the alternative — a sixth argument on `heston_probabilities` — would have broken §7's limit for
something that is not vocabulary.

**The extraction is a pure code move, proved by bit-identity rather than by the suite staying
green.** The expressions and their order are unchanged, so every Heston number is the same double it
was: 120 reference values (3 parameter sets x 4 maturities x 2 grids x 5 strikes) were captured as
raw bit patterns before the move and compared after, **in the same interpreter** — 360 doubles, every
bit equal. §4 permits the bit-identity claim exactly when a refactor preserves the order of the
floating-point operations, and a verbatim move does.

**That proof is in-process, and the committed regression is therefore a ULP bound rather than
equality** — which is not a weakening of the argument but the limit of what a test running on other
machines may claim. §9 states the rule: transcendental results differ 1-2 ULP across runtimes, and
`cmath.exp`/`log`/`sqrt` are exactly that. The first version of `tests/L4/test_fourier_inversion.py`
asserted the captured bits for equality and failed on both CI runners by 1 to 4 ULP. A second form
asserted a ULP *distance*, which sounded stronger and was the wrong instrument: a Fourier price is a
sum of 4001 transcendental terms, so its cross-platform difference accumulates as a **relative**
quantity (a price of 0.00158 came out 25,475 ULP away while being 3.5e-12 relative). The committed
form is a relative tolerance of 1e-9 against a measured worst of 3.5e-12 — six orders tighter than
the tightest behavioural oracle here.

**The truncation is ADAPTIVE, added at tranche 50 to repair OI-4.** A single committed `upper` cannot
serve this inversion: Heston's characteristic function decays *exponentially* in `u` (not Gaussian, as
Black's does), at a rate that collapses as `T -> 0` and the vol-of-vol grows. Measured at `u = 200`, the
integrand `|phi(u)|/u` is **4.2e-47** for a one-year benign slice and **1.7e-04** in the seven-day
high-vol wing - so the committed `upper = 200` truncated while the tail was larger than the price, and
returned **negative call prices** there. `adaptive_grid` steps the range out until the integrand is
genuinely negligible, which is regime-aware by construction: it measures the decay of the actual
characteristic function rather than keying off a formula in the model's parameters, so it serves Heston
and SVJ (whose jump tails decay slower still) without a second constant. A flat wide constant was
refused: it would make every benign pricing call sixteen times dearer to repair a corner.

**The arithmetic floor, and why a price can be refused.** Truncation error falls under refinement until
it reaches a floor and then stalls: `P1 = 1/2 + I/pi` is a difference of two `O(1/2)` quantities, so it
carries an absolute noise floor of ~2e-13 whatever the grid, and a price inherits `forward * 2e-13`.
`checked_call_price` reads a raw inversion output against that floor in three branches - a price above
it, `0.0` within it (which is *more* accurate than the noise, not a cosmetic clamp), and a **refusal**
below it, because a materially negative call price means the method failed in that regime rather than
that the option is cheap. The third branch is what keeps the second honest: a bare clamp would have
turned OI-4's -2.5e-04 into a clean-looking `0.0` and hidden it, and seven orders separate that defect
from the floor.

**`phi(u)` is evaluated once per maturity, not once per strike (tranche 53).** The integrand is
`phi` times the Gil-Pelaez phase `exp(-iuk)`, and only the phase depends on the strike - so an
inversion that rebuilt `phi` for every strike was paying `N` times for one quantity, and `phi` is the
dear half (a complex `sqrt`, `log` and two `exp` per node for Heston, strictly more for SVJ, against
one `exp` for the phase). `InversionKernel` holds `phi` at the grid's nodes and `probabilities_at`
inverts it at any strike, so a smile costs one `phi` sweep. The win is asserted rather than described
(`tests/L4/test_phi_across_strikes.py` counts the calls): a rationale that is not a test erodes, as
tranche 50's first threshold demonstrated.

**The characteristic function arrives already bound to its maturity and carry**, as `phi(u)` alone.
The inversion has no business knowing what parameters a model carries, and a signature taking
`(params, u, years, carry)` would need widening every time a model's bundle differed. Binding
happens at the caller, where the model is known.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 16.2 - the Gil-Pelaez form of the inversion
          integral, unchanged from tranche 33; this module is an extraction, not a new
          transcription
  oracle: tests/L4/test_oi4_adaptive_truncation.py (tranche 50) - the no-arbitrage requirement over
          the mapped defect region, convergence against a high-precision reference IN the failing
          corner, the fix's locality (the benign region selects the same grid and returns the same
          number), and the floor's three branches with the floor measured rather than chosen;
          tests/L4/test_fourier_inversion.py - the 54-point relative-tolerance regression that the
          extraction changed no Heston number (bit-identical in-process; see above for why the
          committed form is a ULP bound), and the inversion's own requirements against a Gaussian
          characteristic function, whose two probabilities have exact closed forms
  crosscheck: none - the numerics are cross-checked where the characteristic functions are
          (models/heston.py, models/bates.py)
  slice:  tranche 49 - the SVJ/Bates model
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable
from dataclasses import dataclass

#: A characteristic function already bound to its maturity and carry: `u -> phi_T(u)`.
CharacteristicFunction = Callable[[complex], complex]


#: Where the quadrature starts, since `1/(iu)` is undefined at 0 while the integrand's limit there
#: is finite. **The omitted sliver `[0, _LOWER]` is a bias, and it is LINEAR in this constant** -
#: measured against the `sigma_v = 0` Black reduction, the relative price error is 1.16e-06 at
#: `1e-6`, 1.16e-08 at `1e-8`, 1.16e-10 at `1e-10` and 1.16e-12 at `1e-12`, reaching the
#: double-precision floor (1.4e-14) by `1e-14`. One-directional and grid-independent, so refining or
#: extending the grid never removes it - which is exactly how it was identified: the reduction's
#: error sat unchanged across three grids and only moved when this did.
#:
#: Set to `1e-12`, where the bias is five orders below the tightest oracle in this tranche. Not
#: `1e-14`: the integrand is evaluated at `u` down there and a smaller start buys nothing once the
#: bias is under the floating-point noise of everything else.
_LOWER = 1e-12


@dataclass(frozen=True)
class FourierGrid:
    """Where the inversion integral is truncated, and how finely it is sampled.

    Both are error sources and they fail differently: too low an `upper` **omits** tail
    contribution (a one-directional bias), too small a `count` adds trapezoid error of either sign.
    The oracle asserts each separately rather than hiding them in one tolerance.
    """

    upper: float
    count: int

    def __post_init__(self) -> None:
        if self.upper <= 0.0:
            raise ValueError(f"the integration upper limit must be positive, got {self.upper}")
        if self.count < 2:
            raise ValueError(f"the quadrature needs at least two nodes, got {self.count}")

    def nodes(self) -> tuple[float, ...]:
        step = (self.upper - _LOWER) / (self.count - 1)
        return tuple(_LOWER + step * k for k in range(self.count))


def _trapezoid(values: list[float], nodes: tuple[float, ...]) -> float:
    area = 0.0
    for i in range(len(nodes) - 1):
        area += 0.5 * (values[i] + values[i + 1]) * (nodes[i + 1] - nodes[i])
    return area


#: Where the adaptive search starts, and the node spacing it holds fixed.
#:
#: `_BASE_UPPER` is tranche 33's committed upper bound, kept as the *starting point* rather than
#: discarded: everywhere it was already sufficient the search stops immediately, so the benign path
#: selects exactly the grid it used before and its prices do not move at all. The fix is local to
#: the defect by construction, not by tolerance.
#:
#: `_SPACING` is that same grid's spacing (`200/4000`). Extending the range at **fixed spacing** is
#: what keeps the discretisation error unchanged while the truncation error falls - the two are
#: separate error sources (`FourierGrid` says so) and this tranche is only about the second.
_BASE_UPPER = 200.0
_SPACING = 200.0 / 4000.0

#: The integrand magnitude `|phi(u)|/u` below which the remaining tail is negligible, and the range
#: past which the search gives up.
#:
#: `1e-13` is chosen against the arithmetic floor rather than for its own sake, and the choice was
#: **measured rather than guessed - the first value was wrong in the direction that matters.** The
#: threshold tests the integrand *at* the cutoff, while what the price feels is the integral of the
#: remaining tail, which for an exponentially decaying `phi` is larger by the decay length (an order
#: or so). Setting it at `1e-14` therefore overshot: a standard tree fixture sits at `3.7e-14`, so it
#: extended to `upper = 400` and **doubled the cost of every benign pricing call** to change the
#: price by 9.6e-15 relative - nothing, and precisely the "tax every call to repair a corner" the
#: ruling refused.
#:
#: At `1e-13` the mapped defect region reprices exactly as well (worst relative error 2.0e-03 at both
#: thresholds, and that residual is the cancellation floor below, not truncation), with zero
#: materially-negative prices, while the ordinary fixtures stay at `upper = 200` and cost what they
#: did. What still extends, extends because its price genuinely moves: a high-vol-of-vol one-year
#: slice goes to 400 and shifts by 6.5e-10 relative, some 250x the floor.
_DECAY_THRESHOLD = 1e-13
_UPPER_CAP = 25_600.0

#: `P1`'s absolute noise floor, as a probability. `P1 = 1/2 + I/pi` cancels two `O(1/2)` quantities,
#: so refining the grid stops helping once the result is this small - measured at a point whose true
#: price is zero to double precision, where the computed value stalls at ~2e-13 and does not shrink
#: when the range is tripled. Set to `1e-12`, a few multiples above the measurement, and a price
#: inherits it scaled by the forward.
_PROBABILITY_NOISE = 1e-12


def adaptive_grid(characteristic: CharacteristicFunction) -> FourierGrid:
    """A grid wide enough that the truncated tail of `|phi(u)|/u` is negligible.

    Doubles the range from `_BASE_UPPER` until the integrand at the upper limit falls below
    `_DECAY_THRESHOLD`, holding the node spacing fixed so only the truncation error moves. The
    integrand's bound is strike-independent (the `exp(-iuk)` phase has unit modulus), so one grid
    serves every strike of a given model and maturity - which is also why this takes the
    characteristic function and nothing about the product.
    """
    upper = _BASE_UPPER
    while upper < _UPPER_CAP and abs(characteristic(complex(upper))) / upper > _DECAY_THRESHOLD:
        upper *= 2.0
    return FourierGrid(upper=upper, count=int(round(upper / _SPACING)) + 1)


def checked_call_price(raw: float, forward: float) -> float:
    """A raw inversion output read against the method's arithmetic floor - see the module docstring.

    Three branches: a price above the floor is returned as computed; a magnitude *within* the floor
    returns `0.0`, because zero is the honest rendering of a number smaller than the arithmetic that
    produced it; and a value materially below zero **raises**, because a negative call price is an
    arbitrage violation that means the inversion failed in that regime, not that the option is cheap.

    The refusal is the branch that makes the zero trustworthy. Clamping alone would be the §4 trap -
    it would convert a real truncation defect into a plausible `0.0` - and the two cases are
    separated by orders of magnitude, not by a margin: OI-4's defect was -2.5e-04 against a floor of
    ~1e-10 on a forward of 100.
    """
    floor = forward * _PROBABILITY_NOISE
    if raw < -floor:
        raise ValueError(
            f"the Fourier inversion returned a materially negative call price ({raw}); the "
            f"truncation is insufficient in this regime (floor {floor})"
        )
    return raw if raw > floor else 0.0


@dataclass(frozen=True)
class InversionKernel:
    """`phi` evaluated at a grid's nodes - everything in the Gil-Pelaez integrand that does not
    depend on the strike.

    `tilted` carries `phi(u - i)/phi(-i)` rather than `phi(u - i)`, because the share-measure
    normalisation is strike-independent too and dividing once per node beats dividing once per node
    per strike. Frozen and holding only numbers, so one kernel is safely reused across a maturity's
    whole smile.
    """

    nodes: tuple[float, ...]
    plain: tuple[complex, ...]
    tilted: tuple[complex, ...]


def inversion_kernel(
    characteristic: CharacteristicFunction, grid: FourierGrid
) -> InversionKernel:
    """Evaluate `phi` across the grid once, ready to be inverted at any strike."""
    nodes = grid.nodes()
    shift = characteristic(-1j)
    return InversionKernel(
        nodes=nodes,
        plain=tuple(characteristic(complex(u)) for u in nodes),
        tilted=tuple(characteristic(complex(u) - 1j) / shift for u in nodes),
    )


def probabilities_at(kernel: InversionKernel, log_strike: float) -> tuple[float, float]:
    """`(P1, P2)` for one strike, from a kernel that may serve many.

    The operations run in the order the per-strike version used - the same `phi` values, the same
    phase, the same sequential trapezoid - so a price computed here is bit-identical in-process to
    one computed before the hoist. (§4: bit-identity is claimable exactly when a refactor preserves
    the order of the floating-point operations, and hoisting a loop-invariant out of a loop does.)
    """
    first_values: list[float] = []
    second_values: list[float] = []
    for u, plain, tilted in zip(kernel.nodes, kernel.plain, kernel.tilted, strict=True):
        phase = cmath.exp(-1j * u * log_strike)
        first_values.append((tilted * phase / (1j * u)).real)
        second_values.append((plain * phase / (1j * u)).real)
    first = _trapezoid(first_values, kernel.nodes)
    second = _trapezoid(second_values, kernel.nodes)
    return 0.5 + first / math.pi, 0.5 + second / math.pi


def gil_pelaez_probabilities(
    characteristic: CharacteristicFunction, log_strike: float, grid: FourierGrid
) -> tuple[float, float]:
    """`(P1, P2)` by Gil-Pelaez inversion - named surface, because the oracles reach for them.

    Returning the probabilities rather than only the price is what lets a suite assert they are
    *probabilities* (in `[0,1]`, falling in the strike) and lets put-call parity be checked from the
    same two numbers the call is built from (§4 - extract the step a test must see).

    `log_strike` is formed against the **spot**, because the characteristic function handed in is
    `E[exp(iu log(S_T/S_0))]` - the carry already lives inside it. Forming it against the forward
    would double-count the drift.
    """
    return probabilities_at(inversion_kernel(characteristic, grid), log_strike)
