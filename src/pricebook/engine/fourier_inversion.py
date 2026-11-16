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
asserted the captured bits for equality and failed on both CI runners by 1 to 4 ULP; it now asserts
a ULP distance with a stated budget, still four orders tighter than any behavioural tolerance here.

**The characteristic function arrives already bound to its maturity and carry**, as `phi(u)` alone.
The inversion has no business knowing what parameters a model carries, and a signature taking
`(params, u, years, carry)` would need widening every time a model's bundle differed. Binding
happens at the caller, where the model is known.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 16.2 - the Gil-Pelaez form of the inversion
          integral, unchanged from tranche 33; this module is an extraction, not a new
          transcription
  oracle: tests/L4/test_fourier_inversion.py - the 120-point ULP-bounded regression that the
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
    shift = characteristic(-1j)

    def integrands(u: float) -> tuple[float, float]:
        phase = cmath.exp(-1j * u * log_strike)
        plain = characteristic(complex(u))
        tilted = characteristic(complex(u) - 1j) / shift
        return (tilted * phase / (1j * u)).real, (plain * phase / (1j * u)).real

    nodes = grid.nodes()
    values = [integrands(u) for u in nodes]
    first = _trapezoid([v[0] for v in values], nodes)
    second = _trapezoid([v[1] for v in values], nodes)
    return 0.5 + first / math.pi, 0.5 + second / math.pi
