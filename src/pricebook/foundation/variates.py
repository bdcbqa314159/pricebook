"""Standard normal variates by Box-Muller (L0). JAECKEL-2002 eq 9.6.

    x = √(−2 ln u) · sin(2πv)
    y = √(−2 ln u) · cos(2πv)        for `u, v` independent uniform on `(0,1)²`

Transcribed in **both** renderings, which agree exactly, and Jäckel states the joint density of
`(x, y)` is *"that of two independent standard normal variates!"*.

**Why Box-Muller and not the inverse CDF Jäckel prefers.** He recommends Acklam's or Moro's
interpolation formula — and **does not print either**. The coefficients live in `[Ack00]`/`[Mor95]`,
which `docs/references.md` records as `referenced-by` only, and his implementation is *"included on
the accompanying CD"* we do not have. Charter §7 forbids citing a work we do not hold, so `Φ⁻¹` is
not transcribable from what is in hand. Eq 9.6 **is** printed in full, is **exact** rather than an
approximation — so it carries no accuracy caveat and admits an exact oracle — and needs no
acquisition. *(Ruled by Cowork, T15. Acklam is deferred to the Sobol unit, where a monotone
transform genuinely earns its keep: Box-Muller's trigonometric wrapping does not preserve
low-discrepancy structure, which is precisely where inversion pays.)*

**The anchor oracle is an exact identity.** Since `sin² + cos² = 1`,

    x² + y² = −2 ln u

for every pair, to machine precision — so a wrong `√`, a missing factor of 2, a `log10` for `ln`, or
`sin`/`cos` in the wrong slots all show with no sampling noise at all. The N(0,1) moment checks and
the pair-independence check are **companions**, not the anchor: they catch what the identity cannot,
and `pi*v` for `2*pi*v` is the concrete case — it preserves `x² + y²` exactly, so only the pointwise
transcription sees it. That pairing is §4's "properties never replace a pointwise transcription" and
its converse in one place.

**Jäckel's two caveats, read and handled rather than assumed away.** Figure 9.5's warning is against
the *rejection* variant (eq 9.8) combined with low-discrepancy numbers — not this trigonometric
form; and the Neave effect (§9.3.2) is an interaction with *simple multiplicative congruential*
generators, which PCG64 is not. Neither is taken on trust: the N(0,1) and independence oracles would
fail if either applied here.

**The domain guard lives here, and that placement was earned.** `ln u` diverges at `u = 0`, so a
uniform must be strictly interior. Tranche 15 first put this guard in the uniform source, where an
exact `0.0` has probability ~2⁻⁵³ — statistically unreachable, so the branch was unoracled and a
mutant breaking it passed all 468 tests. Here it is one call away from a test (`standard_normals([0.0,
0.5])`), which is the difference between a guard and dead code (§6, and §4's third survivor kind).

Provenance:
  stock:  _stock/quarry/pricebook/statistics/rng.py — mined for the SHAPE only (uniforms in,
          normals out); its normal generation is not transcribed from here (§5)
  source: JAECKEL-2002 (docs/references.md) §9.3.1 eq 9.6, read in `-raw` and `-layout`; §9.3 for
          why inversion is preferred and §9.3.2/Fig 9.5 for the two caveats above
  oracle: tests/L0/test_variates.py — the exact `x²+y² = −2 ln u` identity per pair; the pointwise
          transcription of 9.6; a wrong-angle variant shown to satisfy the identity but not the
          transcription; N(0,1) in mean and variance with the band MEASURED from the sample size;
          pair independence by sample covariance; the tails populated; `u ∈ {0,1}` refused; an odd
          count refused
  crosscheck: none — an exact transformation transcribed from the source; the identity is the check
  slice:  tranche 15 — the Monte Carlo spine
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_TWO_PI = 2.0 * math.pi


def standard_normals(uniforms: Sequence[float]) -> list[float]:
    """Standard normal draws from `uniforms`, consumed **in pairs** by eq 9.6.

    Returns exactly as many normals as uniforms given. An odd count is refused rather than silently
    dropping the last uniform — a caller that believed it had consumed `n` draws of a shared stream
    would otherwise be wrong about the stream's position, which is the kind of error that makes two
    "reproducible" runs disagree."""
    count = len(uniforms)
    if count < 2 or count % 2 != 0:
        raise ValueError(
            f"Box-Muller consumes uniforms in pairs, so an even count of at least two is "
            f"required; got {count}"
        )
    for u in uniforms:
        if not 0.0 < u < 1.0:
            raise ValueError(
                f"a uniform must lie strictly inside (0, 1) for Box-Muller — `ln u` diverges at "
                f"0 — got {u}"
            )

    draws: list[float] = []
    for i in range(0, count, 2):
        u, v = uniforms[i], uniforms[i + 1]
        radius = math.sqrt(-2.0 * math.log(u))
        angle = _TWO_PI * v
        draws.append(radius * math.sin(angle))
        draws.append(radius * math.cos(angle))
    return draws
