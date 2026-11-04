"""A deterministic seeded uniform source (L0). T15 — the root of the Monte Carlo dependency graph.

`UniformSource(seed).uniforms(n)` yields reproducible uniforms on `(0, 1)` from numpy's
`Generator(PCG64(seed))`, **wrapped so no global or default RNG is ever reachable**.

**The wrapping is the whole point, not a style preference.** Charter §3 forbids ambient state: a
pricer must be a pure function of its arguments. A simulating pricer that drew from `np.random` would
depend on whatever ran before it, so §3.1's referential transparency and §9's byte-identical
statelessness check would both become untestable. With the seed threaded as explicit config, a
fixed-seed Monte Carlo *is* a pure function — which is what lets an MC oracle be a hard assertion
rather than a flaky one.

**What §4-transcription covers here: nothing, and that is ruled rather than assumed.** PCG64
postdates JAECKEL-2002 and is not in it, so its uniform bits are not transcribed from any source.
They are **trusted** — a documented, version-pinned library stream — and **pinned by a
cross-platform reproducibility regression**: the recorded first draws in the oracle are the contract,
checked on both CI operating systems. A hand-rolled generator is what §6 forbids: reproducible
*normals* are the real problem, a novel RNG is not. *(Ruled by Cowork, T15 §3.)*

**Open interval, not half-open.** numpy's `random()` returns `[0, 1)`, which admits an exact 0 — and
`Φ⁻¹(0)` is `−∞`, so one zero draw in a million would surface as a `-inf` path rather than an error.
The source therefore rejects the endpoints by redrawing, and the oracle asserts `min > 0` over 20,000
draws. The probability of needing a redraw is ~2⁻⁵³ per draw, so this costs nothing and removes a
failure mode that would be very hard to diagnose downstream.

Provenance:
  stock:  _stock/quarry/pricebook/statistics/rng.py (73 lines) — mined for the SHAPE only ("seed in,
          deterministic draws out") and for its migration-hardening lesson that seeding must be
          reproducible across calls; its Sobol half and its placement carry no authority (§5)
  source: JAECKEL-2002 (docs/references.md) ch 7 for *why* a deterministic, reproducible source is
          the foundation of a usable MC — but NOT for the generator: PCG64 is not in Jaeckel and its
          bits are trusted rather than transcribed (above). numpy's PCG64 stream is version-pinned
          in pyproject, since the stream is part of the reproducibility contract.
  oracle: tests/L0/test_rng.py — same seed ⇒ byte-identical draws; different seeds differ and the
          stream is non-degenerate; the first five draws pinned as the cross-platform contract;
          draws strictly inside (0,1); consecutive calls CONTINUE the stream; and the global numpy
          RNG proved untouched after 10,000 draws
  crosscheck: none — a seeded-generator wrapper, no finance convention
  slice:  tranche 15 — the Monte Carlo spine
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class UniformSource:
    """A seeded uniform generator, carrying its own state and nothing else's.

    Frozen in the sense that matters — the *seed* is immutable, so the stream a source produces is
    determined at construction. The generator behind it necessarily advances as it is drawn from;
    that is what `test_CONSECUTIVE_calls_continue_the_stream` pins, and it is why two sources built
    from one seed give the same sequence while one source gives a fresh slice per call."""

    seed: int
    _generator: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"a seed must be non-negative, got {self.seed}")
        # `object.__setattr__` because the dataclass is frozen: the seed is the identity, and the
        # generator is the state that seed determines.
        object.__setattr__(self, "_generator", np.random.Generator(np.random.PCG64(self.seed)))

    def uniforms(self, count: int) -> list[float]:
        """`count` draws on the OPEN interval `(0, 1)`, continuing this source's stream."""
        if count < 1:
            raise ValueError(f"a draw needs at least one sample, got {count}")
        drawn = self._generator.random(count)
        # Reject an exact endpoint rather than let it reach an inverse CDF as +/-inf. numpy's
        # `random()` is [0,1), so only 0.0 is reachable, with probability ~2^-53 per draw.
        while True:
            degenerate = np.nonzero((drawn <= 0.0) | (drawn >= 1.0))[0]
            if degenerate.size == 0:
                break
            drawn[degenerate] = self._generator.random(degenerate.size)
        return [float(x) for x in drawn]
