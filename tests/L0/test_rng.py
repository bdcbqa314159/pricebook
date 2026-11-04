"""The deterministic seeded uniform source (L0). T15, RNG ruled by Cowork.

`UniformSource(seed)` yields reproducible uniforms from `numpy Generator(PCG64(seed))`, wrapped so
**no global or default RNG is ever reachable**. That wrapping is the whole point: a fixed-seed Monte
Carlo is then a *pure function* of its seed, so charter §3's referential transparency and §9's
byte-identical statelessness check both survive a simulating engine.

**What is oracled here, and what is deliberately NOT.** PCG64's uniform bits are **not**
transcribed — PCG64 postdates Jaeckel (2002) and is not in it. They are *trusted* (a documented,
version-pinned library stream) and *pinned by a reproducibility regression*: the same seed must give
byte-identical draws, and the recorded first draws below are the cross-platform contract CI checks on
both operating systems. So these oracles test **our wrapper's determinism and its refusal to touch
global state**, never numpy's statistical quality.

The recorded values are a REGRESSION, not a reference: they come from the pinned numpy, and if a
numpy bump moves them that is a contract change to notice, not a bug to hide. That is why they are
written down rather than computed.
"""

import pytest

from pricebook.foundation.rng import UniformSource


def test_the_same_seed_gives_byte_identical_draws():
    """Determinism, the property everything else rests on."""
    a = UniformSource(seed=12345).uniforms(1000)
    b = UniformSource(seed=12345).uniforms(1000)
    assert [repr(x) for x in a] == [repr(x) for x in b]


def test_DIFFERENT_seeds_give_different_draws():
    """The companion: without it, a source returning a constant would pass the test above."""
    a = UniformSource(seed=1).uniforms(500)
    b = UniformSource(seed=2).uniforms(500)
    assert a != b
    assert len(set(a)) == 500, "the draws are not distinct — the stream is degenerate"


def test_the_FIRST_DRAWS_are_pinned_as_a_cross_platform_contract():
    """**The regression that makes "reproducible on both OSes" an assertion rather than a hope.**
    These are the pinned numpy's values; CI runs ubuntu and windows, so a stream difference between
    them fails here. A numpy bump that moves them is a contract change to notice deliberately."""
    draws = UniformSource(seed=42).uniforms(5)
    expected = [
        0.7739560485559633,
        0.4388784397520523,
        0.8585979199113825,
        0.6973680290593639,
        0.09417734788764953,
    ]
    assert draws == pytest.approx(expected, rel=0.0, abs=0.0), draws


def test_draws_lie_strictly_INSIDE_the_unit_interval():
    """`Phi^-1(0)` is `-inf` and `Phi^-1(1)` is `+inf`, so a variate built by inversion needs the
    open interval. numpy's `random()` is `[0,1)`, which still admits an exact 0 — asserted here
    because a single zero draw in a million would surface as a `-inf` path, not as an error."""
    draws = UniformSource(seed=7).uniforms(20_000)
    assert min(draws) > 0.0, "a zero draw would invert to -inf"
    assert max(draws) < 1.0


def test_CONSECUTIVE_calls_continue_the_stream_rather_than_restarting():
    """A source that restarted per call would silently repeat the same paths across batches —
    reproducible and wrong."""
    source = UniformSource(seed=99)
    first = source.uniforms(10)
    second = source.uniforms(10)
    assert first != second
    assert UniformSource(seed=99).uniforms(20) == first + second


def test_the_source_NEVER_touches_the_global_numpy_RNG():
    """The §3 requirement, asserted rather than assumed: drawing from our source must leave numpy's
    global state untouched, or two unrelated pricings could influence each other."""
    import numpy as np

    np.random.seed(0)
    before = np.random.get_state()[1][:8].tolist()
    UniformSource(seed=5).uniforms(10_000)
    after = np.random.get_state()[1][:8].tolist()
    assert before == after


def test_a_NON_POSITIVE_count_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        UniformSource(seed=1).uniforms(0)


def test_a_NEGATIVE_seed_is_refused():
    """numpy accepts a negative seed in some paths and raises in others; refusing it here keeps the
    contract ours rather than numpy's."""
    with pytest.raises(ValueError, match="seed"):
        UniformSource(seed=-1)
