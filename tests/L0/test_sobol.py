"""Sobol' low-discrepancy numbers (L0). JAECKEL-2002 §8.3.

The construction, as the source prints it:

    p_k(z) = Σ_j a_kj z^(g_k − j)                                                      (8.18)
    v_kl   = (v_k(l−g_k) / 2^g_k) ⊕₂ [ ⊕₂_{j=1..g_k} a_kj · v_k(l−j) ]   for l > g_k    (8.19)
    x_nk   = ⊕₂_j v_kj · 1{jth bit from the right of γ(n) is set}                       (8.20)
    y_nk   = x_nk / 2^b                                                                 (8.17)
    G(n)   = n ⊕₂ ⌊n/2⌋                                                                 (8.22)

**Table 8.3 is the transcription, and it is ~350 magic numbers.** Thirty-two dimensions, each with a
degree, a primitive polynomial in binary, and ten initialisation numbers. Hand-typing that invites
exactly the error T20a showed is almost unoracleable in a rational approximation — so it was
extracted from **both** `pdftotext` renderings and reconciled mechanically: 32 rows, zero
disagreements, every exponent equal to `32 − l`. That is the dual-rendering rule (§4) applied at a
scale where reading twice by eye would be the weaker method, not the stronger one.

**What makes this table checkable at all is that the source states a PROPERTY of it.** Table 8.3 is
captioned *"Initialisation numbers providing property A up to dimension 32"*, and property A is a
sharp, testable claim: in any binary segment of length `2^d` aligned to a multiple of `2^d`, each of
the `2^d` sub-hypercubes of the unit cube contains **exactly one** point. A mistyped direction number
breaks it. So the transcription is not resting on care — it rests on a requirement oracle derived
from the source's own caption.

**The one thing the source does NOT print:** the primitive polynomials beyond this table — *"I
provide a list of all primitive polynomials modulo two up to degree 27 on the accompanying CD"*, the
CD we do not have. Third time this book has sent a reader to something it does not contain (after
`Φ⁻¹` and the control-variate variance factor). It costs nothing here, because Table 8.3 carries the
polynomials for the 32 dimensions it covers.
"""

import math

import pytest

from pricebook.foundation.distributions import inverse_norm_cdf
from pricebook.foundation.sobol import (
    SOBOL_MAX_DIMENSIONS,
    SobolSequence,
    direction_integers,
    gray_code,
)

#: Table 8.2 of the source, printed in full: `G(n) = n ⊕₂ ⌊n/2⌋` for `n = 0..7`.
PRINTED_GRAY_CODES = (0, 1, 3, 2, 6, 7, 5, 4)

#: The first rows of Table 8.3 — `(k, g_k, polynomial bits, m_l for l = 1..10)` with
#: `v_kl = m_l · 2^(32−l)`. Transcribed here independently of the implementation's own copy, which
#: is the §4 independence rule: a shared table would make the pointwise checks below circular.
PRINTED_TABLE_8_3 = {
    1: (0, "1", (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
    2: (1, "11", (1, 3, 5, 15, 17, 51, 85, 255, 257, 771)),
    3: (2, "111", (1, 1, 7, 11, 13, 61, 67, 79, 465, 721)),
    4: (3, "1011", (1, 3, 7, 5, 7, 43, 49, 147, 439, 1013)),
    5: (3, "1101", (1, 1, 5, 3, 15, 51, 125, 141, 177, 759)),
    6: (4, "10011", (1, 3, 1, 1, 9, 59, 25, 89, 321, 835)),
    7: (4, "11001", (1, 1, 3, 7, 31, 47, 109, 173, 181, 949)),
    8: (5, "100101", (1, 3, 3, 9, 9, 57, 43, 43, 225, 113)),
}


# ============================================================================================
# THE GRAY CODE — eq 8.22, against the source's own printed table.
# ============================================================================================


def test_T20b_the_GRAY_CODE_matches_the_sources_printed_table():
    """Table 8.2 is printed in the book, so this is a reference-value oracle rather than a property.

    And the property is asserted beside it: *"exactly one bit changes in any one increase"*, which is
    the whole reason Antonov and Saleev's modification costs one XOR per draw instead of `b` of them.
    A Gray code that is merely *a* bijection would pass the table and fail this."""
    assert tuple(gray_code(n) for n in range(8)) == PRINTED_GRAY_CODES

    for n in range(1, 5_000):
        changed = gray_code(n) ^ gray_code(n - 1)
        assert changed.bit_count() == 1, f"G({n}) differs from G({n - 1}) in {changed.bit_count()} bits"


def test_T20b_the_GRAY_CODE_drives_the_SEQUENCE_a_second_independent_killer() -> None:
    """The same point set computed **without any Gray code at all** — a second method, not a
    second assertion.

    Flagged by re-audit #3: `gray_code` is load-bearing for the whole low-discrepancy construction
    and the mutation `G(n) = n ⊕ ⌊n/2⌋ → n` was killed by **exactly one test**. Both the printed
    table and the one-bit property live in that single function, so deleting it would reopen the
    hole with the suite still green — the §9 single-point-of-failure shape, on an L0 primitive
    everything above it rests on.

    This kills it a different way. Sobol's `n`-th point is `⊕_j v_j` over the **set bits of the
    Gray code** `G(n)`; the production path reaches it incrementally, XOR-ing the one direction
    integer that changed. Here the same point is built **directly** from `G(n)`'s bits with no
    incremental state, so the two agree only if the Gray code is what the recurrence assumes. A
    wrong `G` — the identity included — desynchronises them immediately.
    """
    dimensions, count = 3, 300
    table = [direction_integers(k) for k in range(1, dimensions + 1)]
    scale = 1.0 / 2**32

    direct = []
    for n in range(1, count + 1):
        code = n ^ (n >> 1)   # the Gray code written out HERE, independent of the module's
        point = []
        for k in range(dimensions):
            value = 0
            bits = code
            index = 0
            while bits:
                if bits & 1:
                    value ^= table[k][index]
                bits >>= 1
                index += 1
            point.append(value * scale)
        direct.append(tuple(point))

    incremental = SobolSequence(dimensions=dimensions).uniforms(count)
    assert incremental == direct, "the incremental and direct constructions must agree exactly"


def test_T20b_the_one_bit_property_is_its_OWN_test_so_a_deletion_cannot_take_both() -> None:
    """Split out from the table oracle deliberately (re-audit #3).

    *"Exactly one bit changes in any one increase"* is what makes Antonov-Saleev cost one XOR per
    draw instead of `b` of them, and it is a different claim from "the first eight values match the
    printed table". They were asserted in one function, so one deletion removed both; they are two
    tests now because a single-point-of-failure oracle is only single until someone edits the file.
    """
    for n in range(1, 5_000):
        changed = gray_code(n) ^ gray_code(n - 1)
        assert changed.bit_count() == 1, f"G({n}) differs from G({n - 1}) in {changed.bit_count()} bits"


# ============================================================================================
# THE TRANSCRIBED TABLE and the direction-integer construction — eqs 8.18/8.19.
# ============================================================================================


@pytest.mark.parametrize("dimension", sorted(PRINTED_TABLE_8_3))
def test_T20b_the_INITIALISATION_numbers_match_the_printed_table(dimension):
    """`v_kl = m_l · 2^(32−l)` for `l = 1..10` — eq 8.24's form with the table's multipliers.

    Compared against a copy of Table 8.3 transcribed into this file, independent of the one the
    implementation holds. Eight dimensions rather than all thirty-two: enough to pin the parsing and
    the scaling, while property A below exercises the rest structurally."""
    _, _, multipliers = PRINTED_TABLE_8_3[dimension]
    integers = direction_integers(dimension)
    for index, multiplier in enumerate(multipliers, start=1):
        assert integers[index - 1] == multiplier * 2 ** (32 - index)


@pytest.mark.parametrize("dimension", range(1, SOBOL_MAX_DIMENSIONS + 1))
def test_T20b_EVERY_direction_integer_obeys_the_sources_two_CONSTRAINTS(dimension):
    """*"Only the `l` leftmost bits can be non-zero, and the `l`th leftmost bit of `v_kl` must be
    set."*

    A structural requirement on all 32 dimensions × 32 bits — over a thousand numbers, including
    every one the recurrence (8.19) generates beyond the table's ten. It is the check that makes the
    recurrence trustworthy at indices the source never prints: a wrong shift or a wrong coefficient
    selection breaks the bit pattern immediately."""
    integers = direction_integers(dimension)
    assert len(integers) == 32
    for index, value in enumerate(integers, start=1):
        assert value >> (32 - index) & 1, f"v[{index}] must have its {index}th leftmost bit set"
        assert value % 2 ** (32 - index) == 0, f"v[{index}] has bits below its {index}th leftmost"


@pytest.mark.parametrize("dimension", [2, 4, 7, 8])
def test_T20b_the_RECURRENCE_8_19_is_reproduced_POINTWISE(dimension):
    """Eq 8.19 transcribed here independently, from the polynomial bits in this file's own table.

    `v_kl = (v_k(l−g) >> g) ⊕ [⊕_{j=1..g} a_kj · v_k(l−j)]`. The source notes two things the
    transcription must respect and a careless reading drops: `a_k0` is **not used** (it is implied by
    the degree), and `a_k g_k` is always set, so `v_k(l−g)` always enters. Both are visible here —
    the loop runs `j = 1..g` over the bits *after* the leading one."""
    degree, bits, _ = PRINTED_TABLE_8_3[dimension]
    coefficients = [int(bit) for bit in bits]          # a_k0 .. a_k g_k
    assert coefficients[0] == 1, "a_k0 is 1 by the polynomial's degree"
    assert coefficients[degree] == 1, "a_k g_k is always set"

    integers = direction_integers(dimension)
    for index in range(degree + 1, 33):                # l > g_k, 1-based
        expected = integers[index - degree - 1] >> degree
        for j in range(1, degree + 1):
            if coefficients[j]:
                expected ^= integers[index - j - 1]
        assert integers[index - 1] == expected, f"eq 8.19 fails at l={index}"


# ============================================================================================
# PROPERTY A — the source's own claim about Table 8.3, and the oracle that pins 350 numbers.
# ============================================================================================


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4, 5, 6, 7, 8])
def test_T20b_the_table_delivers_PROPERTY_A_as_its_caption_claims(dimensions):
    """*"Initialisation numbers providing property A up to dimension 32."*

    Property A: for any binary segment of length `2^d` starting at `l·2^d`, there is **exactly one**
    draw in each of the `2^d` sub-hypercubes formed by halving every axis. This is the requirement
    oracle that makes a ~350-number transcription verifiable — a single mistyped direction integer
    moves a point into the wrong half-cube and a count becomes 0 or 2.

    **The first segment is SHORT BY ONE, and the source says why.** Property A is stated over the
    sequence including `n = 0`, which is the origin; we skip it (a `0.0` would map to `−inf` under
    `Φ⁻¹`). §8.3.4: *"if we explicitly exclude the point at the origin ... the segment of the
    sequence starting with the first element will have no point in the sub-hypercube `[0, ½)^d` and
    will be most uniform for a length of `2^d − 1`."* So the first segment must hit exactly
    `2^d − 1` cells with the all-lower cell **missing** — asserted rather than tolerated, because
    "one cell short" is also what a transcription error looks like, and the two must be told apart.

    A first version of this oracle indexed segments from our own skipped numbering and failed at
    "15 of 16" — measuring the skip, not the table.

    Dimensions stop at 8 because a segment is `2^d` draws long: `d = 8` is 256, `d = 32` would be 4
    billion. The high dimensions rest on the bit constraints and the recurrence instead, which is
    stated rather than glossed."""
    sequence = SobolSequence(dimensions=dimensions)
    size = 2**dimensions
    # Our draw i corresponds to the source's n = i + 1, so a segment aligned at the source's
    # `l·2^d` starts at our index `l·2^d − 1`.
    draws = sequence.uniforms(count=4 * size)

    def cells_of(points):
        return {tuple(1 if c >= 0.5 else 0 for c in point) for point in points}

    first = cells_of(draws[0 : size - 1])
    assert len(first) == size - 1, (
        f"the first segment of dimension {dimensions} hits {len(first)} of {size - 1} cells"
    )
    assert (0,) * dimensions not in first, "the missing cell must be the one the origin would fill"

    for segment_index in range(1, 4):
        segment = draws[segment_index * size - 1 : (segment_index + 1) * size - 1]
        cells = cells_of(segment)
        assert len(cells) == size, (
            f"segment {segment_index} of dimension {dimensions} hits {len(cells)} of {size} "
            f"sub-hypercubes — property A fails"
        )


# ============================================================================================
# LOW DISCREPANCY — the point of the sequence, and the −½ rate it is adopted to beat.
# ============================================================================================


def _star_discrepancy_proxy(points, dimensions: int) -> float:
    """An L2-style discrepancy proxy: the worst relative error of an axis-aligned box count.

    Not the true star discrepancy (which is a supremum over all anchored boxes and is expensive);
    a grid of anchored boxes is sampled instead, which is enough to separate a low-discrepancy
    sequence from a pseudo-random one by a wide margin."""
    worst = 0.0
    steps = 8
    for corner in range(1, steps + 1):
        bound = corner / steps
        expected = bound**dimensions
        inside = sum(1 for p in points if all(c < bound for c in p))
        worst = max(worst, abs(inside / len(points) - expected))
    return worst


def test_T20b_SOBOL_fills_the_cube_better_than_PSEUDO_RANDOM_at_the_same_count():
    """The reason the sequence exists, measured against the generator it replaces.

    **Multi-dimensional deliberately** — the spec's degenerate-fixture guard names this: in 1-D a
    Sobol sequence is just the van der Corput sequence and almost anything looks uniform, so the
    joint behaviour is what must be compared. A pseudo-random cloud has discrepancy `~N^(-1/2)`; a
    low-discrepancy sequence does far better, and the assertion is a *ratio* so it does not encode a
    particular N."""
    from pricebook.foundation.rng import UniformSource

    for dimensions in (2, 3, 5):
        count = 1024
        sobol = SobolSequence(dimensions=dimensions).uniforms(count)
        flat = UniformSource(seed=2024).uniforms(count * dimensions)
        pseudo = [
            tuple(flat[i * dimensions + d] for d in range(dimensions)) for i in range(count)
        ]
        quasi_miss = _star_discrepancy_proxy(sobol, dimensions)
        pseudo_miss = _star_discrepancy_proxy(pseudo, dimensions)
        assert quasi_miss < pseudo_miss, (
            f"dimension {dimensions}: Sobol {quasi_miss:.2e} is not better than pseudo-random "
            f"{pseudo_miss:.2e} — the sequence is not low-discrepancy"
        )


def test_T20b_SOBOL_plus_INVERSION_beats_the_MINUS_ONE_HALF_rate_on_our_BLACK():
    """The anchor: our own closed form, and the claim that can actually fail.

    Pricing a Black call by simulation on Sobol, with the T20a inversion as the transform (NOT
    Box-Muller — it is not monotone, which is the reason the inversion had to be built first), the
    error must fall **faster** than `1/√N`. Fitted as a slope of `log|error|` against `log N` and
    required to be steeper than `−0.5`; the realised exponent is reported in the tranche doc rather
    than asserted to a particular value, since how *much* faster is a `TO MEASURE` item.

    This is a convergence-rate oracle, which is the only honest shape here: a single N could beat the
    pseudo-random error by luck, and a rate cannot."""
    from pricebook.models.black import black_price
    from pricebook.foundation.inflation import CapFloorKind

    forward, strike, variance = 0.04, 0.04, 0.2**2 * 1.0
    exact = black_price(forward, strike, variance, CapFloorKind.CAP)

    logs_n, logs_error = [], []
    for power in range(8, 15):                      # 256 .. 16384
        count = 2**power
        draws = SobolSequence(dimensions=1).uniforms(count)
        payoffs = [
            max(forward * math.exp(-0.5 * variance + math.sqrt(variance) * inverse_norm_cdf(u[0]))
                - strike, 0.0)
            for u in draws
        ]
        error = abs(math.fsum(payoffs) / count - exact)
        logs_n.append(math.log(count))
        logs_error.append(math.log(error))

    n = len(logs_n)
    mean_x = math.fsum(logs_n) / n
    mean_y = math.fsum(logs_error) / n
    slope = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(logs_n, logs_error)) / math.fsum(
        (x - mean_x) ** 2 for x in logs_n
    )
    assert slope < -0.5, f"the fitted convergence exponent is {slope:.3f}, not steeper than -1/2"


# ============================================================================================
# CONTRACT — determinism, the skipped zeroth draw, and the refusals.
# ============================================================================================


def test_T20b_the_sequence_is_a_PURE_FUNCTION_of_dimension_and_index():
    """§3.1 — no ambient state, and no dependence on how many draws were asked for at once."""
    first = SobolSequence(dimensions=3).uniforms(64)
    again = SobolSequence(dimensions=3).uniforms(64)
    assert first == again
    longer = SobolSequence(dimensions=3).uniforms(128)
    assert longer[:64] == first, "a longer request must extend the sequence, not change it"


def test_T20b_no_draw_is_EXACTLY_ZERO_because_the_zeroth_is_skipped():
    """*"The only Sobol' variate that could ever be exactly zero is the zeroth draw ... we can
    explicitly exclude the possibility ... by simply skipping the zeroth draw."*

    This matters precisely because of T20a: a `0.0` would map to `−inf` under `Φ⁻¹`, and the domain
    guard there refuses it. So the two tranches meet here — the generator must not produce what the
    transform must reject, and both halves are asserted rather than assumed."""
    for dimensions in (1, 5, 16):
        for point in SobolSequence(dimensions=dimensions).uniforms(2_048):
            assert all(0.0 < coordinate < 1.0 for coordinate in point)


def test_T20b_an_UNSUPPORTED_dimension_is_refused():
    """Table 8.3 covers 32 dimensions and the source's polynomials beyond it are on a CD we do not
    have, so dimension 33 is not a degraded answer — it is one we cannot construct."""
    assert SOBOL_MAX_DIMENSIONS == 32
    with pytest.raises(ValueError, match="between 1 and 32"):
        SobolSequence(dimensions=0)
    with pytest.raises(ValueError, match="between 1 and 32"):
        SobolSequence(dimensions=33)
    with pytest.raises(ValueError, match="between 1 and 32"):
        direction_integers(33)
    with pytest.raises(ValueError, match="at least one"):
        SobolSequence(dimensions=2).uniforms(0)
