"""Sobol' low-discrepancy numbers (L0). JAECKEL-2002 §8.3.

A low-discrepancy sequence fills the unit hypercube more evenly than pseudo-random draws, so an
integral estimated on it converges **faster than `1/√N`**. That is the whole reason it is here, and
it is the one claim the oracle fits a rate to rather than asserting at a single `N`.

The construction, as the source prints it:

    p_k(z) = Σ_j a_kj z^(g_k − j)                                                      (8.18)
    v_kl   = (v_k(l−g_k) >> g_k) ⊕₂ [ ⊕₂_{j=1..g_k} a_kj · v_k(l−j) ]     for l > g_k   (8.19)
    x_nk   = ⊕₂_j v_kj · 1{jth bit from the right of γ(n) is set}                       (8.20)
    y_nk   = x_nk / 2^b                                                                 (8.17)
    G(n)   = n ⊕₂ ⌊n/2⌋                                                                 (8.22)

**Why the inverse CDF and not Box-Muller.** The evenness is a property of the *cube*, and it survives
into normal variates only under a **monotone** transform. Box-Muller consumes two uniforms through a
sine and a square root and scrambles the ordering, destroying the property the sequence is adopted
for. That is why tranche 20a built `inverse_norm_cdf` first; this module is its consumer.

**Table 8.3 is ~350 transcribed numbers, extracted twice and reconciled mechanically.** Thirty-two
dimensions, each with a degree, a primitive polynomial in binary, and ten initialisation numbers.
Reading that by eye twice would be the *weaker* method, not the stronger one — so it was parsed from
both `pdftotext -layout` and `pdftotext -raw`, which disagreed on **nothing** across all 32 rows, and
every exponent was confirmed to equal `32 − l`.

**What makes the table checkable is that the source states a property of it.** Its caption is
*"Initialisation numbers providing property A up to dimension 32"*, and property A is sharp: in any
binary segment of length `2^d` aligned to a multiple of `2^d`, each of the `2^d` sub-hypercubes
contains **exactly one** point. One mistyped direction integer breaks it. Unit initialisation
(`v_kl = 2^(b−l)`, eq 8.24) would also be *valid* yet the source warns it *"leads to surprisingly bad
results ... and should be avoided"* — so the table is not a convenience, it is the content.

**The zeroth draw is skipped**, because it is the only draw that can be exactly zero in every
dimension — and `inverse_norm_cdf` refuses `0.0`, mapping as it would to `−inf`. The generator must
not produce what its transform must reject.

**What the source does not print:** primitive polynomials beyond this table — *"I provide a list of
all primitive polynomials modulo two up to degree 27 on the accompanying CD"*, which we do not have.
Third time this book has sent a reader to something it does not contain (after `Φ⁻¹` and the
control-variate variance factor). It costs nothing at 32 dimensions; it is the reason there is no
33rd.

Provenance:
  stock:  _stock/quarry/pricebook/numerical/_qmc.py — read for SHAPE only; the direction numbers,
          the recurrence and the Gray code are read from the source (§5)
  source: JAECKEL-2002 (docs/references.md) §8.3 — eq 8.18 (the polynomial), eq 8.19 (the direction-
          integer recurrence), eq 8.20 (the draw), eq 8.17 (the scaling), eq 8.22 (the Gray code),
          Table 8.2 (Gray codes 0..7, printed) and Table 8.3 (initialisation numbers, property A to
          dimension 32). Table 8.3 parsed from `-layout` AND `-raw` and reconciled: zero
          disagreements across 32 rows
  oracle: tests/L0/test_sobol.py — the Gray code against the source's printed Table 8.2 plus the
          one-bit-changes property; the initialisation numbers against an independently transcribed
          copy of Table 8.3; the source's TWO bit constraints on all 32x32 direction integers
          (which is what makes the recurrence trustworthy where the source prints nothing); eq 8.19
          pointwise; PROPERTY A over several aligned segments up to dimension 8 (the requirement
          oracle that pins the whole table); discrepancy better than pseudo-random in 2/3/5
          dimensions; and a FITTED convergence exponent steeper than -1/2 against our `black_price`
  crosscheck: quantlib SobolRsg -- deferred to the optional QuantLib tier
  slice:  tranche 20b -- Sobol / low-discrepancy numbers
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bits in the integer representation the sequence is built on — the source's `b`, *"typically 32"*.
_BITS = 32

#: How many dimensions Table 8.3 covers. Not a soft limit: beyond it the source's polynomials are on
#: a CD we do not hold, so a 33rd dimension is one we cannot construct rather than one we decline.
SOBOL_MAX_DIMENSIONS = 32

#: Table 8.3, as `(g_k, polynomial bits, m_l for l = 1..10)` with `v_kl = m_l · 2^(32−l)`.
#: Machine-extracted from two renderings and reconciled; see the module note.
_TABLE_8_3: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (0, 0b1, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
    (1, 0b11, (1, 3, 5, 15, 17, 51, 85, 255, 257, 771)),
    (2, 0b111, (1, 1, 7, 11, 13, 61, 67, 79, 465, 721)),
    (3, 0b1011, (1, 3, 7, 5, 7, 43, 49, 147, 439, 1013)),
    (3, 0b1101, (1, 1, 5, 3, 15, 51, 125, 141, 177, 759)),
    (4, 0b10011, (1, 3, 1, 1, 9, 59, 25, 89, 321, 835)),
    (4, 0b11001, (1, 1, 3, 7, 31, 47, 109, 173, 181, 949)),
    (5, 0b100101, (1, 3, 3, 9, 9, 57, 43, 43, 225, 113)),
    (5, 0b101001, (1, 3, 7, 7, 21, 61, 55, 19, 59, 761)),
    (5, 0b101111, (1, 1, 5, 11, 27, 53, 69, 25, 103, 615)),
    (5, 0b110111, (1, 1, 7, 3, 29, 51, 47, 97, 233, 39)),
    (5, 0b111011, (1, 3, 7, 13, 3, 35, 89, 9, 235, 929)),
    (5, 0b111101, (1, 3, 5, 1, 15, 19, 113, 115, 411, 157)),
    (6, 0b1000011, (1, 1, 1, 9, 23, 37, 97, 97, 353, 169)),
    (6, 0b1011011, (1, 1, 3, 13, 11, 7, 37, 101, 463, 657)),
    (6, 0b1100001, (1, 3, 3, 5, 19, 33, 3, 197, 329, 983)),
    (6, 0b1100111, (1, 1, 7, 13, 25, 5, 27, 71, 377, 719)),
    (6, 0b1101101, (1, 1, 1, 3, 13, 39, 7, 23, 391, 389)),
    (6, 0b1110011, (1, 3, 5, 11, 7, 11, 43, 25, 187, 825)),
    (7, 0b10000011, (1, 3, 1, 7, 3, 23, 79, 65, 451, 321)),
    (7, 0b10001001, (1, 3, 1, 15, 17, 63, 13, 113, 147, 881)),
    (7, 0b10001111, (1, 3, 3, 3, 25, 17, 115, 17, 179, 883)),
    (7, 0b10010001, (1, 3, 7, 9, 31, 29, 17, 121, 363, 783)),
    (7, 0b10011101, (1, 1, 3, 15, 29, 15, 41, 249, 201, 923)),
    (7, 0b10100111, (1, 3, 1, 9, 5, 21, 119, 53, 319, 693)),
    (7, 0b10101011, (1, 1, 5, 5, 1, 27, 33, 253, 341, 385)),
    (7, 0b10111001, (1, 1, 3, 1, 23, 13, 75, 29, 181, 895)),
    (7, 0b10111111, (1, 1, 7, 7, 19, 25, 105, 173, 509, 75)),
    (7, 0b11000001, (1, 3, 5, 5, 21, 9, 7, 143, 157, 959)),
    (7, 0b11001011, (1, 1, 1, 15, 5, 49, 59, 71, 31, 111)),
    (7, 0b11010011, (1, 3, 5, 15, 17, 19, 21, 227, 413, 727)),
    (7, 0b11010101, (1, 1, 7, 11, 13, 29, 3, 15, 279, 17)),
)


def gray_code(n: int) -> int:
    """`G(n) = n ⊕₂ ⌊n/2⌋` — eq 8.22.

    Antonov and Saleev's contribution: exactly one bit changes between `G(n)` and `G(n+1)`, so each
    successive draw costs **one** XOR instead of one per set bit. The sequence of *points* is the
    same set as plain `γ(n) = n` would give; only the order and the cost differ."""
    if n < 0:
        raise ValueError(f"a Gray code index must be non-negative, got {n}")
    return n ^ (n >> 1)


def direction_integers(dimension: int) -> tuple[int, ...]:
    """The 32 direction integers `v_kl` for a dimension — Table 8.3 then eq 8.19 beyond it.

    `dimension` is the source's `k`, counted from 1. The first `g_k` come from the table; the rest
    are generated by the recurrence, which is where the two bit constraints earn their keep: the
    source prints only ten per dimension, so everything past `l = 10` is verified structurally
    rather than against a printed value.

    Note the two subtleties the source flags and a careless reading drops: `a_k0` is **not used** in
    the recurrence (the degree implies it), and `a_k g_k` is always set, so `v_k(l−g_k)` always
    enters."""
    if not 1 <= dimension <= SOBOL_MAX_DIMENSIONS:
        raise ValueError(
            f"Sobol dimensions must lie between 1 and {SOBOL_MAX_DIMENSIONS} (Table 8.3's reach), "
            f"got {dimension}"
        )
    degree, polynomial, multipliers = _TABLE_8_3[dimension - 1]

    # **Dimension 1 is the degenerate case, and the source's own structure says so.** Its polynomial
    # is `p_1(z) = 1`, of degree 0, so there are ZERO freely-chosen initialisation numbers and eq
    # 8.19's recurrence reads `v_1l = (v_1l >> 0)` — it references itself and says nothing. What is
    # left is `v_1l = 2^(b−l)` for every `l`, i.e. unit initialisation, which is precisely what Table
    # 8.3 prints for `k = 1` (every multiplier is 1). That makes dimension 1 the van der Corput /
    # radical-inverse sequence in base 2, as it should be — not a special case bolted on, but the
    # construction evaluated at `g_k = 0`.
    if degree == 0:
        return tuple(2 ** (_BITS - l) for l in range(1, _BITS + 1))

    # The table's own initialisation numbers, scaled as eq 8.24's form: v_kl = m_l * 2^(b-l).
    values = [multipliers[index] * 2 ** (_BITS - index - 1) for index in range(len(multipliers))]

    # The polynomial's bits a_k0..a_k g_k, most significant first; a_k0 is dropped on purpose.
    coefficients = [(polynomial >> (degree - j)) & 1 for j in range(degree + 1)]

    for l in range(len(values) + 1, _BITS + 1):          # 1-based l, continuing past the table
        value = values[l - degree - 1] >> degree
        for j in range(1, degree + 1):
            if coefficients[j]:
                value ^= values[l - j - 1]
        values.append(value)
    return tuple(values[:_BITS])


@dataclass(frozen=True)
class SobolSequence:
    """A Sobol' sequence in `dimensions` dimensions — a pure function of dimension and index.

    Frozen and stateless: `uniforms(count)` always returns the same prefix, so asking for more draws
    extends the sequence rather than changing it. That is §3.1 for a generator, and it is what lets a
    convergence oracle compare `N` and `4N` on the *same* sequence rather than two samples of it."""

    dimensions: int

    def __post_init__(self) -> None:
        if not 1 <= self.dimensions <= SOBOL_MAX_DIMENSIONS:
            raise ValueError(
                f"a Sobol sequence needs between 1 and {SOBOL_MAX_DIMENSIONS} dimensions, got "
                f"{self.dimensions}"
            )

    def uniforms(self, count: int) -> list[tuple[float, ...]]:
        """`count` points in `(0,1)^d` — eqs 8.20 and 8.17, skipping the zeroth draw.

        Built incrementally off the Gray code: between consecutive draws exactly one bit of `γ(n)`
        changes, so each coordinate is updated by a single XOR against the direction integer for
        that bit position. The running integers are carried rather than recomputed, which is the
        point of Antonov-Saleev."""
        if count < 1:
            raise ValueError(f"a Sobol sequence needs at least one draw, got {count}")

        table = [direction_integers(k) for k in range(1, self.dimensions + 1)]
        scale = 1.0 / 2**_BITS
        running = [0] * self.dimensions
        points: list[tuple[float, ...]] = []
        previous = 0
        for n in range(1, count + 1):
            code = gray_code(n)
            # The single bit that changed between G(n-1) and G(n) names the direction integer to XOR.
            bit = (code ^ previous).bit_length() - 1
            for k in range(self.dimensions):
                running[k] ^= table[k][bit]
            points.append(tuple(value * scale for value in running))
            previous = code
        return points
