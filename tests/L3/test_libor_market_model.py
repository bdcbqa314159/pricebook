"""The LIBOR market model's drift (L3). JAECKEL-2002 eq 12.8.

Transcribed from BOTH renderings, which agree:

    mu_i = { -sigma_i * SUM_{k=i+1}^{N-1} (f_k tau_k)/(1 + f_k tau_k) sigma_k rho_ik   for i <  N-1
           {  0                                                                        for i =  N-1
           { +sigma_i * SUM_{k=N}^{i}     (f_k tau_k)/(1 + f_k tau_k) sigma_k rho_ik   for i >= N

`N` indexes the numeraire bond `P_N`, i.e. whose forward measure we are in. Three things make this
formula easy to get wrong and each has its own oracle: the **sign flips** between the branches, the
**summation bounds differ** (`i+1..N-1` upward versus `N..i` downward), and the middle branch is
**exactly zero**.

**That middle branch is the anchor, and it is exact rather than statistical.** `mu_{N-1} = 0` says the
forward spanning the numeraire's own period is a martingale under that measure — which is the finance,
not the formula, so it survives a transcription error the way a requirement oracle should. Its
mandatory companion is that a NON-terminal forward's drift is non-zero and changes with the state:
a `return 0.0` stub would satisfy the anchor perfectly.

**The state dependence is the other half of what makes this a market model.** Jaeckel: *"the drift
terms given by equation (12.8) are clearly state-dependent and thus indirectly stochastic"* — so the
drift must move when the forwards move, and a drift computed from `f(0)` once and frozen would be a
different model. Oracled.
"""

import math
from datetime import date

import pytest

from pricebook.models.libor_market_model import LiborMarketModel, LmmCorrelation, LmmTenor

# Four spanning forwards, DISTINCT rates and vols so a mis-indexed sum cannot hide.
TENOR = LmmTenor(
    accruals=(0.5, 0.5, 0.5, 0.5),
    forwards=(0.030, 0.034, 0.038, 0.042),
    volatilities=(0.22, 0.20, 0.18, 0.16),
)
RHO = 0.80


def _reference_drift(i: int, numeraire: int, tenor: LmmTenor = TENOR, rho: float = RHO) -> float:
    """Eq 12.8 transcribed independently, with a flat correlation `rho_ik = rho` for `i != k`."""
    f, tau, sigma = tenor.forwards, tenor.accruals, tenor.volatilities

    def term(k: int) -> float:
        correlation = 1.0 if k == i else rho
        return (f[k] * tau[k]) / (1.0 + f[k] * tau[k]) * sigma[k] * correlation

    if i < numeraire - 1:
        return -sigma[i] * math.fsum(term(k) for k in range(i + 1, numeraire))
    if i == numeraire - 1:
        return 0.0
    return sigma[i] * math.fsum(term(k) for k in range(numeraire, i + 1))


def _model(numeraire: int) -> LiborMarketModel:
    return LiborMarketModel(tenor=TENOR, correlation=RHO, numeraire_index=numeraire)


# ============================================================================================
# THE ANCHOR — the terminal forward is a martingale under its own measure. EXACT.
# ============================================================================================

@pytest.mark.parametrize("numeraire", [1, 2, 3, 4])
def test_the_forward_spanning_the_NUMERAIRE_period_has_EXACTLY_zero_drift(numeraire):
    """`mu_{N-1} = 0` — the finance, not the formula. A requirement oracle that survives a
    transcription error in either summation branch."""
    assert _model(numeraire).drift(numeraire - 1) == 0.0


def test_a_NON_terminal_drift_is_NON_zero_so_the_anchor_is_not_vacuous():
    """The mandatory companion: a `return 0.0` stub satisfies the anchor at every numeraire."""
    model = _model(numeraire=2)
    assert model.drift(0) != 0.0
    assert model.drift(2) != 0.0
    assert model.drift(3) != 0.0


# ============================================================================================
# The pointwise transcription, and the three things it pins.
# ============================================================================================

@pytest.mark.parametrize("numeraire", [1, 2, 3, 4])
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_the_drift_matches_an_independent_transcription_of_eq_12_8(i, numeraire):
    assert _model(numeraire).drift(i) == pytest.approx(
        _reference_drift(i, numeraire), rel=1e-14, abs=1e-18
    )


def test_the_SIGN_flips_either_side_of_the_numeraire():
    """Forwards below the numeraire drift one way and those above the other — the sign is the first
    thing a transcription loses, and a single-branch implementation would fail here."""
    model = _model(numeraire=2)
    assert model.drift(0) < 0.0, "a forward below the numeraire must drift negatively"
    assert model.drift(2) > 0.0, "a forward above the numeraire must drift positively"
    assert model.drift(3) > 0.0


def test_the_SUMMATION_BOUNDS_differ_between_the_branches():
    """`i+1..N-1` upward against `N..i` downward. With the numeraire at 2, forward 0's drift sums
    only `k=1`, and forward 3's sums `k=2,3` — so the two have different term counts and swapping
    the bounds changes both."""
    model = _model(numeraire=2)
    f, tau, sigma = TENOR.forwards, TENOR.accruals, TENOR.volatilities

    def term(k: int, i: int) -> float:
        rho = 1.0 if k == i else RHO
        return (f[k] * tau[k]) / (1.0 + f[k] * tau[k]) * sigma[k] * rho

    assert model.drift(0) == pytest.approx(-sigma[0] * term(1, 0), rel=1e-14)
    assert model.drift(3) == pytest.approx(
        sigma[3] * (term(2, 3) + term(3, 3)), rel=1e-14
    )


def test_the_SELF_correlation_term_is_UNITY_not_rho():
    """In the `i >= N` branch the sum includes `k = i`, where `rho_ii = 1`. Using `rho` there would
    be a plausible-looking error that only a case with `k = i` in range can catch."""
    model = _model(numeraire=3)
    f, tau, sigma = TENOR.forwards, TENOR.accruals, TENOR.volatilities
    own = (f[3] * tau[3]) / (1.0 + f[3] * tau[3]) * sigma[3] * 1.0
    with_rho = (f[3] * tau[3]) / (1.0 + f[3] * tau[3]) * sigma[3] * RHO
    assert model.drift(3) == pytest.approx(sigma[3] * own, rel=1e-14)
    assert model.drift(3) != pytest.approx(sigma[3] * with_rho, rel=1e-6)


# ============================================================================================
# State dependence — Jaeckel's "indirectly stochastic".
# ============================================================================================

def test_the_drift_MOVES_when_the_FORWARDS_move():
    """*"The drift terms given by equation (12.8) are clearly state-dependent and thus indirectly
    stochastic."* A drift frozen at `f(0)` would be a different model."""
    base = _model(numeraire=2).drift(3)
    shifted = LiborMarketModel(
        tenor=LmmTenor(TENOR.accruals, (0.030, 0.034, 0.060, 0.070), TENOR.volatilities),
        correlation=RHO,
        numeraire_index=2,
    ).drift(3)
    assert shifted != pytest.approx(base, rel=1e-9)
    assert abs(shifted) > abs(base), "higher forwards must give a larger drift magnitude"


def test_the_drift_vanishes_as_the_FORWARDS_go_to_zero():
    """An exact reduction: every term carries `f_k tau_k / (1 + f_k tau_k)`, which is zero at
    `f_k = 0`. So a zero yield curve has no drift at all, whatever the vols."""
    flat = LiborMarketModel(
        tenor=LmmTenor(TENOR.accruals, (0.0, 0.0, 0.0, 0.0), TENOR.volatilities),
        correlation=RHO,
        numeraire_index=2,
    )
    for i in range(4):
        assert flat.drift(i) == 0.0


def test_the_drift_vanishes_at_ZERO_volatility():
    """The other exact reduction — `sigma_i` is an overall factor on both branches."""
    zero_vol = LiborMarketModel(
        tenor=LmmTenor(TENOR.accruals, TENOR.forwards, (0.0, 0.0, 0.0, 0.0)),
        correlation=RHO,
        numeraire_index=2,
    )
    for i in range(4):
        assert zero_vol.drift(i) == 0.0


# ============================================================================================
# Construction contracts.
# ============================================================================================

def test_mismatched_tenor_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        LmmTenor(accruals=(0.5, 0.5), forwards=(0.03,), volatilities=(0.2, 0.2))


def test_a_NON_POSITIVE_accrual_is_refused():
    with pytest.raises(ValueError, match="positive"):
        LmmTenor(accruals=(0.5, 0.0), forwards=(0.03, 0.03), volatilities=(0.2, 0.2))


def test_a_NEGATIVE_volatility_is_refused():
    with pytest.raises(ValueError, match="negative"):
        LmmTenor(accruals=(0.5,), forwards=(0.03,), volatilities=(-0.1,))


def test_an_OUT_OF_RANGE_correlation_is_refused():
    with pytest.raises(ValueError, match="must lie"):
        LiborMarketModel(tenor=TENOR, correlation=1.5, numeraire_index=2)


def test_a_BAD_numeraire_index_is_refused():
    with pytest.raises(ValueError, match="numeraire"):
        LiborMarketModel(tenor=TENOR, correlation=RHO, numeraire_index=0)
    with pytest.raises(ValueError, match="numeraire"):
        LiborMarketModel(tenor=TENOR, correlation=RHO, numeraire_index=5)


def test_a_BAD_forward_index_is_refused():
    with pytest.raises(IndexError, match="forward"):
        _model(numeraire=2).drift(4)


# ---------------------------------------------------------------------------------------------
# Tranche 17 — the correlation becomes a MATRIX. T16 shipped a scalar `ρ` and said so in the
# module docstring: *"a real correlation structure arrives with the unit that needs one (a swaption
# fit discriminates correlation shapes; a caplet strip does not)"*. T17 is that unit — its spec's
# degenerate-fixture guard requires non-uniform `ρ_ij`, because a FLAT correlation makes the
# INDEXING of `ρ` invisible: swap `ρ_ij` for `ρ_ji`, or read the wrong pair entirely, and a flat
# matrix returns the same number. That is the recurring trap, so the fixtures below are non-uniform.
# ---------------------------------------------------------------------------------------------


def _reference_drift_matrix(
    i: int, numeraire: int, rho: tuple[tuple[float, ...], ...], tenor: LmmTenor = TENOR
) -> float:
    """Eq 12.8 transcribed independently again, now reading `ρ_ik` from a full MATRIX.

    Separate from `_reference_drift` on purpose: that one hard-codes the flat `1.0 if k == i else ρ`
    rule, which is exactly the assumption under test here. Reusing it would let the flat rule into
    both sides of the comparison."""
    f, tau, sigma = tenor.forwards, tenor.accruals, tenor.volatilities

    def term(k: int) -> float:
        return (f[k] * tau[k]) / (1.0 + f[k] * tau[k]) * sigma[k] * rho[i][k]

    if i < numeraire - 1:
        return -sigma[i] * math.fsum(term(k) for k in range(i + 1, numeraire))
    if i == numeraire - 1:
        return 0.0
    return sigma[i] * math.fsum(term(k) for k in range(numeraire, i + 1))


def _decaying(size: int, beta: float = 0.35) -> LmmCorrelation:
    """Jäckel eq 12.12's shape with `γ = 1`: `ρ_ij = exp(−β|t_i − t_j|)`. Non-uniform by
    construction, which is the point — every off-diagonal entry differs from every other."""
    return LmmCorrelation(
        tuple(tuple(math.exp(-beta * abs(i - j)) for j in range(size)) for i in range(size))
    )


def test_T17_a_flat_correlation_reproduces_the_scalar_case_BIT_IDENTICALLY():
    """The widening is a PURE REFACTOR in the flat case, and bit-identity is the available proof.

    `LmmCorrelation.flat` must put exactly `ρ` off the diagonal and exactly `1.0` on it — the same
    floats the scalar branch produced, consumed in the same order — so the drift's operations are
    not merely equivalent but identical. §4 permits the exact-arithmetic-identity substitute only
    when a refactor reorders the operations; this one does not, so the stronger proof is owed."""
    flat = LmmCorrelation.flat(len(TENOR), 0.80)
    for i in range(len(TENOR)):
        for j in range(len(TENOR)):
            assert flat.matrix[i][j] == (1.0 if i == j else 0.80)

    for numeraire in range(1, len(TENOR) + 1):
        model = LiborMarketModel(TENOR, flat, numeraire_index=numeraire)
        for index in range(len(TENOR)):
            # The reference values this file already pins for the scalar case, recomputed here from
            # eq 12.8 — not read back from the model (§4 independence).
            assert model.drift(index) == _reference_drift(index, numeraire, rho=0.80)


def test_T17_the_matrix_is_REFUSED_unless_it_is_a_correlation_matrix():
    with pytest.raises(ValueError, match="square"):
        LmmCorrelation(((1.0, 0.5),))
    with pytest.raises(ValueError, match="symmetric"):
        LmmCorrelation(((1.0, 0.5), (0.4, 1.0)))
    with pytest.raises(ValueError, match="unit diagonal"):
        LmmCorrelation(((0.9, 0.5), (0.5, 1.0)))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        LmmCorrelation(((1.0, 1.5), (1.5, 1.0)))
    with pytest.raises(ValueError, match="at least one"):
        LmmCorrelation(())


def test_T17_a_NON_UNIFORM_correlation_is_read_at_the_RIGHT_pair():
    """The oracle a flat matrix cannot provide: that `ρ_ik` is indexed correctly.

    Under a decaying correlation every pair differs, so reading `ρ_ji` for `ρ_ij`, or `ρ_i,k+1` for
    `ρ_ik`, changes the drift. The reference transcribes eq 12.8 with the matrix explicitly."""
    rho = _decaying(len(TENOR))
    model = LiborMarketModel(TENOR, rho, numeraire_index=2)
    for index in range(len(TENOR)):
        expected = _reference_drift_matrix(index, 2, rho.matrix)
        assert model.drift(index) == pytest.approx(expected, rel=1e-13)

    # Non-degeneracy asserted as a PREMISE: if the fixture were flat this test would be vacuous.
    off_diagonal = {rho.matrix[i][j] for i in range(len(TENOR)) for j in range(len(TENOR)) if i != j}
    assert len(off_diagonal) > 1, "a flat fixture cannot see the indexing — the trap this guards"
