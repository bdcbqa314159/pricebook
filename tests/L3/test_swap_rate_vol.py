"""Rebonato's swaption-volatility approximation (L3). MCINERNEY-ZASTAWNIAK eq 5.40.

    v²_{0,n} = 1/(T₀−t) · Σ_{i,j} [w_i(t)w_j(t)F_i(t)F_j(t) / S_{0,n}(t)²] ∫_t^{T₀} σ_iσ_jρ_ij ds

with the weights `w_i = τ_i B(t,T_i)/A_{0,n}`, the annuity `A_{0,n} = Σ_k τ_k B(t,T_k)` (5.32) and
the forward swap rate `S_{0,n} = (B(t,T₀) − B(t,T_n))/A_{0,n}` (5.31). Our `σ_i` are constant, so
the integral is `σ_iσ_jρ_ij(T₀−t)` and the `1/(T₀−t)` cancels.

**Why this is an approximation at all, and the one thing it must not be dressed up as.** McInerney
§5.11 proves the LMM swap rate is NOT lognormal — `σ_{0,n}` is random — so a European swaption has no
exact LMM closed form. Eq 5.40 freezes two sets of martingales (`w_iF_i` and `S`) at their time-`t`
values. So oracle 3 is a *bounded* check whose error is measured and reported, never a tuned
tolerance (the MM-I freezing lesson).

**The exact reduction is what gives this file a closed-form anchor.** For a ONE-period swap the
weight is exactly 1, the swap rate is exactly the forward, and `v = σ_i` exactly — so a one-period
swaption is a caplet, which T16 already reprices to our Black. That is checked here to machine
precision, not to a tolerance.
"""

import math

import pytest

from pricebook.models.libor_market_model import LiborMarketModel, LmmCorrelation, LmmTenor
from pricebook.models.swap_rate_vol import (
    forward_swap_rate,
    rebonato_swaption_vol,
    swap_annuity,
    swap_rate_weights,
    tenor_discount_factors,
)

# **NON-UNIFORM accruals and a correlation built on real TIMES, and both are deliberate.** Three
# degeneracies have to be excluded here, not two:
#   * `τ = 1` makes the accrual invisible (five prior defects in this project), so no `τ` is 1;
#   * a FLAT `τ` makes the span OFFSET invisible — `accruals[:count]` and `accruals[first:first+count]`
#     are then the same slice, and a mutation dropping the offset survives;
#   * a correlation `exp(−β|i−j|)` is non-uniform yet TRANSLATION-INVARIANT, so `ρ[first+m][first+l]`
#     equals `ρ[m][l]` and the span offset is invisible there too. Jäckel eq 12.12 writes the decay in
#     the actual times `t_i`, not in index distance; with non-uniform accruals that is no longer
#     translation-invariant, which is both the faithful reading and the non-degenerate one.
# All three were found by mutation, not by inspection — two mutants survived the whole suite first.
TENOR = LmmTenor(
    accruals=(0.50, 0.25, 0.50, 0.75, 0.50, 0.25),
    forwards=(0.030, 0.034, 0.038, 0.042, 0.045, 0.047),
    volatilities=(0.22, 0.20, 0.18, 0.16, 0.15, 0.14),
)
#: `T_i`, the cumulative accruals — the times Jäckel's eq 12.12 decays in.
TIMES = tuple(math.fsum(TENOR.accruals[:i]) for i in range(len(TENOR)))
RHO = LmmCorrelation(
    tuple(tuple(math.exp(-0.35 * abs(TIMES[i] - TIMES[j])) for j in range(6)) for i in range(6))
)


def _model(first: int, count: int) -> LiborMarketModel:
    return LiborMarketModel(TENOR, RHO, numeraire_index=first + count)


def _reference_pieces(first: int, count: int):
    """`B`, `A`, `S` and `w` transcribed here from eqs 5.31/5.32 — the test's own arithmetic."""
    tau, f = TENOR.accruals, TENOR.forwards
    bonds = [1.0]
    for i in range(len(TENOR)):
        bonds.append(bonds[-1] / (1.0 + f[i] * tau[i]))
    annuity = math.fsum(tau[first + m] * bonds[first + m + 1] for m in range(count))
    swap_rate = (bonds[first] - bonds[first + count]) / annuity
    weights = tuple(tau[first + m] * bonds[first + m + 1] / annuity for m in range(count))
    return bonds, annuity, swap_rate, weights


# ============================================================================================
# THE EXACT REDUCTION — a one-period swap's rate IS the forward, so its vol IS that forward's.
# ============================================================================================


@pytest.mark.parametrize("first", [0, 1, 2, 3, 4, 5])
def test_O1_a_ONE_period_swap_rate_is_EXACTLY_the_forward_and_its_vol_EXACTLY_the_forward_vol(first):
    """The exact anchor, and it is the finance rather than the formula.

    For `count = 1` the annuity is the single `τ_i B(T_{i+1})`, so `w = 1`; and
    `B(T_i) − B(T_{i+1}) = τ_i f_i B(T_{i+1})` identically, so `S = f_i`. Then `Z = wF/S = 1` and
    `v² = σ_i²ρ_ii = σ_i²`. Exact by construction at every index — which is what makes T16's caplet
    the exact anchor for the swaption engine, rather than a numerical coincidence."""
    assert forward_swap_rate(TENOR, first, 1) == pytest.approx(TENOR.forwards[first], rel=1e-15)
    assert swap_rate_weights(TENOR, first, 1) == pytest.approx((1.0,), rel=1e-15)
    assert rebonato_swaption_vol(_model(first, 1), first, 1) == pytest.approx(
        TENOR.volatilities[first], rel=1e-15
    )


# ============================================================================================
# REQUIREMENT ORACLES — exact identities the weights must satisfy for ALL parameters.
# ============================================================================================


@pytest.mark.parametrize(("first", "count"), [(0, 6), (0, 3), (1, 4), (2, 4), (3, 3), (4, 2)])
def test_O4_the_weights_REPRODUCE_the_swap_rate_exactly(first, count):
    """`Σ_i w_i F_i = S` to machine precision — eq 1.16, and not a tolerance.

    The identity is `τ_i F_i B(T_i) = B(T_{i-1}) − B(T_i)`, which telescopes to the numerator of
    5.31. It is a **requirement** oracle: it is derived from the finance, so it survives a
    transcription error in 5.40 that a source-value check would share. It is also the sharpest
    single check on the weights, which is where the magnitude risk lives (T16's lesson)."""
    weights = swap_rate_weights(TENOR, first, count)
    rebuilt = math.fsum(weights[m] * TENOR.forwards[first + m] for m in range(count))
    assert rebuilt == pytest.approx(forward_swap_rate(TENOR, first, count), rel=1e-14)


@pytest.mark.parametrize(("first", "count"), [(0, 6), (1, 4), (2, 4), (3, 3)])
def test_O4_the_normalised_weights_SUM_TO_ONE_exactly(first, count):
    """`Σ_i w_i F_i / S = 1` — the previous identity divided by `S`.

    Worth asserting separately because `w_iF_i/S` is precisely the `Z_i` that enters 5.40
    quadratically, so this pins the vector 5.40 actually consumes. A weight scaled by a constant
    `c` would leave oracle `Σ w_i F_i = S` red, but a weight vector that is *individually* wrong
    while summing correctly is caught only by the pointwise transcription below — which is why both
    are here (§4: a requirement never replaces the pointwise check)."""
    weights = swap_rate_weights(TENOR, first, count)
    swap_rate = forward_swap_rate(TENOR, first, count)
    normalised = [weights[m] * TENOR.forwards[first + m] / swap_rate for m in range(count)]
    assert math.fsum(normalised) == pytest.approx(1.0, rel=1e-14)


def test_O4_the_annuity_and_discount_factors_match_their_transcription():
    bonds, annuity, swap_rate, weights = _reference_pieces(0, 6)
    assert tenor_discount_factors(TENOR) == pytest.approx(tuple(bonds), rel=1e-15)
    assert swap_annuity(TENOR, 0, 6) == pytest.approx(annuity, rel=1e-15)
    assert forward_swap_rate(TENOR, 0, 6) == pytest.approx(swap_rate, rel=1e-15)
    assert swap_rate_weights(TENOR, 0, 6) == pytest.approx(weights, rel=1e-15)


# ============================================================================================
# THE POINTWISE TRANSCRIPTION of eq 5.40 — the load-bearing formula, aimed at the ρ term.
# ============================================================================================


@pytest.mark.parametrize(("first", "count"), [(0, 6), (0, 3), (1, 4), (2, 4), (3, 3), (4, 2)])
def test_O4_rebonato_matches_the_POINTWISE_transcription_of_eq_5_40(first, count):
    """Eq 5.40's **value**, not its factors' presence — T16's lesson applied to the swaption.

    The reference is written from the printed formula with the correlation read from the matrix at
    `[first+m][first+l]`, so a `ρ` indexed off by the span offset, transposed, or weighted wrongly
    shows up here. That is the factor the spec singles out: most likely to be present and correctly
    signed yet wrong in magnitude."""
    _, _, swap_rate, weights = _reference_pieces(first, count)
    f, sigma = TENOR.forwards, TENOR.volatilities
    expected_variance = math.fsum(
        (weights[m] * f[first + m] / swap_rate)
        * (weights[l] * f[first + l] / swap_rate)
        * sigma[first + m]
        * sigma[first + l]
        * RHO.matrix[first + m][first + l]
        for m in range(count)
        for l in range(count)
    )
    assert rebonato_swaption_vol(_model(first, count), first, count) == pytest.approx(
        math.sqrt(expected_variance), rel=1e-13
    )


def test_O4_the_CORRELATION_fixture_is_not_degenerate():
    """The premise the oracle above rests on, asserted rather than assumed.

    Under a flat `ρ` the transcription test cannot see a transposed or offset index, because every
    off-diagonal entry is the same number. Five of this project's defects came from a fixture whose
    key parameter sat where it could not be seen, so the non-degeneracy is a test, not a comment."""
    off_diagonal = {RHO.matrix[i][j] for i in range(6) for j in range(6) if i != j}
    assert len(off_diagonal) > 1
    assert 1.0 not in TENOR.accruals, "τ ≠ 1 — the accrual visible"
    assert len(set(TENOR.accruals)) > 1, "τ NON-FLAT — the span offset visible"
    # And the correlation must not be TRANSLATION-INVARIANT, stated as the property the surviving
    # mutant actually needed: the ρ block a span reads must differ from the block at offset 0. An
    # `exp(−β|i−j|)` correlation fails this while looking perfectly non-uniform, which is how the
    # offset-dropping mutant passed all 603 tests.
    for first, count in ((2, 4), (1, 4), (3, 3)):
        offset = [[RHO.matrix[first + m][first + l] for l in range(count)] for m in range(count)]
        unoffset = [[RHO.matrix[m][l] for l in range(count)] for m in range(count)]
        assert offset != unoffset, f"ρ is translation-invariant over span ({first}, {count})"
    assert len(set(TENOR.volatilities)) == len(TENOR.volatilities), "σ_i all distinct"


# ============================================================================================
# THE CROSS-SOURCE COMPANION — Jäckel's exact-Z form, and the gap to Rebonato's frozen weights.
# ============================================================================================


def _jaeckel_exact_z(first: int, count: int) -> tuple[float, ...]:
    """Jäckel eq 12.18: `Z_k = (∂SR/∂f_k)·(f_k/SR)` — the EXACT derivative, weights not frozen.

    Computed here by central differences on `forward_swap_rate`'s own transcription, so it is
    independent of the implementation. This is NOT eq 5.40: Rebonato freezes `w_iF_i` at time `t`,
    which drops the terms where the weights themselves depend on the forwards (Jäckel eq 12.21's
    second and third terms). It is the companion, and the *gap* is the quantity of interest."""
    tau, f = TENOR.accruals, TENOR.forwards

    def swap_rate(forwards) -> float:
        bonds = [1.0]
        for i in range(len(TENOR)):
            bonds.append(bonds[-1] / (1.0 + forwards[i] * tau[i]))
        annuity = math.fsum(tau[first + m] * bonds[first + m + 1] for m in range(count))
        return (bonds[first] - bonds[first + count]) / annuity

    base = swap_rate(f)
    out = []
    for m in range(count):
        k = first + m
        step = 1e-7
        up, down = list(f), list(f)
        up[k] += step
        down[k] -= step
        out.append((swap_rate(up) - swap_rate(down)) / (2.0 * step) * f[k] / base)
    return tuple(out)


@pytest.mark.parametrize(("first", "count"), [(0, 6), (1, 4), (2, 4), (3, 3)])
def test_O3_rebonato_agrees_with_JAECKELS_EXACT_Z_to_a_MEASURED_bound(first, count):
    """The two sources do **not** print one formula — and this oracle states the gap rather than
    hiding it.

    McInerney 5.40 uses the frozen weight `w_kF_k/S`; Jäckel 12.18/12.21 uses the exact
    `(∂SR/∂f_k)(f_k/SR)`. McInerney's own Remark 5.13 says 5.40 *can be improved* by not freezing —
    so they are two approximations of different order, not two renderings of one. The relationship
    is still a real check, because the frozen form is the exact form with the weight-derivative
    terms dropped: the two must agree to the size of those terms, which is **sub-1%** here. A
    transcription error in either would not land inside that band.

    **The bound is measured, not tuned** — the assertion is that the gap is small and non-zero, and
    the actual size is reported in the tranche doc. A zero gap would mean the companion had been
    computed from the same frozen weights, which is the failure this guards."""
    _, _, swap_rate, weights = _reference_pieces(first, count)
    f, sigma = TENOR.forwards, TENOR.volatilities
    frozen = tuple(weights[m] * f[first + m] / swap_rate for m in range(count))
    exact = _jaeckel_exact_z(first, count)

    def variance(z) -> float:
        return math.fsum(
            z[m] * z[l] * sigma[first + m] * sigma[first + l] * RHO.matrix[first + m][first + l]
            for m in range(count)
            for l in range(count)
        )

    v_frozen, v_exact = math.sqrt(variance(frozen)), math.sqrt(variance(exact))
    gap = abs(v_frozen - v_exact) / v_exact
    assert gap < 0.01, f"Rebonato vs Jäckel exact-Z differs by {gap:.3%}, beyond the frozen-weight slack"
    assert gap > 1e-6, "a ZERO gap means the companion is not independent — it froze the weights too"
    # The frozen weights sum to exactly 1; the exact ones do NOT. That asymmetry is the signature of
    # the approximation, and it is the cleanest proof the two Z vectors are genuinely different.
    assert math.fsum(frozen) == pytest.approx(1.0, rel=1e-14)
    assert abs(math.fsum(exact) - 1.0) > 1e-4


# ============================================================================================
# REQUIREMENT ORACLES on the vol itself, and the refusals.
# ============================================================================================


def test_O2_the_vol_MOVES_with_the_correlation_and_is_bounded_by_the_flat_limits():
    """Non-vacuity: a swaption vol that ignored `ρ` would pass every reduction above.

    And two requirement bounds from the finance: at `ρ ≡ 1` the swap rate is a perfectly correlated
    basket, so `v` attains its maximum — the weighted average of the forward vols; a decaying `ρ`
    must sit strictly below it. This is the correlation dependence that is the swaption's whole
    reason to need the LMM rather than a caplet strip."""
    perfect = rebonato_swaption_vol(
        LiborMarketModel(TENOR, LmmCorrelation.flat(6, 1.0), numeraire_index=6), 0, 6
    )
    decaying = rebonato_swaption_vol(_model(0, 6), 0, 6)
    flat_80 = rebonato_swaption_vol(
        LiborMarketModel(TENOR, LmmCorrelation.flat(6, 0.80), numeraire_index=6), 0, 6
    )
    assert decaying < flat_80 < perfect

    _, _, swap_rate, weights = _reference_pieces(0, 6)
    average = math.fsum(
        weights[m] * TENOR.forwards[m] / swap_rate * TENOR.volatilities[m] for m in range(6)
    )
    assert perfect == pytest.approx(average, rel=1e-13)


def test_O6_the_vol_is_NON_NEGATIVE_and_a_bad_span_is_REFUSED():
    """`v² ≥ 0` is a requirement; a non-positive-semidefinite `ρ` is what could break it."""
    assert rebonato_swaption_vol(_model(0, 6), 0, 6) >= 0.0
    with pytest.raises(ValueError, match="span"):
        forward_swap_rate(TENOR, 0, 7)
    with pytest.raises(ValueError, match="span"):
        forward_swap_rate(TENOR, 5, 2)
    with pytest.raises(ValueError, match="span"):
        forward_swap_rate(TENOR, 0, 0)
    with pytest.raises(ValueError, match="span"):
        forward_swap_rate(TENOR, -1, 2)
