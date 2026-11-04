"""The pseudo-square-root of a covariance matrix (L0). JAECKEL-2002 eqs 12.4, 12.6, 12.7.

A covariance matrix `C` is factorised as `C ~ A A^T`, optionally **truncated to m factors**, and then
rescaled so that

    sum_j a_ij^2 = c_ii                                                                 (12.6)

holds EXACTLY. Jaeckel's words for why: *"in order to retain the calibration of the options on the
FRAs, i.e. the caplets"* — the diagonal of `C` is each forward's own total variance, so if truncation
disturbed it, every caplet would reprice wrong. Equation 12.7 is the rescaling that preserves it:

    a_ij = atilde_ij * sqrt( c_ii / sum_k atilde_ik^2 )                                 (12.7)

**The pointwise oracle is 12.6 as an exact identity**, not a tolerance: after rescaling, each row's
squared norm must equal the diagonal entry to machine precision, at any factor count. That is the
caplet-calibration guarantee, and it is the one property truncation must not break.

**And Jaeckel states the trade-off, so the companion asserts it rather than hoping:** *"the individual
variances of each of the rates are still correct, even if we have reduced the number of driving
factors to one, but the effective covariances will differ. For instance, for a single factor model,
all of the correlation coefficients will be unity."* So a one-factor rescaling must reproduce the
diagonal exactly AND drive every implied correlation to one — both asserted.

This is **finance-free linear algebra**: the finance is *which* matrix, which is L3's business.
"""

import math

import pytest

from pricebook.foundation.linalg import pseudo_square_root

# A 3x3 covariance with distinct diagonal entries, so a rescaling error cannot hide behind symmetry.
C = (
    (0.040, 0.012, 0.006),
    (0.012, 0.090, 0.018),
    (0.006, 0.018, 0.160),
)


def _product(a: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    n, m = len(a), len(a[0])
    return [[math.fsum(a[i][k] * a[j][k] for k in range(m)) for j in range(n)] for i in range(n)]


def test_the_FULL_factorisation_reconstructs_the_covariance():
    """With all factors retained, `A A^T = C` — the defining property."""
    a = pseudo_square_root(C, factors=3)
    product = _product(a)
    for i in range(3):
        for j in range(3):
            assert product[i][j] == pytest.approx(C[i][j], abs=1e-14), (i, j)


@pytest.mark.parametrize("factors", [1, 2, 3])
def test_eq_12_6_holds_EXACTLY_at_every_factor_count(factors):
    """**The caplet-calibration guarantee, and it is exact.** Each row's squared norm must equal the
    diagonal entry to machine precision however hard the factorisation is truncated — that is what
    eq 12.7's rescaling is for, and what a caplet oracle downstream depends on."""
    a = pseudo_square_root(C, factors=factors)
    for i in range(3):
        assert math.fsum(x * x for x in a[i]) == pytest.approx(C[i][i], rel=1e-15), (i, factors)


def test_a_ONE_FACTOR_model_drives_every_correlation_to_UNITY():
    """Jaeckel's own stated consequence: *"for a single factor model, all of the correlation
    coefficients will be unity"*. Asserted rather than assumed — and it is the companion that proves
    the diagonal-preservation above is not vacuous, since a one-factor fit genuinely changes the
    off-diagonals while leaving the diagonal exact."""
    a = pseudo_square_root(C, factors=1)
    product = _product(a)
    for i in range(3):
        for j in range(3):
            implied = product[i][j] / math.sqrt(product[i][i] * product[j][j])
            assert implied == pytest.approx(1.0, abs=1e-12), (i, j, implied)
    # and the diagonal is still exactly C's
    for i in range(3):
        assert product[i][i] == pytest.approx(C[i][i], rel=1e-15)


def test_TRUNCATION_changes_the_off_diagonals_so_the_trade_off_is_real():
    """The other half of Jaeckel's sentence: the *covariances* differ under truncation. Without this
    the one-factor test could pass on a matrix that was already rank one."""
    full = _product(pseudo_square_root(C, factors=3))
    one = _product(pseudo_square_root(C, factors=1))
    assert abs(one[0][2] - full[0][2]) > 1e-3, (one[0][2], full[0][2])


def test_the_factorisation_has_the_REQUESTED_number_of_columns():
    for factors in (1, 2, 3):
        a = pseudo_square_root(C, factors=factors)
        assert len(a) == 3
        assert all(len(row) == factors for row in a)


def test_a_DIAGONAL_covariance_factorises_to_its_square_roots():
    """A hand-checkable case: with no correlation the factorisation is just the standard deviations
    on the diagonal, so the answer is known without any decomposition."""
    diagonal = ((0.04, 0.0), (0.0, 0.09))
    a = pseudo_square_root(diagonal, factors=2)
    product = _product(a)
    assert product[0][0] == pytest.approx(0.04, rel=1e-15)
    assert product[1][1] == pytest.approx(0.09, rel=1e-15)
    assert product[0][1] == pytest.approx(0.0, abs=1e-15)


def test_a_NON_SYMMETRIC_matrix_is_refused():
    with pytest.raises(ValueError, match="symmetric"):
        pseudo_square_root(((1.0, 0.5), (0.4, 1.0)), factors=2)


def test_a_NON_POSITIVE_DEFINITE_matrix_is_refused_not_silently_truncated():
    """A matrix with a negative eigenvalue is not a covariance. Silently clipping it would produce a
    factorisation of something the caller never asked for."""
    with pytest.raises(ValueError, match="positive"):
        pseudo_square_root(((1.0, 2.0), (2.0, 1.0)), factors=2)


def test_a_NON_POSITIVE_DIAGONAL_is_refused():
    """Eq 12.7 divides by the row norm and multiplies by `sqrt(c_ii)`; a zero variance makes the
    rescaling undefined rather than merely degenerate."""
    with pytest.raises(ValueError, match="positive"):
        pseudo_square_root(((0.0, 0.0), (0.0, 1.0)), factors=1)


def test_a_BAD_factor_count_is_refused():
    with pytest.raises(ValueError, match="between 1"):
        pseudo_square_root(C, factors=0)
    with pytest.raises(ValueError, match="between 1"):
        pseudo_square_root(C, factors=4)
