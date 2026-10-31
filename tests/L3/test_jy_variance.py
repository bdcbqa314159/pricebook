"""The JY cap/floor log-ratio variance V^2. BERHAN-2012 eq 2.6.4, read in the PDF.

Eight terms, which is where every sign, kappa and factor-of-2 error lives -- so the primary
correctness oracle is an **independent transcription**, swept. The `sigma_n = sigma_r = 0`
reduction kills six of the eight, so it is reduction-*sanity*, not the thing that checks their
signs (Cowork's framing, thread 0031).

`rho_nI` appears in the eighth term and nowhere else in the whole model, which is why it lands in
this tranche and not in T4.
"""

import math

import pytest

from pricebook.models import HullWhiteParams, JYCorrelations, JYParams, hw_beta, yoy_variance

KN, SN, KR, SR, SI = 0.05, 0.010, 0.04, 0.008, 0.012
P = JYParams(HullWhiteParams(KN, SN), HullWhiteParams(KR, SR), SI,
             JYCorrelations(nominal_real=0.30, real_index=-0.20, nominal_index=0.15))


def _reference_V2(t, t0, t1, params=P):
    """Eq 2.6.4 transcribed from the PDF, independently of the implementation."""
    kn, sn = params.nominal.kappa, params.nominal.sigma
    kr, sr = params.real.kappa, params.real.sigma
    si = params.index_vol
    rho_nr = params.correlations.nominal_real
    rho_ri = params.correlations.real_index
    rho_ni = params.correlations.nominal_index

    d = t1 - t0                      # Ti - Ti-1
    tau = t0 - t                     # Ti-1 - t
    bn = (1.0 - math.exp(-kn * d)) / kn
    br = (1.0 - math.exp(-kr * d)) / kr

    v = sn * sn / (2.0 * kn) * bn * bn * (1.0 - math.exp(-2.0 * kn * tau))
    v += si * si * d
    v += sr * sr / (2.0 * kr) * br * br * (1.0 - math.exp(-2.0 * kr * tau))
    v -= 2.0 * rho_nr * (sn * sr / (kn + kr)) * bn * br * (1.0 - math.exp(-(kn + kr) * tau))
    v += (sn * sn / kn**2) * (
        d + (2.0 / kn) * math.exp(-kn * d) - (1.0 / (2.0 * kn)) * math.exp(-2.0 * kn * d)
        - 3.0 / (2.0 * kn)
    )
    v += (sr * sr / kr**2) * (
        d + (2.0 / kr) * math.exp(-kr * d) - (1.0 / (2.0 * kr)) * math.exp(-2.0 * kr * d)
        - 3.0 / (2.0 * kr)
    )
    v -= 2.0 * rho_nr * (sn * sr / (kn * kr)) * (
        d - bn - br + (1.0 - math.exp(-(kn + kr) * d)) / (kn + kr)
    )
    v += 2.0 * rho_ni * (sn * si / kn) * (d - bn)
    v -= 2.0 * rho_ri * (sr * si / kr) * (d - br)
    return v


# ── ORACLE 1 (primary): the eight terms against an independent transcription ────────────────
def test_the_variance_matches_eq_2_6_4_across_a_parameter_sweep() -> None:
    """**The primary correctness oracle** -- the reduction below kills six of the eight terms, so
    only this checks their signs and factors."""
    for t, t0, t1 in ((0.0, 1.0, 2.0), (0.0, 2.0, 3.0), (0.5, 4.0, 5.0), (0.0, 9.0, 10.0)):
        for sn in (0.0, 0.006, 0.02):
            for sr in (0.0, 0.004, 0.015):
                for rho_nr, rho_ri, rho_ni in (
                    (0.0, 0.0, 0.0), (0.4, -0.3, 0.2), (-0.7, 0.5, -0.6), (0.9, 0.9, 0.9),
                ):
                    p = JYParams(
                        HullWhiteParams(KN, sn), HullWhiteParams(KR, sr), SI,
                        JYCorrelations(rho_nr, rho_ri, rho_ni),
                    )
                    got = yoy_variance(t, t0, t1, p)
                    assert got == pytest.approx(_reference_V2(t, t0, t1, p), rel=1e-12, abs=1e-20)


# ── ORACLE 2 (reduction sanity) + its companion ─────────────────────────────────────────────
def test_with_both_rate_vols_zero_the_variance_IS_index_vol_squared_times_the_period() -> None:
    """`sigma_n = sigma_r = 0` leaves only the `sigma_I^2 * (Ti - Ti-1)` term: six of the eight
    carry a rate vol and the eighth carries both cross terms. Exact, not a limit."""
    flat = JYParams(HullWhiteParams(KN, 0.0), HullWhiteParams(KR, 0.0), SI, P.correlations)
    for t0, t1 in ((1.0, 2.0), (4.0, 5.5), (9.0, 10.0)):
        assert yoy_variance(0.0, t0, t1, flat) == pytest.approx(SI * SI * (t1 - t0), rel=1e-14)


def test_the_rate_vols_MOVE_the_variance_so_the_reduction_is_not_vacuous() -> None:
    """The charter's companion requirement. Without this, the reduction above passes an
    implementation that returns `sigma_I^2 * delta` and ignores the other seven terms entirely."""
    flat = JYParams(HullWhiteParams(KN, 0.0), HullWhiteParams(KR, 0.0), SI, P.correlations)
    base = yoy_variance(0.0, 2.0, 3.0, flat)
    full = yoy_variance(0.0, 2.0, 3.0, P)
    assert abs(full - base) > 0.01 * base


# ── ORACLE 5: a variance cannot be negative ─────────────────────────────────────────────────
def test_the_variance_is_NON_NEGATIVE_even_at_adverse_correlations() -> None:
    """An eight-term expression with three signed correlations is exactly where a negative
    variance hides -- and it would surface far from here, as a domain error inside a sqrt."""
    for rho in (-0.95, -0.5, 0.0, 0.5, 0.95):
        for t0, t1 in ((0.5, 1.5), (3.0, 4.0), (9.0, 10.0)):
            p = JYParams(
                HullWhiteParams(KN, 0.02), HullWhiteParams(KR, 0.02), 0.03,
                JYCorrelations(rho, rho, rho),
            )
            assert yoy_variance(0.0, t0, t1, p) >= 0.0, (rho, t0, t1)


def test_rho_nominal_index_MOVES_the_variance_its_first_consumer() -> None:
    """`rho_nI` enters only the eighth term. T4 omitted it because its convexity never reads it;
    this asserts that it is genuinely consumed here rather than carried."""
    lo = JYParams(P.nominal, P.real, SI, JYCorrelations(0.3, -0.2, -0.8))
    hi = JYParams(P.nominal, P.real, SI, JYCorrelations(0.3, -0.2, 0.8))
    assert yoy_variance(0.0, 2.0, 3.0, lo) != pytest.approx(yoy_variance(0.0, 2.0, 3.0, hi))


def test_the_variance_grows_with_the_period_length() -> None:
    by_span = [yoy_variance(0.0, 2.0, 2.0 + d, P) for d in (0.25, 1.0, 3.0)]
    assert by_span == sorted(by_span)
    assert hw_beta(KN, 2.0, 3.0) > 0.0   # the shared beta, already oracled in T4
