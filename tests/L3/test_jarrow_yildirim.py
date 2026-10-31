"""Jarrow-Yildirim's convexity correction. BERHAN-2012 ch 2, eqs 2.4.3, 2.4.16, 2.5.11.

Every formula here is **read in the held PDF**, not inferred from the stock -- which matters,
because the stock's `yoy_convexity_adjustment` is a first-order approximation in which `rho` is a
factor, and taking the anchor from it produced a wrong oracle in the original handoff.

    beta(t,T) = (1 - e^{-k(T-t)})/k                                  (2.4.3, 2.4.16)

    C(t,T0,T1) = sr*beta_r(T0,T1)*[ beta_r(t,T0)*(rho_rI*sI - 0.5*beta_r(t,T0))
                                  + (rho_nr*sn/(kn+kr))*(1 + kr*beta_n(t,T0))
                                  - (rho_nr*sn/(kn+kr))*beta_n(t,T0) ]   (2.5.11)
"""

import math

import pytest

from pricebook.models import HullWhiteParams, JYCorrelations, JYParams, hw_beta, yoy_convexity

KN, SN, KR, SR, SI = 0.05, 0.010, 0.04, 0.008, 0.012
RHO_NR, RHO_RI = 0.30, -0.20
NOMINAL = HullWhiteParams(KN, SN)
REAL = HullWhiteParams(KR, SR)
CORR = JYCorrelations(nominal_real=RHO_NR, real_index=RHO_RI, nominal_index=0.15)


def _reference_C(t, t0, t1, sr=SR, rho_nr=RHO_NR, rho_ri=RHO_RI):
    """Eq 2.5.11, written independently here so the library's version has something to be wrong
    against. Transcribed from the PDF, not from the implementation."""
    def beta(k, a, b):
        return (1.0 - math.exp(-k * (b - a))) / k

    br_period, br_t, bn_t = beta(KR, t0, t1), beta(KR, t, t0), beta(KN, t, t0)
    cross = rho_nr * SN / (KN + KR)
    return sr * br_period * (
        br_t * (rho_ri * SI - 0.5 * br_t) + cross * (1.0 + KR * bn_t) - cross * bn_t
    )


def test_beta_matches_its_closed_form_and_its_small_kappa_limit() -> None:
    """`beta(t,T) = (1-e^{-k(T-t)})/k`, and `beta -> (T-t)` as `k -> 0`. The limit is what catches
    a `kappa` in the wrong place: a wrong-denominator version still looks plausible at k=0.05."""
    for k in (0.001, 0.05, 0.5):
        for span in (0.25, 1.0, 7.0):
            assert hw_beta(k, 0.0, span) == pytest.approx((1.0 - math.exp(-k * span)) / k, rel=1e-14)
    for span in (0.5, 3.0):
        assert hw_beta(1e-9, 0.0, span) == pytest.approx(span, rel=1e-6)


def test_the_convexity_VANISHES_at_zero_real_vol_exactly() -> None:
    """**The T4 anchor.** `sigma_r` multiplies the whole of 2.5.11, and the thesis says so in the
    line after it ("it vanishes for sigma_r = 0") -- so this is exact by construction rather than a
    numerical limit."""
    flat = HullWhiteParams(KR, 0.0)
    assert yoy_convexity(0.0, 2.0, 3.0, JYParams(NOMINAL, flat, SI, CORR)) == 0.0


def test_the_convexity_does_NOT_vanish_at_zero_correlation() -> None:
    """**Pinned deliberately.** The original oracle plan specified `rho -> 0` as the anchor, taken
    from the stock's first-order approximation where `rho` *is* a factor. In JY it is not: the
    surviving term is `sr*beta_r(T0,T1)*beta_r(t,T0)*(-0.5*beta_r(t,T0))`, a **pure real-rate
    convexity with no correlation in it at all**. This test exists so nobody "simplifies" 2.5.11
    back toward the approximation."""
    zero = JYCorrelations(nominal_real=0.0, real_index=0.0, nominal_index=0.0)
    got = yoy_convexity(0.0, 2.0, 3.0, JYParams(NOMINAL, REAL, SI, zero))
    survivor = SR * hw_beta(KR, 2.0, 3.0) * hw_beta(KR, 0.0, 2.0) * (-0.5 * hw_beta(KR, 0.0, 2.0))
    assert got == pytest.approx(survivor, rel=1e-14)
    assert abs(got) > 1e-6


def test_the_convexity_matches_eq_2_5_11_across_a_parameter_sweep() -> None:
    for t, t0, t1 in ((0.0, 1.0, 2.0), (0.0, 2.0, 3.0), (0.5, 4.0, 5.0), (0.0, 9.0, 10.0)):
        for sr in (0.0, 0.004, 0.02):
            for rho_nr in (-0.8, 0.0, 0.5):
                for rho_ri in (-0.5, 0.0, 0.7):
                    got = yoy_convexity(
                        t, t0, t1,
                        JYParams(NOMINAL, HullWhiteParams(KR, sr), SI,
                                 JYCorrelations(rho_nr, rho_ri, 0.15)),
                    )
                    want = _reference_C(t, t0, t1, sr, rho_nr, rho_ri)
                    assert got == pytest.approx(want, rel=1e-13, abs=1e-18), (t, t0, t1, sr)


def test_the_convexity_grows_with_real_vol_and_with_the_period_length() -> None:
    """Monotonicity a scalar match misses. Magnitude, because the sign depends on the bracket."""
    by_vol = [abs(yoy_convexity(0.0, 2.0, 3.0, JYParams(NOMINAL, HullWhiteParams(KR, s), SI, CORR)))
              for s in (0.002, 0.008, 0.02)]
    assert by_vol == sorted(by_vol)
    by_span = [abs(yoy_convexity(0.0, 2.0, 2.0 + d, JYParams(NOMINAL, REAL, SI, CORR)))
               for d in (0.25, 1.0, 3.0)]
    assert by_span == sorted(by_span)


def test_the_convexity_is_INVARIANT_to_the_nominal_index_correlation() -> None:
    """Pins tranche 4's claim, now that `rho_nI` exists. `C` (2.5.11) reads `rho_rI` and `rho_nr`
    and **not** `rho_nI`, which enters only the cap/floor variance (2.6.4). Asserted rather than
    asserted-in-prose: `rho_nI` became a required field in T5, and this is what stops it quietly
    leaking into the convexity later."""
    swept = [
        yoy_convexity(0.0, 2.0, 3.0, JYParams(NOMINAL, REAL, SI, JYCorrelations(RHO_NR, RHO_RI, r)))
        for r in (-0.9, 0.0, 0.9)
    ]
    assert swept[0] == swept[1] == swept[2]
