"""L3 oracle — SABR lognormal implied vol (Hagan et al. 2002, eq 2.17a).

Closed-form top-tier oracle: `sabr_vol` reprices to an INDEPENDENT inline Hagan evaluation (a
different code path), plus two analytic anchors — the β=1/ν=0 degenerate collapses to a flat `α`
at every strike, and the ATM (F=K) branch matches the published ATM formula (eq 2.18). The skew
sign (ρ<0 ⇒ higher vol on low strikes) pins the smile's orientation.
"""

import math

from pricebook_ng.market.vol_surface import SabrParams
from pricebook_ng.models.sabr import sabr_vol


def _ref_sabr(f: float, k: float, t: float, p: SabrParams) -> float:
    """Independent Hagan 2002 lognormal σ_B (inline) — the oracle's reference code path."""
    a, b, r, nu = p.alpha, p.beta, p.rho, p.nu
    if f == k:  # ATM, eq 2.18
        term = ((1 - b) ** 2 / 24) * a * a / f ** (2 - 2 * b) + 0.25 * r * b * nu * a / f ** (1 - b) + (
            2 - 3 * r * r
        ) / 24 * nu * nu
        return a / f ** (1 - b) * (1 + term * t)
    log_fk = math.log(f / k)
    fk = (f * k) ** ((1 - b) / 2)
    z = nu / a * fk * log_fk
    chi = math.log((math.sqrt(1 - 2 * r * z + z * z) + z - r) / (1 - r))
    denom = fk * (1 + (1 - b) ** 2 / 24 * log_fk**2 + (1 - b) ** 4 / 1920 * log_fk**4)
    term = ((1 - b) ** 2 / 24) * a * a / (f * k) ** (1 - b) + 0.25 * r * b * nu * a / fk + (
        2 - 3 * r * r
    ) / 24 * nu * nu
    return a / denom * (z / chi) * (1 + term * t)


PARAMS = SabrParams(alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
F, T = 0.035, 1.0


def test_sabr_matches_independent_hagan_across_strikes() -> None:
    for k in (0.020, 0.030, 0.035, 0.040, 0.060):
        assert abs(sabr_vol(F, k, T, PARAMS) - _ref_sabr(F, k, T, PARAMS)) < 1e-14


def test_sabr_atm_matches_published_atm_formula() -> None:
    # at K=F the smile evaluates to the ATM branch (eq 2.18) — no z/χ removable singularity blow-up
    assert abs(sabr_vol(F, F, T, PARAMS) - _ref_sabr(F, F, T, PARAMS)) < 1e-14


def test_sabr_atm_limit_is_continuous() -> None:
    # approaching ATM, σ→σ_ATM smoothly (the z/χ(z)→1 limit) — no discontinuity at the F=K seam
    atm = sabr_vol(F, F, T, PARAMS)
    near = sabr_vol(F, F + 1e-8, T, PARAMS)
    assert abs(near - atm) < 1e-4


def test_sabr_beta1_nu0_collapses_to_flat_alpha() -> None:
    # β=1, ν=0: (F K)^0 = 1, all correction terms vanish ⇒ σ_B ≡ α at every strike (degenerate lognormal)
    flat = SabrParams(alpha=0.18, beta=1.0, rho=0.0, nu=0.0)
    for k in (0.020, 0.035, 0.070):
        assert abs(sabr_vol(F, k, T, flat) - 0.18) < 1e-14


def test_sabr_negative_rho_skews_low_strikes_up() -> None:
    # ρ<0 tilts the smile: deep-OTM-put (low strike) vol exceeds deep-OTM-call (high strike) vol
    assert sabr_vol(F, 0.020, T, PARAMS) > sabr_vol(F, 0.060, T, PARAMS)
