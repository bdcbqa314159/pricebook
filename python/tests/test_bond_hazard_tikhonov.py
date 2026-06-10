"""Tests for the Tikhonov-regularised hazard bootstrap.

Covers:
- lam=0 reproduces the original unregularised behaviour exactly (regression).
- lam > 0 reduces roughness vs lam = 0 on a noisy close-maturity universe.
- lam="auto" resolves to a positive λ via find_lcurve_lambda.
- find_lcurve_lambda standalone.
- Edge cases (n_pillars < 3 → no curvature penalty; invalid lam).
- HazardBootstrapResult.lam and .roughness fields populated.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.credit.bond_hazard_bootstrap import (
    BondInput,
    bootstrap_hazard_from_bonds,
    find_lcurve_lambda,
    _bootstrap_global,
    _price_risky_bond,
)


REF = date(2026, 6, 11)


def _flat_rf():
    return DiscountCurve.flat(REF, 0.04)


def _truth_survival():
    """Piecewise-constant 2%/3%/4% hazard on [0,3]/[3,7]/[7,15]."""
    dates = [REF + timedelta(days=365 * y) for y in [3, 7, 15]]
    survs = []
    cum = 1.0
    prev_t = 0.0
    for t_y, h in zip([3.0, 7.0, 15.0], [0.02, 0.03, 0.04]):
        cum *= math.exp(-h * (t_y - prev_t))
        survs.append(cum)
        prev_t = t_y
    return SurvivalCurve(REF, dates, survs)


def _close_maturity_bonds(noise_5y_bp: float = 0.0) -> list[BondInput]:
    """Five bonds (1, 3, 5, 5+2mo, 10y), optionally with price noise on the 5y."""
    rf = _flat_rf()
    truth = _truth_survival()
    specs = [(1.0, 0.040), (3.0, 0.045), (5.0, 0.050), (5.0 + 2 / 12, 0.050), (10.0, 0.055)]
    out = []
    for y, c in specs:
        bm = REF + timedelta(days=int(round(365 * y)))
        pr = _price_risky_bond(REF, bm, c, 2, 0.40, rf, truth)
        out.append(
            BondInput(maturity=bm, coupon=c, market_price=pr, frequency=2, recovery=0.40)
        )
    if noise_5y_bp != 0.0:
        out[2] = BondInput(
            maturity=out[2].maturity,
            coupon=out[2].coupon,
            market_price=out[2].market_price + noise_5y_bp / 100.0,
            frequency=out[2].frequency,
            recovery=out[2].recovery,
        )
    return out


class TestRegression:
    """lam=0 must reproduce the existing unregularised behaviour exactly."""

    def test_lam_zero_matches_default_global(self):
        bonds = _close_maturity_bonds()
        rf = _flat_rf()
        # The bootstrap_hazard_from_bonds default for lam is 0.0
        r_default = bootstrap_hazard_from_bonds(REF, bonds, rf, method="global", n_pillars=5)
        r_explicit_zero = bootstrap_hazard_from_bonds(
            REF, bonds, rf, method="global", n_pillars=5, lam=0.0
        )
        assert r_default.pillar_hazards == r_explicit_zero.pillar_hazards
        assert r_default.rmse_bp == r_explicit_zero.rmse_bp
        assert r_default.lam == 0.0
        assert r_explicit_zero.lam == 0.0
        # method label unchanged when lam=0
        assert r_default.method == "global_ls"

    def test_lam_zero_reports_roughness(self):
        """Even at lam=0, the result must populate the diagnostic .roughness field."""
        bonds = _close_maturity_bonds(noise_5y_bp=5.0)
        rf = _flat_rf()
        r = bootstrap_hazard_from_bonds(REF, bonds, rf, method="global", n_pillars=5, lam=0.0)
        # Unregularised noisy fit should have a noticeably nonzero roughness
        assert r.roughness > 0


class TestSmoothing:
    """lam > 0 must reduce roughness and increase rmse vs lam = 0."""

    def test_increasing_lam_decreases_roughness(self):
        bonds = _close_maturity_bonds(noise_5y_bp=5.0)
        rf = _flat_rf()
        rough_seq = []
        rmse_seq = []
        for lam in [0.0, 1e5, 1e7, 1e9]:
            r = bootstrap_hazard_from_bonds(
                REF, bonds, rf, method="global", n_pillars=5, lam=lam
            )
            rough_seq.append(r.roughness)
            rmse_seq.append(r.rmse_bp)
        # Roughness decreases monotonically (strictly) as lam grows
        for i in range(len(rough_seq) - 1):
            assert rough_seq[i + 1] < rough_seq[i] + 1e-12
        # rmse increases (or stays flat in pathological cases)
        assert rmse_seq[-1] >= rmse_seq[0]

    def test_tikhonov_drives_curve_toward_linear_at_strong_lam(self):
        """At very large lam, the second-difference penalty drives the hazard
        curve toward zero curvature, i.e., a near-linear ramp. The maximum
        absolute discrete second-difference of the hazards must shrink
        substantially vs the unregularised noisy fit. (Note: lam→∞ does NOT
        give a flat curve — that's what a first-difference penalty would
        produce. A second-difference penalty gives a linear curve.)"""
        bonds = _close_maturity_bonds(noise_5y_bp=5.0)
        rf = _flat_rf()
        r0 = bootstrap_hazard_from_bonds(REF, bonds, rf, method="global", n_pillars=5, lam=0.0)
        r_strong = bootstrap_hazard_from_bonds(
            REF, bonds, rf, method="global", n_pillars=5, lam=1e10
        )

        def max_second_diff(h):
            return max(
                abs(h[i - 1] - 2 * h[i] + h[i + 1])
                for i in range(1, len(h) - 1)
            )

        unreg_d2 = max_second_diff(r0.pillar_hazards)
        strong_d2 = max_second_diff(r_strong.pillar_hazards)
        # Strong-lam max-|d2| must be at least 100× smaller than unreg
        assert strong_d2 < unreg_d2 / 100.0
        # And the regularised fit's method label should reflect Tikhonov
        assert r_strong.method == "global_ls_tikhonov"
        assert r_strong.lam == 1e10


class TestAutoLambda:
    """lam='auto' must call find_lcurve_lambda and resolve to a positive value."""

    def test_auto_returns_positive_lambda(self):
        bonds = _close_maturity_bonds(noise_5y_bp=5.0)
        rf = _flat_rf()
        r = bootstrap_hazard_from_bonds(
            REF, bonds, rf, method="global", n_pillars=5, lam="auto"
        )
        assert r.lam > 0
        assert r.method == "global_ls_tikhonov"

    def test_auto_matches_find_lcurve_lambda(self):
        bonds = _close_maturity_bonds(noise_5y_bp=5.0)
        rf = _flat_rf()
        lam_picker = find_lcurve_lambda(REF, bonds, rf, n_pillars=5)
        r_auto = bootstrap_hazard_from_bonds(
            REF, bonds, rf, method="global", n_pillars=5, lam="auto"
        )
        assert r_auto.lam == pytest.approx(lam_picker, rel=1e-9)

    def test_find_lcurve_lambda_on_clean_data(self):
        """With essentially noise-free data, the picker should still return a
        positive lambda (it will likely be at the lower end of the sweep)."""
        bonds = _close_maturity_bonds(noise_5y_bp=0.0)
        rf = _flat_rf()
        lam = find_lcurve_lambda(REF, bonds, rf, n_pillars=5)
        assert lam > 0


class TestEdgeCases:
    def test_n_pillars_less_than_3_ignores_lam(self):
        """With fewer than 3 pillars there is no second-difference to penalise;
        lam should have no effect, and roughness should be 0."""
        bonds = _close_maturity_bonds(noise_5y_bp=5.0)
        rf = _flat_rf()
        r_no_pen = _bootstrap_global(REF, bonds, rf, n_pillars=2, lam=0.0)
        r_with_lam = _bootstrap_global(REF, bonds, rf, n_pillars=2, lam=1e9)
        assert r_no_pen.roughness == 0.0
        assert r_with_lam.roughness == 0.0
        # Hazards should be the same since lam has no effect
        for a, b in zip(r_no_pen.pillar_hazards, r_with_lam.pillar_hazards):
            assert a == pytest.approx(b, abs=1e-6)

    def test_negative_lam_raises(self):
        bonds = _close_maturity_bonds()
        rf = _flat_rf()
        with pytest.raises(ValueError, match="lam must be"):
            bootstrap_hazard_from_bonds(REF, bonds, rf, method="global", lam=-1.0)

    def test_unknown_lam_string_raises(self):
        bonds = _close_maturity_bonds()
        rf = _flat_rf()
        with pytest.raises(ValueError, match="must be 'auto'"):
            bootstrap_hazard_from_bonds(REF, bonds, rf, method="global", lam="something")

    def test_sequential_ignores_lam(self):
        """Sequential method does not use lam — it must produce identical results
        regardless of what lam is passed."""
        bonds = _close_maturity_bonds()
        rf = _flat_rf()
        r1 = bootstrap_hazard_from_bonds(REF, bonds, rf, method="sequential", lam=0.0)
        r2 = bootstrap_hazard_from_bonds(REF, bonds, rf, method="sequential", lam=1e6)
        assert r1.pillar_hazards == r2.pillar_hazards
        assert r1.rmse_bp == r2.rmse_bp
