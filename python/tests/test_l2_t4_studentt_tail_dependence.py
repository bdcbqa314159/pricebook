"""Regression for L2 Wave-2 audit — StudentT.tail_dependence formula.

Pre-fix the method took no `rho` argument and computed the formula

    2 · T_{ν+1}(-√((ν+1) / (ν-1+ε)))

which has no copula interpretation.  Tail dependence is a BIVARIATE
concept that requires the linear correlation between marginals.

Post-fix uses the standard formula
(Embrechts-McNeil-Straumann 2002, McNeil-Frey-Embrechts 2005 §5.3.1):

    λ_L = 2·T_{ν+1}(-√((ν+1)·(1-ρ)/(1+ρ)))

and validates ρ ∈ [-1, 1].
"""

from __future__ import annotations

import math

import pytest

from pricebook.numerical._distributions import StudentT
from pricebook.statistics.copulas import StudentTCopula


class TestStudentTTailDependence:
    def test_signature_requires_rho(self):
        """Pre-fix `tail_dependence()` took no args.  Post-fix it
        requires `rho` — calling without it must raise."""
        d = StudentT(df=4.0)
        with pytest.raises(TypeError):
            d.tail_dependence()  # type: ignore[call-arg]

    def test_rho_one_gives_unit_dependence(self):
        d = StudentT(df=4.0)
        assert d.tail_dependence(1.0) == pytest.approx(1.0)

    def test_rho_minus_one_gives_zero_dependence(self):
        d = StudentT(df=4.0)
        assert d.tail_dependence(-1.0) == pytest.approx(0.0)

    def test_rho_zero_is_positive(self):
        """Student-t copula has positive tail dependence even at ρ=0
        — the key property distinguishing it from Gaussian."""
        d = StudentT(df=4.0)
        lam = d.tail_dependence(0.0)
        assert 0.0 < lam < 1.0

    def test_matches_copula_implementation(self):
        """Must reproduce `StudentTCopula.tail_dependence` for matching (ν, ρ).

        Note: `StudentTCopula` requires ``rho ∈ [0, 1]`` (one-factor
        param requires non-negative ρ for both √ρ and √(1−ρ) to be real)
        so the test only spans the non-negative half — the distribution
        method is well-defined for ``ρ ∈ [-1, 1]`` separately.
        """
        for nu in [2.5, 3.0, 5.0, 10.0]:
            for rho in [0.0, 0.2, 0.7, 0.95]:
                d = StudentT(df=nu)
                c = StudentTCopula(rho=rho, nu=nu)
                assert d.tail_dependence(rho) == pytest.approx(
                    c.tail_dependence, abs=1e-12
                ), f"ν={nu}, ρ={rho}: dist={d.tail_dependence(rho)}, cop={c.tail_dependence}"

    def test_decreases_with_df(self):
        """Heavier tails (smaller ν) → stronger tail dependence at fixed ρ."""
        rho = 0.5
        lams = [StudentT(df=nu).tail_dependence(rho) for nu in [2.5, 5.0, 10.0, 30.0]]
        for a, b in zip(lams, lams[1:]):
            assert a > b, f"expected monotone decrease, got {lams}"

    def test_invalid_rho_raises(self):
        d = StudentT(df=4.0)
        with pytest.raises(ValueError):
            d.tail_dependence(1.5)
        with pytest.raises(ValueError):
            d.tail_dependence(-1.5)
