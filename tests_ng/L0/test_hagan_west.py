"""L0 oracle — Hagan–West monotone-convex reconstruction (AMF 2006), equation-anchored.

The paper publishes the worked example only as figures (Fig 2/3/9/10), so — per the ratified
oracle — we anchor on the paper's EQUATIONS: the eq-33 interval-average identity to machine
precision, and per-region pointwise values (regions i–iv) computed BY HAND from eq 47/49–56
(hard-coded here, independent of the implementation, so a wrong-region transcription is caught).
"""

from pricebook_ng.foundation import monotone_convex
from pricebook_ng.foundation.hagan_west import _region_g


def test_eq33_interval_average_reproduction() -> None:
    mc = monotone_convex((0.0, 1.0, 2.5, 4.0, 6.0, 9.0), (0.05, 0.08, 0.03, 0.06, 0.02))
    for i in range(1, 6):
        avg = (mc.integral(mc.knots[i]) - mc.integral(mc.knots[i - 1])) / (mc.knots[i] - mc.knots[i - 1])
        assert abs(avg - mc.averages[i - 1]) < 1e-14


def test_knot_values_eq30_32() -> None:
    mc = monotone_convex((0.0, 1.0, 3.0, 6.0), (0.04, 0.06, 0.05))
    # eq 30: fᵢ = (tᵢ−tᵢ₋₁)/(tᵢ₊₁−tᵢ₋₁)·aᵢ₊₁ + (tᵢ₊₁−tᵢ)/(tᵢ₊₁−tᵢ₋₁)·aᵢ  (unequal spacing)
    assert abs(mc.knot_values[1] - (1 / 3 * 0.06 + 2 / 3 * 0.04)) < 1e-14  # knots 0,1,3
    assert abs(mc.knot_values[2] - (2 / 5 * 0.05 + 3 / 5 * 0.06)) < 1e-14  # knots 1,3,6
    assert abs(mc.knot_values[0] - (0.04 - 0.5 * (mc.knot_values[1] - 0.04))) < 1e-14  # eq 31 collar
    assert mc.value(mc.knots[1]) == mc.knot_values[1]  # value reproduces the knot


def test_per_region_pointwise_hand_derived() -> None:
    # expected values computed by hand from AMF eq 47/49–56 (independent of the code)
    assert abs(_region_g(0.5, 1.0, -1.0) - 0.0) < 1e-12          # (i)   quadratic
    assert abs(_region_g(0.1, 1.0, -3.0) - 1.0) < 1e-12          # (ii)  flat head (η=0.25)
    assert abs(_region_g(0.5, 1.0, -3.0) - 5 / 9) < 1e-12        # (ii)  quadratic tail
    assert abs(_region_g(0.2, 1.0, -0.2) - 0.232) < 1e-12        # (iii) quadratic head (η=0.5)
    assert abs(_region_g(0.7, 1.0, -0.2) - (-0.2)) < 1e-12       # (iii) flat tail
    assert abs(_region_g(0.3, 1.0, 2.0) - (-0.1625)) < 1e-12     # (iv)  first quad (η=2/3, A=-2/3)
    assert abs(_region_g(0.8, 1.0, 2.0) - (-0.24)) < 1e-12       # (iv)  second quad
    for g0, g1 in [(1.0, -1.0), (1.0, -3.0), (1.0, -0.2), (1.0, 2.0), (-1.0, -2.0)]:
        assert abs(_region_g(0.0, g0, g1) - g0) < 1e-12 and abs(_region_g(1.0, g0, g1) - g1) < 1e-12


def test_positivity_clamps_knots_and_preserves_averages() -> None:
    knots, avgs = (0.0, 1.0, 2.0, 3.0), (0.08, 0.01, 0.08)
    off = monotone_convex(knots, avgs)
    on = monotone_convex(knots, avgs, positive=True)
    assert on.knot_values[1] < off.knot_values[1]  # eq 61 clamps f₁ down to 2·min(avgs)
    assert on.knot_values[1] >= 0.0
    assert all(on.value(0.1 * k) >= -1e-12 for k in range(1, 30))  # forwards stay positive
    for i in range(1, 4):  # eq-33 still holds — positivity changes shape, not the average
        avg = (on.integral(knots[i]) - on.integral(knots[i - 1])) / (knots[i] - knots[i - 1])
        assert abs(avg - avgs[i - 1]) < 1e-14


def test_degeneracy_constant_averages_is_constant() -> None:
    mc = monotone_convex((0.0, 1.0, 3.0, 7.0), (0.04, 0.04, 0.04))
    assert max(abs(mc.value(t) - 0.04) for t in [0.5, 2.0, 5.0, 6.9]) < 1e-15
