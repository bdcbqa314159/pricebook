"""The extracted Gil-Pelaez inversion (L4) — tranche 49.

**This file guards one thing: extracting the inversion out of `heston_fourier.py` changed no Heston
number.** The references below were produced by `heston_call_price` **before** the extraction, and
they are stored as raw IEEE-754 bit patterns rather than decimals so that the comparison is made in
the integer domain, where a 1-ULP drift is visible and a decimal round-trip would have hidden it.

**But the assertion is a ULP bound, not equality, and the first version of this file got that
wrong.** §4 permits a bit-identity claim when a refactor preserves the order of the floating-point
operations, and the move does — so the *purity proof* was in-process bit-identity: 360 doubles, the
same interpreter, before versus after, every bit equal. That proof is sound and it is what justified
the refactor. Baking those bits into a **committed** test is a different and stronger claim, because
the test then runs on other machines, and §9 states the rule this breaks: *"Tolerance-based oracles
are a hard rule (transcendental results differ 1-2 ULP across runtimes)."* `cmath.exp`, `cmath.log`
and `cmath.sqrt` are exactly that, and CI duly disagreed with this machine by **1 to 4 ULP** on both
runners.

So the committed form is a **ULP-distance** assertion with a stated margin — still four orders of
magnitude tighter than any behavioural tolerance in the tree, and tight enough that a real change
to the inversion could not hide inside it, while being a claim the test is actually entitled to
make. The in-process bit-identity measurement stays the purity argument; this is its portable
regression.

The test also guards the move in the other direction: `_LOWER` and `FourierGrid` moved with the
inversion, and `tests/L4/test_heston_fourier.py`'s cutoff test had been patching
`heston_fourier._LOWER`, which after the move would have been a **silent no-op** — the test passing
while measuring nothing. That is the §9 "a gate that cannot fail" shape inside a test, and it is why
the constant's home is asserted here.
"""

import cmath
import struct
from datetime import date, timedelta
from statistics import NormalDist

import pytest

from pricebook.engine import fourier_inversion
from pricebook.engine.fourier_inversion import FourierGrid, gil_pelaez_probabilities
from pricebook.engine.heston_fourier import heston_call_price
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import DiscountCurve
from pricebook.market.equity import DividendSchedule, EquityUnderlying
from pricebook.models.heston import HestonParams

VAL = date(2026, 1, 1)
CURVE = DiscountCurve.flat(VAL, 0.025, DayCountConvention.ACT_365_FIXED)
UNDERLYING = EquityUnderlying(100.0, 0.0, DividendSchedule(continuous_yield=0.01))
GRID = FourierGrid(200.0, 4001)

#: `(kappa, theta, sigma_v, rho, v0, days, strike, the call price's exact bits)` — captured from
#: the pre-extraction implementation.
#:
#: **54 of 60, and the 6 that are excluded are named rather than quietly dropped (§4).** A
#: regression needs a number the engine can actually resolve, and six of the sweep's deep
#: out-of-the-money short-dated points are below `1e-3` on a spot of 100 — quadrature residue, not
#: prices. At `sigma_v = 0.9`, 30 days, `K = 130` the committed grid returns **-2.5e-04** where an
#: `upper=1200, count=24001` reference gives **+1.7e-05**: a *negative call price*, wrong by 2.7e-04
#: absolute. That is a pre-existing truncation limit of tranche 33's grid in the far short-dated
#: wing at high vol-of-vol — the same regime and the same cause as tranche 49's wider SVJ grid, in
#: code that merged sixteen tranches ago — and it is raised for ruling rather than silently
#: regression-locked here. Pinning those six would be preserving a wrong number.
REFERENCES = [
    (1.5, 0.04, 0.6, -0.7, 0.05, 30, 60.0, "ec9307a83f054440"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 30, 95.0, "2ac3bd60cde51740"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 30, 100.0, "7694a17086580440"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 180, 60.0, "8b280c9e142d4440"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 180, 95.0, "c71ab7378a342240"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 180, 100.0, "5c1f8dc0fa341740"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 180, 130.0, "2d35ff4ed014a43f"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 730, 60.0, "d2ed66f62f004540"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 730, 95.0, "acf076a363e32c40"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 730, 100.0, "7f62de77dccb2640"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 730, 130.0, "f560b0fb1cb7f43f"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 730, 200.0, "83ad54c05ad3763f"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 1825, 60.0, "41c9b82d3a494640"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 1825, 95.0, "d4bfb1d007553540"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 1825, 100.0, "e40ddf57d4ad3240"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 1825, 130.0, "22076e1d79ff1b40"),
    (1.5, 0.04, 0.6, -0.7, 0.05, 1825, 200.0, "2b83d5f0d7d1d13f"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 30, 60.0, "5ea2f90a44054440"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 30, 95.0, "43cc27e223da1540"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 30, 100.0, "733621bed5e3fa3f"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 180, 60.0, "781daa2cc8254440"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 180, 95.0, "29c910c6d48e2040"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 180, 100.0, "8179b872ce4a1440"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 180, 130.0, "7aa09e11d389c83f"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 180, 200.0, "de10987254f6593f"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 730, 60.0, "6cad7af7bff54440"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 730, 95.0, "41c1cd98c3ff2e40"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 730, 100.0, "683544a1f69c2940"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 730, 130.0, "df64b3f60a2b0c40"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 730, 200.0, "a344b9efe60ad23f"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 1825, 60.0, "5a28df960a924640"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 1825, 95.0, "6e50e0aa5a1f3840"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 1825, 100.0, "3789dd6eced83540"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 1825, 130.0, "d5fae78a86722740"),
    (2.5, 0.06, 0.9, -0.3, 0.02, 1825, 200.0, "531df96c60750640"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 30, 60.0, "95b6c3963f054440"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 30, 95.0, "f44f2f8e86bf1940"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 30, 100.0, "01c93fd6b9930b40"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 30, 130.0, "b055f0f01c09783f"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 180, 60.0, "e6e56a7577204440"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 180, 95.0, "660f6ac14d582540"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 180, 100.0, "4b125964654e2040"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 180, 130.0, "21d9f2f8cfbaf43f"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 180, 200.0, "6fa396deee838f3f"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 730, 60.0, "9810ca11edb04440"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 730, 95.0, "54e2024a12c73040"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 730, 100.0, "f0674f471c1a2d40"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 730, 130.0, "6fe52f93080b1940"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 730, 200.0, "c74b15ccd354f23f"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 1825, 60.0, "07d4ffbeb0b94540"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 1825, 95.0, "343b62fe65823640"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 1825, 100.0, "7b3a9d2055753440"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 1825, 130.0, "66f7686301992740"),
    (0.8, 0.03, 0.2, 0.4, 0.09, 1825, 200.0, "65f500bd731b1040"),
]


#: How far a reference value may drift. Measured: the extraction is bit-identical in-process, and
#: CI disagreed with the capturing machine by **1 to 4 ULP** on both runners — pure libm variation
#: in `cmath.exp`/`log`/`sqrt`, which §9 names as a hard rule rather than a surprise. 32 leaves room
#: for a slower runtime or a future libm without admitting anything real: 32 ULP on a price near 10
#: is about 6e-14 relative, four orders tighter than the tightest behavioural oracle in the tree,
#: and any actual change to the inversion moves prices by vastly more.
_ULP_BUDGET = 32


def _ulp_distance(left: float, right: float) -> int:
    """Signed-magnitude doubles compared as integers — the distance in representable steps.

    Comparing bit patterns as *integers* rather than as floats is what makes "within 4 ULP" a
    statement one can assert; a relative tolerance at this scale is dominated by the exponent."""
    def ordered(value: float) -> int:
        bits = int.from_bytes(struct.pack("<d", value), "little")
        return bits if bits < (1 << 63) else (1 << 63) - bits
    return abs(ordered(left) - ordered(right))


@pytest.mark.parametrize("kappa,theta,sigma_v,rho,v0,days,strike,expected", REFERENCES)
def test_the_extraction_changed_no_heston_number(
    kappa: float, theta: float, sigma_v: float, rho: float, v0: float,
    days: int, strike: float, expected: str,
) -> None:
    """Within `_ULP_BUDGET` of the pre-extraction value — see the module docstring for why this is
    a ULP bound and not equality."""
    params = HestonParams(kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, v0=v0)
    expiry = VAL + timedelta(days=days)
    price = heston_call_price(params, UNDERLYING, strike, expiry, (CURVE, GRID))
    reference = struct.unpack("<d", bytes.fromhex(expected))[0]
    distance = _ulp_distance(price, reference)
    assert distance <= _ULP_BUDGET, (
        f"{price!r} is {distance} ULP from the pre-extraction {reference!r} "
        f"(budget {_ULP_BUDGET}) — a real change, not libm variation"
    )


def test_THE_EXCLUDED_REGIME_is_named_and_still_fails(self=None) -> None:
    """The six excluded points are excluded for a measured reason, asserted here so the exclusion
    cannot quietly become a hole: in that regime the committed grid genuinely disagrees with a far
    finer one, and at `sigma_v = 0.9` it returns a negative call price."""
    params = HestonParams(kappa=2.5, theta=0.06, sigma_v=0.9, rho=-0.3, v0=0.02)
    expiry = VAL + timedelta(days=30)
    committed = heston_call_price(params, UNDERLYING, 130.0, expiry, (CURVE, GRID))
    fine = heston_call_price(params, UNDERLYING, 130.0, expiry, (CURVE, FourierGrid(1200.0, 24001)))
    assert committed < 0.0 < fine, f"committed={committed}, fine={fine}"
    assert abs(committed - fine) > 1e-4, "the disagreement must be real, not rounding"
    # and every kept reference is comfortably above that residue
    for row in REFERENCES:
        assert abs(struct.unpack("<d", bytes.fromhex(row[7]))[0]) >= 1e-3


def test_the_ulp_budget_would_catch_a_real_change() -> None:
    """The budget is only meaningful if it rejects something. A 1e-9 *relative* perturbation — far
    below any behavioural tolerance in this tree, and the sort of thing a quietly reordered
    quadrature produces — is already millions of ULP away."""
    reference = struct.unpack("<d", bytes.fromhex(REFERENCES[0][7]))[0]
    assert _ulp_distance(reference, reference) == 0
    assert _ulp_distance(reference * (1.0 + 1e-9), reference) > 1_000 * _ULP_BUDGET


def test_the_cutoff_constant_lives_where_the_quadrature_reads_it() -> None:
    """`FourierGrid.nodes()` starts at `_LOWER`, and both now live in the inversion module. If the
    constant drifted back without the grid, the cutoff test one file over would patch a name
    nothing reads and pass while measuring nothing."""
    assert fourier_inversion._LOWER > 0.0
    assert FourierGrid(1.0, 3).nodes()[0] == fourier_inversion._LOWER


class TestTheInversionAgainstAClosedFormItCanBeCheckedOn:
    """The inversion is generic now, so it is testable **with no model at all** — which a
    Heston-only test could never do, and which is the real prize of the extraction.

    Hand it a Gaussian characteristic function, `phi(u) = exp(iu*mu - sigma^2 u^2/2)`, the law of a
    lognormal `S_T`. Then both probabilities have exact closed forms: `P2 = N((mu - k)/sigma)`, and
    `P1` is the same under the **Esscher tilt** `phi(u-i)/phi(-i)`, which shifts the mean by exactly
    `sigma^2`, so `P1 = N((mu + sigma^2 - k)/sigma)`. That second identity is the valuable one — it
    pins the share-measure tilt, the subtlest line in the inversion, against a formula derived here
    rather than against another run of the same code.

    (A point mass, `phi = 1`, looks like the obvious degenerate check and is a trap: its
    characteristic function never decays, so the truncated quadrature cannot converge and the test
    would be measuring the grid, not the inversion.)
    """

    SIGMA = 0.3
    MU = -0.5 * 0.3 * 0.3  # martingale: E[S_T] = S_0

    def _characteristic(self, u: complex) -> complex:
        return cmath.exp(1j * u * self.MU - self.SIGMA * self.SIGMA * u * u / 2.0)

    @pytest.mark.parametrize("log_strike", [-0.4, -0.1, 0.0, 0.15, 0.5])
    def test_both_probabilities_match_their_closed_forms(self, log_strike: float) -> None:
        normal = NormalDist()
        first, second = gil_pelaez_probabilities(self._characteristic, log_strike, GRID)
        assert second == pytest.approx(normal.cdf((self.MU - log_strike) / self.SIGMA), abs=1e-8)
        assert first == pytest.approx(
            normal.cdf((self.MU + self.SIGMA**2 - log_strike) / self.SIGMA), abs=1e-8
        )

    def test_the_share_measure_tilt_actually_moves_the_answer(self) -> None:
        """The premise: without it `P1 == P2` and the tilt would be unobservable."""
        first, second = gil_pelaez_probabilities(self._characteristic, 0.0, GRID)
        assert first - second > 0.1, f"the tilt is invisible here: {first} vs {second}"

    def test_it_reads_the_function_it_is_HANDED(self) -> None:
        """A mutant ignoring `characteristic` and reaching for a module-level default would pass
        every Heston test in the tree; it cannot pass this one."""
        narrow = gil_pelaez_probabilities(
            lambda u: cmath.exp(1j * u * -0.005 - 0.01 * u * u / 2.0), 0.2, GRID
        )
        wide = gil_pelaez_probabilities(self._characteristic, 0.2, GRID)
        assert wide[1] > 5.0 * narrow[1], "a wider law must put far more mass above the strike"
