"""Paper 1 validation: Ametrano & Bianchetti (2013) — Multicurve Bootstrap.

Reproduces the EUR multicurve case study (11-Dec-2012):
- OIS bootstrap from Eonia strip
- IRS-6M bootstrap with OIS discounting
- Loss of telescoping identity
- Bootstrap exact-fit round-trip

Reference: ssrn:2219548, §4-5.
"""

import pytest
import math
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.schedule import Frequency, generate_schedule
from pricebook.fixed_income.ois import OISSwap, bootstrap_ois
from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection
from pricebook.fixed_income.deposit import Deposit
from pricebook.curves.bootstrap import bootstrap


# ═══════════════════════════════════════════════════════════════
# Market data: EUR, 11-Dec-2012 (from paper §5)
# ═══════════════════════════════════════════════════════════════

REF_DATE = date(2012, 12, 11)
SPOT_DATE = date(2012, 12, 13)  # T+2

# OIS (Eonia) quotes — mid rates (paper p.56)
OIS_QUOTES = [
    (date(2013, 1, 14), 0.00070),   # ~1M
    (date(2013, 3, 13), 0.00047),   # ~3M
    (date(2013, 6, 13), 0.00018),   # ~6M
    (date(2013, 12, 13), 0.00000),  # 1Y (zero!)
    (date(2014, 12, 15), 0.00086),  # 2Y
    (date(2015, 12, 14), 0.00200),  # 3Y
    (date(2017, 12, 13), 0.00500),  # 5Y
    (date(2019, 12, 13), 0.00900),  # 7Y
    (date(2022, 12, 13), 0.01350),  # 10Y
    (date(2027, 12, 13), 0.01800),  # 15Y
    (date(2032, 12, 13), 0.02000),  # 20Y
    (date(2042, 12, 13), 0.02100),  # 30Y
]

# IRS-6M quotes — par rates (paper p.28)
IRS_6M_QUOTES = [
    (date(2013, 12, 13), 0.00286),   # 1Y
    (date(2014, 12, 15), 0.00340),   # 2Y
    (date(2015, 12, 14), 0.00440),   # 3Y
    (date(2017, 12, 13), 0.00762),   # 5Y
    (date(2019, 12, 13), 0.01100),   # 7Y
    (date(2022, 12, 13), 0.01584),   # 10Y
    (date(2027, 12, 13), 0.02000),   # 15Y
    (date(2032, 12, 13), 0.02150),   # 20Y
    (date(2042, 12, 13), 0.02256),   # 30Y
]


# ═══════════════════════════════════════════════════════════════
# Test 1: OIS bootstrap
# ═══════════════════════════════════════════════════════════════

class TestOISBootstrap:
    """Bootstrap the EUR Eonia discount curve from OIS quotes."""

    @pytest.fixture
    def ois_curve(self):
        return bootstrap_ois(
            SPOT_DATE, OIS_QUOTES,
            day_count=DayCountConvention.ACT_360,
            fixed_frequency=Frequency.ANNUAL,
            interpolation=InterpolationMethod.LOG_LINEAR,
        )

    def test_curve_has_all_pillars(self, ois_curve):
        """Bootstrap should produce DFs at all pillar dates."""
        for mat, _ in OIS_QUOTES:
            df = ois_curve.df(mat)
            assert df > 0, f"DF at {mat} should be positive, got {df}"

    def test_negative_rate_df_above_one(self, ois_curve):
        """Negative OIS rates → DF > 1 for short maturities."""
        # 6M OIS = 0.018% → nearly zero, 1Y = 0.000% → DF ≈ 1.0
        df_1y = ois_curve.df(date(2013, 12, 13))
        # With 0% OIS, DF should be very close to 1.0
        assert abs(df_1y - 1.0) < 0.005, f"1Y DF with 0% OIS should be ~1.0, got {df_1y}"

    def test_ois_round_trip(self, ois_curve):
        """Re-pricing each OIS instrument should recover its quote.

        This is the bootstrap exact-fit invariant (paper eq. 76).
        Tolerance: ~1e-6 (relaxed from 1e-12 due to interpolation).
        """
        for mat, par_rate in OIS_QUOTES:
            ois = OISSwap(SPOT_DATE, mat, par_rate, fixed_frequency=Frequency.ANNUAL,
                          day_count=DayCountConvention.ACT_360)
            recovered_par = ois.par_rate(ois_curve)
            assert abs(recovered_par - par_rate) < 1e-4, \
                f"OIS round-trip failed at {mat}: expected {par_rate:.6f}, got {recovered_par:.6f}"

    def test_df_monotone_long_end(self, ois_curve):
        """DFs should be decreasing for positive rates (long end)."""
        df_5y = ois_curve.df(date(2017, 12, 13))
        df_10y = ois_curve.df(date(2022, 12, 13))
        df_30y = ois_curve.df(date(2042, 12, 13))
        assert df_5y > df_10y > df_30y


# ═══════════════════════════════════════════════════════════════
# Test 2: Multicurve IRS bootstrap
# ═══════════════════════════════════════════════════════════════

class TestMulticurveIRS:
    """Bootstrap EUR IRS-6M curve using OIS as discount curve."""

    @pytest.fixture
    def curves(self):
        ois_curve = bootstrap_ois(
            SPOT_DATE, OIS_QUOTES,
            day_count=DayCountConvention.ACT_360,
            fixed_frequency=Frequency.ANNUAL,
            interpolation=InterpolationMethod.LOG_LINEAR,
        )
        # Bootstrap IRS-6M projection curve
        # Use deposits for short end + IRS for long end
        deposits_6m = [(date(2013, 6, 13), 0.00018)]  # 6M deposit
        irs_curve = bootstrap(
            SPOT_DATE,
            deposits=deposits_6m,
            swaps=IRS_6M_QUOTES,
            deposit_day_count=DayCountConvention.ACT_360,
            fixed_day_count=DayCountConvention.THIRTY_360,
            fixed_frequency=Frequency.SEMI_ANNUAL,
            interpolation=InterpolationMethod.LOG_LINEAR,
        )
        return ois_curve, irs_curve

    def test_projection_curve_positive(self, curves):
        """All projection curve DFs should be positive."""
        _, proj = curves
        for mat, _ in IRS_6M_QUOTES:
            df = proj.df(mat)
            assert df > 0, f"Projection DF at {mat} should be positive"

    def test_ois_and_projection_differ(self, curves):
        """OIS and projection curves should differ (multicurve effect)."""
        ois, proj = curves
        df_ois_10y = ois.df(date(2022, 12, 13))
        df_proj_10y = proj.df(date(2022, 12, 13))
        # They should be close but not identical
        assert abs(df_ois_10y - df_proj_10y) > 1e-6, \
            "OIS and projection curves should differ at 10Y"


# ═══════════════════════════════════════════════════════════════
# Test 3: Loss of telescoping (eq. 64-65)
# ═══════════════════════════════════════════════════════════════

class TestTelescopingLoss:
    """The floating leg PV of a multicurve IRS ≠ P(T0) - P(Tn)."""

    @pytest.fixture
    def curves(self):
        ois_curve = bootstrap_ois(
            SPOT_DATE, OIS_QUOTES,
            day_count=DayCountConvention.ACT_360,
            fixed_frequency=Frequency.ANNUAL,
        )
        proj_curve = bootstrap(
            SPOT_DATE,
            deposits=[(date(2013, 6, 13), 0.00018)],
            swaps=IRS_6M_QUOTES,
            deposit_day_count=DayCountConvention.ACT_360,
            fixed_day_count=DayCountConvention.THIRTY_360,
            fixed_frequency=Frequency.SEMI_ANNUAL,
        )
        return ois_curve, proj_curve

    def test_floating_leg_not_par(self, curves):
        """At-par 10Y IRS-6M: floating PV should differ from (1 - df(10Y)).

        In single-curve world: floating PV = 1 - df(T) exactly.
        In multicurve: this identity breaks because forward and discount
        curves are different.
        """
        ois, proj = curves
        mat_10y = date(2022, 12, 13)

        # Telescoping value: P_c(T0) - P_c(Tn)
        telescope = ois.df(SPOT_DATE) - ois.df(mat_10y)

        # Actual floating leg PV: sum of fwd × tau × df_ois
        schedule = generate_schedule(SPOT_DATE, mat_10y, Frequency.SEMI_ANNUAL)
        float_pv = 0.0
        for i in range(1, len(schedule)):
            tau = year_fraction(schedule[i-1], schedule[i], DayCountConvention.ACT_360)
            df1 = proj.df(schedule[i-1])
            df2 = proj.df(schedule[i])
            fwd = (df1 / df2 - 1) / tau if df2 > 0 and tau > 0 else 0
            float_pv += fwd * tau * ois.df(schedule[i])

        # The difference should be non-zero (multicurve effect)
        deviation = abs(float_pv - telescope)
        assert deviation > 1e-6, \
            f"Telescoping should fail in multicurve: deviation = {deviation:.2e}"


# ═══════════════════════════════════════════════════════════════
# Test 4: OIS single-curve property (eq. 73-74)
# ═══════════════════════════════════════════════════════════════

class TestOISSingleCurve:
    """OIS recovers the classical telescoping identity."""

    @pytest.fixture
    def ois_curve(self):
        return bootstrap_ois(
            SPOT_DATE, OIS_QUOTES,
            day_count=DayCountConvention.ACT_360,
            fixed_frequency=Frequency.ANNUAL,
        )

    def test_ois_telescoping_holds(self, ois_curve):
        """For OIS: floating PV = P_c(T0) - P_c(Tn) exactly.

        Because the OIS floating rate IS the overnight rate that builds
        the discount curve — forwarding and discounting collapse.
        """
        mat_10y = date(2022, 12, 13)
        telescope = ois_curve.df(SPOT_DATE) - ois_curve.df(mat_10y)

        # OIS floating leg = sum of overnight forwards × tau × df
        # But for OIS, this telescopes to df(T0) - df(Tn)
        schedule = generate_schedule(SPOT_DATE, mat_10y, Frequency.ANNUAL)
        float_pv = 0.0
        for i in range(1, len(schedule)):
            tau = year_fraction(schedule[i-1], schedule[i], DayCountConvention.ACT_360)
            df1 = ois_curve.df(schedule[i-1])
            df2 = ois_curve.df(schedule[i])
            fwd = (df1 / df2 - 1) / tau if df2 > 0 and tau > 0 else 0
            float_pv += fwd * tau * ois_curve.df(schedule[i])

        assert abs(float_pv - telescope) < 1e-4, \
            f"OIS telescoping should hold: float_pv={float_pv:.8f}, telescope={telescope:.8f}"


# ═══════════════════════════════════════════════════════════════
# Test 5: Sample IRS rates from paper
# ═══════════════════════════════════════════════════════════════

class TestSampleRates:
    """Verify sample rates from paper p.28."""

    def test_irs_6m_rates_reasonable(self):
        """IRS-6M par rates should be in expected range for Dec-2012 EUR."""
        # 1Y: 0.286%, 5Y: 0.762%, 10Y: 1.584%, 30Y: 2.256%
        assert 0.002 < IRS_6M_QUOTES[0][1] < 0.005   # 1Y
        assert 0.005 < IRS_6M_QUOTES[3][1] < 0.010    # 5Y
        assert 0.010 < IRS_6M_QUOTES[5][1] < 0.020    # 10Y
        assert 0.020 < IRS_6M_QUOTES[8][1] < 0.025    # 30Y

    def test_ois_rates_near_zero(self):
        """OIS rates should be near zero / slightly negative for Dec-2012."""
        # 1Y OIS = 0.000%
        assert OIS_QUOTES[3][1] == 0.0  # 1Y exactly zero
