"""OIS-IBOR basis: term structure with credit/liquidity decomposition.

The OIS-IBOR basis is the spread between IBOR forward rates and OIS
forward rates at the same tenor. It reflects bank credit risk and
term liquidity premium.

    from pricebook.ois_ibor_basis import OISIBORBasis

    basis = OISIBORBasis.from_curves(ibor_3m, ois_curve, pillar_dates)
    spread = basis.forward_basis(d1, d2)

    # Decompose into credit + liquidity
    basis.decompose(cds_spreads={"BankA": 0.005, "BankB": 0.008})

References:
    Mercurio, F. (2009). Interest Rates and The Credit Crunch.
    Ametrano & Bianchetti (2013), §2.4 — OIS-IBOR spread decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.ibor_curve import IBORCurve, IBORConventions
from pricebook.rfr import SpreadCurve, bootstrap_spread_curve


@dataclass
class OISIBORBasis:
    """OIS-IBOR basis term structure with optional decomposition.

    total_basis(T) = IBOR_forward(T) - OIS_forward(T)
                   = credit_component(T) + liquidity_component(T)

    The credit component reflects panel bank default risk (proxied by
    average CDS spread). The liquidity component is the residual.

    Attributes:
        spread_curve: the total basis spread term structure.
        credit_component: credit-driven portion (if decomposed).
        liquidity_component: liquidity-driven portion (if decomposed).
    """
    reference_date: date
    spread_curve: SpreadCurve
    tenor: str = "3M"
    credit_component: SpreadCurve | None = None
    liquidity_component: SpreadCurve | None = None

    def basis(self, d: date) -> float:
        """Total OIS-IBOR basis at date d."""
        return self.spread_curve.spread(d)

    def forward_basis(self, start: date, end: date) -> float:
        """Forward basis over [start, end] (midpoint evaluation)."""
        mid_ord = (start.toordinal() + end.toordinal()) // 2
        mid = date.fromordinal(mid_ord)
        return self.spread_curve.spread(mid)

    @classmethod
    def from_curves(
        cls,
        ibor_curve: IBORCurve,
        ois_curve: DiscountCurve,
        pillar_dates: list[date],
        tenor: str = "3M",
    ) -> OISIBORBasis:
        """Extract basis from two calibrated curves.

        At each pillar, basis = IBOR_forward - OIS_forward for a
        period matching the index tenor.
        """
        from datetime import timedelta

        ibor_conv = ibor_curve.conventions
        tenor_days = (ibor_conv.tenor_months or 3) * 30

        dates = []
        spreads = []
        for d in pillar_dates:
            d_end = d + timedelta(days=tenor_days)
            ibor_fwd = ibor_curve.forward_rate(d, d_end)
            ois_fwd = ois_curve.forward_rate(d, d_end)
            dates.append(d)
            spreads.append(ibor_fwd - ois_fwd)

        sc = SpreadCurve(ibor_curve.reference_date, dates, spreads)
        return cls(reference_date=ibor_curve.reference_date, spread_curve=sc, tenor=tenor)

    @classmethod
    def from_swap_rates(
        cls,
        reference_date: date,
        ibor_swap_rates: list[tuple[date, float]],
        ois_curve: DiscountCurve,
        conventions: IBORConventions | None = None,
        tenor: str = "3M",
    ) -> OISIBORBasis:
        """Bootstrap basis from IBOR swap rates vs OIS curve.

        Delegates to the upgraded bootstrap_spread_curve() which uses
        proper iterative root-finding.
        """
        kwargs = {}
        if conventions is not None:
            kwargs.update(
                float_frequency=conventions.float_frequency,
                float_day_count=conventions.float_day_count,
                fixed_frequency=conventions.fixed_frequency,
                fixed_day_count=conventions.fixed_day_count,
            )

        sc = bootstrap_spread_curve(reference_date, ibor_swap_rates, ois_curve, **kwargs)
        return cls(reference_date=reference_date, spread_curve=sc, tenor=tenor)

    def decompose(
        self,
        cds_spreads: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> None:
        """Decompose basis into credit + liquidity using panel bank CDS spreads.

        credit_component ≈ weighted average CDS spread of IBOR panel banks
        liquidity_component = total_basis - credit_component

        This is a simplified structural decomposition (Mercurio 2009).
        In practice the relationship is non-linear and maturity-dependent.

        Args:
            cds_spreads: {bank_name: 5Y CDS spread} for panel banks.
            weights: {bank_name: weight}. If None, equal weights.
        """
        if not cds_spreads:
            return

        if weights is None:
            n = len(cds_spreads)
            weights = {k: 1.0 / n for k in cds_spreads}

        avg_cds = sum(cds_spreads[k] * weights.get(k, 0.0) for k in cds_spreads)

        # Credit component = avg_cds at all pillars (flat approximation)
        credit_dates = self.spread_curve.dates[:]
        credit_spreads = [min(avg_cds, self.spread_curve.spread(d)) for d in credit_dates]

        # Liquidity = total - credit
        liq_spreads = [self.spread_curve.spread(d) - c for d, c in zip(credit_dates, credit_spreads)]

        self.credit_component = SpreadCurve(
            self.reference_date, credit_dates, credit_spreads,
        )
        self.liquidity_component = SpreadCurve(
            self.reference_date, credit_dates, liq_spreads,
        )
