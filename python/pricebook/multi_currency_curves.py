"""Multi-currency curve set: complete curve environment for pricing.

Builds and stores OIS + IBOR + tenor basis + xccy basis for one or
more currencies. Provides factory methods for standard configurations.

    from pricebook.multi_currency_curves import MultiCurrencyCurveSet

    curves = MultiCurrencyCurveSet.eur_with_euribor(
        ref, estr_rates, euribor_3m_swaps)
    ctx = curves.to_pricing_context()

References:
    Henrard (2014), Ch. 2-5 — multi-curve framework.
    Ametrano & Bianchetti (2013) — bootstrapping across currencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.ibor_curve import IBORCurve, IBORConventions, bootstrap_ibor
from pricebook.ibor_curve import EURIBOR_3M_CONVENTIONS, EURIBOR_6M_CONVENTIONS
from pricebook.ois import bootstrap_ois
from pricebook.pricing_context import PricingContext
from pricebook.tenor_basis import TenorBasis, bootstrap_tenor_basis


@dataclass
class IBORCurveSpec:
    """Specification for one IBOR curve within a currency set."""
    conventions: IBORConventions
    deposits: list[tuple[date, float]] | None = None
    fras: list[tuple[date, date, float]] | None = None
    futures: list[tuple[date, date, float]] | None = None
    swaps: list[tuple[date, float]] | None = None


@dataclass
class CurrencyCurveSetSpec:
    """Specification for all curves in one currency."""
    currency: str
    ois_rates: list[tuple[date, float]]
    ibor_specs: list[IBORCurveSpec] = field(default_factory=list)
    basis_swap_quotes: dict[str, list[tuple[date, float]]] = field(default_factory=dict)


class MultiCurrencyCurveSet:
    """Complete curve environment: OIS + IBOR per tenor + basis + xccy.

    Hierarchical structure:
    - Per currency: OIS curve, one or more IBORCurves, tenor basis
    - Cross-currency: xccy basis curves

    Usage:
        curves = MultiCurrencyCurveSet()
        curves.add_currency("EUR", estr_curve, {"EURIBOR_3M": ibor_3m})
        ctx = curves.to_pricing_context()
    """

    def __init__(self):
        self._ois: dict[str, DiscountCurve] = {}
        self._ibor: dict[str, IBORCurve] = {}           # keyed by index name
        self._tenor_basis: dict[str, TenorBasis] = {}    # keyed by "CCY_3M_6M"
        self._xccy: dict[str, DiscountCurve] = {}        # keyed by "BASE_QUOTE"
        self._currencies: set[str] = set()

    def add_currency(
        self,
        currency: str,
        ois_curve: DiscountCurve,
        ibor_curves: dict[str, IBORCurve] | None = None,
        tenor_basis: dict[str, TenorBasis] | None = None,
    ) -> None:
        """Add a currency with its curves."""
        self._currencies.add(currency)
        self._ois[currency] = ois_curve
        if ibor_curves:
            self._ibor.update(ibor_curves)
        if tenor_basis:
            for key, tb in tenor_basis.items():
                self._tenor_basis[f"{currency}_{key}"] = tb

    def add_xccy_basis(
        self,
        base_ccy: str,
        quote_ccy: str,
        basis_curve: DiscountCurve,
    ) -> None:
        """Add cross-currency basis curve."""
        self._xccy[f"{base_ccy}_{quote_ccy}"] = basis_curve

    @property
    def currencies(self) -> set[str]:
        return self._currencies.copy()

    def ois(self, currency: str) -> DiscountCurve:
        """OIS discount curve for a currency."""
        if currency not in self._ois:
            raise KeyError(f"No OIS curve for {currency}")
        return self._ois[currency]

    def ibor(self, index_name: str) -> IBORCurve:
        """IBOR projection curve by index name (e.g. 'EURIBOR_3M')."""
        if index_name not in self._ibor:
            raise KeyError(f"No IBOR curve for {index_name}")
        return self._ibor[index_name]

    def tenor_basis_for(self, currency: str, pair: str) -> TenorBasis:
        """Tenor basis for a currency and pair (e.g. 'EUR', '3M_6M')."""
        key = f"{currency}_{pair}"
        if key not in self._tenor_basis:
            raise KeyError(f"No tenor basis for {key}")
        return self._tenor_basis[key]

    def xccy_curve(self, base_ccy: str, quote_ccy: str) -> DiscountCurve:
        """Cross-currency basis-adjusted curve."""
        key = f"{base_ccy}_{quote_ccy}"
        if key not in self._xccy:
            raise KeyError(f"No xccy curve for {key}")
        return self._xccy[key]

    def to_pricing_context(
        self,
        valuation_date: date | None = None,
        reporting_currency: str = "USD",
    ) -> PricingContext:
        """Build a PricingContext with all curves populated.

        Uses the reporting currency's OIS as the primary discount curve.
        Projection curves are keyed by index name.
        """
        if reporting_currency not in self._ois:
            # Fall back to first available
            reporting_currency = next(iter(self._ois))

        disc = self._ois[reporting_currency]
        ref = valuation_date or disc.reference_date

        projection_curves = {
            name: curve.projection_curve for name, curve in self._ibor.items()
        }

        return PricingContext(
            valuation_date=ref,
            discount_curve=disc,
            projection_curves=projection_curves if projection_curves else None,
        )

    # ---- Factory methods ----

    @classmethod
    def build(
        cls,
        reference_date: date,
        specs: list[CurrencyCurveSetSpec],
    ) -> MultiCurrencyCurveSet:
        """Build complete multi-currency curve set from specs.

        Construction order per currency:
        1. Bootstrap OIS from OIS rates
        2. Bootstrap IBOR curves (using OIS as discount)
        3. Bootstrap tenor basis where applicable
        """
        result = cls()

        for spec in specs:
            # Step 1: OIS
            ois = bootstrap_ois(reference_date, spec.ois_rates)

            # Step 2: IBOR curves
            ibor_curves: dict[str, IBORCurve] = {}
            for ibor_spec in spec.ibor_specs:
                curve = bootstrap_ibor(
                    reference_date, ibor_spec.conventions, ois,
                    deposits=ibor_spec.deposits,
                    fras=ibor_spec.fras,
                    futures=ibor_spec.futures,
                    swaps=ibor_spec.swaps,
                )
                ibor_curves[ibor_spec.conventions.name] = curve

            # Step 3: Tenor basis
            tenor_bases: dict[str, TenorBasis] = {}
            for pair_key, quotes in spec.basis_swap_quotes.items():
                # Parse pair: "3M_6M" → find short IBOR in already-built curves
                parts = pair_key.split("_")
                if len(parts) != 2:
                    continue
                # Find the short-tenor curve among already-built IBOR curves
                short_ibor = None
                long_conv = None
                for name, curve in ibor_curves.items():
                    if parts[0] in name:
                        short_ibor = curve
                    if parts[1] in name:
                        long_conv = curve.conventions
                if short_ibor is None:
                    continue
                # If long conventions not found among specs, try known conventions
                if long_conv is None:
                    if spec.currency == "EUR" and "6M" in parts[1]:
                        long_conv_obj = EURIBOR_6M_CONVENTIONS
                    else:
                        continue
                else:
                    long_conv_obj = long_conv
                long_ibor, tb = bootstrap_tenor_basis(
                    reference_date, short_ibor, ois, quotes, long_conv_obj,
                )
                ibor_curves[long_conv_obj.name] = long_ibor
                tenor_bases[pair_key] = tb

            result.add_currency(spec.currency, ois, ibor_curves, tenor_bases)

        return result

    @classmethod
    def usd_post_libor(
        cls,
        reference_date: date,
        ois_rates: list[tuple[date, float]],
    ) -> MultiCurrencyCurveSet:
        """USD post-LIBOR: SOFR OIS only, no IBOR projection curves."""
        ois = bootstrap_ois(reference_date, ois_rates)
        result = cls()
        result.add_currency("USD", ois)
        return result

    @classmethod
    def eur_with_euribor(
        cls,
        reference_date: date,
        estr_rates: list[tuple[date, float]],
        euribor_3m_swaps: list[tuple[date, float]],
        euribor_6m_swaps: list[tuple[date, float]] | None = None,
        basis_3m_6m: list[tuple[date, float]] | None = None,
    ) -> MultiCurrencyCurveSet:
        """EUR with ESTR OIS + EURIBOR 3M + optional EURIBOR 6M."""
        ois = bootstrap_ois(reference_date, estr_rates)

        ibor_3m = bootstrap_ibor(
            reference_date, EURIBOR_3M_CONVENTIONS, ois,
            swaps=euribor_3m_swaps,
        )

        ibor_curves = {"EURIBOR_3M": ibor_3m}
        tenor_bases = {}

        if euribor_6m_swaps and not basis_3m_6m:
            # Direct bootstrap of 6M
            ibor_6m = bootstrap_ibor(
                reference_date, EURIBOR_6M_CONVENTIONS, ois,
                swaps=euribor_6m_swaps,
            )
            ibor_curves["EURIBOR_6M"] = ibor_6m
        elif basis_3m_6m:
            # 6M via tenor basis from 3M
            ibor_6m, tb = bootstrap_tenor_basis(
                reference_date, ibor_3m, ois,
                basis_3m_6m, EURIBOR_6M_CONVENTIONS,
            )
            ibor_curves["EURIBOR_6M"] = ibor_6m
            tenor_bases["3M_6M"] = tb

        result = cls()
        result.add_currency("EUR", ois, ibor_curves, tenor_bases)
        return result
