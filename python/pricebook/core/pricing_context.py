"""
PricingContext — bundles market data for instrument pricing.

A PricingContext is "what date, what curves, what vols" in one object,
replacing the growing parameter lists in pricing methods. Instruments
query the context for what they need.

    ctx = PricingContext(
        valuation_date=date(2024, 1, 15),
        discount_curve=ois_curve,
        projection_curves={"USD.3M": libor_3m_curve},
        vol_surfaces={"ir": swaption_vol},
        credit_curves={"ACME": acme_survival},
        fx_spots={("EUR", "USD"): 1.0850},
    )

Multi-currency support: ``discount_curves``, ``inflation_curves``, and
``repo_curves`` are keyed by currency code. The original ``discount_curve``
field is kept for backward compatibility — ``get_discount_curve(ccy)``
checks ``discount_curves`` first, then falls back to ``discount_curve``.

FX translation: ``fx_translate()`` converts a PV from one currency to
``reporting_currency`` using the ``fx_spots`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.numerical_config import (
    DEFAULT_NUMERICAL_CONFIG,
    NumericalConfig,
)
from pricebook.core.survival_curve import SurvivalCurve


@dataclass
class PricingContext:
    """Immutable snapshot of market data for pricing.

    Attributes:
        valuation_date: the "as-of" date for the pricing run.
        discount_curve: the risk-free (OIS) discount curve (single-ccy shortcut).
        discount_curves: per-currency discount curves (e.g. ``{"USD": usd_ois, "EUR": eur_ois}``).
        projection_curves: named forward projection curves (e.g. "USD.3M").
        vol_surfaces: named volatility surfaces (e.g. "ir", "fx").
            Each must expose a ``vol(expiry, strike)`` method.
        credit_curves: named survival curves (e.g. issuer name).
        fx_spots: spot rates keyed by (base, quote) currency code tuples.
        inflation_curves: per-currency CPI/inflation curves.
        repo_curves: per-currency repo/funding curves.
        reporting_currency: base currency for P&L aggregation.
    """

    valuation_date: date
    discount_curve: DiscountCurve | None = None
    discount_curves: dict[str, DiscountCurve] = field(default_factory=dict)
    projection_curves: dict[str, DiscountCurve] = field(default_factory=dict)
    vol_surfaces: dict[str, object] = field(default_factory=dict)
    credit_curves: dict[str, SurvivalCurve] = field(default_factory=dict)
    fx_spots: dict[tuple[str, str], float] = field(default_factory=dict)
    inflation_curves: dict[str, object] = field(default_factory=dict)
    repo_curves: dict[str, DiscountCurve] = field(default_factory=dict)
    reporting_currency: str = "USD"
    # Stochastic credit models (CIR++, HW hazard rate, etc.)
    stochastic_credit_models: dict[str, object] = field(default_factory=dict)
    # CDS spread vol surfaces (CDSSpreadSmile per name)
    credit_vol_surfaces: dict[str, object] = field(default_factory=dict)
    # Credit correlations (for basket CLN)
    credit_correlations: dict[str, float] = field(default_factory=dict)
    # Numerical hyperparameters (MC paths, PDE grid, COS N, integration tol, ...).
    # `None` means the library default (`DEFAULT_NUMERICAL_CONFIG`) — pricers
    # should call `get_numerical_config()` rather than reading this field
    # directly so they get the default automatically.
    numerical_config: NumericalConfig | None = None

    # ---- Curve accessors ----

    def get_discount_curve(self, ccy: str | None = None) -> DiscountCurve:
        """Return a discount curve by currency.

        Looks up ``discount_curves[ccy]`` first; if not found (or *ccy*
        is ``None``), falls back to the single ``discount_curve`` field.
        Raises ``KeyError`` if neither is available.
        """
        if ccy is not None and ccy in self.discount_curves:
            return self.discount_curves[ccy]
        if self.discount_curve is not None:
            return self.discount_curve
        raise KeyError(
            f"No discount curve for '{ccy}' and no default discount_curve"
        )

    def get_projection_curve(self, name: str) -> DiscountCurve:
        """Return a projection curve by name, or raise KeyError."""
        if name not in self.projection_curves:
            raise KeyError(f"Projection curve '{name}' not in context")
        return self.projection_curves[name]

    def get_vol_surface(self, name: str) -> object:
        """Return a vol surface by name, or raise KeyError."""
        if name not in self.vol_surfaces:
            raise KeyError(f"Vol surface '{name}' not in context")
        return self.vol_surfaces[name]

    def get_credit_curve(self, name: str) -> SurvivalCurve:
        """Return a credit/survival curve by name, or raise KeyError."""
        if name not in self.credit_curves:
            raise KeyError(f"Credit curve '{name}' not in context")
        return self.credit_curves[name]

    def get_fx_spot(self, base: str, quote: str) -> float:
        """Return an FX spot rate, or raise KeyError."""
        key = (base, quote)
        if key not in self.fx_spots:
            raise KeyError(f"FX spot '{base}/{quote}' not in context")
        return self.fx_spots[key]

    def get_inflation_curve(self, ccy: str) -> object:
        """Return an inflation/CPI curve by currency, or raise KeyError."""
        if ccy not in self.inflation_curves:
            raise KeyError(f"Inflation curve '{ccy}' not in context")
        return self.inflation_curves[ccy]

    def get_repo_curve(self, ccy: str) -> DiscountCurve:
        """Return a repo/funding curve by currency, or raise KeyError."""
        if ccy not in self.repo_curves:
            raise KeyError(f"Repo curve '{ccy}' not in context")
        return self.repo_curves[ccy]

    # ---- Numerical config ----

    def get_numerical_config(self) -> NumericalConfig:
        """Return the attached `NumericalConfig`, or the library default.

        Pricers should read numerical hyperparameters through this accessor.
        Hard-coded defaults in pricer entry points should fall back to the
        accessor's value when the caller has not overridden them.
        """
        return (
            self.numerical_config
            if self.numerical_config is not None
            else DEFAULT_NUMERICAL_CONFIG
        )

    # ---- FX translation ----

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        """FX rate to convert ``from_ccy`` to ``to_ccy``.

        Tries direct lookup, then inverse. Returns 1.0 if
        ``from_ccy == to_ccy``.
        """
        if from_ccy == to_ccy:
            return 1.0
        key = (from_ccy, to_ccy)
        if key in self.fx_spots:
            return self.fx_spots[key]
        inv = (to_ccy, from_ccy)
        if inv in self.fx_spots:
            return 1.0 / self.fx_spots[inv]
        raise KeyError(
            f"No FX rate for '{from_ccy}/{to_ccy}' or its inverse"
        )

    def fx_translate(
        self,
        value: float,
        from_ccy: str,
        to_ccy: str | None = None,
    ) -> float:
        """Translate a value from *from_ccy* to *to_ccy*.

        Defaults to ``reporting_currency`` if *to_ccy* is ``None``.
        """
        target = to_ccy or self.reporting_currency
        return value * self.fx_rate(from_ccy, target)

    # ---- Copy with replacements ----

    def replace(self, **kwargs) -> "PricingContext":
        """Return a new PricingContext with specified fields replaced.

        Mutable containers (dicts) are defensively copied so the returned
        context cannot share state with `self`. This honours the
        "Immutable snapshot" contract from the class docstring.
        (Fix D.1 B3.)
        """
        def _pick_dict(name: str) -> dict:
            v = kwargs.get(name, getattr(self, name))
            return dict(v) if v else {}
        return PricingContext(
            valuation_date=kwargs.get("valuation_date", self.valuation_date),
            discount_curve=kwargs.get("discount_curve", self.discount_curve),
            discount_curves=_pick_dict("discount_curves"),
            projection_curves=_pick_dict("projection_curves"),
            vol_surfaces=_pick_dict("vol_surfaces"),
            credit_curves=_pick_dict("credit_curves"),
            fx_spots=_pick_dict("fx_spots"),
            inflation_curves=_pick_dict("inflation_curves"),
            repo_curves=_pick_dict("repo_curves"),
            reporting_currency=kwargs.get("reporting_currency", self.reporting_currency),
            stochastic_credit_models=_pick_dict("stochastic_credit_models"),
            credit_vol_surfaces=_pick_dict("credit_vol_surfaces"),
            credit_correlations=_pick_dict("credit_correlations"),
            numerical_config=kwargs.get("numerical_config", self.numerical_config),
        )

    @classmethod
    def simple(
        cls,
        valuation_date: date,
        rate: float = 0.05,
        vol: float | None = None,
        hazard: float | None = None,
    ) -> "PricingContext":
        """Build a simple PricingContext from flat rates.

        Convenience for quick pricing and testing.
        """
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.options.vol_surface import FlatVol

        curve = DiscountCurve.flat(valuation_date, rate)
        vol_surfaces = {}
        if vol is not None:
            vol_surfaces["ir"] = FlatVol(vol)

        credit_curves = {}
        if hazard is not None:
            from pricebook.core.survival_curve import SurvivalCurve
            credit_curves["default"] = SurvivalCurve.flat(valuation_date, hazard)

        return cls(
            valuation_date=valuation_date,
            discount_curve=curve,
            vol_surfaces=vol_surfaces,
            credit_curves=credit_curves,
        )


    def to_dict(self) -> dict:
        return dict(vars(self))   # copy so external mutation doesn't leak
from pricebook.core.serialisable import _register, make_payload, read_payload

PricingContext._SERIAL_TYPE = "pricing_context"

def _ctx_to_dict(self):
    """Emit every dataclass-declared field in the payload (Fix D.1 B2).

    Empty containers are emitted as empty dicts (not omitted), so the
    round-trip exactly reconstructs the original (Fix D.1 B1).
    """
    params = {"valuation_date": self.valuation_date.isoformat()}

    # Single-curve shortcut (legacy single-currency field).
    if self.discount_curve is not None:
        params["discount_curve"] = self.discount_curve.to_dict()

    # Per-currency curves — multi-currency support.
    params["discount_curves"] = {
        ccy: c.to_dict() for ccy, c in self.discount_curves.items()
    }
    params["projection_curves"] = {
        n: c.to_dict() for n, c in self.projection_curves.items()
    }
    params["credit_curves"] = {
        n: c.to_dict() for n, c in self.credit_curves.items()
    }
    params["inflation_curves"] = {
        ccy: (c.to_dict() if hasattr(c, "to_dict") else None)
        for ccy, c in self.inflation_curves.items()
    }
    params["repo_curves"] = {
        ccy: c.to_dict() for ccy, c in self.repo_curves.items()
    }

    # Vol surfaces — skip entries that aren't to_dict-capable (e.g. opaque
    # closures used in some sandbox paths). Recorded as None.
    params["vol_surfaces"] = {
        n: (v.to_dict() if hasattr(v, "to_dict") else None)
        for n, v in self.vol_surfaces.items()
    }
    params["credit_vol_surfaces"] = {
        n: (v.to_dict() if hasattr(v, "to_dict") else None)
        for n, v in self.credit_vol_surfaces.items()
    }
    params["stochastic_credit_models"] = {
        n: (m.to_dict() if hasattr(m, "to_dict") else None)
        for n, m in self.stochastic_credit_models.items()
    }

    # FX spots — stored as (base, quote) tuples in memory, serialised as "B/Q" strings.
    params["fx_spots"] = {
        f"{b}/{q}": r for (b, q), r in self.fx_spots.items()
    }

    # Plain scalar / numeric fields.
    params["credit_correlations"] = dict(self.credit_correlations)
    params["reporting_currency"] = self.reporting_currency

    # NumericalConfig (G1 P3 Slice 1). Dataclass not serialisable via the
    # registry; emit as a plain dict of its dataclass fields.
    if self.numerical_config is not None:
        from dataclasses import asdict
        params["numerical_config"] = asdict(self.numerical_config)
    else:
        params["numerical_config"] = None

    return make_payload(self, params)


@classmethod
def _ctx_from_dict(cls, d):
    """Round-trip every dataclass-declared field. Empty containers stay empty
    (no `or None` collapse — Fix D.1 B1)."""
    from datetime import date as _d
    from pricebook.core.serialisable import from_dict as _fd
    from pricebook.core.numerical_config import NumericalConfig
    p = read_payload(d, cls)

    def _fd_dict(payload: dict | None) -> dict:
        """Reconstruct each value via the serialisation registry; pass
        through `None` and non-dict values unchanged."""
        if not payload:
            return {}
        out = {}
        for k, v in payload.items():
            if isinstance(v, dict) and "type" in v:
                out[k] = _fd(v)
            else:
                out[k] = v   # preserve opaque values (e.g. None for non-serialisable vol surfaces)
        return out

    disc = _fd(p["discount_curve"]) if "discount_curve" in p else None
    discount_curves = _fd_dict(p.get("discount_curves"))
    projection_curves = _fd_dict(p.get("projection_curves"))
    credit_curves = _fd_dict(p.get("credit_curves"))
    inflation_curves = _fd_dict(p.get("inflation_curves"))
    repo_curves = _fd_dict(p.get("repo_curves"))
    vol_surfaces = _fd_dict(p.get("vol_surfaces"))
    credit_vol_surfaces = _fd_dict(p.get("credit_vol_surfaces"))
    stochastic_credit_models = _fd_dict(p.get("stochastic_credit_models"))

    fx_spots = {}
    for ps, r in p.get("fx_spots", {}).items():
        base_quote = ps.split("/")
        if len(base_quote) == 2:
            fx_spots[(base_quote[0], base_quote[1])] = r

    nc_payload = p.get("numerical_config")
    numerical_config = NumericalConfig(**nc_payload) if isinstance(nc_payload, dict) else None

    return cls(
        valuation_date=_d.fromisoformat(p["valuation_date"]),
        discount_curve=disc,
        discount_curves=discount_curves,
        projection_curves=projection_curves,
        vol_surfaces=vol_surfaces,
        credit_curves=credit_curves,
        fx_spots=fx_spots,
        inflation_curves=inflation_curves,
        repo_curves=repo_curves,
        reporting_currency=p.get("reporting_currency", "USD"),
        stochastic_credit_models=stochastic_credit_models,
        credit_vol_surfaces=credit_vol_surfaces,
        credit_correlations=dict(p.get("credit_correlations", {})),
        numerical_config=numerical_config,
    )

PricingContext.to_dict = _ctx_to_dict
PricingContext.from_dict = _ctx_from_dict
_register(PricingContext)
