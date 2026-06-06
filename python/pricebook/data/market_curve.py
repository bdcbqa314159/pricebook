"""MarketCurve: builds a DiscountCurve from any RateSource.

The user-facing API. Handles convention lookup, bootstrapping,
calibration, and visualisation. Works with any data provider
that implements the RateSource protocol.

    from pricebook.data.market_curve import MarketCurve

    curve = MarketCurve.euribor()                    # today, EUR
    curve = MarketCurve.euribor(date(2024, 6, 3))    # historical
    history = MarketCurve.euribor_year(2024)          # full year

    # Same pattern for any future source:
    # curve = MarketCurve.sofr()
    # curve = MarketCurve.sonia()

    # Or generic:
    curve = MarketCurve.from_source(my_source, date.today())

    # Use it:
    curve.df(d)
    curve.zero_rate(d)
    curve.forward_rate(d1, d2)
    curve.plot()
    curve.ns_fit()
    curve.dv01()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.curves.bootstrap import bootstrap
from pricebook.curves.curve_builder import get_conventions
from pricebook.curves.nelson_siegel import calibrate_nelson_siegel
from pricebook.data.rate_source import (
    RateSource, RateFixing, RateType, TenorDefinition, DEPOSIT_TENORS,
)


class MarketCurve:
    """Discount curve built from market rate fixings.

    Wraps a ``DiscountCurve`` with the source metadata, conventions,
    and built-in analytics/visualisation.

    The user never touches bootstrap parameters or day count conventions.
    """

    def __init__(
        self,
        curve: DiscountCurve,
        fixing: RateFixing,
        tenors: list[TenorDefinition],
        source_name: str,
        attribution: str,
    ):
        self._curve = curve
        self._fixing = fixing
        self._tenors = tenors
        self._source_name = source_name
        self._attribution = attribution

    # ═══════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def from_source(cls, source: RateSource, d: date | None = None) -> MarketCurve:
        """Build a curve from any RateSource for a given date.

        Looks up currency conventions automatically.
        """
        d = d or date.today()
        fixing = source.fetch(d)
        if fixing is None:
            raise ValueError(f"No data from {source.source_name} for {d}")
        return cls._build(fixing, source.tenors, source.source_name, source.attribution)

    @classmethod
    def from_source_year(cls, source: RateSource, year: int) -> MarketCurveHistory:
        """Build curves for every business day in a year."""
        fixings = source.fetch_year(year)
        curves = [cls._build(f, source.tenors, source.source_name, source.attribution)
                  for f in fixings]
        return MarketCurveHistory(year, curves, source.source_name, source.attribution)

    @classmethod
    def _build(cls, fixing: RateFixing, tenors: list[TenorDefinition],
               source_name: str, attribution: str) -> MarketCurve:
        """Internal: bootstrap curve from fixing using currency conventions."""
        ccy = fixing.currency
        conv = get_conventions(ccy)

        ref = fixing.date
        deposits = []
        active_tenors = []
        for t in tenors:
            rate = fixing.rates.get(t.key)
            if rate is not None:
                deposits.append((t.maturity(ref), rate))
                active_tenors.append(t)

        if not deposits:
            raise ValueError(f"No rates available for {ref} from {source_name}")

        curve = bootstrap(
            ref, deposits, swaps=[],
            deposit_day_count=conv.deposit_day_count,
            interpolation=conv.interpolation,
        )

        return cls(curve, fixing, active_tenors, source_name, attribution)

    # ── Shortcuts for known sources ──

    @classmethod
    def euribor(cls, d: date | None = None) -> MarketCurve:
        """EUR Euribor curve. Data from euriborrates.com."""
        from pricebook.data.euribor_source import EuriborSource
        return cls.from_source(EuriborSource(), d)

    @classmethod
    def euribor_year(cls, year: int) -> MarketCurveHistory:
        """Full year of EUR Euribor curves."""
        from pricebook.data.euribor_source import EuriborSource
        return cls.from_source_year(EuriborSource(), year)

    # Future shortcuts (when sources are built):
    # @classmethod
    # def sofr(cls, d=None) -> MarketCurve: ...
    # @classmethod
    # def sonia(cls, d=None) -> MarketCurve: ...

    # ═══════════════════════════════════════════════════════════════
    # DiscountCurve delegation
    # ═══════════════════════════════════════════════════════════════

    @property
    def reference_date(self) -> date:
        return self._curve.reference_date

    @property
    def currency(self) -> str:
        return self._fixing.currency

    @property
    def rates(self) -> dict[str, float]:
        """Raw fixings (decimal)."""
        return dict(self._fixing.rates)

    @property
    def curve(self) -> DiscountCurve:
        """The underlying DiscountCurve."""
        return self._curve

    def df(self, d: date) -> float:
        return self._curve.df(d)

    def zero_rate(self, d: date) -> float:
        return self._curve.zero_rate(d)

    def forward_rate(self, d1: date, d2: date) -> float:
        return self._curve.forward_rate(d1, d2)

    def instantaneous_forward(self, d: date) -> float:
        return self._curve.instantaneous_forward(d)

    def bumped(self, shift: float) -> DiscountCurve:
        return self._curve.bumped(shift)

    # ═══════════════════════════════════════════════════════════════
    # Analytics
    # ═══════════════════════════════════════════════════════════════

    def summary(self) -> dict:
        """Pillar-by-pillar: tenor, maturity, fixing, DF, zero, forward."""
        ref = self.reference_date
        rows = []
        prev = ref
        for t in self._tenors:
            mat = t.maturity(ref)
            rows.append({
                "tenor": t.label,
                "maturity": mat,
                "fixing": self._fixing.rates.get(t.key, 0),
                "df": self._curve.df(mat),
                "zero_rate": self._curve.zero_rate(mat),
                "forward_rate": self._curve.forward_rate(prev, mat),
            })
            prev = mat
        return {
            "date": ref, "currency": self.currency,
            "source": self._source_name, "pillars": rows,
        }

    def ns_fit(self) -> dict[str, float]:
        """Nelson-Siegel calibration to zero rates."""
        ref = self.reference_date
        tenors_yr = [t.years for t in self._tenors]
        zeros = [self._curve.zero_rate(t.maturity(ref)) for t in self._tenors]
        return calibrate_nelson_siegel(tenors_yr, zeros)

    def dv01(self, notional: float = 10_000.0) -> dict[str, float]:
        """DV01 per tenor (1bp parallel shift)."""
        ref = self.reference_date
        bumped = self._curve.bumped(0.0001)
        result = {}
        for t in self._tenors:
            mat = t.maturity(ref)
            result[t.label] = (bumped.df(mat) - self._curve.df(mat)) * notional
        return result

    # ═══════════════════════════════════════════════════════════════
    # Visualisation
    # ═══════════════════════════════════════════════════════════════

    def plot(self, show: bool = True):
        """Plot: DF, zero rates, instantaneous forwards."""
        import matplotlib.pyplot as plt
        from pricebook.viz import configure_theme
        from pricebook.viz._theme import get_theme

        configure_theme()
        theme = get_theme()
        ref = self.reference_date

        max_days = max(int(t.years * 365) for t in self._tenors) + 10
        plot_dates = [ref + timedelta(days=d) for d in range(1, max_days)]
        days = [(d - ref).days for d in plot_dates]
        dfs = [self._curve.df(d) for d in plot_dates]
        zeros = [self._curve.zero_rate(d) * 100 for d in plot_dates]
        fwds = [self._curve.instantaneous_forward(d) * 100 for d in plot_dates]

        pillar_days = [(t.maturity(ref) - ref).days for t in self._tenors]
        pillar_zeros = [self._curve.zero_rate(t.maturity(ref)) * 100 for t in self._tenors]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(days, dfs, color=theme.colors[0], linewidth=theme.line_width)
        axes[0].set_xlabel("Days"); axes[0].set_ylabel("DF")
        axes[0].set_title("Discount Factors", fontsize=theme.title_size)
        axes[0].grid(True, alpha=theme.grid_alpha)

        axes[1].plot(days, zeros, color=theme.colors[2], linewidth=theme.line_width)
        axes[1].scatter(pillar_days, pillar_zeros, color=theme.colors[1],
                        s=60, zorder=5, label="Pillars")
        axes[1].set_xlabel("Days"); axes[1].set_ylabel("Rate (%)")
        axes[1].set_title("Zero Rates", fontsize=theme.title_size)
        axes[1].legend(); axes[1].grid(True, alpha=theme.grid_alpha)

        axes[2].plot(days, fwds, color=theme.colors[3], linewidth=theme.line_width)
        axes[2].set_xlabel("Days"); axes[2].set_ylabel("Rate (%)")
        axes[2].set_title("Inst. Forward", fontsize=theme.title_size)
        axes[2].grid(True, alpha=theme.grid_alpha)

        fig.suptitle(f"{self.currency} Curve — {ref}  (Source: {self._source_name})",
                     fontsize=theme.title_size)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def __repr__(self) -> str:
        tenors = [t.key for t in self._tenors]
        return f"MarketCurve({self.currency}, {self.reference_date}, {tenors}, source={self._source_name})"


class MarketCurveHistory:
    """A year of daily MarketCurves from the same source."""

    def __init__(self, year: int, curves: list[MarketCurve],
                 source_name: str, attribution: str):
        self.year = year
        self.curves = curves
        self._source_name = source_name
        self._attribution = attribution

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, idx) -> MarketCurve:
        return self.curves[idx]

    @property
    def dates(self) -> list[date]:
        return [c.reference_date for c in self.curves]

    @property
    def currency(self) -> str:
        return self.curves[0].currency if self.curves else ""

    def to_dataframe(self) -> Any:
        """Rates DataFrame (tenors as columns, %)."""
        import pandas as pd
        rows = []
        for c in self.curves:
            row = {"date": c.reference_date}
            for t in c._tenors:
                rate = c.rates.get(t.key)
                if rate is not None:
                    row[t.label] = rate * 100
            rows.append(row)
        return pd.DataFrame(rows).set_index("date")

    def plot_rates(self, show: bool = True):
        """All tenor rates through time."""
        import matplotlib.pyplot as plt
        from pricebook.viz import configure_theme
        from pricebook.viz._theme import get_theme

        configure_theme()
        theme = get_theme()
        df = self.to_dataframe()

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, col in enumerate(df.columns):
            ax.plot(df.index, df[col], label=col,
                    color=theme.colors[i % len(theme.colors)],
                    linewidth=theme.line_width)
        ax.set_xlabel("Date"); ax.set_ylabel("Rate (%)")
        ax.set_title(f"{self.currency} Rates — {self.year}  (Source: {self._source_name})",
                     fontsize=theme.title_size)
        ax.legend(loc="best")
        ax.grid(True, alpha=theme.grid_alpha)
        fig.autofmt_xdate()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_slope(self, long: str = "12m", short: str = "1w", show: bool = True):
        """Curve slope: long − short tenor."""
        import matplotlib.pyplot as plt
        from pricebook.viz import configure_theme
        from pricebook.viz._theme import get_theme

        configure_theme()
        theme = get_theme()
        df = self.to_dataframe()

        # Find column names from tenor keys
        long_col = next((t.label for c in self.curves for t in c._tenors if t.key == long), long)
        short_col = next((t.label for c in self.curves for t in c._tenors if t.key == short), short)
        slope = df[long_col] - df[short_col]

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.fill_between(slope.index, slope.values, 0,
                        where=slope >= 0, color=theme.colors[2], alpha=0.4, label="Normal")
        ax.fill_between(slope.index, slope.values, 0,
                        where=slope < 0, color=theme.colors[1], alpha=0.4, label="Inverted")
        ax.plot(slope.index, slope.values, color=theme.foreground, linewidth=0.8)
        ax.axhline(0, color=theme.foreground, linewidth=0.5)
        ax.set_ylabel("Spread (pp)")
        ax.set_title(f"{self.currency} Slope: {long_col} − {short_col} ({self.year})"
                     f"\nSource: {self._source_name}", fontsize=theme.title_size)
        ax.legend(); ax.grid(True, alpha=theme.grid_alpha)
        fig.autofmt_xdate()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def correlation(self, show: bool = True):
        """Correlation heatmap of daily rate changes."""
        from pricebook.viz import correlation_heatmap
        import matplotlib.pyplot as plt

        df = self.to_dataframe().diff().dropna()
        fig = correlation_heatmap(
            df, title=f"{self.currency} Daily Change Correlation — {self.year}"
                      f"\nSource: {self._source_name}",
        )
        if show:
            plt.show()
        return fig

    def __repr__(self) -> str:
        return f"MarketCurveHistory({self.currency}, {self.year}, {len(self.curves)} days)"
