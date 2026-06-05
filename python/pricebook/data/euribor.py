"""Euribor curve: from market data to DiscountCurve in one step.

The user-facing API. Handles data fetching, convention mapping,
bootstrapping, and calibration internally.

    from pricebook.data.euribor import EuriborCurve

    curve = EuriborCurve.today()          # live curve
    curve = EuriborCurve.from_date(d)     # historical
    history = EuriborCurve.year(2024)     # full year

    curve.df(mat)                         # discount factor
    curve.zero_rate(mat)                  # zero rate
    curve.forward_rate(d1, d2)            # forward rate
    curve.ns_fit()                        # Nelson-Siegel parameters
    curve.plot()                          # visualise

DATA SOURCE: https://euriborrates.com/
    An independent, non-commercial, non-profit information resource.
    All Euribor data is sourced from and attributed to euriborrates.com.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
from dateutil.relativedelta import relativedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.day_count import DayCountConvention
from pricebook.curves.bootstrap import bootstrap
from pricebook.curves.nelson_siegel import (
    calibrate_nelson_siegel, nelson_siegel_yield,
)
from pricebook.data.euribor_loader import (
    fetch_date, fetch_year_all_tenors, EuriborFixing,
    attribution as _attribution, TENORS, TENOR_LABELS,
)

# EUR money market conventions (internal — user never sees these)
_TENOR_OFFSETS = {
    "1w": relativedelta(weeks=1),
    "1m": relativedelta(months=1),
    "3m": relativedelta(months=3),
    "6m": relativedelta(months=6),
    "12m": relativedelta(months=12),
}
_TENOR_YEARS = {"1w": 7 / 365, "1m": 1 / 12, "3m": 0.25, "6m": 0.5, "12m": 1.0}


class EuriborCurve:
    """EUR discount curve built from Euribor fixings.

    All data sourced from https://euriborrates.com/

    The curve is a full ``DiscountCurve`` — call ``.df()``,
    ``.zero_rate()``, ``.forward_rate()``, ``.instantaneous_forward()``,
    ``.bumped()`` directly.
    """

    def __init__(self, fixing: EuriborFixing, curve: DiscountCurve):
        self._fixing = fixing
        self._curve = curve

    # ── Constructors ──

    @classmethod
    def today(cls) -> EuriborCurve:
        """Fetch today's Euribor fixings and build a curve."""
        fixing = fetch_date(date.today())
        if fixing is None:
            raise ValueError(f"No Euribor fixing available for {date.today()} (weekend/holiday?)")
        return cls._from_fixing(fixing)

    @classmethod
    def from_date(cls, d: date) -> EuriborCurve:
        """Build a curve from a historical date's fixings."""
        fixing = fetch_date(d)
        if fixing is None:
            raise ValueError(f"No Euribor fixing available for {d}")
        return cls._from_fixing(fixing)

    @classmethod
    def year(cls, yr: int, delay: float = 1.5) -> EuriborHistory:
        """Fetch a full year of daily curves (all 5 tenors).

        Returns an ``EuriborHistory`` with one curve per business day.
        """
        fixings = fetch_year_all_tenors(yr, delay_between_tenors=delay)
        curves = [cls._from_fixing(f) for f in fixings]
        return EuriborHistory(yr, curves)

    @classmethod
    def _from_fixing(cls, fixing: EuriborFixing) -> EuriborCurve:
        """Internal: bootstrap a DiscountCurve from an EuriborFixing."""
        ref = fixing.date
        deposits = [
            (ref + _TENOR_OFFSETS[t], fixing.rates[t])
            for t in TENORS if t in fixing.rates
        ]
        curve = bootstrap(
            ref, deposits, swaps=[],
            deposit_day_count=DayCountConvention.ACT_360,
            interpolation=InterpolationMethod.LOG_LINEAR,
        )
        return cls(fixing, curve)

    # ── DiscountCurve delegation ──

    @property
    def reference_date(self) -> date:
        return self._curve.reference_date

    @property
    def rates(self) -> dict[str, float]:
        """Raw Euribor fixings (decimal)."""
        return dict(self._fixing.rates)

    def df(self, d: date) -> float:
        """Discount factor at date d."""
        return self._curve.df(d)

    def zero_rate(self, d: date) -> float:
        """Continuously compounded zero rate to date d."""
        return self._curve.zero_rate(d)

    def forward_rate(self, d1: date, d2: date) -> float:
        """Simply compounded forward rate from d1 to d2."""
        return self._curve.forward_rate(d1, d2)

    def instantaneous_forward(self, d: date) -> float:
        """Instantaneous forward rate at date d."""
        return self._curve.instantaneous_forward(d)

    def bumped(self, shift: float) -> DiscountCurve:
        """Parallel-shifted curve (for DV01)."""
        return self._curve.bumped(shift)

    @property
    def curve(self) -> DiscountCurve:
        """The underlying DiscountCurve object."""
        return self._curve

    # ── Analytics ──

    def ns_fit(self) -> dict[str, float]:
        """Calibrate Nelson-Siegel to the Euribor zero rates.

        Returns dict with beta0, beta1, beta2, tau, rmse.
        """
        tenors = [_TENOR_YEARS[t] for t in TENORS if t in self._fixing.rates]
        zeros = [
            self._curve.zero_rate(self.reference_date + _TENOR_OFFSETS[t])
            for t in TENORS if t in self._fixing.rates
        ]
        return calibrate_nelson_siegel(tenors, zeros)

    def dv01(self, notional: float = 10_000.0) -> dict[str, float]:
        """DV01 per tenor (1bp parallel shift)."""
        bumped = self._curve.bumped(0.0001)
        result = {}
        for t in TENORS:
            if t not in self._fixing.rates:
                continue
            mat = self.reference_date + _TENOR_OFFSETS[t]
            sens = (bumped.df(mat) - self._curve.df(mat)) * notional
            result[TENOR_LABELS[t]] = sens
        return result

    def summary(self) -> dict:
        """Summary table: tenor, maturity, rate, DF, zero, forward."""
        rows = []
        prev = self.reference_date
        for t in TENORS:
            if t not in self._fixing.rates:
                continue
            mat = self.reference_date + _TENOR_OFFSETS[t]
            rows.append({
                "tenor": TENOR_LABELS[t],
                "maturity": mat,
                "fixing": self._fixing.rates[t],
                "df": self._curve.df(mat),
                "zero_rate": self._curve.zero_rate(mat),
                "forward_rate": self._curve.forward_rate(prev, mat),
            })
            prev = mat
        return {"date": self.reference_date, "source": "euriborrates.com", "pillars": rows}

    # ── Visualisation ──

    def plot(self, show: bool = True):
        """Plot the curve: DF, zero rate, and instantaneous forward."""
        import matplotlib.pyplot as plt
        from pricebook.viz import configure_theme
        from pricebook.viz._theme import get_theme

        configure_theme()
        theme = get_theme()
        ref = self.reference_date

        plot_dates = [ref + timedelta(days=d) for d in range(1, 370)]
        days = [(d - ref).days for d in plot_dates]
        dfs = [self._curve.df(d) for d in plot_dates]
        zeros = [self._curve.zero_rate(d) * 100 for d in plot_dates]
        fwds = [self._curve.instantaneous_forward(d) * 100 for d in plot_dates]

        pillar_days = [(ref + _TENOR_OFFSETS[t] - ref).days for t in TENORS if t in self._fixing.rates]
        pillar_zeros = [self._curve.zero_rate(ref + _TENOR_OFFSETS[t]) * 100
                        for t in TENORS if t in self._fixing.rates]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(days, dfs, color=theme.colors[0], linewidth=theme.line_width)
        axes[0].set_xlabel("Days")
        axes[0].set_ylabel("DF")
        axes[0].set_title("Discount Factors", fontsize=theme.title_size)
        axes[0].grid(True, alpha=theme.grid_alpha)

        axes[1].plot(days, zeros, color=theme.colors[2], linewidth=theme.line_width)
        axes[1].scatter(pillar_days, pillar_zeros, color=theme.colors[1], s=60, zorder=5, label="Pillars")
        axes[1].set_xlabel("Days")
        axes[1].set_ylabel("Rate (%)")
        axes[1].set_title("Zero Rates", fontsize=theme.title_size)
        axes[1].legend()
        axes[1].grid(True, alpha=theme.grid_alpha)

        axes[2].plot(days, fwds, color=theme.colors[3], linewidth=theme.line_width)
        axes[2].set_xlabel("Days")
        axes[2].set_ylabel("Rate (%)")
        axes[2].set_title("Inst. Forward", fontsize=theme.title_size)
        axes[2].grid(True, alpha=theme.grid_alpha)

        fig.suptitle(f"EUR Euribor Curve — {ref}  (Source: euriborrates.com)",
                     fontsize=theme.title_size)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def __repr__(self) -> str:
        return f"EuriborCurve({self.reference_date}, tenors={list(self._fixing.rates.keys())})"


class EuriborHistory:
    """A year of daily EuriborCurves.

    Data sourced from https://euriborrates.com/
    """

    def __init__(self, year: int, curves: list[EuriborCurve]):
        self.year = year
        self.curves = curves

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, idx) -> EuriborCurve:
        return self.curves[idx]

    @property
    def dates(self) -> list[date]:
        return [c.reference_date for c in self.curves]

    def to_dataframe(self) -> Any:
        """Rates DataFrame (tenors as columns, dates as index, in %)."""
        import pandas as pd
        rows = []
        for c in self.curves:
            row = {"date": c.reference_date}
            for t in TENORS:
                if t in c.rates:
                    row[TENOR_LABELS[t]] = c.rates[t] * 100
            rows.append(row)
        return pd.DataFrame(rows).set_index("date")

    def plot_rates(self, show: bool = True):
        """Plot all tenor rates through the year."""
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
        ax.set_xlabel("Date")
        ax.set_ylabel("Rate (%)")
        ax.set_title(f"Euribor Rates — {self.year}  (Source: euriborrates.com)",
                     fontsize=theme.title_size)
        ax.legend(loc="best")
        ax.grid(True, alpha=theme.grid_alpha)
        fig.autofmt_xdate()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_slope(self, long: str = "12m", short: str = "1w", show: bool = True):
        """Plot the curve slope (long tenor − short tenor)."""
        import matplotlib.pyplot as plt
        from pricebook.viz import configure_theme
        from pricebook.viz._theme import get_theme

        configure_theme()
        theme = get_theme()
        df = self.to_dataframe()
        long_col = TENOR_LABELS[long]
        short_col = TENOR_LABELS[short]
        slope = df[long_col] - df[short_col]

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.fill_between(slope.index, slope.values, 0,
                        where=slope >= 0, color=theme.colors[2], alpha=0.4, label="Normal")
        ax.fill_between(slope.index, slope.values, 0,
                        where=slope < 0, color=theme.colors[1], alpha=0.4, label="Inverted")
        ax.plot(slope.index, slope.values, color=theme.foreground, linewidth=0.8)
        ax.axhline(0, color=theme.foreground, linewidth=0.5)
        ax.set_ylabel("Spread (pp)")
        ax.set_title(f"Euribor Slope: {long_col} − {short_col} ({self.year})"
                     f"\nSource: euriborrates.com", fontsize=theme.title_size)
        ax.legend()
        ax.grid(True, alpha=theme.grid_alpha)
        fig.autofmt_xdate()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def correlation(self, show: bool = True):
        """Correlation heatmap of daily rate changes."""
        from pricebook.viz import correlation_heatmap
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        changes = df.diff().dropna()
        fig = correlation_heatmap(
            changes,
            title=f"Euribor Daily Change Correlation — {self.year}\nSource: euriborrates.com",
        )
        if show:
            plt.show()
        return fig

    def __repr__(self) -> str:
        return f"EuriborHistory({self.year}, {len(self.curves)} days)"


def attribution() -> str:
    """Data source attribution — always display when using Euribor data."""
    return _attribution()
