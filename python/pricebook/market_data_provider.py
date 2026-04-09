"""Market data providers: FRED, ECB, and sample data.

Fetch rates, yields, and CPI from external sources. Providers return
standardised time series that can be converted to curves.

    from pricebook.market_data_provider import (
        FREDProvider, ECBProvider, SampleProvider,
        build_curve_from_series,
    )

    provider = FREDProvider(api_key="...")
    sofr = provider.fetch("SOFR")

    # Or use bundled sample data (no network)
    provider = SampleProvider()
    sofr = provider.fetch("SOFR")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from pricebook.discount_curve import DiscountCurve


# ---- Time series ----

@dataclass
class RateSeries:
    """A time series of rate observations."""
    name: str
    currency: str
    unit: str  # "rate", "index", "yield"
    dates: list[date]
    values: list[float]

    def latest(self) -> tuple[date, float]:
        if not self.dates:
            raise ValueError(f"Empty series: {self.name}")
        return self.dates[-1], self.values[-1]

    def on_date(self, d: date) -> float | None:
        """Look up value on a specific date."""
        for dt, v in zip(self.dates, self.values):
            if dt == d:
                return v
        return None

    def between(self, start: date, end: date) -> "RateSeries":
        """Filter to date range [start, end]."""
        filtered = [(d, v) for d, v in zip(self.dates, self.values) if start <= d <= end]
        if not filtered:
            return RateSeries(self.name, self.currency, self.unit, [], [])
        dates, values = zip(*filtered)
        return RateSeries(self.name, self.currency, self.unit, list(dates), list(values))


# ---- Provider interface ----

class MarketDataProvider:
    """Base class for market data providers."""

    def fetch(self, series_name: str, start: date | None = None, end: date | None = None) -> RateSeries:
        raise NotImplementedError

    def available_series(self) -> list[str]:
        raise NotImplementedError


# ---- FRED provider ----

# FRED series IDs for common rates
FRED_SERIES = {
    "SOFR": "SOFR",
    "FED_FUNDS": "EFFR",
    "UST_3M": "DGS3MO",
    "UST_6M": "DGS6MO",
    "UST_1Y": "DGS1",
    "UST_2Y": "DGS2",
    "UST_5Y": "DGS5",
    "UST_10Y": "DGS10",
    "UST_30Y": "DGS30",
    "CPI": "CPIAUCSL",
}


class FREDProvider(MarketDataProvider):
    """Fetch data from FRED (Federal Reserve Economic Data).

    Requires an API key from https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        api_key: FRED API key.
        cache: if True, cache results in memory for the session.
    """

    def __init__(self, api_key: str, cache: bool = True):
        self.api_key = api_key
        self._cache: dict[str, RateSeries] = {} if cache else None

    def available_series(self) -> list[str]:
        return list(FRED_SERIES.keys())

    def fetch(self, series_name: str, start: date | None = None, end: date | None = None) -> RateSeries:
        if series_name not in FRED_SERIES:
            raise KeyError(f"Unknown FRED series: {series_name}. Available: {list(FRED_SERIES.keys())}")

        cache_key = f"{series_name}:{start}:{end}"
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        fred_id = FRED_SERIES[series_name]
        result = self._fetch_fred(fred_id, series_name, start, end)

        if self._cache is not None:
            self._cache[cache_key] = result
        return result

    def _fetch_fred(self, fred_id: str, name: str, start: date | None, end: date | None) -> RateSeries:
        """Fetch from FRED API."""
        import urllib.request
        import urllib.parse

        params = {
            "series_id": fred_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start.isoformat()
        if end:
            params["observation_end"] = end.isoformat()

        url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        dates = []
        values = []
        for obs in data.get("observations", []):
            if obs["value"] == ".":
                continue
            dates.append(date.fromisoformat(obs["date"]))
            values.append(float(obs["value"]) / 100.0 if "CPI" not in name else float(obs["value"]))

        currency = "USD"
        unit = "index" if "CPI" in name else "rate"

        return RateSeries(name, currency, unit, dates, values)


# ---- ECB provider ----

ECB_SERIES = {
    "ESTR": "FM.B.U2.EUR.4F.KR.DFR.LEV",
    "EURIBOR_3M": "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
    "EURIBOR_6M": "FM.M.U2.EUR.RT.MM.EURIBOR6MD_.HSTA",
    "EUR_GOVT_2Y": "FM.M.U2.EUR.4F.BB.U2_2Y.YLD",
    "EUR_GOVT_5Y": "FM.M.U2.EUR.4F.BB.U2_5Y.YLD",
    "EUR_GOVT_10Y": "FM.M.U2.EUR.4F.BB.U2_10Y.YLD",
}


class ECBProvider(MarketDataProvider):
    """Fetch data from ECB Statistical Data Warehouse.

    No API key required — public access.
    """

    def __init__(self, cache: bool = True):
        self._cache: dict[str, RateSeries] = {} if cache else None

    def available_series(self) -> list[str]:
        return list(ECB_SERIES.keys())

    def fetch(self, series_name: str, start: date | None = None, end: date | None = None) -> RateSeries:
        if series_name not in ECB_SERIES:
            raise KeyError(f"Unknown ECB series: {series_name}. Available: {list(ECB_SERIES.keys())}")

        cache_key = f"{series_name}:{start}:{end}"
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        ecb_id = ECB_SERIES[series_name]
        result = self._fetch_ecb(ecb_id, series_name, start, end)

        if self._cache is not None:
            self._cache[cache_key] = result
        return result

    def _fetch_ecb(self, ecb_id: str, name: str, start: date | None, end: date | None) -> RateSeries:
        """Fetch from ECB SDW API (CSV format)."""
        import urllib.request
        import csv
        import io

        url = f"https://data-api.ecb.europa.eu/service/data/{ecb_id}?format=csvdata"
        if start:
            url += f"&startPeriod={start.isoformat()}"
        if end:
            url += f"&endPeriod={end.isoformat()}"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")

        dates = []
        values = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            period = row.get("TIME_PERIOD", "")
            value = row.get("OBS_VALUE", "")
            if not period or not value:
                continue
            try:
                d = date.fromisoformat(period) if "-" in period and len(period) == 10 else None
                if d is None and len(period) == 7:
                    d = date.fromisoformat(period + "-01")
                if d:
                    dates.append(d)
                    values.append(float(value) / 100.0)
            except (ValueError, TypeError):
                continue

        return RateSeries(name, "EUR", "rate", dates, values)


# ---- Sample provider (no network) ----

def _generate_sample_rates(name: str, ref: date, n_days: int, base_rate: float, vol: float = 0.001) -> RateSeries:
    """Generate synthetic rate data for testing."""
    import random
    rng = random.Random(hash(name))
    dates = []
    values = []
    rate = base_rate
    for i in range(n_days):
        d = ref - timedelta(days=n_days - i)
        if d.weekday() < 5:  # business days only
            rate = max(rate + rng.gauss(0, vol), 0.001)
            dates.append(d)
            values.append(rate)
    currency = "USD" if name.startswith("UST") or name in ("SOFR", "FED_FUNDS", "CPI") else "EUR"
    unit = "index" if "CPI" in name else "rate"
    return RateSeries(name, currency, unit, dates, values)


SAMPLE_BASE_RATES = {
    "SOFR": 0.043, "FED_FUNDS": 0.043,
    "UST_3M": 0.044, "UST_6M": 0.045, "UST_1Y": 0.042,
    "UST_2Y": 0.040, "UST_5Y": 0.039, "UST_10Y": 0.041, "UST_30Y": 0.044,
    "ESTR": 0.035, "EURIBOR_3M": 0.036, "EURIBOR_6M": 0.037,
    "EUR_GOVT_2Y": 0.028, "EUR_GOVT_5Y": 0.027, "EUR_GOVT_10Y": 0.029,
}


class SampleProvider(MarketDataProvider):
    """Bundled sample data for testing and demos. No network required."""

    def __init__(self, reference_date: date | None = None, history_days: int = 252):
        self.reference_date = reference_date or date.today()
        self.history_days = history_days

    def available_series(self) -> list[str]:
        return list(SAMPLE_BASE_RATES.keys())

    def fetch(self, series_name: str, start: date | None = None, end: date | None = None) -> RateSeries:
        if series_name not in SAMPLE_BASE_RATES:
            raise KeyError(f"Unknown sample series: {series_name}")
        base = SAMPLE_BASE_RATES[series_name]
        series = _generate_sample_rates(series_name, self.reference_date, self.history_days, base)
        if start or end:
            s = start or series.dates[0]
            e = end or series.dates[-1]
            return series.between(s, e)
        return series


# ---- Curve building from series ----

def build_curve_from_yields(
    reference_date: date,
    yields: dict[float, float],
) -> DiscountCurve:
    """Build a discount curve from par yields at standard tenors.

    Args:
        yields: tenor_years -> yield (e.g. {2: 0.04, 5: 0.039, 10: 0.041}).
    """
    pillar_dates = []
    dfs = []
    for tenor in sorted(yields.keys()):
        y = yields[tenor]
        d = date(reference_date.year + int(tenor), reference_date.month, reference_date.day)
        if tenor != int(tenor):
            d = date.fromordinal(reference_date.toordinal() + int(tenor * 365))
        pillar_dates.append(d)
        dfs.append(math.exp(-y * tenor))

    return DiscountCurve(reference_date, pillar_dates, dfs)
