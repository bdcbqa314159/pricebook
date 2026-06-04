"""Euribor rate loader: historical and daily fixings.

Fetches Euribor fixings from euriborrates.com and caches locally.
Five tenors: 1W, 1M, 3M, 6M, 12M. Daily data since 1999.

DATA SOURCE: https://euriborrates.com/
    euriborrates.com is an independent, non-commercial, non-profit
    information resource. All Euribor data used in this module is
    sourced from and attributed to euriborrates.com.

* :func:`fetch_year` — fetch all daily fixings for a given year.
* :func:`fetch_date` — fetch fixings for a specific date.
* :func:`fetch_all_history` — fetch complete history 1999–present.
* :func:`load_local` — load from local CSV cache.
* :func:`update_latest` — fetch today's fixing and append.

References:
    European Money Markets Institute (EMMI), *Euribor Benchmark*.
    Data source: https://euriborrates.com/
"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# Attribution: data sourced from https://euriborrates.com/
_SOURCE_URL = "https://euriborrates.com"
_SOURCE_ATTRIBUTION = (
    "Euribor data sourced from euriborrates.com — "
    "an independent, non-commercial, non-profit information resource. "
    "See https://euriborrates.com/"
)

# Default cache directory
_DEFAULT_CACHE_DIR = Path(__file__).parent / "euribor"

# Tenors available
TENORS = ["1w", "1m", "3m", "6m", "12m"]
TENOR_LABELS = {"1w": "1 Week", "1m": "1 Month", "3m": "3 Months", "6m": "6 Months", "12m": "12 Months"}


@dataclass
class EuriborFixing:
    """Single day's Euribor fixings across all tenors."""
    date: date
    rates: dict[str, float]  # tenor → rate (decimal, e.g. 0.03782)

    def rate(self, tenor: str) -> float:
        return self.rates.get(tenor, 0.0)

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "source": _SOURCE_URL, **self.rates}


def _parse_daily_page(html: str, target_date: date) -> EuriborFixing | None:
    """Parse a daily rates page for all 5 tenors."""
    rates = {}

    # Pattern: look for tenor labels and their associated rates
    # The page structure has rows with tenor name and rate percentage
    # HTML has comments between number and %: "3.828<!-- -->%"
    pct = r"(-?\d+\.\d+)(?:\s|<!--.*?-->)*%"
    patterns = [
        (r"1\s*[Ww]eek.*?" + pct, "1w"),
        (r"1\s*[Mm]onth.*?" + pct, "1m"),
        (r"3\s*[Mm]onth.*?" + pct, "3m"),
        (r"6\s*[Mm]onth.*?" + pct, "6m"),
        (r"12\s*[Mm]onth.*?" + pct, "12m"),
    ]

    for pattern, tenor in patterns:
        match = re.search(pattern, html, re.DOTALL)
        if match:
            rates[tenor] = float(match.group(1)) / 100.0  # convert % to decimal

    if not rates:
        return None

    return EuriborFixing(date=target_date, rates=rates)


def _parse_year_page(html: str, year: int) -> list[EuriborFixing]:
    """Parse a yearly page. These show 3M rate by default with a date/rate table."""
    fixings = []

    # HTML table rows: <td>M/D/YYYY</td><td ...>X.XXX%</td>
    row_pattern = r"(\d{1,2}/\d{1,2}/\d{4})</td>.*?(-?\d+\.\d+)%"
    for match in re.finditer(row_pattern, html, re.DOTALL):
        try:
            d = datetime.strptime(match.group(1), "%m/%d/%Y").date()
            rate = float(match.group(2)) / 100.0
            fixings.append(EuriborFixing(date=d, rates={"3m": rate}))
        except (ValueError, IndexError):
            continue

    return fixings


def fetch_date(target_date: date, retries: int = 3, delay: float = 1.0) -> EuriborFixing | None:
    """Fetch Euribor fixings for a specific date.

    Fetches from: https://euriborrates.com/en/euribor-rates/YYYY-MM-DD

    Args:
        target_date: business date to fetch.
        retries: number of retry attempts.
        delay: delay between retries (seconds).

    Returns:
        EuriborFixing with all available tenors, or None if not available.

    Attribution: Data from https://euriborrates.com/
    """
    import urllib.request

    url = f"{_SOURCE_URL}/en/euribor-rates/{target_date.isoformat()}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pricebook/euribor-loader"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8")
            fixing = _parse_daily_page(html, target_date)
            if fixing:
                return fixing
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)

    return None


def fetch_year(year: int, tenor: str = "3m", retries: int = 3, delay: float = 1.0) -> list[EuriborFixing]:
    """Fetch all daily fixings for a given year and tenor.

    Uses the ?term= parameter to select tenor.

    Attribution: Data from https://euriborrates.com/

    Args:
        year: calendar year (1999–present).
        tenor: "1w", "1m", "3m", "6m", or "12m".
    """
    import urllib.request

    url = f"{_SOURCE_URL}/en/historical-euribor/{year}/?term={tenor}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pricebook/euribor-loader"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")
            fixings = _parse_year_page(html, year)
            # Tag with correct tenor (parser defaults to 3m key)
            for f in fixings:
                if "3m" in f.rates and tenor != "3m":
                    f.rates[tenor] = f.rates.pop("3m")
            return fixings
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)

    return []


def fetch_year_all_tenors(
    year: int,
    delay_between_tenors: float = 2.0,
) -> list[EuriborFixing]:
    """Fetch all 5 tenors for a given year, merge into single fixings per date.

    Makes 5 requests (one per tenor) and combines into complete daily fixings.

    Attribution: Data from https://euriborrates.com/

    Args:
        year: calendar year.
        delay_between_tenors: polite delay between requests (seconds).
    """
    # Collect all tenors
    tenor_data: dict[date, dict[str, float]] = {}

    for tenor in TENORS:
        fixings = fetch_year(year, tenor)
        for f in fixings:
            if f.date not in tenor_data:
                tenor_data[f.date] = {}
            tenor_data[f.date].update(f.rates)
        if tenor != TENORS[-1]:
            time.sleep(delay_between_tenors)

    # Merge into EuriborFixing list
    result = []
    for d in sorted(tenor_data.keys()):
        result.append(EuriborFixing(date=d, rates=tenor_data[d]))

    return result


def fetch_all_history(
    start_year: int = 1999,
    end_year: int | None = None,
    cache_dir: Path | str | None = None,
    all_tenors: bool = True,
    delay_between_years: float = 2.0,
) -> list[EuriborFixing]:
    """Fetch complete Euribor history from 1999 to present.

    With all_tenors=True (default), fetches all 5 tenors per year
    (135 requests for 27 years, ~5 minutes with polite delays).

    Saves to local CSV cache after fetching.

    Attribution: Data from https://euriborrates.com/

    Args:
        start_year: first year to fetch (default 1999).
        end_year: last year (default: current year).
        cache_dir: directory to save CSV cache.
        all_tenors: if True, fetch all 5 tenors (recommended).
        delay_between_years: polite delay between years.
    """
    if end_year is None:
        end_year = date.today().year

    all_fixings = []

    for year in range(start_year, end_year + 1):
        if all_tenors:
            fixings = fetch_year_all_tenors(year)
        else:
            fixings = fetch_year(year, "3m")
        all_fixings.extend(fixings)
        if year < end_year:
            time.sleep(delay_between_years)

    # Sort by date
    all_fixings.sort(key=lambda f: f.date)

    # Cache locally
    save_to_csv(all_fixings, cache_dir or _DEFAULT_CACHE_DIR)

    return all_fixings


def save_to_csv(fixings: list[EuriborFixing], cache_dir: Path | str):
    """Save fixings to local CSV file.

    File includes attribution header.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filepath = cache_dir / "euribor_history.csv"

    with open(filepath, "w", newline="") as f:
        f.write(f"# {_SOURCE_ATTRIBUTION}\n")
        f.write(f"# Downloaded: {datetime.now().isoformat()}\n")

        writer = csv.writer(f)
        writer.writerow(["date", "1w", "1m", "3m", "6m", "12m"])

        for fixing in fixings:
            writer.writerow([
                fixing.date.isoformat(),
                fixing.rates.get("1w", ""),
                fixing.rates.get("1m", ""),
                fixing.rates.get("3m", ""),
                fixing.rates.get("6m", ""),
                fixing.rates.get("12m", ""),
            ])

    return filepath


def load_local(cache_dir: Path | str | None = None) -> list[EuriborFixing]:
    """Load Euribor fixings from local CSV cache.

    Returns:
        List of EuriborFixing, sorted by date.
    """
    cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
    filepath = cache_dir / "euribor_history.csv"

    if not filepath.exists():
        return []

    fixings = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("date"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                d = date.fromisoformat(parts[0])
                rates = {}
                for i, tenor in enumerate(TENORS):
                    if i + 1 < len(parts) and parts[i + 1]:
                        rates[tenor] = float(parts[i + 1])
                fixings.append(EuriborFixing(date=d, rates=rates))
            except (ValueError, IndexError):
                continue

    fixings.sort(key=lambda f: f.date)
    return fixings


def update_latest(cache_dir: Path | str | None = None) -> EuriborFixing | None:
    """Fetch today's fixing and append to local cache.

    Designed to be called by a daily cron job.

    Attribution: Data from https://euriborrates.com/

    Returns:
        Today's fixing, or None if not available (weekend/holiday).
    """
    today = date.today()
    fixing = fetch_date(today)

    if fixing is None:
        return None

    # Load existing and append
    cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
    existing = load_local(cache_dir)

    # Check if today already exists
    existing_dates = {f.date for f in existing}
    if today not in existing_dates:
        existing.append(fixing)
        existing.sort(key=lambda f: f.date)
        save_to_csv(existing, cache_dir)

    return fixing


def attribution() -> str:
    """Return the data source attribution string.

    Always display this when using Euribor data from this module.
    """
    return _SOURCE_ATTRIBUTION


def summary(fixings: list[EuriborFixing]) -> dict:
    """Summary statistics for a list of fixings."""
    if not fixings:
        return {"count": 0, "source": _SOURCE_URL}

    dates = [f.date for f in fixings]
    rates_3m = [f.rates.get("3m", 0) for f in fixings if "3m" in f.rates]

    return {
        "count": len(fixings),
        "date_range": f"{min(dates)} to {max(dates)}",
        "years": (max(dates) - min(dates)).days / 365.25,
        "tenors_available": list(set(t for f in fixings for t in f.rates.keys())),
        "rate_3m_current": rates_3m[-1] if rates_3m else None,
        "rate_3m_min": min(rates_3m) if rates_3m else None,
        "rate_3m_max": max(rates_3m) if rates_3m else None,
        "source": _SOURCE_URL,
        "attribution": _SOURCE_ATTRIBUTION,
    }
