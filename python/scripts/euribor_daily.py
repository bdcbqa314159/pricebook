#!/usr/bin/env python3
"""Daily Euribor update cron job.

Fetches today's Euribor fixings, stores in the rate database,
and calibrates the Nelson-Siegel curve.

Usage:
    python scripts/euribor_daily.py                  # default DB path
    python scripts/euribor_daily.py --db /path/to.db # custom DB path

Cron (weekdays at 18:00 CET, after ECB publishes):
    0 18 * * 1-5 cd /path/to/pricebook/python && .venv/bin/python scripts/euribor_daily.py

Data source: https://euriborrates.com/
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Add pricebook to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pricebook.data.euribor_source import EuriborSource
from pricebook.data.rate_database import RateDatabase


def main():
    parser = argparse.ArgumentParser(description="Daily Euribor update")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--historical", action="store_true", help="Fetch full history (1999–present)")
    parser.add_argument("--year", type=int, default=None, help="Fetch a specific year")
    args = parser.parse_args()

    source = EuriborSource()
    db = RateDatabase(args.db)

    print(f"[{datetime.now().isoformat()}] Euribor daily update")
    print(f"  Database: {db.path}")
    print(f"  Source: {source.attribution}")
    print()

    try:
        if args.historical:
            db.ingest_history(source, start_year=1999)
        elif args.year:
            db.ingest_year(source, args.year)
        else:
            today = date.today()
            stored = db.ingest_today(source)
            if stored:
                rates = db.query_date(today, "EUR", source.source_name)
                print(f"  Stored fixings for {today}:")
                for tenor, rate in sorted(rates.items()):
                    print(f"    {tenor}: {rate*100:.3f}%")
            else:
                print(f"  No new data for {today} (already stored or weekend/holiday)")
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    s = db.summary()
    print(f"  DB: {s['total_fixing_records']} fixing records, {s['total_curve_params']} curve params")
    dr = db.date_range()
    if dr:
        print(f"  Range: {dr[0]} to {dr[1]}")

    db.close()


if __name__ == "__main__":
    main()
