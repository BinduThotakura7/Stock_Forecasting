import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

SYMBOLS = ["NEE", "DUK", "SO", "JNJ", "MRK", "ABBV", "XLU", "XLV", "SPY"]
CENTRAL = ZoneInfo("America/Chicago")

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["history", "daily"])
parser.add_argument(
    "--date",
    type=date.fromisoformat,
    help="Daily session date in YYYY-MM-DD format",
)
args = parser.parse_args()

now = datetime.now(CENTRAL)
today = now.date()
closed = False

if args.mode == "history":
    if args.date is not None:
        parser.error("--date applies to daily collection only.")

    start = date(2021, 1, 1)
    last_day = date(2026, 11, 11)

else:
    start = last_day = args.date or today

    if not date(2026, 11, 12) <= last_day <= date(2026, 12, 11):
        parser.error("Requested date is outside the collection window.")

    if last_day > today:
        parser.error("Choose today or an earlier date.")

    calendar = xcals.get_calendar(
        "XNYS", start="2026-11-01", end="2026-12-31"
    )
    closed = not calendar.is_session(last_day.isoformat())

    if last_day == today and not closed and now.time() < time(18):
        parser.error("Run today's collection after 6:00 p.m. Central.")

end = last_day + timedelta(days=1)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
folder = Path("data/raw") / args.mode / stamp
folder.mkdir(parents=True)

frames = []
errors = {}

if not closed:
    for symbol in SYMBOLS:
        try:
            bars = yf.download(
                symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=True,
                multi_level_index=False,
                progress=False,
                threads=False,
                timeout=15,
            )

            if bars.empty:
                raise ValueError("The provider returned no data.")

            bars = bars.reset_index()
            dates = pd.to_datetime(bars["Date"]).dt.date
            bars = bars.loc[dates.between(start, last_day)].copy()

            if bars.empty:
                raise ValueError("No rows fall within the requested dates.")

            bars.insert(0, "symbol", symbol)
            frames.append(bars)

            if last_day not in pd.to_datetime(bars["Date"]).dt.date.values:
                errors[symbol] = f"Missing expected session {last_day}"

        except Exception as exc:
            errors[symbol] = str(exc)

rows = sum(len(frame) for frame in frames)

if frames:
    pd.concat(frames, ignore_index=True).to_csv(
        folder / "bars.csv", index=False
    )

if closed:
    status = "market_closed"
elif errors:
    status = "partial" if rows else "failed"
else:
    status = "complete"

downloaded_at = datetime.now(timezone.utc)
timing = "historical"

if args.mode == "daily":
    download_day = downloaded_at.astimezone(CENTRAL).date()
    timing = "same_day" if download_day == last_day else "backfill"

manifest = {
    "mode": args.mode,
    "status": status,
    "timing": timing,
    "source": "Yahoo Finance via yfinance",
    "downloaded_at_utc": downloaded_at.isoformat(),
    "first_requested_date": start.isoformat(),
    "last_requested_date": last_day.isoformat(),
    "rows": rows,
    "errors": errors,
}

(folder / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)

print(json.dumps(manifest, indent=2))
print(f"Saved to {folder}")

if status in {"partial", "failed"}:
    raise SystemExit(1)