import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

SYMBOLS = ["NEE", "DUK", "SO", "JNJ", "MRK", "ABBV", "XLU", "XLV", "SPY"]

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["history", "daily"])
args = parser.parse_args()

now = datetime.now(ZoneInfo("America/Chicago"))
today = now.date()
closed = False

if args.mode == "history":
    start = date(2021, 1, 1)
    last_day = date(2026, 11, 11)
else:
    if not date(2026, 11, 12) <= today <= date(2026, 12, 11):
        parser.error("Today is outside the study's daily collection window.")
    start = last_day = today
    closed = not xcals.get_calendar("XNYS").is_session(today.isoformat())
    if not closed and now.time() < time(18):
        parser.error("Run daily collection at or after 6:00 p.m. Central.")

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
            bars.insert(0, "symbol", symbol)
            frames.append(bars)
            if pd.Timestamp(last_day) not in pd.to_datetime(bars["Date"]).values:
                errors[symbol] = f"Missing expected session {last_day}"
        except Exception as exc:
            errors[symbol] = str(exc)

rows = sum(len(frame) for frame in frames)
if frames:
    pd.concat(frames, ignore_index=True).to_csv(folder / "bars.csv", index=False)

status = "market_closed" if closed else "failed" if not rows else "partial" if errors else "complete"
manifest = {
    "mode": args.mode,
    "status": status,
    "source": "Yahoo Finance via yfinance",
    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    "first_requested_date": start.isoformat(),
    "last_requested_date": last_day.isoformat(),
    "rows": rows,
    "errors": errors,
}
(folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
print(f"Saved to {folder}")