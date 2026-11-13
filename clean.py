from pathlib import Path

import exchange_calendars as xcals
from pyspark.sql import SparkSession, functions as F

SOURCE = Path("data/raw/history/20261112T203543496892Z/bars.csv")
OUTPUT = "data/clean/history.parquet"
SYMBOLS = ["NEE", "DUK", "SO", "JNJ", "MRK", "ABBV", "XLU", "XLV", "SPY"]

if not SOURCE.is_file():
    raise SystemExit(f"Historical CSV not found: {SOURCE}")

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("COMP670 Clean Historical Prices")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    raw = spark.read.option("header", True).csv(str(SOURCE))

    price_columns = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
    }

    bars = raw.select(
        F.upper(F.trim("symbol")).alias("symbol"),
        F.col("Date").try_cast("date").alias("session_date"),
        *[
            F.col(original).try_cast("double").alias(clean)
            for original, clean in price_columns.items()
        ],
        F.col("Volume").try_cast("bigint").alias("volume"),
        F.col("Dividends").try_cast("double").alias("dividends"),
        F.col("Stock Splits").try_cast("double").alias("stock_splits"),
    ).cache()

    # Compare each symbol's dates with the NYSE trading calendar.
    calendar = xcals.get_calendar(
        "XNYS", start="2020-12-28", end="2026-11-20"
    )
    sessions = calendar.sessions_in_range("2021-01-01", "2026-11-11")
    dates = spark.createDataFrame(
        [(session.date(),) for session in sessions], "session_date date"
    )
    symbols = spark.createDataFrame([(symbol,) for symbol in SYMBOLS], "symbol string")
    expected = symbols.crossJoin(dates)
    keys = ["symbol", "session_date"]
    actual = bars.select(*keys).distinct()

    missing = expected.join(actual, keys, "left_anti")
    unexpected = actual.join(expected, keys, "left_anti")
    duplicates = bars.groupBy(*keys).count().filter(F.col("count") > 1)

    invalid = (
        F.col("volume").isNull()
        | (F.col("volume") < 0)
        | (F.col("high") < F.greatest("open", "low", "close"))
        | (F.col("low") > F.least("open", "close"))
    )
    for name in price_columns.values():
        price = F.col(name)
        invalid = (
            invalid | price.isNull() | F.isnan(price)
            | (price <= 0) | (price == float("inf"))
        )

    checks = {
        "Missing symbol/date pairs": missing,
        "Unexpected symbol/date pairs": unexpected,
        "Duplicate symbol/date pairs": duplicates,
        "Invalid price or volume rows": bars.filter(invalid),
    }

    print(f"Input rows: {bars.count()}")
    print(f"Expected symbol/date pairs: {len(SYMBOLS) * len(sessions)}")

    problems = 0
    for label, result in checks.items():
        count = result.count()
        print(f"{label}: {count}")
        if count:
            result.show(5, truncate=False)
        problems += count

    if problems:
        raise SystemExit("Review the issues above before saving the clean dataset.")

    bars.write.mode("overwrite").parquet(OUTPUT)
    print(f"Saved {spark.read.parquet(OUTPUT).count()} rows to {OUTPUT}")
finally:
    spark.stop()