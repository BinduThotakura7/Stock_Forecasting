from pyspark.sql import SparkSession, Window, functions as F

INPUT = "data/clean/history.parquet"
OUTPUT = "data/features/history.parquet"

UTILITIES = ["NEE", "DUK", "SO", "XLU"]
HEALTHCARE = ["JNJ", "MRK", "ABBV", "XLV"]

MODEL_FEATURES = [
    "return_1", "return_5", "return_20",
    "price_vs_ma_5", "price_vs_ma_20",
    "volatility_20", "intraday_volatility_20",
    "range_ratio", "volume_ratio_20",
    "spy_return", "spy_return_lag_1", "sector_return",
]

spark = SparkSession.builder.master("local[2]").appName("COMP670 Features").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

try:
    bars = spark.read.parquet(INPUT)
    by_symbol = Window.partitionBy("symbol").orderBy("session_date")
    last_5 = by_symbol.rowsBetween(-4, 0)
    last_20 = by_symbol.rowsBetween(-19, 0)

    # Calculate each symbol's features using its own price history.
    for days in (1, 5, 20):
        bars = bars.withColumn(
            f"return_{days}",
            F.col("adj_close") / F.lag("adj_close", days).over(by_symbol) - 1,
        )

    bars = (
        bars.withColumn("intraday_return", F.col("close") / F.col("open") - 1)
        .withColumn("price_vs_ma_5", F.col("adj_close") / F.avg("adj_close").over(last_5) - 1)
        .withColumn("price_vs_ma_20", F.col("adj_close") / F.avg("adj_close").over(last_20) - 1)
        .withColumn("volatility_20", F.stddev_samp("return_1").over(last_20))
        .withColumn("intraday_volatility_20", F.stddev_samp("intraday_return").over(last_20))
        .withColumn("range_ratio", (F.col("high") - F.col("low")) / F.col("close"))
        .withColumn("volume_ratio_20", F.try_divide(F.col("volume"), F.avg("volume").over(last_20)))
        .withColumn("return_1_lag", F.lag("return_1").over(by_symbol))
        .withColumn("baseline_20", F.avg("intraday_return").over(last_20))
        # The next session's date and return are used for evaluation.
        .withColumn("target_date", F.lead("session_date").over(by_symbol))
        .withColumn("target_return", F.lead("intraday_return").over(by_symbol))
    )

    market = bars.filter(F.col("symbol") == "SPY").select(
        "session_date",
        F.col("return_1").alias("spy_return"),
        F.col("return_1_lag").alias("spy_return_lag_1"),
    )

    sectors = bars.filter(F.col("symbol").isin("XLU", "XLV")).select(
        "session_date",
        F.col("symbol").alias("sector_etf"),
        F.col("return_1").alias("sector_return"),
    )

    features = (
        bars.filter(F.col("symbol").isin(*(UTILITIES + HEALTHCARE)))
        .withColumn("sector", F.when(F.col("symbol").isin(*UTILITIES), "utilities").otherwise("healthcare"))
        .withColumn("sector_etf", F.when(F.col("sector") == "utilities", "XLU").otherwise("XLV"))
        .join(market, "session_date")
        .join(sectors, ["session_date", "sector_etf"])
        .select(
            "symbol", "sector", "session_date", "target_date",
            *MODEL_FEATURES, "baseline_20", "target_return",
        )
        .dropna(subset=[*MODEL_FEATURES, "baseline_20", "target_date", "target_return"])
        .cache()
    )

    rows = features.count()
    if rows == 0:
        raise SystemExit("No usable rows. Check the clean historical data.")

    features.groupBy("symbol").count().orderBy("symbol").show()
    features.filter(F.col("symbol") == "XLU").orderBy(F.desc("session_date")).select(
        "symbol", "session_date", "target_date", "target_return", "baseline_20"
    ).show(5)

    features.write.mode("overwrite").parquet(OUTPUT)
    print(f"Saved {rows} labeled rows to {OUTPUT}")
finally:
    spark.stop()