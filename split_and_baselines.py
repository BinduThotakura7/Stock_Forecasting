import csv
from datetime import date
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("COMP670 Splits and Baselines")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    data = spark.read.parquet("data/features/history.parquet")
    target_date = F.col("target_date")
    year = F.year(target_date)

    validation_dates = year == 2025
    test_dates = target_date.between(
        F.lit(date(2026, 1, 1)), F.lit(date(2026, 10, 30))
    )

    first_validation_feature = (
        data.filter(validation_dates).agg(F.min("session_date")).first()[0]
    )
    first_test_feature = (
        data.filter(test_dates).agg(F.min("session_date")).first()[0]
    )
    if first_validation_feature is None or first_test_feature is None:
        raise SystemExit("Check that the feature file covers 2025 and 2026.")

    split_data = data.withColumn(
        "split",
        F.when(
            year.between(2021, 2024)
            & (target_date < F.lit(first_validation_feature)),
            "train",
        )
        .when(
            validation_dates & (target_date < F.lit(first_test_feature)),
            "validation",
        )
        .when(test_dates, "test"),
    ).filter(F.col("split").isNotNull())

    print("Split summary")
    split_data.groupBy("split").agg(
        F.count("*").alias("rows"),
        F.min("target_date").alias("first_target"),
        F.max("target_date").alias("last_target"),
    ).orderBy("first_target").show()

    validation = split_data.filter(F.col("split") == "validation")
    forecasts = {
        "zero_return": F.lit(0.0),
        "trailing_20_session_mean": F.col("baseline_20"),
    }

    results = []
    for name, prediction in forecasts.items():
        error = F.col("target_return") - prediction
        metrics = validation.agg(
            F.count("*").alias("rows"),
            (F.avg(F.abs(error)) * 10_000).alias("mae_bps"),
            (F.sqrt(F.avg(error * error)) * 10_000).alias("rmse_bps"),
        ).first()
        results.append({"baseline": name, **metrics.asDict()})

    split_data.write.mode("overwrite").parquet("data/splits/history.parquet")

    report = Path("reports/baseline_validation.csv")
    report.parent.mkdir(exist_ok=True)
    with report.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("2025 validation results in basis points")
    for row in results:
        print(
            f"{row['baseline']}: {row['rows']} rows, "
            f"MAE = {row['mae_bps']:.2f}, RMSE = {row['rmse_bps']:.2f}"
        )
    print(f"Saved baseline scores to {report}")
finally:
    spark.stop()