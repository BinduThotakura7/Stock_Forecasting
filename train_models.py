import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT = Path("data/splits/history.parquet")

FEATURES = [
    "return_1",
    "return_5",
    "return_20",
    "price_vs_ma_5",
    "price_vs_ma_20",
    "volatility_20",
    "intraday_volatility_20",
    "range_ratio",
    "volume_ratio_20",
    "spy_return",
    "spy_return_lag_1",
    "sector_return",
]

INPUT_COLUMNS = FEATURES + ["symbol"]


def build_pipeline(estimator):
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), FEATURES),
            (
                "symbol",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["symbol"],
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", clone(estimator)),
        ]
    )


def calculate_metrics(name, actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if not np.isfinite(predicted).all():
        raise ValueError(f"{name} produced missing or infinite predictions.")

    error = actual - predicted

    # Direction accuracy is undefined for the always-zero baseline.
    direction_accuracy = None
    if not np.all(predicted == 0):
        direction_accuracy = float(
            np.mean(np.sign(actual) == np.sign(predicted)) * 100
        )

    return {
        "model": name,
        "rows": len(actual),
        "mae_bps": float(np.mean(np.abs(error)) * 10_000),
        "rmse_bps": float(np.sqrt(np.mean(error ** 2)) * 10_000),
        "direction_accuracy_pct": direction_accuracy,
    }


def main():
    if not INPUT.exists():
        raise SystemExit(
            "Missing split data. Run split_and_baselines.py first."
        )

    data = pd.read_parquet(INPUT)

    for column in ["session_date", "target_date"]:
        data[column] = pd.to_datetime(data[column])

    # Model selection uses only the train and validation partitions.
    development = (
        data.loc[data["split"].isin(["train", "validation"])]
        .sort_values(["target_date", "symbol"])
        .reset_index(drop=True)
    )

    train = development.loc[development["split"] == "train"]
    validation = development.loc[
        development["split"] == "validation"
    ].copy()

    if train.empty or validation.empty:
        raise SystemExit("The training or validation partition is empty.")

    if not (development["session_date"] < development["target_date"]).all():
        raise SystemExit("Every target date must follow its feature date.")

    if development.duplicated(["symbol", "session_date"]).any():
        raise SystemExit("Duplicate symbol/session_date rows were found.")

    numeric_columns = FEATURES + ["baseline_20", "target_return"]
    if not np.isfinite(
        development[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise SystemExit("Missing or infinite numeric values were found.")

    if train["target_date"].max() >= validation["session_date"].min():
        raise SystemExit("Training outcomes overlap the validation boundary.")

    print(f"Initial training rows: {len(train)}", flush=True)
    print(f"Validation rows: {len(validation)}", flush=True)

    models = {
        f"ridge_alpha_{alpha:g}": Ridge(alpha=alpha)
        for alpha in (0.1, 1.0, 10.0)
    }

    for depth in (3, 6):
        for leaf_size in (20, 50):
            name = f"forest_depth_{depth}_leaf_{leaf_size}"
            models[name] = RandomForestRegressor(
                n_estimators=200,
                max_depth=depth,
                min_samples_leaf=leaf_size,
                random_state=42,
                n_jobs=2,
            )

    predictions = pd.DataFrame(index=validation.index)
    predictions["zero_return"] = 0.0
    predictions["trailing_20_session_mean"] = validation["baseline_20"]

    for name in models:
        predictions[name] = np.nan

    months = validation["target_date"].dt.to_period("M")

    for month, block in validation.groupby(months, sort=True):
        # Keep every symbol from a session together.
        # Training outcomes must precede this month's earliest feature date.
        cutoff = block["session_date"].min()
        monthly_train = development.loc[
            development["target_date"] < cutoff
        ]

        if monthly_train.empty:
            raise SystemExit(f"No training data available for {month}.")

        print(
            f"Validating {month}: "
            f"{len(monthly_train)} training rows, "
            f"{len(block)} validation rows",
            flush=True,
        )

        for name, estimator in models.items():
            pipeline = build_pipeline(estimator)
            pipeline.fit(
                monthly_train[INPUT_COLUMNS],
                monthly_train["target_return"],
            )

            predictions.loc[block.index, name] = pipeline.predict(
                block[INPUT_COLUMNS]
            )

    results = [
        calculate_metrics(
            name,
            validation["target_return"],
            predictions[name],
        )
        for name in predictions.columns
    ]

    # Choose by validation MAE. Stable sorting preserves baseline-first ties.
    report = (
        pd.DataFrame(results)
        .sort_values("mae_bps", kind="stable")
        .reset_index(drop=True)
    )

    winner = str(report.iloc[0]["model"])

    print("\n2025 monthly validation results")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSelected model: {winner}", flush=True)

    # Refit a selected learned model on all development data.
    # A selected baseline needs no fitted estimator.
    fitted_pipeline = None
    if winner in models:
        fitted_pipeline = build_pipeline(models[winner])
        fitted_pipeline.fit(
            development[INPUT_COLUMNS],
            development["target_return"],
        )

    selection = {
        "selected_model": winner,
        "selection_metric": "mae_bps",
        "validation_method": "expanding monthly windows during 2025",
        "validation_rows": len(validation),
        "validation_sessions": int(validation["target_date"].nunique()),
        "validation_mae_bps": float(report.iloc[0]["mae_bps"]),
        "validation_rmse_bps": float(report.iloc[0]["rmse_bps"]),
        "refit_rows": len(development),
        "last_training_target": (
            development["target_date"].max().date().isoformat()
        ),
        "target_definition": "next_session_close / next_session_open - 1",
        "feature_columns": INPUT_COLUMNS,
        "selected_parameters": (
            models[winner].get_params() if winner in models else {}
        ),
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/predictions").mkdir(parents=True, exist_ok=True)

    report.to_csv("reports/model_validation.csv", index=False)

    prediction_details = validation[
        ["symbol", "sector", "session_date", "target_date", "target_return"]
    ].join(predictions)

    prediction_details.to_csv(
        "data/predictions/validation.csv",
        index=False,
    )

    Path("reports/selection.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )

    joblib.dump(
        {
            "selection": selection,
            "pipeline": fitted_pipeline,
        },
        "models/selected_model.joblib",
    )

    print("\nSaved validation report, predictions, selection, and model.")


if __name__ == "__main__":
    main()