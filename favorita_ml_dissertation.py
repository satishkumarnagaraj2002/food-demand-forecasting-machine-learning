from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit("Missing package: xgboost. Install it with: pip install xgboost") from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise SystemExit("Missing package: lightgbm. Install it with: pip install lightgbm") from exc

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
TEST_FRACTION = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Favorita demand forecasting dissertation pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing the extracted Kaggle CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Folder where figures, metrics and models will be saved",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="Maximum rows loaded from train.csv. Use 0 for the full dataset",
    )
    parser.add_argument("--fast", action="store_true", help="Use a smaller tuning search for the first run")
    return parser.parse_args()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(c).strip().lower().replace(" ", "_") for c in result.columns]
    return result


def locate_file(folder: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        direct = folder / candidate
        if direct.exists():
            return direct
        matches = list(folder.rglob(candidate))
        if matches:
            return matches[0]
    return None


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    for folder in (output_dir, figure_dir, model_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return figure_dir, model_dir


def load_train(path: Path, max_rows: int) -> pd.DataFrame:
    """Read the first N rows while preserving date order."""
    nrows = None if max_rows == 0 else max_rows
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def read_optional(folder: Path, names: list[str]) -> pd.DataFrame | None:
    path = locate_file(folder, names)
    if path is None:
        return None
    print(f"Found supporting file: {path.name}")
    return normalise_columns(pd.read_csv(path, low_memory=False))


def detect_schema(df: pd.DataFrame) -> tuple[str, list[str]]:
    if "sales" in df.columns:
        target = "sales"
    elif "unit_sales" in df.columns:
        target = "unit_sales"
    else:
        raise ValueError("Target not found. Expected a 'sales' or 'unit_sales' column in train.csv.")

    keys = [c for c in ("store_nbr", "family", "item_nbr") if c in df.columns]
    if "store_nbr" not in keys or len(keys) < 2:
        raise ValueError("Expected store_nbr and either family or item_nbr in train.csv.")
    return target, keys


def merge_supporting_tables(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    result = df.copy()

    stores = read_optional(data_dir, ["stores.csv"])
    if stores is not None and "store_nbr" in result.columns:
        stores = stores.drop_duplicates("store_nbr")
        result = result.merge(stores, on="store_nbr", how="left", suffixes=("", "_store"))

    items = read_optional(data_dir, ["items.csv"])
    if items is not None and "item_nbr" in result.columns:
        items = items.drop_duplicates("item_nbr")
        result = result.merge(items, on="item_nbr", how="left", suffixes=("", "_item"))

    oil = read_optional(data_dir, ["oil.csv"])
    if oil is not None and "date" in result.columns:
        oil["date"] = pd.to_datetime(oil["date"], errors="coerce")
        oil = oil.drop_duplicates("date")
        result = result.merge(oil, on="date", how="left")

    transactions = read_optional(data_dir, ["transactions.csv"])
    if transactions is not None and {"date", "store_nbr"}.issubset(result.columns):
        transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")
        transactions = transactions.drop_duplicates(["date", "store_nbr"])
        result = result.merge(transactions, on=["date", "store_nbr"], how="left")

    holidays = read_optional(data_dir, ["holidays_events.csv", "holidays.csv"])
    if holidays is not None and "date" in result.columns:
        holidays["date"] = pd.to_datetime(holidays["date"], errors="coerce")
        holiday_type_col = "type" if "type" in holidays.columns else None
        if holiday_type_col:
            summary = holidays.groupby("date", as_index=False).agg(
                holiday_count=("date", "size"),
                holiday_type=(holiday_type_col, lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
            )
        else:
            summary = holidays.groupby("date", as_index=False).size().rename(columns={"size": "holiday_count"})
            summary["holiday_type"] = "Holiday"
        summary["is_holiday"] = (summary["holiday_count"] > 0).astype(int)
        result = result.merge(summary, on="date", how="left")

    return result


def clean_data(df: pd.DataFrame, target: str, output_dir: Path) -> pd.DataFrame:
    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if target in result.columns:
        result[target] = pd.to_numeric(result[target], errors="coerce")
    result = result.dropna(subset=["date", target])

    negative_targets = int((result[target] < 0).sum())
    result[target] = result[target].clip(lower=0)

    if "onpromotion" in result.columns:
        result["onpromotion"] = result["onpromotion"].fillna(False).astype(int)

    if "dcoilwtico" in result.columns:
        result = result.sort_values("date")
        result["dcoilwtico"] = result["dcoilwtico"].ffill().bfill()

    for col in result.select_dtypes(include=["object", "category"]).columns:
        result[col] = result[col].fillna("Unknown").astype(str)

    before = len(result)
    result = result.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(result)

    quality = {
        "rows_after_cleaning": int(len(result)),
        "columns_after_merging": int(len(result.columns)),
        "negative_targets_clipped": negative_targets,
        "duplicate_rows_removed": int(duplicates_removed),
        "date_min": str(result["date"].min().date()),
        "date_max": str(result["date"].max().date()),
    }
    (output_dir / "data_quality_summary.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return result


def create_time_and_lag_features(df: pd.DataFrame, target: str, keys: list[str]) -> pd.DataFrame:
    result = df.sort_values(keys + ["date"]).copy()
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["iso_week"] = result["date"].dt.isocalendar().week.astype(int)
    result["day_of_month"] = result["date"].dt.day
    result["day_of_week"] = result["date"].dt.dayofweek
    result["quarter"] = result["date"].dt.quarter
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

    grouped = result.groupby(keys, sort=False)[target]
    for lag in (1, 7, 14, 28):
        result[f"lag_{lag}"] = grouped.shift(lag)
    for window in (7, 28):
        result[f"rolling_mean_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).mean()
        )
        result[f"rolling_std_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )

    return result.sort_values("date").reset_index(drop=True)


def save_dataset_summary(df: pd.DataFrame, target: str, keys: list[str], output_dir: Path) -> None:
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target": target,
        "keys": keys,
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "mean_target": float(df[target].mean()),
        "median_target": float(df[target].median()),
        "std_target": float(df[target].std()),
        "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def perform_eda(df: pd.DataFrame, target: str, figure_dir: Path, output_dir: Path) -> None:
    sales = df[target].dropna()
    plt.figure(figsize=(10, 4))
    plt.hist(sales, bins=40, edgecolor="black")
    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figure_dir / "target_distribution.png", dpi=200)
    plt.close()

    date_counts = df.groupby("date").size()
    plt.figure(figsize=(10, 4))
    date_counts.plot()
    plt.title("Daily observation count")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(figure_dir / "daily_row_count.png", dpi=200)
    plt.close()

    eda_summary = {
        "target_mean": float(sales.mean()),
        "target_median": float(sales.median()),
        "target_std": float(sales.std()),
        "min_date": str(df["date"].min().date()),
        "max_date": str(df["date"].max().date()),
        "non_null_rows": int(len(sales)),
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")


def chronological_split(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index, pd.Index]:
    ordered = df.sort_values("date").copy()
    dates = pd.Index(sorted(ordered["date"].dropna().unique()))

    if len(dates) < 2:
        n_test = max(1, int(len(ordered) * TEST_FRACTION))
        n_train = len(ordered) - n_test
        train = ordered.iloc[:n_train].copy()
        test = ordered.iloc[n_train:].copy()
        train_dates = pd.Index(train["date"].dropna().unique())
        test_dates = pd.Index(test["date"].dropna().unique())
    else:
        split_point = max(1, int(len(dates) * (1 - TEST_FRACTION)))
        train_dates = dates[:split_point]
        test_dates = dates[split_point:]
        train = ordered[ordered["date"].isin(train_dates)].copy()
        test = ordered[ordered["date"].isin(test_dates)].copy()

    X_train = train.drop(columns=[target, "date"]).copy()
    X_test = test.drop(columns=[target, "date"]).copy()
    y_train = train[target].copy()
    y_test = test[target].copy()
    return X_train, X_test, y_train, y_test, train_dates, test_dates


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    low_card = [c for c in categorical_cols if X_train[c].nunique(dropna=True) <= 10]
    high_card = [c for c in categorical_cols if c not in low_card]

    transformers = [
        (
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_cols,
        )
    ]

    if low_card:
        transformers.append(
            (
                "low_card",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                low_card,
            )
        )

    if high_card:
        transformers.append(
            (
                "high_card",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                high_card,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, low_card, high_card


def make_searches(preprocessor: ColumnTransformer, fast: bool) -> dict[str, RandomizedSearchCV]:
    search_spaces = {
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        },
        "XGBoost": {
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.7, 0.9],
            },
        },
        "LightGBM": {
            "estimator": LGBMRegressor(
                objective="regression",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
                "model__max_depth": [-1, 8, 16],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.7, 0.9],
            },
        },
    }

    searches = {}
    for name, config in search_spaces.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", config["estimator"]),
        ])
        n_iter = 3 if fast else 6
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=3),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        searches[name] = search
    return searches


def evaluate(name: str, model: RandomizedSearchCV, X_test: pd.DataFrame, y_test: pd.Series, test_dates: pd.Index, figure_dir: Path) -> tuple[dict, np.ndarray]:
    preds = model.predict(X_test)
    preds_series = pd.Series(preds, index=y_test.index)
    y_true = pd.Series(y_test, index=y_test.index)
    mae = mean_absolute_error(y_true, preds_series)
    mse = mean_squared_error(y_true, preds_series)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, preds_series)
    msle = mean_squared_log_error(y_true.clip(lower=0), preds_series.clip(lower=0))
    r2 = r2_score(y_true, preds_series)

    metrics = {
        "model": name,
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "MSLE": float(msle),
        "R2": float(r2),
    }

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.2)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1)
    plt.title(f"{name}: actual vs predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(figure_dir / f"{name.lower().replace(' ', '_')}_actual_vs_predicted.png", dpi=200)
    plt.close()

    return metrics, preds


def export_feature_importance(name: str, search: RandomizedSearchCV, output_dir: Path, figure_dir: Path) -> None:
    model = search.best_estimator_
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = np.array(preprocessor.get_feature_names_out())
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        feature_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        feature_df = feature_df.sort_values("importance", ascending=False).reset_index(drop=True)
        feature_df.to_csv(output_dir / f"{name.lower().replace(' ', '_')}_feature_importance.csv", index=False)
        plt.figure(figsize=(10, 6))
        top = feature_df.head(15)
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title(f"Top feature importance for {name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(figure_dir / f"{name.lower().replace(' ', '_')}_feature_importance.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    figure_dir, model_dir = ensure_output_dirs(output_dir)

    if args.max_rows == 0:
        print("Loading full dataset (this may take time).")
    else:
        print(f"Loading first {args.max_rows} rows from train.csv.")

    train_path = locate_file(data_dir, ["train.csv"])
    if train_path is None:
        raise FileNotFoundError(f"train.csv not found in {data_dir}")

    print(f"Loading: {train_path}")
    data = normalise_columns(load_train(train_path, args.max_rows))
    if "date" not in data.columns:
        raise ValueError("train.csv must contain a date column.")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    target, keys = detect_schema(data)
    print(f"Detected target: {target}; series keys: {keys}")

    data = merge_supporting_tables(data, data_dir)
    data = normalise_columns(data)
    data = clean_data(data, target, output_dir)
    data = create_time_and_lag_features(data, target, keys)

    print("\n" + "=" * 70)
    print("FIGURE 4.1 - INITIAL STRUCTURE OF THE PROCESSED DATASET")
    print("=" * 70)
    print(f"\nDataset shape: {data.shape}")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    important_cols = [
        c for c in ["date", "store_nbr", "item_nbr", "family", "unit_sales", "sales", "onpromotion"] if c in data.columns
    ]
    print("\nMain columns:")
    print(important_cols)
    print("\nFirst 5 rows:")
    print(data[important_cols].head().to_string(index=False))
    print("=" * 70)

    save_dataset_summary(data, target, keys, output_dir)
    perform_eda(data, target, figure_dir, output_dir)

    X_train, X_test, y_train, y_test, train_dates, test_dates = chronological_split(data, target)
    print("\n" + "=" * 70)
    print("CHRONOLOGICAL TRAINING AND TESTING SPLIT")
    print("=" * 70)
    print(f"Total observations: {len(data)}")
    print(f"Training observations: {len(X_train)} ({len(X_train) / len(data) * 100:.1f}%)")
    print(f"Testing observations: {len(X_test)} ({len(X_test) / len(data) * 100:.1f}%)")
    print(f"\nTraining date range: {train_dates.min().date()} to {train_dates.max().date()}")
    print(f"Testing date range: {test_dates.min().date()} to {test_dates.max().date()}")
    print("Split strategy: chronological by unique date")
    print("Random shuffling: No")
    print("=" * 70)

    preprocessor, numeric, low_card, high_card = build_preprocessor(X_train)
    split_summary = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_start_date": str(train_dates.min().date()),
        "train_end_date": str(train_dates.max().date()),
        "test_start_date": str(test_dates.min().date()),
        "test_end_date": str(test_dates.max().date()),
        "numeric_features": numeric,
        "one_hot_features": low_card,
        "ordinal_encoded_features": high_card,
        "fast_mode": bool(args.fast),
        "max_rows": int(args.max_rows),
    }
    (output_dir / "split_and_feature_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    searches = make_searches(preprocessor, args.fast)
    metrics_rows: list[dict] = []
    predictions = pd.DataFrame({"date": test_dates.to_series().reset_index(drop=True), "actual": y_test.reset_index(drop=True)})
    best_params: dict[str, dict] = {}

    for name, search in searches.items():
        print(f"\nTraining and tuning {name}...")
        search.fit(X_train, y_train)
        metrics, pred = evaluate(name, search, X_test, y_test, test_dates, figure_dir)
        metrics_rows.append(metrics)
        predictions[f"predicted_{name}"] = pred
        best_params[name] = search.best_params_
        export_feature_importance(name, search, output_dir, figure_dir)
        joblib.dump(search.best_estimator_, model_dir / f"{name.lower().replace(' ', '_')}_best_model.joblib")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE")
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "best_parameters.json").write_text(json.dumps(best_params, indent=2, default=str), encoding="utf-8")

    plt.figure(figsize=(9, 5))
    plt.bar(metrics_df["model"], metrics_df["RMSE"])
    plt.title("Model Comparison by RMSE")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(figure_dir / "model_comparison_rmse.png", dpi=200)
    plt.close()

    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))
    print(f"\nCompleted. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Check the data folder, CSV filenames, and installed packages.", file=sys.stderr)
        raise
