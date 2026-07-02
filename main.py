"""
Day-Ahead Electricity Load Forecaster
Forecasts the next 24 hours of household electricity consumption
(Global_active_power) using the UCI Individual Household Electric
Power Consumption dataset.

Pipeline:
  1. Load & clean raw minute-level data
  2. Resample to hourly resolution (proper aggregation per column type)
  3. Engineer time, lag, and rolling features
  4. Chronological train/test split (no leakage)
  5. Naive baselines vs. XGBoost regressor
  6. Evaluate (MAE / RMSE) and plot results

Usage:
    python main.py --data data/household_power_consumption.txt
"""

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor



# Step 1: Data Loading & Cleaning

def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw UCI text file, parse dates, and clean missing values."""
    print("Step 1: Loading & cleaning raw data...")

    usecols = [
        "Date", "Time", "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]

    df = pd.read_csv(
        "household_power_consumption.txt",
        sep=";",
        usecols=usecols,
        na_values=["?"],
        low_memory=False,
    )

    # Combine Date + Time into a single Datetime index.
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("Datetime").sort_index()

   
    numeric_cols = [c for c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    missing_counts = df[numeric_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        print("  Missing values found per column:")
        for col, cnt in missing_counts.items():
            pct = cnt / len(df) * 100
            print(f"    {col:<25s} {cnt:>7,} missing ({pct:.2f}%)")
    else:
        print("  No missing values found.")


    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f"  Loaded {n_before:,} minute-level rows, "
          f"dropped {n_before - n_after:,} rows with missing values "
          f"({n_after:,} remain).")

    return df


# Step 2: Resampling to Hourly Data

def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level data to hourly bins with metric-appropriate math."""
    print("Step 2: Resampling to hourly resolution...")

   
    sum_cols = [
        "Global_active_power", "Global_reactive_power",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    # Instantaneous / steady-state metrics -> average.
    mean_cols = ["Voltage", "Global_intensity"]

    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "mean" for c in mean_cols})

    hourly = df.resample("h").agg(agg)

    # Convert summed per-minute kW readings to kWh for the hour
    # (sum of 60 one-minute kW samples / 60 = kWh consumed that hour).
    hourly["Global_active_power"] = hourly["Global_active_power"] / 60.0
    hourly["Global_reactive_power"] = hourly["Global_reactive_power"] / 60.0

    hourly = hourly.dropna()
    print(f"  Produced {len(hourly):,} hourly rows spanning "
          f"{hourly.index.min()} to {hourly.index.max()}.")

    return hourly


# Step 2b: Outlier Detection & Handling

def detect_and_handle_outliers(
    hourly: pd.DataFrame,
    cols: list = None,
    method: str = "iqr",
    iqr_multiplier: float = 3.0,
    strategy: str = "cap",
) -> pd.DataFrame:

    if cols is None:
        cols = ["Global_active_power", "Global_reactive_power",
                 "Voltage", "Global_intensity"]

    df_out = hourly.copy()
    outlier_mask = pd.Series(False, index=df_out.index)

    for col in cols:
        if method == "iqr":
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
        elif method == "zscore":
            mean = df_out[col].mean()
            std = df_out[col].std()
            lower = mean - iqr_multiplier * std   # here iqr_multiplier acts as z-threshold
            upper = mean + iqr_multiplier * std
        else:
            raise ValueError(f"Unknown method: {method}")

        col_mask = (df_out[col] < lower) | (df_out[col] > upper)
        n_col_outliers = col_mask.sum()

        if n_col_outliers > 0:
            pct = n_col_outliers / len(df_out) * 100
            print(f"  {col:<25s} {n_col_outliers:>6,} outliers "
                  f"({pct:.2f}%)  bounds=[{lower:.2f}, {upper:.2f}]")

        if strategy == "cap":
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)
        elif strategy == "remove":
            outlier_mask = outlier_mask | col_mask
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    if strategy == "remove":
        n_before = len(df_out)
        df_out = df_out[~outlier_mask]
        print(f"  Removed {n_before - len(df_out):,} rows containing at "
              f"least one outlier ({len(df_out):,} rows remain).")
    else:
        print(f"  Capped outlier values to their nearest boundary "
              f"({len(df_out):,} rows retained).")

    return df_out


# Step 3: Feature Engineering

def engineer_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical time features, lag features, and rolling statistics."""
    print("Step 3: Engineering features...")

    df_feat = hourly.copy()

    # --- Cyclical time encodings ---
    df_feat["hour"] = df_feat.index.hour
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month

    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["dow_sin"] = np.sin(2 * np.pi * df_feat["dayofweek"] / 7)
    df_feat["dow_cos"] = np.cos(2 * np.pi * df_feat["dayofweek"] / 7)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
    df_feat["is_weekend"] = (df_feat["dayofweek"] >= 5).astype(int)

    target = "Global_active_power"

    # --- Lag features (only past target values -> no leakage) ---
    df_feat["lag_24h"] = df_feat[target].shift(24)     # same hour, yesterday
    df_feat["lag_48h"] = df_feat[target].shift(48)      # same hour, 2 days ago
    df_feat["lag_168h"] = df_feat[target].shift(168)    # same hour, last week

    # --- Rolling statistics (shifted by 24 so the window only uses data
    #     available strictly *before* the forecast target time; this
    #     preserves a true day-ahead forecasting setup) ---
    shifted = df_feat[target].shift(24)
    df_feat["roll_mean_6h"] = shifted.rolling(window=6).mean()
    df_feat["roll_mean_24h"] = shifted.rolling(window=24).mean()
    df_feat["roll_std_24h"] = shifted.rolling(window=24).std()
    df_feat["roll_mean_168h"] = shifted.rolling(window=168).mean()

    n_before = len(df_feat)
    df_feat = df_feat.dropna()
    print(f"  Created {df_feat.shape[1] - hourly.shape[1]} new features; "
          f"dropped {n_before - len(df_feat):,} warm-up rows "
          f"({len(df_feat):,} rows remain).")

    return df_feat



# Step 4: Chronological Train/Test Split

def train_test_split_chrono(df_feat: pd.DataFrame, test_fraction: float = 0.2):
    """Split strictly by time: all training rows precede all test rows."""
    print("Step 4: Performing chronological train/test split...")

    split_idx = int(len(df_feat) * (1 - test_fraction))
    split_date = df_feat.index[split_idx]

    train_df = df_feat.iloc[:split_idx]
    test_df = df_feat.iloc[split_idx:]

    print(f"  Split at {split_date}: "
          f"{len(train_df):,} train rows / {len(test_df):,} test rows.")

    return train_df, test_df



# Step 5: Baselines & Model Training

LEAKAGE_COLS = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
TARGET = "Global_active_power"


def get_feature_columns(df_feat: pd.DataFrame):
    """Every column except the target and same-timestamp leakage columns."""
    drop_cols = [TARGET] + LEAKAGE_COLS
    return [c for c in df_feat.columns if c not in drop_cols]


def evaluate(y_true, y_pred, label: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {label:<28s} MAE = {mae:6.4f} kWh   RMSE = {rmse:6.4f} kWh")
    return mae, rmse


def run_baselines(test_df: pd.DataFrame):
    """Naive baselines: 'tomorrow = today' (lag 24h) and 'same hour last week'."""
    print("Step 5a: Evaluating naive baselines...")
    y_true = test_df[TARGET]

    results = {}
    results["Naive (same hour yesterday)"] = evaluate(
        y_true, test_df["lag_24h"], "Naive (lag 24h)"
    )
    results["Naive (same hour last week)"] = evaluate(
        y_true, test_df["lag_168h"], "Naive (lag 168h)"
    )
    return results


def train_and_predict(train_df, test_df, feature_cols):
    print("Step 5b: Training XGBoost regressor...")

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    preds = model.predict(X_test)
    preds = np.clip(preds, a_min=0, a_max=None)  # power can't be negative

    mae, rmse = evaluate(y_test, preds, "XGBoost")
    print(f"  (trained in {fit_time:.1f}s on {len(X_train):,} rows, "
          f"{len(feature_cols)} features)")

    return model, preds, mae, rmse



# Step 6: Visualization

def plot_results(test_df, preds, out_path="forecast_results.png", n_hours=24 * 7):
    print("Step 6: Generating visualization...")

    plot_df = test_df.iloc[:n_hours].copy()
    plot_preds = preds[:n_hours]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    axes[0].plot(plot_df.index, plot_df[TARGET], label="Actual", color="black", lw=1.5)
    axes[0].plot(plot_df.index, plot_preds, label="XGBoost Forecast", color="tab:orange", lw=1.2)
    axes[0].plot(plot_df.index, plot_df["lag_24h"], label="Naive Baseline (lag 24h)",
                 color="tab:blue", lw=1, alpha=0.6, linestyle="--")
    axes[0].set_title("Day-Ahead Load Forecast vs. Actual (first week of test set)")
    axes[0].set_ylabel("Global Active Power (kWh)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    errors_model = np.abs(test_df[TARGET].values - preds)
    errors_naive = np.abs(test_df[TARGET].values - test_df["lag_24h"].values)
    axes[1].hist(errors_naive, bins=50, alpha=0.5, label="Naive baseline |error|", color="tab:blue")
    axes[1].hist(errors_model, bins=50, alpha=0.5, label="XGBoost |error|", color="tab:orange")
    axes[1].set_title("Distribution of Absolute Errors: Full Test Set")
    axes[1].set_xlabel("Absolute Error (kWh)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved plot to {out_path}")



# Main

def main():
    parser = argparse.ArgumentParser(description="Day-ahead electricity load forecaster")
    parser.add_argument(
        "--data", type=str,
        default="data/household_power_consumption.txt",
        help="Path to the raw UCI household_power_consumption.txt file",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="forecast_results.png")
    args = parser.parse_args()

    df = load_and_clean(args.data)
    hourly = resample_hourly(df)
    df_feat = engineer_features(hourly)
    train_df, test_df = train_test_split_chrono(df_feat, args.test_fraction)

    baseline_results = run_baselines(test_df)

    feature_cols = get_feature_columns(df_feat)
    model, preds, model_mae, model_rmse = train_and_predict(train_df, test_df, feature_cols)

    best_baseline_mae = min(v[0] for v in baseline_results.values())
    improvement = (best_baseline_mae - model_mae) / best_baseline_mae * 100

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, (mae, rmse) in baseline_results.items():
        print(f"  {label:<30s} MAE = {mae:.4f} kWh")
    print(f"  {'XGBoost model':<30s} MAE = {model_mae:.4f} kWh")
    print(f"\n  Improvement over best naive baseline: {improvement:.1f}%")
    print("=" * 60)

    plot_results(test_df, preds, args.out)


if __name__ == "__main__":
    main()