"""Manual Baseline Workflows (Ad-Hoc / Legacy Approaches)."""


import numpy as np
import pandas as pd


def prepare_manual_baseline_a_naive(df: pd.DataFrame, target_horizon: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    """Baseline A (Naive): Simple global mean imputation and raw unengineered features.

    Reflects the most basic student/ad-hoc notebook approach:
    - Sentinel codes replaced with NaN.
    - Global column mean imputation.
    - Raw features: temp, humidity, pressure, wind_speed, wind_dir.
    - No temporal alignment, no cyclical encodings, no multi-scale lags.
    """
    df_clean = df.copy()

    # Sentinel replacement
    for col in ["temp", "humidity", "pressure", "wind_speed", "wind_dir"]:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col].isin([-999, -9999, 999, 9999]), col] = np.nan
            mean_val = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(mean_val if pd.notna(mean_val) else 0.0)

    feature_cols = [c for c in ["temp", "humidity", "pressure", "wind_speed", "wind_dir"] if c in df_clean.columns]
    X = df_clean[feature_cols].copy()
    y = df_clean["temp"].shift(-target_horizon)
    return X, y


def prepare_manual_baseline_b_analyst(df: pd.DataFrame, target_horizon: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    """Baseline B (Ad-hoc Analyst): Forward-fill, raw calendar integers, 1 lag, 1 rolling mean.

    Reflects a typical practitioner's manual feature exploration:
    - Forward-fill imputation (ffill) with backward-fill fallback.
    - Integer hour (0..23) and integer month (1..12).
    - 1 lag feature: temp(t - 1).
    - 1 rolling average: 6-hour moving average of temp.
    - Raw physical columns.
    """
    df_clean = df.copy()

    # Sentinel replacement
    for col in ["temp", "humidity", "pressure", "wind_speed", "wind_dir"]:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col].isin([-999, -9999, 999, 9999]), col] = np.nan
            df_clean[col] = df_clean[col].ffill().bfill().fillna(0.0)

    ts = pd.to_datetime(df_clean["timestamp"])
    df_clean["hour_raw"] = ts.dt.hour
    df_clean["month_raw"] = ts.dt.month
    df_clean["temp_lag_1"] = df_clean["temp"].shift(1)
    df_clean["temp_roll_6h"] = df_clean["temp"].rolling(window=6, min_periods=1, closed="right").mean()

    feature_cols = [
        "temp", "humidity", "pressure", "wind_speed",
        "hour_raw", "month_raw", "temp_lag_1", "temp_roll_6h",
    ]
    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    X = df_clean[feature_cols].copy()
    y = df_clean["temp"].shift(-target_horizon)
    return X, y
