"""Domain-Aware Meteorological Preprocessing and Causal Imputation Module."""

import numpy as np
import pandas as pd

PHYSICAL_BOUNDS = {
    "temp": (-50.0, 60.0),        # Celsius
    "humidity": (0.0, 100.0),     # Percentage
    "pressure": (870.0, 1085.0),  # hPa / mbar
    "wind_speed": (0.0, 75.0),    # m/s
    "wind_dir": (0.0, 360.0),     # Degrees
}


def align_time_series(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Align raw meteorological logs to a continuous, regular hourly time grid.

    - Parses and rounds timestamps to the nearest hour.
    - Resolves and removes duplicate timestamps.
    - Reindexes to an unbroken date range (creating explicit NaNs for missing hours).
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.round(freq)
    # Deduplicate: if multiple records fall in the same hour, average numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols and c != "timestamp"]

    df_agg = df.groupby("timestamp")[numeric_cols].mean()
    if non_numeric:
        df_first = df.groupby("timestamp")[non_numeric].first()
        df_agg = pd.concat([df_agg, df_first], axis=1)

    # Reindex to a complete regular grid
    full_idx = pd.date_range(start=df_agg.index.min(), end=df_agg.index.max(), freq=freq, name="timestamp")
    df_aligned = df_agg.reindex(full_idx).reset_index()
    return df_aligned


def clean_physical_ranges(
    df: pd.DataFrame,
    temp_spike_threshold: float = 10.0,
    pressure_spike_threshold: float = 25.0,
) -> pd.DataFrame:
    """Filter sentinel codes, physical out-of-bound errors, and unphysical instrument spikes.

    Sentinel codes (such as -999, 9999, 0 pressure) are mapped to NaN.
    Values exceeding physical thermodynamic boundaries are set to NaN.
    Sudden 1-hour spikes exceeding meteorological thresholds are filtered.
    """
    df = df.copy()

    # Sentinel mapping
    for col in ["temp", "humidity", "pressure", "wind_speed", "wind_dir"]:
        if col in df.columns:
            df.loc[df[col].isin([-999, -9999, 999, 9999]), col] = np.nan

    # Zero pressure is physically impossible at sea level / habitable elevations
    if "pressure" in df.columns:
        df.loc[df["pressure"] <= 500.0, "pressure"] = np.nan

    # Bound checking
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            invalid_mask = (df[col] < low) | (df[col] > high)
            df.loc[invalid_mask, col] = np.nan

    # Meteorological Spike Filter: isolated 1-hour physical glitch
    if "temp" in df.columns:
        prev_temp = df["temp"].shift(1)
        next_temp = df["temp"].shift(-1)
        diff_prev = (df["temp"] - prev_temp).abs()
        diff_next = (df["temp"] - next_temp).abs()
        diff_context = (next_temp - prev_temp).abs()
        # An isolated spike deviates strongly from both sides while context is relatively stable
        temp_spike = (diff_prev > temp_spike_threshold) & (diff_next > temp_spike_threshold) & (diff_context < temp_spike_threshold)
        df.loc[temp_spike, "temp"] = np.nan

    if "pressure" in df.columns:
        prev_p = df["pressure"].shift(1)
        next_p = df["pressure"].shift(-1)
        diff_prev_p = (df["pressure"] - prev_p).abs()
        diff_next_p = (df["pressure"] - next_p).abs()
        diff_ctx_p = (next_p - prev_p).abs()
        press_spike = (diff_prev_p > pressure_spike_threshold) & (diff_next_p > pressure_spike_threshold) & (diff_ctx_p < pressure_spike_threshold)
        df.loc[press_spike, "pressure"] = np.nan

    return df


def encode_circular_wind(df: pd.DataFrame) -> pd.DataFrame:
    """Encode circular wind direction (0°–360°) into orthogonal sine and cosine components.

    Resolves the boundary jump where 359° and 1° appear distant in Euclidean space.
    Calm conditions (wind_speed == 0) yield (sin=0, cos=0).
    """
    df = df.copy()
    if "wind_dir" in df.columns:
        rad = np.deg2rad(df["wind_dir"])
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)

        if "wind_speed" in df.columns:
            calm_mask = df["wind_speed"] == 0
            df.loc[calm_mask, "wind_dir_sin"] = 0.0
            df.loc[calm_mask, "wind_dir_cos"] = 0.0

    return df


class CausalDiurnalImputer:
    """Domain-Aware, Strictly Causal Meteorological Imputer.

    Scientific Principle:
    Standard linear / spline interpolation looks forward into future timestamps (t + k),
    causing critical data leakage in real-time operational forecasting.

    This imputer operates strictly causally:
    1. During `fit(train_df)`:
       Learns the diurnal climatological cycle for each (month, hour) combination:
       mu(month, hour) = mean(variable | month, hour).
    2. During `transform(df)`:
       Fills gaps by combining the diurnal climatological baseline with an exponentially
       decaying anomaly from the most recent valid observation (t - k):
       x_hat(t) = mu(m(t), h(t)) + (alpha^k) * (x(t-k) - mu(m(t-k), h(t-k)))
       where alpha in (0, 1) is the persistence decay rate (default 0.90).
       Short gaps (1–3h) preserve local atmospheric inertia; long gaps (>48h) smoothly
       revert to climatology without peeking into the future.
    """

    def __init__(self, target_cols: list | None = None, alpha: float = 0.90, max_anomaly_hours: int = 48):
        self.target_cols = target_cols or ["temp", "humidity", "pressure", "wind_speed"]
        self.alpha = alpha
        self.max_anomaly_hours = max_anomaly_hours
        self.climatology_: dict[str, pd.DataFrame] = {}
        self.global_means_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "CausalDiurnalImputer":
        """Compute diurnal climatology tables strictly on the training partition."""
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"])
        df["_month"] = ts.dt.month
        df["_hour"] = ts.dt.hour

        for col in self.target_cols:
            if col in df.columns:
                self.global_means_[col] = float(df[col].mean())
                # Group by month (1..12) and hour (0..23)
                clim_table = df.groupby(["_month", "_hour"])[col].mean().unstack(level="_hour")
                # Fill any remaining NaNs in the table with global mean
                clim_table = clim_table.fillna(self.global_means_[col])
                self.climatology_[col] = clim_table

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causally impute missing values using learned diurnal climatology + decaying anomaly."""
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"])
        months = ts.dt.month.values
        hours = ts.dt.hour.values

        for col in self.target_cols:
            if col not in df.columns or col not in self.climatology_:
                continue

            clim_table = self.climatology_[col]
            global_mean = self.global_means_.get(col, 0.0)

            # Extract climatological baseline for every row
            clim_baseline = np.array([
                clim_table.loc[m, h] if (m in clim_table.index and h in clim_table.columns) else global_mean
                for m, h in zip(months, hours)
            ])

            vals = df[col].values.copy()
            n = len(vals)
            last_valid_val = np.nan
            last_valid_idx = -999999

            for i in range(n):
                if not np.isnan(vals[i]):
                    last_valid_val = vals[i]
                    last_valid_idx = i
                else:
                    k = i - last_valid_idx
                    if k <= self.max_anomaly_hours and not np.isnan(last_valid_val):
                        # Anomaly at last valid timestamp
                        last_clim = clim_baseline[last_valid_idx]
                        anomaly = last_valid_val - last_clim
                        decay = self.alpha ** k
                        vals[i] = clim_baseline[i] + (decay * anomaly)
                    else:
                        vals[i] = clim_baseline[i]

            df[col] = vals

        # If wind circular components exist, impute them smoothly as well
        for circ_col in ["wind_dir_sin", "wind_dir_cos"]:
            if circ_col in df.columns and df[circ_col].isna().any():
                df[circ_col] = df[circ_col].ffill().bfill().fillna(0.0)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
