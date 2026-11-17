"""Automated Domain-Aware Feature Engineering Module for WeatherPipe."""


import numpy as np
import pandas as pd


def compute_dew_point(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """Calculate the meteorological dew point temperature using the Magnus-Tetens formula.

    Formula:
        gamma(T, RH) = (17.27 * T) / (237.7 + T) + ln(RH / 100)
        T_dew = (237.7 * gamma) / (17.27 - gamma)
    """
    rh_clipped = humidity_pct.clip(lower=1.0, upper=100.0) / 100.0
    a = 17.27
    b = 237.7
    gamma = (a * temp_c) / (b + temp_c) + np.log(rh_clipped)
    dew_point = (b * gamma) / (a - gamma)
    return dew_point


class FeatureEngineer:
    """Automated, Strictly Causal Meteorological Feature Extraction & Selection Pipeline.

    Strict Causality Rule:
    Every feature at timestamp t depends strictly on data observed at or prior to t (<= t).
    Lags utilize positive shift intervals; rolling windows are backward-facing.
    """

    def __init__(
        self,
        lag_config: dict | None = None,
        rolling_windows: list[int] | None = None,
        collinear_threshold: float = 0.98,
        target_horizon_hours: int = 6,
    ):
        self.lag_config = lag_config or {
            "temp": [1, 2, 3, 6, 12, 24],
            "pressure": [1, 3, 6, 24],
            "humidity": [1, 3, 6, 24],
            "wind_speed": [1, 3, 6],
        }
        self.rolling_windows = rolling_windows or [3, 24]
        self.collinear_threshold = collinear_threshold
        self.target_horizon_hours = target_horizon_hours
        self.selected_features_: list[str] = []

    def generate_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Systematically extract lags, rolling aggregates, cyclical markers, and physical indicators."""
        feat_df = df.copy()
        ts = pd.to_datetime(feat_df["timestamp"])

        # 1. Temporal Cyclical Encodings (sin / cos)
        hour = ts.dt.hour
        doy = ts.dt.dayofyear
        feat_df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        feat_df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        feat_df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        feat_df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # 2. Multi-scale Causal Lags
        for col, lags in self.lag_config.items():
            if col in feat_df.columns:
                for lag in lags:
                    feat_df[f"{col}_lag_{lag}"] = feat_df[col].shift(lag)

        # 3. Backward-Only Rolling Statistics (3h and 24h)
        for w in self.rolling_windows:
            if "temp" in feat_df.columns:
                feat_df[f"temp_roll_mean_{w}h"] = (
                    feat_df["temp"].rolling(window=w, min_periods=1, closed="right").mean()
                )
                feat_df[f"temp_roll_std_{w}h"] = (
                    feat_df["temp"].rolling(window=w, min_periods=2, closed="right").std().fillna(0.0)
                )
            if "pressure" in feat_df.columns:
                feat_df[f"pressure_roll_mean_{w}h"] = (
                    feat_df["pressure"].rolling(window=w, min_periods=1, closed="right").mean()
                )

        # 4. Domain Meteorological Physical Interactions
        if "temp" in feat_df.columns and "humidity" in feat_df.columns:
            dew_point = compute_dew_point(feat_df["temp"], feat_df["humidity"])
            feat_df["dew_point"] = dew_point
            feat_df["dew_point_spread"] = feat_df["temp"] - dew_point

        # 5. Pressure Tendency (Barometric shift over 3 hours)
        if "pressure" in feat_df.columns:
            feat_df["pressure_tendency_3h"] = feat_df["pressure"] - feat_df["pressure"].shift(3)

        # 6. Target Variable Creation (Forward shift of horizon hours)
        if "temp" in feat_df.columns:
            feat_df["target_temp_h6"] = feat_df["temp"].shift(-self.target_horizon_hours)

        return feat_df

    def fit_feature_selection(self, df_features: pd.DataFrame) -> "FeatureEngineer":
        """Learn collinear feature pruning mask strictly on training set."""
        exclude_cols = {"timestamp", "target_temp_h6", "weather_desc"}
        candidate_cols = [c for c in df_features.columns if c not in exclude_cols]
        numeric_df = df_features[candidate_cols].dropna()

        if len(numeric_df) < 10:
            self.selected_features_ = candidate_cols
            return self

        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.collinear_threshold)]
        self.selected_features_ = [c for c in candidate_cols if c not in to_drop]
        return self

    def transform_features(self, df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Apply feature selection and return (X, y)."""
        if not self.selected_features_:
            raise ValueError("FeatureEngineer must be fit before transforming.")

        X = df_features[self.selected_features_].copy()
        y = df_features["target_temp_h6"].copy() if "target_temp_h6" in df_features.columns else None
        return X, y
