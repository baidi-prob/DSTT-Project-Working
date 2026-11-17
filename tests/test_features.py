"""Unit Tests for Automated Feature Generation and Collinearity Pruning."""

import numpy as np
import pandas as pd

from weatherpipe.features import FeatureEngineer, compute_dew_point


def test_dew_point_calculation():
    """Verify Magnus-Tetens dew point calculation matches meteorological standards."""
    # At T=20C, RH=100%, Dew Point = 20C
    dp_100 = compute_dew_point(pd.Series([20.0]), pd.Series([100.0])).iloc[0]
    assert np.isclose(dp_100, 20.0, atol=0.2)

    # At T=20C, RH=50%, Dew Point is approximately 9.3C
    dp_50 = compute_dew_point(pd.Series([20.0]), pd.Series([50.0])).iloc[0]
    assert 8.5 < dp_50 < 10.0


def test_feature_generation_shapes_and_lags():
    """Verify that multi-scale lags, rolling stats, and cyclical encodings are produced."""
    dates = pd.date_range("2023-01-01", periods=50, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "temp": np.linspace(10, 25, 50),
        "humidity": np.linspace(50, 80, 50),
        "pressure": np.linspace(1010, 1020, 50),
        "wind_speed": np.full(50, 5.0),
    })

    fe = FeatureEngineer(target_horizon_hours=6)
    feat_df = fe.generate_raw_features(df)

    # Check that lags exist and match expected shift
    assert "temp_lag_1" in feat_df.columns
    assert "temp_lag_6" in feat_df.columns
    assert np.isnan(feat_df["temp_lag_1"].iloc[0])
    assert feat_df["temp_lag_1"].iloc[1] == df["temp"].iloc[0]

    # Check rolling stats
    assert "temp_roll_mean_3h" in feat_df.columns
    assert "temp_roll_mean_24h" in feat_df.columns

    # Check cyclical markers
    assert "hour_sin" in feat_df.columns
    assert "hour_cos" in feat_df.columns

    # Check target is forward shifted
    assert "target_temp_h6" in feat_df.columns
    assert feat_df["target_temp_h6"].iloc[0] == df["temp"].iloc[6]


def test_collinear_feature_pruning():
    """Verify that highly correlated redundant features are pruned."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    x = np.linspace(1, 100, 100)
    # x_duplicate has r=1.0 with x
    df = pd.DataFrame({
        "timestamp": dates,
        "x1": x,
        "x1_redundant": x * 1.0001,
        "x2_independent": np.sin(x),
        "target_temp_h6": x,
    })

    fe = FeatureEngineer(collinear_threshold=0.98)
    fe.fit_feature_selection(df)

    # One of x1 or x1_redundant must be dropped
    selected = fe.selected_features_
    assert not ("x1" in selected and "x1_redundant" in selected)
    assert "x2_independent" in selected
