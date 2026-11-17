"""Unit Tests for Domain-Aware Causal Imputation and Circular Wind Encodings."""

import numpy as np
import pandas as pd

from weatherpipe.preprocessing import CausalDiurnalImputer, encode_circular_wind


def test_circular_wind_encoding():
    """Verify circular wind direction is properly decomposed without boundary jumps."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=4, freq="1h"),
        "wind_dir": [0.0, 90.0, 180.0, 270.0],
        "wind_speed": [5.0, 5.0, 5.0, 0.0],  # Last is calm
    })

    df_circ = encode_circular_wind(df)
    assert np.isclose(df_circ.loc[0, "wind_dir_sin"], 0.0, atol=1e-5)
    assert np.isclose(df_circ.loc[0, "wind_dir_cos"], 1.0, atol=1e-5)
    assert np.isclose(df_circ.loc[1, "wind_dir_sin"], 1.0, atol=1e-5)
    assert np.isclose(df_circ.loc[1, "wind_dir_cos"], 0.0, atol=1e-5)

    # Calm wind (speed == 0) must be 0, 0
    assert df_circ.loc[3, "wind_dir_sin"] == 0.0
    assert df_circ.loc[3, "wind_dir_cos"] == 0.0


def test_causal_diurnal_imputer_no_nans():
    """Verify that causal imputer replaces all gaps and honors diurnal pattern."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    # Temperature with synthetic diurnal cycle: 15 + 5*sin(2*pi*h/24)
    hours = dates.hour.values
    temps = 15.0 + 5.0 * np.sin(2 * np.pi * hours / 24.0)

    train_df = pd.DataFrame({
        "timestamp": dates[:80],
        "temp": temps[:80],
        "humidity": np.full(80, 60.0),
        "pressure": np.full(80, 1013.0),
        "wind_speed": np.full(80, 5.0),
    })

    test_df = pd.DataFrame({
        "timestamp": dates[80:],
        "temp": temps[80:],
        "humidity": np.full(20, 60.0),
        "pressure": np.full(20, 1013.0),
        "wind_speed": np.full(20, 5.0),
    })
    # Inject missing values into test
    test_df.loc[test_df.index[2:5], "temp"] = np.nan

    imputer = CausalDiurnalImputer(target_cols=["temp", "humidity", "pressure", "wind_speed"])
    imputer.fit(train_df)
    imputed_test = imputer.transform(test_df)

    assert not imputed_test["temp"].isna().any()
    # Imputed values should be close to true diurnal temperature
    imputed_vals = imputed_test.loc[test_df.index[2:5], "temp"].values
    true_vals = temps[80:][2:5]
    assert np.all(np.abs(imputed_vals - true_vals) < 2.0)
