"""Unit Tests for Quality Filtering, Range Bounds, and Spike Removal."""

import pandas as pd

from weatherpipe.preprocessing import clean_physical_ranges


def test_clean_sentinels_and_bounds():
    """Verify that sentinel codes and unphysical thermodynamic values are mapped to NaN."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=6, freq="1h"),
        "temp": [15.0, -999.0, 75.0, -60.0, 20.0, 9999.0],  # -999 & 9999 are sentinels, 75C & -60C out of bound
        "humidity": [50.0, 110.0, -5.0, 85.0, 99.0, 0.0],   # 110 & -5 out of bound
        "pressure": [1013.25, 0.0, 450.0, 1020.0, 1200.0, 1010.0], # 0, 450, 1200 out of bound
    })

    df_cleaned = clean_physical_ranges(df)

    # Temperature checks
    assert pd.isna(df_cleaned.loc[1, "temp"])  # -999
    assert pd.isna(df_cleaned.loc[2, "temp"])  # 75C > 60C
    assert pd.isna(df_cleaned.loc[3, "temp"])  # -60C < -50C
    assert pd.isna(df_cleaned.loc[5, "temp"])  # 9999

    # Humidity checks
    assert pd.isna(df_cleaned.loc[1, "humidity"]) # 110%
    assert pd.isna(df_cleaned.loc[2, "humidity"]) # -5%
    assert df_cleaned.loc[0, "humidity"] == 50.0

    # Pressure checks
    assert pd.isna(df_cleaned.loc[1, "pressure"]) # 0 hPa
    assert pd.isna(df_cleaned.loc[2, "pressure"]) # 450 hPa
    assert pd.isna(df_cleaned.loc[4, "pressure"]) # 1200 hPa


def test_clean_meteorological_spike():
    """Verify that sudden unphysical 1-hour temperature jumps are removed."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="1h"),
        "temp": [15.0, 15.5, 32.0, 16.0, 16.2], # Jump of 16.5C in 1 hour is a sensor glitch
        "pressure": [1013.0, 1013.2, 1013.1, 1013.0, 1013.2],
    })

    df_cleaned = clean_physical_ranges(df, temp_spike_threshold=10.0)
    assert pd.isna(df_cleaned.loc[2, "temp"])
    assert df_cleaned.loc[1, "temp"] == 15.5
    assert df_cleaned.loc[3, "temp"] == 16.0
