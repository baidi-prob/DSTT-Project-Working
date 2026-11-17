"""Unit Tests for Temporal Alignment and Sequence Normalization."""

import numpy as np
import pandas as pd

from weatherpipe.preprocessing import align_time_series


def test_align_regular_hourly_grid():
    """Verify that irregular timestamps are rounded and reindexed to a continuous 1-hour grid."""
    timestamps = [
        "2023-01-01 00:02:15",
        "2023-01-01 01:00:00",
        "2023-01-01 03:00:00",  # Hour 02:00:00 is missing
        "2023-01-01 04:00:00",
    ]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "temp": [15.0, 16.0, 18.0, 17.5],
        "humidity": [80.0, 75.0, 70.0, 72.0],
    })

    df_aligned = align_time_series(df, freq="1h")

    # Should have 5 hourly steps: 00:00, 01:00, 02:00, 03:00, 04:00
    assert len(df_aligned) == 5
    assert (df_aligned["timestamp"].diff().iloc[1:] == pd.Timedelta(hours=1)).all()

    # The missing hour (02:00:00) must exist as an explicit NaN row
    h2_row = df_aligned[df_aligned["timestamp"] == pd.Timestamp("2023-01-01 02:00:00")]
    assert len(h2_row) == 1
    assert pd.isna(h2_row["temp"].iloc[0])


def test_align_deduplication():
    """Verify that multiple readings in the same hourly bucket are aggregated without duplicate rows."""
    timestamps = [
        "2023-01-01 01:05:00",
        "2023-01-01 01:20:00",
        "2023-01-01 02:00:00",
    ]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "temp": [10.0, 12.0, 14.0],
    })

    df_aligned = align_time_series(df, freq="1h")
    assert len(df_aligned) == 2
    # The two readings for 01:00 should be averaged (10.0 + 12.0)/2 = 11.0
    val_1h = df_aligned.loc[df_aligned["timestamp"] == "2023-01-01 01:00:00", "temp"].iloc[0]
    assert np.isclose(val_1h, 11.0)
