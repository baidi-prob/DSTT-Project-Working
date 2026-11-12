"""CRITICAL SCIENTIFIC TEST: Temporal Causality and Future Leakage Verification.

Principle:
In a valid time-series forecasting pipeline, features computed at timestamp t <= T
must be completely invariant to any future observations or alterations occurring
at timestamps t > T.

If changing data in the future alters features in the past, data leakage is present.
This test proves mathematically that WeatherPipe has ZERO future leakage.
"""

import numpy as np
import pandas as pd

from weatherpipe.features import FeatureEngineer
from weatherpipe.preprocessing import CausalDiurnalImputer


def test_zero_future_leakage_in_features():
    """Verify that altering future data (t > T) does NOT alter feature vectors at t <= T."""
    n_hours = 200
    dates = pd.date_range("2023-01-01", periods=n_hours, freq="1h")
    np.random.seed(42)

    df_base = pd.DataFrame({
        "timestamp": dates,
        "temp": 15.0 + 5.0 * np.sin(np.linspace(0, 10, n_hours)) + np.random.normal(0, 0.5, n_hours),
        "humidity": np.clip(60.0 + 10.0 * np.cos(np.linspace(0, 10, n_hours)), 10, 95),
        "pressure": 1013.0 + np.random.normal(0, 2, n_hours),
        "wind_speed": np.clip(np.random.exponential(5, n_hours), 0, 30),
    })

    cutoff_idx = 100  # Cutoff timestamp T

    # 1. Compute features on unmodified dataset
    fe1 = FeatureEngineer(target_horizon_hours=6)
    feat1 = fe1.generate_raw_features(df_base.copy())
    past_features_original = feat1.iloc[:cutoff_idx].drop(columns=["target_temp_h6", "timestamp"])

    # 2. Construct a corrupted future dataset:
    # All values after cutoff T are drastically altered (extreme temperature heatwave, hurricane pressure drop)
    df_altered_future = df_base.copy()
    df_altered_future.loc[cutoff_idx:, "temp"] = 55.0  # Extreme future heat
    df_altered_future.loc[cutoff_idx:, "pressure"] = 890.0  # Extreme hurricane drop
    df_altered_future.loc[cutoff_idx:, "humidity"] = 10.0

    # 3. Re-extract features on altered dataset
    fe2 = FeatureEngineer(target_horizon_hours=6)
    feat2 = fe2.generate_raw_features(df_altered_future)
    past_features_after_future_mutation = feat2.iloc[:cutoff_idx].drop(columns=["target_temp_h6", "timestamp"])

    # 4. Strict Assertion: Past features must match EXACTLY (bit-for-bit identity)
    pd.testing.assert_frame_equal(
        past_features_original,
        past_features_after_future_mutation,
        check_exact=True,
    )
    print("\n[LEAKAGE TEST PASSED] Zero future leakage detected in automated feature engineering.")


def test_zero_future_leakage_in_imputer():
    """Verify that causal diurnal imputation does not peek ahead to future observations."""
    n_hours = 120
    dates = pd.date_range("2023-01-01", periods=n_hours, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "temp": 20.0 + np.random.normal(0, 1, n_hours),
        "humidity": np.full(n_hours, 65.0),
        "pressure": np.full(n_hours, 1015.0),
        "wind_speed": np.full(n_hours, 4.0),
    })

    cutoff = 80
    train_history = df.iloc[:cutoff].copy()

    imputer = CausalDiurnalImputer()
    imputer.fit(train_history)

    # Test slice with a gap at index 10
    test_slice1 = df.iloc[cutoff:].copy().reset_index(drop=True)
    test_slice1.loc[10, "temp"] = np.nan

    imp_result1 = imputer.transform(test_slice1.copy())
    imputed_val1 = imp_result1.loc[10, "temp"]

    # Now alter future data at index 15..25
    test_slice2 = test_slice1.copy()
    test_slice2.loc[15:25, "temp"] = -40.0  # Extreme future cold anomaly

    imp_result2 = imputer.transform(test_slice2.copy())
    imputed_val2 = imp_result2.loc[10, "temp"]

    # Imputed value at index 10 must NOT change despite massive future alteration at index 15
    assert np.isclose(imputed_val1, imputed_val2, atol=1e-6)
    print("\n[LEAKAGE TEST PASSED] Zero future leakage detected in causal imputer.")
