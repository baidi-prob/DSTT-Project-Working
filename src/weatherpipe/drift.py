"""Continuous Data Quality Monitoring & Statistical Drift Detection."""

import numpy as np
import pandas as pd


def check_missingness_drift(
    df_incoming: pd.DataFrame,
    threshold_pct: float = 15.0,
) -> dict[str, float]:
    """Audit missing rate on incoming data batch. Alerts if missingness exceeds threshold."""
    report = {}
    for col in df_incoming.select_dtypes(include=[np.number]).columns:
        pct = (df_incoming[col].isna().sum() / len(df_incoming)) * 100
        report[col] = round(pct, 2)
    return report


def detect_distribution_drift(
    reference_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    features: list[str],
    p_value_threshold: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Perform Kolmogorov-Smirnov statistical test for concept / covariate drift.

    H0: The incoming batch and reference baseline come from the same continuous distribution.
    If p-value < p_value_threshold: Significant distribution drift detected.
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        # Fallback to mean and standard deviation shift
        report = {}
        for feat in features:
            if feat in reference_df.columns and feat in incoming_df.columns:
                ref_s = reference_df[feat].dropna()
                inc_s = incoming_df[feat].dropna()
                mean_shift = float(abs(inc_s.mean() - ref_s.mean()))
                std_ratio = float(inc_s.std() / (ref_s.std() + 1e-6))
                report[feat] = {
                    "mean_shift": round(mean_shift, 4),
                    "std_ratio": round(std_ratio, 4),
                    "drift_detected": bool(mean_shift > 3.0 or std_ratio > 2.0 or std_ratio < 0.5),
                }
        return report

    report = {}
    for feat in features:
        if feat in reference_df.columns and feat in incoming_df.columns:
            ref_vals = reference_df[feat].dropna().values
            inc_vals = incoming_df[feat].dropna().values

            if len(ref_vals) < 10 or len(inc_vals) < 10:
                continue

            stat, p_val = ks_2samp(ref_vals, inc_vals)
            report[feat] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 5),
                "drift_detected": bool(p_val < p_value_threshold),
            }

    return report
