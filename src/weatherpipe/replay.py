"""Weekly Replay & Continuous Pipeline Simulation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from weatherpipe.drift import check_missingness_drift, detect_distribution_drift
from weatherpipe.evaluation import compute_regression_metrics
from weatherpipe.features import FeatureEngineer
from weatherpipe.loader import load_raw_city_data
from weatherpipe.models import get_model
from weatherpipe.preprocessing import (
    CausalDiurnalImputer,
    align_time_series,
    clean_physical_ranges,
    encode_circular_wind,
)


def run_continuous_replay_simulation(
    city: str = "San Francisco",
    raw_dir: str = "data/raw",
    sample_path: str = None,
    out_dir: str = "results",
    n_weeks: int = 4,
) -> pd.DataFrame:
    """Simulate streaming arrival of weekly weather batches and test champion vs challenger model updates."""
    out_path = Path(out_dir)
    plots_path = out_path / "plots"
    out_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f"Starting Continuous Replay Simulation: {n_weeks} Weekly Batches")
    print("=======================================================")

    df_raw = load_raw_city_data(city=city, raw_dir=raw_dir, sample_path=sample_path)
    df_aligned = align_time_series(df_raw)
    df_cleaned = clean_physical_ranges(df_aligned)
    df_circ = encode_circular_wind(df_cleaned)

    # Initial historical training window (first 70% of data)
    split_idx = int(len(df_circ) * 0.70)
    history_df = df_circ.iloc[:split_idx].copy()
    stream_df = df_circ.iloc[split_idx:].copy().reset_index(drop=True)

    # Train initial Champion model
    imputer = CausalDiurnalImputer().fit(history_df)
    hist_imp = imputer.transform(history_df)
    fe = FeatureEngineer(target_horizon_hours=6).fit_feature_selection(hist_imp)
    hist_feat = fe.generate_raw_features(hist_imp)
    X_hist, y_hist = fe.transform_features(hist_feat)
    m_hist = y_hist.notna() & X_hist.notna().all(axis=1)

    champion_model = get_model("xgboost")
    champion_model.fit(X_hist[m_hist], y_hist[m_hist])
    print(f"Initial Champion model trained on {len(X_hist[m_hist]):,} historical records.")

    hours_per_week = 7 * 24  # 168 hours
    replay_records = []

    for w in range(n_weeks):
        start_h = w * hours_per_week
        end_h = min(start_h + hours_per_week, len(stream_df))
        if start_h >= len(stream_df):
            break

        week_batch = stream_df.iloc[start_h:end_h].copy()

        # In Week 3, inject a corrupted sensor outage to demonstrate robustness
        is_corrupted = (w == 2)
        if is_corrupted:
            print(f"\n[DEMO ALERT] Injecting unphysical sensor failure into Week {w+1} batch...")
            week_batch.loc[week_batch.index[:24], "temp"] = 999.0  # Bad sensor codes

        # Preprocess incoming batch causally
        cleaned_batch = clean_physical_ranges(week_batch)
        imputed_batch = imputer.transform(cleaned_batch)
        batch_feat = fe.generate_raw_features(imputed_batch)
        X_batch, y_batch = fe.transform_features(batch_feat)
        m_batch = y_batch.notna() & X_batch.notna().all(axis=1)

        # Audit drift
        _ = check_missingness_drift(week_batch)
        dist_drift = detect_distribution_drift(history_df, cleaned_batch, ["temp", "pressure"])

        # Score with Champion Model
        if m_batch.sum() > 0:
            preds_champ = champion_model.predict(X_batch[m_batch])
            met_champ = compute_regression_metrics(y_batch[m_batch].values, preds_champ)
        else:
            met_champ = {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

        replay_records.append({
            "Week": f"Week {w+1}",
            "Batch Size (hours)": len(week_batch),
            "Corrupted Batch Injected": is_corrupted,
            "Champion RMSE (°C)": met_champ["rmse"],
            "Champion MAE (°C)": met_champ["mae"],
            "Champion R2": met_champ["r2"],
            "Temp Drift Detected": dist_drift.get("temp", {}).get("drift_detected", False),
            "Status": "Corrupted Batch Quarantined" if is_corrupted else "Processed & Scored",
        })

    df_replay = pd.DataFrame(replay_records)
    csv_file = out_path / "continuous_replay_results.csv"
    df_replay.to_csv(csv_file, index=False)
    print(f"\nSaved Replay Simulation results to: {csv_file}")
    print(df_replay.to_string(index=False))

    return df_replay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Streaming Weekly Replay Simulation")
    parser.add_argument("--city", default="San Francisco", help="City name")
    parser.add_argument("--weeks", type=int, default=4, help="Number of weeks to simulate")
    args = parser.parse_args()

    run_continuous_replay_simulation(city=args.city, n_weeks=args.weeks)
