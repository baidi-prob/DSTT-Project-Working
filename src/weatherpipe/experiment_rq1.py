"""RQ1 Experiment: Forecast Accuracy Comparison & Feature Group Ablation Study."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from weatherpipe.evaluation import (
    compute_bootstrap_ci,
    compute_regression_metrics,
    generate_temporal_folds,
)
from weatherpipe.features import FeatureEngineer
from weatherpipe.loader import load_raw_city_data, save_curated_dataset
from weatherpipe.manual_baseline import (
    prepare_manual_baseline_a_naive,
    prepare_manual_baseline_b_analyst,
)
from weatherpipe.models import get_model
from weatherpipe.preprocessing import (
    CausalDiurnalImputer,
    align_time_series,
    clean_physical_ranges,
    encode_circular_wind,
)


def run_rq1_experiment(
    city: str = "San Francisco",
    raw_dir: str = "data/raw",
    sample_path: str = None,
    n_folds: int = 4,
    out_dir: str = "results",
    model_type: str = "xgboost",
    use_mlflow: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute the end-to-end RQ1 evaluation comparing baselines against the automated pipeline."""
    out_path = Path(out_dir)
    plots_path = out_path / "plots"
    out_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f"Starting RQ1 Experiment on [{city}] with model [{model_type}]")
    print("=======================================================")

    # 1. Load data
    df_raw = load_raw_city_data(city=city, raw_dir=raw_dir, sample_path=sample_path)
    print(f"Loaded raw records: {len(df_raw):,} rows from {df_raw['timestamp'].iloc[0]} to {df_raw['timestamp'].iloc[-1]}")

    # Also save a sample for CI if sample directory is empty
    sample_file = Path("data/sample/weather_sample.csv")
    if not sample_file.exists():
        save_curated_dataset(df_raw, sample_file, max_rows=3000)
        print(f"Cached sample dataset at: {sample_file}")

    # Setup MLflow if available
    mlflow_active = False
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(f"WeatherPipe_RQ1_{city}")
            mlflow_active = True
        except Exception as e:
            print(f"MLflow initialization skipped: {e}")

    # -------------------------------------------------------------
    # 2. Build Dataset 1: Automated Pipeline (Proposed)
    # -------------------------------------------------------------
    print("Building Automated Pipeline datasets (aligned, cleaned, causal imputed)...")
    df_aligned = align_time_series(df_raw)
    df_cleaned = clean_physical_ranges(df_aligned)
    df_circ = encode_circular_wind(df_cleaned)

    # Automated Feature Engineering
    fe = FeatureEngineer(target_horizon_hours=6)
    df_auto_raw_feat = fe.generate_raw_features(df_circ)

    # -------------------------------------------------------------
    # 3. Build Dataset 2 & 3: Manual Baselines
    # -------------------------------------------------------------
    print("Building Manual Baseline datasets (Naive & Analyst)...")
    X_base_a, y_base_a = prepare_manual_baseline_a_naive(df_raw, target_horizon=6)
    X_base_b, y_base_b = prepare_manual_baseline_b_analyst(df_raw, target_horizon=6)

    # Synchronize indices on aligned grid for fair comparison
    n_samples = len(df_auto_raw_feat)
    folds = generate_temporal_folds(n_samples=n_samples, n_folds=n_folds, min_train_ratio=0.45)
    print(f"Generated {len(folds)} expanding temporal cross-validation folds.")

    results_folds = []
    y_true_all = []
    y_pred_base_a_all = []
    y_pred_base_b_all = []
    y_pred_auto_all = []
    y_pred_persist_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\n--- Processing Temporal Fold {fold_idx}/{n_folds} (Train: {len(train_idx)}, Test: {len(test_idx)}) ---")

        # Split automated features
        train_slice = df_auto_raw_feat.iloc[train_idx].copy()
        test_slice = df_auto_raw_feat.iloc[test_idx].copy()

        # Fit imputer ONLY on train slice to avoid future leakage
        imputer = CausalDiurnalImputer()
        imputer.fit(train_slice)
        train_imputed = imputer.transform(train_slice)
        test_imputed = imputer.transform(test_slice)

        # Fit feature selection ONLY on train slice
        fe_fold = FeatureEngineer(target_horizon_hours=6)
        fe_fold.fit_feature_selection(train_imputed)
        X_train_auto, y_train_auto = fe_fold.transform_features(train_imputed)
        X_test_auto, y_test_auto = fe_fold.transform_features(test_imputed)

        # Valid rows mask
        train_mask_auto = y_train_auto.notna() & X_train_auto.notna().all(axis=1)

        # Train Automated Pipeline Model
        model_auto = get_model(model_type=model_type)
        model_auto.fit(X_train_auto[train_mask_auto], y_train_auto[train_mask_auto])
        preds_auto = model_auto.predict(X_test_auto)
        metrics_auto = compute_regression_metrics(y_test_auto.values, preds_auto)

        # Baseline A (Naive)
        X_tr_a, y_tr_a = X_base_a.iloc[train_idx], y_base_a.iloc[train_idx]
        X_te_a, y_te_a = X_base_a.iloc[test_idx], y_base_a.iloc[test_idx]
        m_tr_a = y_tr_a.notna() & X_tr_a.notna().all(axis=1)
        model_a = get_model(model_type=model_type)
        model_a.fit(X_tr_a[m_tr_a], y_tr_a[m_tr_a])
        preds_a = model_a.predict(X_te_a.fillna(0.0))
        metrics_a = compute_regression_metrics(y_te_a.values, preds_a)

        # Baseline B (Analyst)
        X_tr_b, y_tr_b = X_base_b.iloc[train_idx], y_base_b.iloc[train_idx]
        X_te_b, y_te_b = X_base_b.iloc[test_idx], y_base_b.iloc[test_idx]
        m_tr_b = y_tr_b.notna() & X_tr_b.notna().all(axis=1)
        model_b = get_model(model_type=model_type)
        model_b.fit(X_tr_b[m_tr_b], y_tr_b[m_tr_b])
        preds_b = model_b.predict(X_te_b.fillna(0.0))
        metrics_b = compute_regression_metrics(y_te_b.values, preds_b)

        # Persistence Reference (y_pred = current temp)
        current_temp_test = df_auto_raw_feat.iloc[test_idx]["temp"].values
        metrics_persist = compute_regression_metrics(y_test_auto.values, current_temp_test)

        results_folds.append({
            "fold": fold_idx,
            "persistence_rmse": metrics_persist["rmse"],
            "persistence_mae": metrics_persist["mae"],
            "persistence_r2": metrics_persist["r2"],
            "naive_rmse": metrics_a["rmse"],
            "naive_mae": metrics_a["mae"],
            "naive_r2": metrics_a["r2"],
            "analyst_rmse": metrics_b["rmse"],
            "analyst_mae": metrics_b["mae"],
            "analyst_r2": metrics_b["r2"],
            "automated_rmse": metrics_auto["rmse"],
            "automated_mae": metrics_auto["mae"],
            "automated_r2": metrics_auto["r2"],
            "rmse_gain_vs_analyst_%": round(((metrics_b["rmse"] - metrics_auto["rmse"]) / metrics_b["rmse"]) * 100, 2),
        })

        y_true_all.extend(y_test_auto.values)
        y_pred_base_a_all.extend(preds_a)
        y_pred_base_b_all.extend(preds_b)
        y_pred_auto_all.extend(preds_auto)
        y_pred_persist_all.extend(current_temp_test)

    df_results_folds = pd.DataFrame(results_folds)

    # Calculate overall summary metrics
    overall_auto = compute_regression_metrics(np.array(y_true_all), np.array(y_pred_auto_all))
    overall_analyst = compute_regression_metrics(np.array(y_true_all), np.array(y_pred_base_b_all))
    overall_naive = compute_regression_metrics(np.array(y_true_all), np.array(y_pred_base_a_all))
    overall_persist = compute_regression_metrics(np.array(y_true_all), np.array(y_pred_persist_all))

    # Bootstrap Confidence Interval on gain over best manual baseline (analyst)
    ci_res = compute_bootstrap_ci(
        y_true=np.array(y_true_all),
        y_pred_baseline=np.array(y_pred_base_b_all),
        y_pred_pipeline=np.array(y_pred_auto_all),
    )

    summary_rows = [
        {"Method": "Persistence (Sanity Baseline)", "RMSE": overall_persist["rmse"], "MAE": overall_persist["mae"], "R2": overall_persist["r2"]},
        {"Method": "Manual Baseline A (Naive Mean)", "RMSE": overall_naive["rmse"], "MAE": overall_naive["mae"], "R2": overall_naive["r2"]},
        {"Method": "Manual Baseline B (Ad-hoc Analyst)", "RMSE": overall_analyst["rmse"], "MAE": overall_analyst["mae"], "R2": overall_analyst["r2"]},
        {"Method": "Automated Pipeline (Proposed)", "RMSE": overall_auto["rmse"], "MAE": overall_auto["mae"], "R2": overall_auto["r2"]},
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_summary["Gain vs Best Manual (%)"] = [
        "-", "-", "-",
        f"{round(((overall_analyst['rmse'] - overall_auto['rmse']) / overall_analyst['rmse']) * 100, 2)}%"
    ]

    # Save CSVs
    folds_csv = out_path / "rq1_forecast_accuracy_by_fold.csv"
    summary_csv = out_path / "rq1_forecast_accuracy.csv"
    df_results_folds.to_csv(folds_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved RQ1 results to:\n- {folds_csv}\n- {summary_csv}")

    # -------------------------------------------------------------
    # 4. Feature Group Ablation Study (Noboa et al. 2024 inspired)
    # -------------------------------------------------------------
    print("\nRunning Feature Group Ablation Study...")
    ablation_groups = {
        "1. Lags Only": [c for c in df_auto_raw_feat.columns if "lag" in c],
        "2. Rolling Only": [c for c in df_auto_raw_feat.columns if "roll" in c],
        "3. Cyclical Only": [c for c in df_auto_raw_feat.columns if "sin" in c or "cos" in c],
        "4. Physical Interactions Only": [c for c in ["dew_point", "dew_point_spread", "pressure_tendency_3h"] if c in df_auto_raw_feat.columns],
        "5. All Combined (Full Pipeline)": [c for c in df_auto_raw_feat.columns if c not in {"timestamp", "target_temp_h6", "weather_desc"}],
    }

    # Evaluate on the final fold for clean ablation comparison
    last_train_idx, last_test_idx = folds[-1]
    train_slice = df_auto_raw_feat.iloc[last_train_idx].copy()
    test_slice = df_auto_raw_feat.iloc[last_test_idx].copy()
    imputer = CausalDiurnalImputer().fit(train_slice)
    train_imp = imputer.transform(train_slice)
    test_imp = imputer.transform(test_slice)
    y_tr = train_imp["target_temp_h6"]
    y_te = test_imp["target_temp_h6"]

    ablation_results = []
    for grp_name, feat_cols in ablation_groups.items():
        if not feat_cols:
            continue
        X_tr = train_imp[feat_cols]
        X_te = test_imp[feat_cols]
        mask_tr = y_tr.notna() & X_tr.notna().all(axis=1)
        mask_te = y_te.notna() & X_te.notna().all(axis=1)

        m = get_model(model_type=model_type)
        m.fit(X_tr[mask_tr], y_tr[mask_tr])
        p = m.predict(X_te[mask_te])
        met = compute_regression_metrics(y_te[mask_te].values, p)

        ablation_results.append({
            "Feature Group": grp_name,
            "Feature Count": len(feat_cols),
            "RMSE": met["rmse"],
            "MAE": met["mae"],
            "R2": met["r2"],
        })

    df_ablation = pd.DataFrame(ablation_results).sort_values("RMSE")
    ablation_csv = out_path / "rq1_ablation_study.csv"
    df_ablation.to_csv(ablation_csv, index=False)
    print(f"Saved Ablation Study to: {ablation_csv}")

    # Generate Visualization Plots
    try:
        import matplotlib.pyplot as plt

        # Plot 1: RMSE Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = df_summary["Method"].tolist()
        rmses = df_summary["RMSE"].tolist()
        colors = ["#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(methods, rmses, color=colors, width=0.55)
        ax.set_ylabel("RMSE (°C) - Lower is Better", fontsize=11, fontweight="bold")
        ax.set_title(f"Forecast Accuracy: Baseline vs Automated Pipeline ({city}, Horizon=6h)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.2f}°C", ha="center", va="bottom", fontweight="bold")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plot1_path = plots_path / "rq1_rmse_comparison.png"
        fig.savefig(plot1_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {plot1_path}")

        # Plot 2: Ablation Bar Chart
        fig, ax = plt.subplots(figsize=(9, 4.5))
        grp_names = df_ablation["Feature Group"].tolist()
        grp_rmse = df_ablation["RMSE"].tolist()
        bars2 = ax.barh(grp_names, grp_rmse, color="#1f77b4", height=0.55)
        ax.set_xlabel("RMSE (°C) - Lower is Better", fontsize=11, fontweight="bold")
        ax.set_title("Feature Group Ablation Study (Noboa et al. 2024 Methodology)", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for bar in bars2:
            xval = bar.get_width()
            ax.text(xval + 0.03, bar.get_y() + bar.get_height() / 2.0, f"{xval:.2f}°C", ha="left", va="center", fontweight="bold")
        plt.tight_layout()
        plot2_path = plots_path / "rq1_ablation_study.png"
        fig.savefig(plot2_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {plot2_path}")

    except Exception as e:
        print(f"Plot generation skipped: {e}")

    # Log to MLflow if enabled
    if mlflow_active:
        try:
            with mlflow.start_run(run_name=f"Full_Evaluation_{city}"):
                mlflow.log_param("city", city)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("n_folds", n_folds)
                mlflow.log_metric("overall_automated_rmse", overall_auto["rmse"])
                mlflow.log_metric("overall_analyst_rmse", overall_analyst["rmse"])
                mlflow.log_metric("overall_naive_rmse", overall_naive["rmse"])
                mlflow.log_metric("ci_mean_gain", ci_res["mean_gain"])
                mlflow.log_metric("ci_lower", ci_res["ci_lower"])
                mlflow.log_metric("ci_upper", ci_res["ci_upper"])
                mlflow.log_artifact(str(summary_csv))
                mlflow.log_artifact(str(ablation_csv))
                if (plots_path / "rq1_rmse_comparison.png").exists():
                    mlflow.log_artifact(str(plots_path / "rq1_rmse_comparison.png"))
            print("Successfully logged experiment run to MLflow.")
        except Exception as e:
            print(f"MLflow run logging skipped: {e}")

    return {"summary": df_summary, "folds": df_results_folds, "ablation": df_ablation, "ci": ci_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute RQ1 Forecast Accuracy Experiment")
    parser.add_argument("--city", default="San Francisco", help="City name")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw dataset path")
    parser.add_argument("--sample", default=None, help="Sample CSV path")
    parser.add_argument("--folds", type=int, default=4, help="Number of temporal folds")
    parser.add_argument("--model", default="xgboost", help="Model family (xgboost, random_forest)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    run_rq1_experiment(
        city=args.city,
        raw_dir=args.raw_dir,
        sample_path=args.sample,
        n_folds=args.folds,
        model_type=args.model,
        use_mlflow=not args.no_mlflow,
    )
