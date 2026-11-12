"""RQ2 Experiment: Imputation Fidelity, Error Benchmark, and Synthetic Bias Evaluation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from weatherpipe.loader import load_raw_city_data
from weatherpipe.preprocessing import (
    CausalDiurnalImputer,
    align_time_series,
    clean_physical_ranges,
)


def evaluate_imputation_fidelity(
    y_true: np.ndarray,
    y_imputed: np.ndarray,
) -> dict[str, float]:
    """Calculate RMSE, MAE, and Mean Bias Error (MBE) over masked coordinates."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_imputed)
    yt = y_true[mask]
    yi = y_imputed[mask]

    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mbe": np.nan, "count": 0}

    rmse = float(np.sqrt(np.mean((yi - yt) ** 2)))
    mae = float(np.mean(np.abs(yi - yt)))
    mbe = float(np.mean(yi - yt))  # Synthetic bias measure

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mbe": round(mbe, 4),
        "count": len(yt),
    }


def run_rq2_experiment(
    city: str = "San Francisco",
    raw_dir: str = "data/raw",
    sample_path: str = None,
    out_dir: str = "results",
    random_state: int = 42,
) -> pd.DataFrame:
    """Run RQ2 Benchmark: Mask known ground truth values and measure imputation accuracy & bias."""
    out_path = Path(out_dir)
    plots_path = out_path / "plots"
    out_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f"Starting RQ2 Imputation Fidelity Benchmark on [{city}]")
    print("=======================================================")

    # 1. Load and align data
    df_raw = load_raw_city_data(city=city, raw_dir=raw_dir, sample_path=sample_path)
    df_aligned = align_time_series(df_raw)
    df_cleaned = clean_physical_ranges(df_aligned)

    # 2. Extract complete contiguous blocks of temperature and humidity for ground-truth testing
    # We use the final 30% of the dataset as the evaluation period
    split_idx = int(len(df_cleaned) * 0.70)
    train_history = df_cleaned.iloc[:split_idx].copy()
    test_slice = df_cleaned.iloc[split_idx:].copy().reset_index(drop=True)

    # Fit proposed imputer on historical training data
    proposed_imputer = CausalDiurnalImputer(target_cols=["temp", "humidity", "pressure", "wind_speed"])
    proposed_imputer.fit(train_history)

    target_var = "temp"
    ground_truth = test_slice[target_var].copy()
    valid_indices = ground_truth.dropna().index.values
    rng = np.random.RandomState(random_state)

    results = []

    # -----------------------------------------------------------------
    # Scenario A: Missing Completely at Random (MCAR) - 10% and 20%
    # -----------------------------------------------------------------
    for missing_rate in [0.10, 0.20]:
        n_mask = int(len(valid_indices) * missing_rate)
        mask_idx = rng.choice(valid_indices, size=n_mask, replace=False)

        masked_slice = test_slice.copy()
        masked_slice.loc[mask_idx, target_var] = np.nan

        # 1. Mean Imputation
        mean_val = train_history[target_var].mean()
        imp_mean = masked_slice[target_var].fillna(mean_val).values

        # 2. Forward Fill
        imp_ffill = masked_slice[target_var].ffill().bfill().values

        # 3. Linear Interpolation (Offline standard)
        imp_linear = masked_slice[target_var].interpolate(method="linear").bfill().ffill().values

        # 4. Proposed Causal Diurnal Imputer
        df_prop = proposed_imputer.transform(masked_slice.copy())
        imp_prop = df_prop[target_var].values

        true_vals = ground_truth.loc[mask_idx].values
        met_mean = evaluate_imputation_fidelity(true_vals, imp_mean[mask_idx])
        met_ffill = evaluate_imputation_fidelity(true_vals, imp_ffill[mask_idx])
        met_lin = evaluate_imputation_fidelity(true_vals, imp_linear[mask_idx])
        met_prop = evaluate_imputation_fidelity(true_vals, imp_prop[mask_idx])

        for m_name, met in [
            ("Global Mean Fill", met_mean),
            ("Forward Fill (ffill)", met_ffill),
            ("Linear Interpolation (Non-causal)", met_lin),
            ("Proposed Causal Diurnal", met_prop),
        ]:
            results.append({
                "Scenario": f"Random Dropout ({int(missing_rate*100)}%)",
                "Imputation Method": m_name,
                "RMSE (°C)": met["rmse"],
                "MAE (°C)": met["mae"],
                "Mean Bias Error (MBE °C)": met["mbe"],
                "Evaluated Points": met["count"],
            })

    # -----------------------------------------------------------------
    # Scenario B: Burst / Block Outages (3h, 6h, 12h, 24h Consecutive Gaps)
    # -----------------------------------------------------------------
    for block_len in [3, 6, 12, 24]:
        # Inject 15 non-overlapping outage blocks
        masked_slice = test_slice.copy()
        block_masks = []
        step = len(valid_indices) // 16
        for b_i in range(1, 16):
            start = b_i * step
            if start + block_len < len(test_slice):
                block_masks.extend(range(start, start + block_len))

        block_masks = np.array([idx for idx in block_masks if idx in valid_indices])
        masked_slice.loc[block_masks, target_var] = np.nan

        imp_mean = masked_slice[target_var].fillna(train_history[target_var].mean()).values
        imp_ffill = masked_slice[target_var].ffill().bfill().values
        imp_lin = masked_slice[target_var].interpolate(method="linear").bfill().ffill().values
        df_prop = proposed_imputer.transform(masked_slice.copy())
        imp_prop = df_prop[target_var].values

        true_vals = ground_truth.loc[block_masks].values
        met_mean = evaluate_imputation_fidelity(true_vals, imp_mean[block_masks])
        met_ffill = evaluate_imputation_fidelity(true_vals, imp_ffill[block_masks])
        met_lin = evaluate_imputation_fidelity(true_vals, imp_lin[block_masks])
        met_prop = evaluate_imputation_fidelity(true_vals, imp_prop[block_masks])

        for m_name, met in [
            ("Global Mean Fill", met_mean),
            ("Forward Fill (ffill)", met_ffill),
            ("Linear Interpolation (Non-causal)", met_lin),
            ("Proposed Causal Diurnal", met_prop),
        ]:
            results.append({
                "Scenario": f"Burst Outage ({block_len}h Block)",
                "Imputation Method": m_name,
                "RMSE (°C)": met["rmse"],
                "MAE (°C)": met["mae"],
                "Mean Bias Error (MBE °C)": met["mbe"],
                "Evaluated Points": met["count"],
            })

    df_rq2 = pd.DataFrame(results)
    csv_path = out_path / "rq2_imputation_benchmark.csv"
    df_rq2.to_csv(csv_path, index=False)
    print(f"\nSaved RQ2 Imputation Benchmark results to: {csv_path}")

    # Generate Visualization Plot
    try:
        import matplotlib.pyplot as plt

        df_blocks = df_rq2[df_rq2["Scenario"].str.contains("Burst")].copy()
        scenarios = df_blocks["Scenario"].unique()
        methods = df_blocks["Imputation Method"].unique()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(scenarios))
        width = 0.20

        color_map = {
            "Global Mean Fill": "#d62728",
            "Forward Fill (ffill)": "#ff7f0e",
            "Linear Interpolation (Non-causal)": "#1f77b4",
            "Proposed Causal Diurnal": "#2ca02c",
        }

        for i, m in enumerate(methods):
            m_data = df_blocks[df_blocks["Imputation Method"] == m]
            ax.bar(x + (i - 1.5) * width, m_data["RMSE (°C)"], width, label=m, color=color_map.get(m, None))

        ax.set_ylabel("Imputation RMSE (°C) - Lower is Better", fontsize=11, fontweight="bold")
        ax.set_title("RQ2 Imputation Fidelity under Burst Sensor Outages (Ground Truth Test)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontweight="bold")
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot_path = plots_path / "rq2_imputation_rmse.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {plot_path}")

    except Exception as e:
        print(f"RQ2 plot generation skipped: {e}")

    return df_rq2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute RQ2 Imputation Study")
    parser.add_argument("--city", default="San Francisco", help="City name")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw dataset path")
    parser.add_argument("--sample", default=None, help="Sample CSV path")
    args = parser.parse_args()

    run_rq2_experiment(city=args.city, raw_dir=args.raw_dir, sample_path=args.sample)
