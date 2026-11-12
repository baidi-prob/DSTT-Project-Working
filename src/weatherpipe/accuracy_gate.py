"""CI/CD Accuracy Gate: Quality assertion to prevent performance regression."""

import argparse
import sys
from pathlib import Path

import pandas as pd


def evaluate_gate(results_csv: str = "results/rq1_forecast_accuracy.csv", simulate_failure: bool = False) -> int:
    """Evaluate performance gate: Automated Pipeline RMSE must be lower than Manual Baselines.

    Returns:
        0: Gate passed (automated pipeline wins).
        1: Gate failed (regression detected or failure simulated).
    """
    csv_file = Path(results_csv)
    if not csv_file.exists():
        print(f"[CI GATE ERROR] Results file not found at: {csv_file}")
        return 1

    df = pd.read_csv(csv_file)
    print("\n=======================================================")
    print("           CI/CD AUTOMATED ACCURACY GATE               ")
    print("=======================================================")
    print(df.to_string(index=False))
    print("-------------------------------------------------------")

    # Locate automated and manual RMSEs
    auto_row = df[df["Method"].str.contains("Automated", case=False, na=False)]
    analyst_row = df[df["Method"].str.contains("Analyst", case=False, na=False)]
    naive_row = df[df["Method"].str.contains("Naive", case=False, na=False)]

    if auto_row.empty or analyst_row.empty:
        print("[CI GATE ERROR] Missing required rows for comparison.")
        return 1

    auto_rmse = float(auto_row["RMSE"].iloc[0])
    analyst_rmse = float(analyst_row["RMSE"].iloc[0])
    naive_rmse = float(naive_row["RMSE"].iloc[0]) if not naive_row.empty else float("inf")

    best_manual_rmse = min(analyst_rmse, naive_rmse)
    delta_rmse = best_manual_rmse - auto_rmse
    gain_pct = (delta_rmse / best_manual_rmse) * 100

    print(f"Automated Pipeline RMSE : {auto_rmse:.4f}°C")
    print(f"Best Manual Baseline RMSE: {best_manual_rmse:.4f}°C")
    print(f"Observed Improvement    : {delta_rmse:+.4f}°C ({gain_pct:+.2f}%)")

    if simulate_failure:
        print("\n[GATE TRIGGERED: SIMULATED BREAKAGE TEST]")
        print("FAIL: Intentional failure test triggered for CI demo.")
        return 1

    if auto_rmse < best_manual_rmse:
        print(f"\n[GATE PASSED] Automated pipeline achieved a {gain_pct:.2f}% improvement over manual baseline.")
        print("Model promotion criteria satisfied.")
        return 0
    else:
        print(f"\n[GATE FAILED] Automated pipeline RMSE ({auto_rmse:.4f}) did not beat manual baseline ({best_manual_rmse:.4f}).")
        print("Regression detected. Blocking model deployment in CI/CD pipeline.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI/CD Accuracy Gate")
    parser.add_argument("--results", default="results/rq1_forecast_accuracy.csv", help="Path to summary results CSV")
    parser.add_argument("--simulate-failure", action="store_true", help="Force fail for CI demo")
    args = parser.parse_args()

    exit_code = evaluate_gate(results_csv=args.results, simulate_failure=args.simulate_failure)
    sys.exit(exit_code)
