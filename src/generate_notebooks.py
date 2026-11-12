"""Script to generate cleanly formatted Jupyter Notebooks for DSTT project demonstration."""

import json
from pathlib import Path


def create_notebook(cells, out_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {out_path}")


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    }


def main():
    # Notebook 1: EDA
    nb1_cells = [
        md_cell("# Meteorological Data Exploratory Analysis & Audit\n## Course: Data Science Tools and Techniques (DSTT)\n**Authors**: Obaid Raja, Zoffishan Sattar, Hamza Ali\n\nThis notebook audits the raw Kaggle Historical Hourly Weather Dataset (2012–2017) across 36 cities, identifying missing rates, sensor dropouts, unit anomalies, and unphysical values."),
        code_cell("import pandas as pd\nimport numpy as np\nfrom weatherpipe.loader import load_raw_city_data\nfrom weatherpipe.eda import run_eda"),
        md_cell("### 1. Execute Comprehensive EDA Audit on Raw CSVs"),
        code_cell("eda_report = run_eda(raw_dir='data/raw', out_path='results/eda_summary.md')\nprint('EDA Audit Complete. Summary saved to results/eda_summary.md')"),
        md_cell("### 2. Inspect Extracted City Series (San Francisco)"),
        code_cell("df_sf = load_raw_city_data(city='San Francisco', raw_dir='data/raw')\nprint(f'Total observations: {len(df_sf):,}')\ndf_sf.head(10)"),
        code_cell("df_sf.describe()")
    ]
    create_notebook(nb1_cells, "notebooks/01_exploratory_data_analysis.ipynb")

    # Notebook 2: Manual Baseline
    nb2_cells = [
        md_cell("# Manual Preprocessing & Baseline Forecasting (Ad-Hoc Workflow)\n## Course: Data Science Tools and Techniques (DSTT)\n\nDemonstrates how weather forecasting models are typically prepared today: ad-hoc scripts, global mean/forward-fill, manual lags, and raw features."),
        code_cell("import numpy as np\nimport pandas as pd\nfrom weatherpipe.loader import load_raw_city_data\nfrom weatherpipe.manual_baseline import prepare_manual_baseline_a_naive, prepare_manual_baseline_b_analyst\nfrom weatherpipe.models import get_model\nfrom weatherpipe.evaluation import compute_regression_metrics, generate_temporal_folds"),
        md_cell("### 1. Load Raw San Francisco Weather Records"),
        code_cell("df_raw = load_raw_city_data('San Francisco', raw_dir='data/raw')\nprint('Raw shape:', df_raw.shape)"),
        md_cell("### 2. Prepare Manual Baselines\n- **Baseline A (Naive)**: Column mean fill, raw columns\n- **Baseline B (Analyst)**: Forward fill, calendar integers, 1 lag, 1 rolling mean"),
        code_cell("X_naive, y_naive = prepare_manual_baseline_a_naive(df_raw, target_horizon=6)\nX_analyst, y_analyst = prepare_manual_baseline_b_analyst(df_raw, target_horizon=6)\nprint('Analyst feature columns:', list(X_analyst.columns))"),
        md_cell("### 3. Evaluate Manual Baseline B on Temporal Fold"),
        code_cell("folds = generate_temporal_folds(len(X_analyst), n_folds=4)\ntrain_idx, test_idx = folds[-1]\n\nmodel_analyst = get_model('xgboost')\nmask_tr = y_analyst.iloc[train_idx].notna() & X_analyst.iloc[train_idx].notna().all(axis=1)\nmodel_analyst.fit(X_analyst.iloc[train_idx][mask_tr], y_analyst.iloc[train_idx][mask_tr])\n\npreds = model_analyst.predict(X_analyst.iloc[test_idx].fillna(0.0))\nmetrics = compute_regression_metrics(y_analyst.iloc[test_idx].values, preds)\nprint(f'Manual Baseline B (Analyst) Test RMSE: {metrics[\"rmse\"]:.2f}°C, MAE: {metrics[\"mae\"]:.2f}°C, R²: {metrics[\"r2\"]:.4f}')")
    ]
    create_notebook(nb2_cells, "notebooks/02_manual_baseline.ipynb")

    # Notebook 3: Automated Pipeline Comparison
    nb3_cells = [
        md_cell("# Automated CI/CD Weather Pipeline vs. Manual Baselines\n## Comparative Evaluation (RQ1 & RQ2)\n\nDemonstrates the end-to-end automated pipeline: alignment, cleaning, causal diurnal imputation, circular wind encodings, multi-scale lags, rolling statistics, physical interaction metrics, and strict temporal validation."),
        code_cell("import pandas as pd\nimport numpy as np\nfrom weatherpipe.experiment_rq1 import run_rq1_experiment\nfrom weatherpipe.experiment_rq2 import run_rq2_experiment\nfrom weatherpipe.accuracy_gate import evaluate_gate"),
        md_cell("### 1. Run Complete RQ1 Forecast Accuracy Benchmark across 4 Folds"),
        code_cell("res_rq1 = run_rq1_experiment(city='San Francisco', raw_dir='data/raw', n_folds=4, use_mlflow=False)\nres_rq1['summary']"),
        md_cell("### 2. Run RQ2 Missing Data Imputation Fidelity Study"),
        code_cell("res_rq2 = run_rq2_experiment(city='San Francisco', raw_dir='data/raw')\nres_rq2.head(10)"),
        md_cell("### 3. CI/CD Accuracy Gate Verification"),
        code_cell("gate_exit_code = evaluate_gate('results/rq1_forecast_accuracy.csv')\nprint(f'Gate Status Code: {gate_exit_code} (0 = Passed)')")
    ]
    create_notebook(nb3_cells, "notebooks/03_automated_pipeline_comparison.ipynb")


if __name__ == "__main__":
    main()
