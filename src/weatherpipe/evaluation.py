"""Evaluation Harness: Temporal Cross-Validation, Metrics, and Statistical Confidence."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate RMSE, MAE, and R² score on valid (non-NaN) targets."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "count": len(yt),
    }


def generate_temporal_folds(
    n_samples: int,
    n_folds: int = 4,
    min_train_ratio: float = 0.40,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window chronological train/test indices.

    Guarantees strict temporal ordering: train indices < test indices.
    No future data is ever included in train partitions.
    """
    total_idx = np.arange(n_samples)
    min_train_size = int(n_samples * min_train_ratio)
    remaining = n_samples - min_train_size
    fold_test_size = remaining // n_folds

    folds = []
    for k in range(n_folds):
        split_point = min_train_size + k * fold_test_size
        end_point = split_point + fold_test_size if k < n_folds - 1 else n_samples

        train_indices = total_idx[:split_point]
        test_indices = total_idx[split_point:end_point]
        folds.append((train_indices, test_indices))

    return folds


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_pipeline: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Calculate the 95% bootstrap confidence interval for RMSE improvement (Baseline - Pipeline).

    A strictly positive confidence interval (lower bound > 0) proves that the
    accuracy gain is statistically significant.
    """
    rng = np.random.RandomState(random_state)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred_baseline) & ~np.isnan(y_pred_pipeline)
    yt = y_true[mask]
    yb = y_pred_baseline[mask]
    yp = y_pred_pipeline[mask]

    n = len(yt)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        rmse_b = np.sqrt(np.mean((yt[idx] - yb[idx]) ** 2))
        rmse_p = np.sqrt(np.mean((yt[idx] - yp[idx]) ** 2))
        diffs.append(rmse_b - rmse_p)

    diffs = np.array(diffs)
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(diffs, alpha * 100))
    high = float(np.percentile(diffs, (1.0 - alpha) * 100))
    mean_gain = float(np.mean(diffs))

    return {
        "mean_gain": round(mean_gain, 4),
        "ci_lower": round(low, 4),
        "ci_upper": round(high, 4),
        "statistically_significant": bool(low > 0.0),
    }
