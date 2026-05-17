from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def metrics_frame(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "R2"])
    return pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)


def naive_last_value_forecast(series: np.ndarray, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=float).reshape(-1)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(series) <= horizon:
        raise ValueError("series is too short for the requested horizon")

    y_pred = series[:-horizon]
    y_true = series[horizon:]
    return y_true, y_pred


def naive_metrics(series: np.ndarray, horizon: int = 1, model_name: str = "Naive (last price)") -> dict[str, float | str]:
    y_true, y_pred = naive_last_value_forecast(series, horizon=horizon)
    return {"model": model_name, **regression_metrics(y_true, y_pred)}


def paired_t_test_errors(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    loss: str = "absolute",
    model_a: str = "model_a",
    model_b: str = "model_b",
) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_a = np.asarray(y_pred_a, dtype=float).reshape(-1)
    y_pred_b = np.asarray(y_pred_b, dtype=float).reshape(-1)

    n = min(len(y_true), len(y_pred_a), len(y_pred_b))
    if n == 0:
        raise ValueError("inputs must be non-empty")

    y_true = y_true[:n]
    y_pred_a = y_pred_a[:n]
    y_pred_b = y_pred_b[:n]

    if loss == "absolute":
        err_a = np.abs(y_true - y_pred_a)
        err_b = np.abs(y_true - y_pred_b)
    elif loss == "squared":
        err_a = (y_true - y_pred_a) ** 2
        err_b = (y_true - y_pred_b) ** 2
    else:
        raise ValueError("loss must be 'absolute' or 'squared'")

    stat, p_value = ttest_rel(err_a, err_b)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "loss": loss,
        "mean_loss_a": float(err_a.mean()),
        "mean_loss_b": float(err_b.mean()),
        "t_stat": float(stat),
        "p_value": float(p_value),
    }
