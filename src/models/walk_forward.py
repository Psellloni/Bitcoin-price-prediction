from __future__ import annotations

from typing import Callable, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.metrics import directional_accuracy


def walk_forward_cv(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    min_train_frac: float = 0.5,
    verbose: bool = True,
    return_predictions: bool = False,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward cross-validation for time-series models.

    The dataset is divided into (n_splits + 1) contiguous blocks.
    The first block (min_train_frac of all data) forms the initial training
    set.  In each subsequent fold k the training set expands by one block and
    the following block is used as the test set.

    Parameters
    ----------
    model_factory : zero-argument callable that returns a fresh unfitted model
                    with .fit(X_train, y_train) and .predict(X_test) methods.
    X             : feature matrix, shape (n_samples, n_features)
    y             : target vector, shape (n_samples,)
    n_splits      : number of walk-forward folds
    min_train_frac: fraction of data used for the initial training window
    verbose       : print fold-level results

    Returns
    -------
    pd.DataFrame with columns [fold, train_size, test_size, MAE, RMSE, R2]
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)

    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = remaining // n_splits

    records = []
    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            test_end = n

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # LSTM returns fewer predictions due to seq_len offset
        reference = None
        if len(y_pred) < len(y_test):
            offset = len(y_test) - len(y_pred)
            if X_test.ndim == 2 and X_test.shape[1] > 0:
                reference = X_test[offset:, 0]
            y_test = y_test[len(y_test) - len(y_pred):]
        elif X_test.ndim == 2 and X_test.shape[1] > 0:
            reference = X_test[: len(y_pred), 0]

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        direction_metrics = (
            directional_accuracy(y_test, y_pred, reference)
            if reference is not None
            else {
                "Directional Accuracy": np.nan,
                "Precision Long": np.nan,
                "Recall Long": np.nan,
            }
        )

        record = {
            "fold": fold + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            **direction_metrics,
        }
        if return_predictions:
            record["y_true"] = np.asarray(y_test, dtype=float)
            record["y_pred"] = np.asarray(y_pred, dtype=float)
            if reference is not None:
                record["reference"] = np.asarray(reference, dtype=float)
        records.append(record)

        if verbose:
            print(
                f"Fold {fold + 1}/{n_splits} | "
                f"train={len(X_train):>5d}  test={len(X_test):>5d} | "
                f"MAE={mae:>8.2f}  RMSE={rmse:>8.2f}  R²={r2:.4f}  "
                f"DA={direction_metrics['Directional Accuracy']:.3f}"
            )

    return pd.DataFrame(records)


def walk_forward_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fold-level results: mean ± std across folds.

    Parameters
    ----------
    df : output of walk_forward_cv

    Returns
    -------
    One-row DataFrame with columns MAE_mean, MAE_std, RMSE_mean, RMSE_std,
    R2_mean, R2_std, DA_mean, PrecisionLong_mean, RecallLong_mean.
    """
    summary = {
        "MAE_mean": df["MAE"].mean(),
        "MAE_std": df["MAE"].std(),
        "RMSE_mean": df["RMSE"].mean(),
        "RMSE_std": df["RMSE"].std(),
        "R2_mean": df["R2"].mean(),
        "R2_std": df["R2"].std(),
        "DA_mean": df["Directional Accuracy"].mean(),
        "PrecisionLong_mean": df["Precision Long"].mean(),
        "RecallLong_mean": df["Recall Long"].mean(),
    }
    return pd.DataFrame([summary])


def walk_forward_naive_baseline(
    series: np.ndarray,
    horizon: int = 1,
    n_splits: int = 5,
    min_train_frac: float = 0.5,
) -> pd.DataFrame:
    series = np.asarray(series, dtype=float).reshape(-1)
    n = len(series)

    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = remaining // n_splits

    records = []
    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        test_series = series[test_start:test_end]
        if len(test_series) <= horizon:
            continue

        y_true = test_series[horizon:]
        y_pred = test_series[:-horizon]
        reference = test_series[:-horizon]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        direction_metrics = directional_accuracy(y_true, y_pred, reference)

        records.append(
            {
                "fold": fold + 1,
                "train_size": train_end,
                "test_size": len(test_series),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                **direction_metrics,
                "y_true": y_true,
                "y_pred": y_pred,
                "reference": reference,
            }
        )

    return pd.DataFrame(records)


def walk_forward_price_model(
    model_factory: Callable[[], Any],
    prices: np.ndarray,
    horizon: int = 10,
    n_splits: int = 5,
    min_train_frac: float = 0.5,
    dt: float = 1.0,
    verbose: bool = True,
    return_predictions: bool = False,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation for models that train directly on
    a 1D price series, such as Hybrid SDE-ML models.

    Expected interface:
        model.fit(train_prices, dt=dt)
        model.predict_point(eval_prices, horizon=horizon, dt=dt)
    """
    prices = np.asarray(prices, dtype=float).reshape(-1)
    n = len(prices)

    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = remaining // n_splits

    records = []
    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        train_prices = prices[:train_end]
        eval_prices = prices[test_start:test_end]
        if len(eval_prices) <= horizon:
            continue

        model = model_factory()
        model.fit(train_prices, dt=dt)

        if hasattr(model, "predict_point"):
            y_pred = model.predict_point(eval_prices, horizon=horizon, dt=dt)
            lag = getattr(model, "n_lag_returns", 0)
            y_true = eval_prices[lag + horizon : lag + horizon + len(y_pred)]
            reference = eval_prices[lag : lag + len(y_pred)]
        else:
            raise AttributeError("price model must expose predict_point(...) for walk-forward evaluation")

        if len(y_true) == 0 or len(y_pred) == 0:
            continue

        n_common = min(len(y_true), len(y_pred), len(reference))
        y_true = y_true[:n_common]
        y_pred = y_pred[:n_common]
        reference = reference[:n_common]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        direction_metrics = directional_accuracy(y_true, y_pred, reference)

        record = {
            "fold": fold + 1,
            "train_size": len(train_prices),
            "test_size": len(eval_prices),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            **direction_metrics,
        }
        if return_predictions:
            record["y_true"] = np.asarray(y_true, dtype=float)
            record["y_pred"] = np.asarray(y_pred, dtype=float)
            record["reference"] = np.asarray(reference, dtype=float)
        records.append(record)

        if verbose:
            print(
                f"Fold {fold + 1}/{n_splits} | "
                f"train={len(train_prices):>5d}  test={len(eval_prices):>5d} | "
                f"MAE={mae:>8.2f}  RMSE={rmse:>8.2f}  R²={r2:.4f}  "
                f"DA={direction_metrics['Directional Accuracy']:.3f}"
            )

    return pd.DataFrame(records)
