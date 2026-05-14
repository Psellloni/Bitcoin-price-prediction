from __future__ import annotations

from typing import Callable, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def walk_forward_cv(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    min_train_frac: float = 0.5,
    verbose: bool = True,
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
        if len(y_pred) < len(y_test):
            y_test = y_test[len(y_test) - len(y_pred):]

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        record = {
            "fold": fold + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
        records.append(record)

        if verbose:
            print(
                f"Fold {fold + 1}/{n_splits} | "
                f"train={len(X_train):>5d}  test={len(X_test):>5d} | "
                f"MAE={mae:>8.2f}  RMSE={rmse:>8.2f}  R²={r2:.4f}"
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
    R2_mean, R2_std.
    """
    summary = {
        "MAE_mean": df["MAE"].mean(),
        "MAE_std": df["MAE"].std(),
        "RMSE_mean": df["RMSE"].mean(),
        "RMSE_std": df["RMSE"].std(),
        "R2_mean": df["R2"].mean(),
        "R2_std": df["R2"].std(),
    }
    return pd.DataFrame([summary])
