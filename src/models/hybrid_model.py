from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HybridSDEMLModel:
    """
    SDE-ML Hybrid Model with time-varying ML-predicted drift (novelty element).

    Core idea
    ---------
    Classical GBM uses constant drift mu and volatility sigma.  This model
    replaces the constant drift with a time-varying mu_t predicted by a
    Gradient Boosting regressor at each time step, while keeping the stochastic
    diffusion term from GBM:

        S_{t+h} = S_t * exp( mu_hat_t * h*dt  +  sigma_t * sqrt(h*dt) * Z )

    where
        mu_hat_t = f_ML(lagged log-returns, realised vol, RSI momentum proxy)
        sigma_t  = rolling realised volatility

    For a point prediction (Z = 0):
        S_hat_{t+h} = S_t * exp( mu_hat_t * h * dt )

    This architecture inherits distributional forecasting from the SDE
    framework while adopting the predictive accuracy of ML, constituting the
    primary novel contribution of the thesis.

    Parameters
    ----------
    rv_window        : rolling window for realised-volatility estimation (bars)
    n_lag_returns    : lagged log-returns as ML features
    gb_n_estimators  : Gradient Boosting trees
    gb_learning_rate : GB shrinkage parameter
    gb_max_depth     : GB tree depth
    random_state     : reproducibility seed
    """

    def __init__(
        self,
        rv_window: int = 20,
        n_lag_returns: int = 10,
        gb_n_estimators: int = 200,
        gb_learning_rate: float = 0.05,
        gb_max_depth: int = 3,
        random_state: int = 42,
    ):
        self.model_name = "SDE-ML Hybrid"
        self.rv_window = rv_window
        self.n_lag_returns = n_lag_returns
        self.random_state = random_state

        self._drift_model = GradientBoostingRegressor(
            n_estimators=gb_n_estimators,
            learning_rate=gb_learning_rate,
            max_depth=gb_max_depth,
            random_state=random_state,
        )
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _compute_rv(self, log_returns: np.ndarray, dt: float) -> np.ndarray:
        return (
            pd.Series(log_returns)
            .rolling(self.rv_window, min_periods=self.rv_window)
            .std()
            .fillna(method="bfill")
            .values
        ) / math.sqrt(dt)

    def _build_features(
        self, log_returns: np.ndarray, rv: np.ndarray
    ) -> np.ndarray:
        """
        Feature matrix for the ML drift predictor.

        Columns: lag_1 ... lag_n  (lagged log-returns, most-recent first)
                 realised_vol     (current rolling RV)
                 rsi_14           (fraction of positive returns over 14 bars)
        """
        lag = self.n_lag_returns
        rows = []
        for i in range(lag, len(log_returns)):
            lags = log_returns[i - lag : i][::-1]
            rv_val = float(rv[i]) if i < len(rv) else float(rv[-1])
            window_14 = log_returns[max(0, i - 14) : i]
            rsi = float(np.sum(window_14 > 0)) / max(len(window_14), 1)
            rows.append(np.concatenate([lags, [rv_val, rsi]]))
        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, prices: np.ndarray, dt: float = 1.0) -> "HybridSDEMLModel":
        """Calibrate the hybrid model on a historical price series."""
        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        self._dt = dt

        rv = self._compute_rv(log_returns, dt)
        self._rv_last = rv

        lag = self.n_lag_returns
        X = self._build_features(log_returns, rv)   # (n-lag, p)
        y = log_returns[lag:]                        # predict next bar log-return

        self._drift_model.fit(X, y)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_point(
        self,
        prices: np.ndarray,
        horizon: int = 10,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Rolling point forecast (Z = 0 deterministic path).

            S_hat_{t+h} ≈ S_t * exp( mu_hat_t * h * dt )

        Returns array of length len(prices) - n_lag_returns - horizon - 1.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        rv = self._compute_rv(log_returns, dt)

        lag = self.n_lag_returns
        X_all = self._build_features(log_returns, rv)
        mu_preds = self._drift_model.predict(X_all)

        forecasts = []
        for i in range(len(mu_preds) - horizon):
            S_t = prices[lag + i]
            mu_t = float(mu_preds[i])
            forecasts.append(S_t * math.exp(mu_t * horizon * dt))

        return np.array(forecasts)

    def predict_distribution(
        self,
        prices: np.ndarray,
        horizon: int = 10,
        n_paths: int = 1000,
        dt: float = 1.0,
        quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> dict[str, np.ndarray]:
        """
        Distributional forecast: for each step draw n_paths GBM paths using
        ML-predicted drift and realised volatility, then return quantiles.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        prices = np.asarray(prices, dtype=float)
        log_returns = np.diff(np.log(prices))
        rv = self._compute_rv(log_returns, dt)

        lag = self.n_lag_returns
        X_all = self._build_features(log_returns, rv)
        mu_preds = self._drift_model.predict(X_all)

        rng = np.random.default_rng(self.random_state)
        results: dict[str, list] = {str(q): [] for q in quantiles}

        for i in range(len(mu_preds) - horizon):
            S_t = prices[lag + i]
            mu_t = float(mu_preds[i])
            sigma_t = float(rv[lag + i]) if (lag + i) < len(rv) else float(rv[-1])

            Z = rng.standard_normal((n_paths, horizon))
            step_log = (mu_t - 0.5 * sigma_t ** 2) * dt + sigma_t * math.sqrt(dt) * Z
            S_T = S_t * np.exp(step_log.sum(axis=1))

            for q in quantiles:
                results[str(q)].append(float(np.quantile(S_T, q)))

        return {k: np.array(v) for k, v in results.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        prices: np.ndarray,
        horizon: int = 10,
        dt: float = 1.0,
    ) -> dict:
        """Point-prediction accuracy on a price series."""
        y_pred = self.predict_point(prices, horizon, dt)
        lag = self.n_lag_returns
        y_true = prices[lag + horizon : lag + horizon + len(y_pred)]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {"model": self.model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self._drift_model.feature_importances_

    def feature_importance_df(self) -> pd.DataFrame:
        lag = self.n_lag_returns
        names = [f"lag_{i+1}" for i in range(lag)] + ["realised_vol", "rsi_14"]
        df = pd.DataFrame({"feature": names, "importance": self.feature_importances_})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_forecast(
        self,
        prices: np.ndarray,
        horizon: int = 10,
        dt: float = 1.0,
        n_points: int = 200,
    ):
        y_pred = self.predict_point(prices, horizon, dt)
        lag = self.n_lag_returns
        y_true = prices[lag + horizon : lag + horizon + len(y_pred)]

        if n_points:
            y_pred = y_pred[:n_points]
            y_true = y_true[:n_points]

        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="Actual", linewidth=1.5)
        plt.plot(y_pred, label="Hybrid (point)", linewidth=1.5, linestyle="--")
        plt.title(f"{self.model_name}: Rolling {horizon}-step Forecast")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
