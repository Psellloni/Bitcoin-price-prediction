from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaseRegressionModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_name} is not fitted yet.")
        return self.model.predict(X)

    def evaluate(self, X, y) -> dict:
        y_pred = self.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return {
            "model": self.model_name,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance_df(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Return a sorted DataFrame of feature importances.

        Works for any sklearn model that exposes:
            - feature_importances_  (tree-based: RF, GB)
            - coef_                 (linear: ElasticNet, Ridge, Lasso)

        Parameters
        ----------
        feature_names : column names corresponding to the training features.
                        If None, generic names F0, F1, … are used.

        Returns
        -------
        pd.DataFrame with columns ['feature', 'importance'] sorted descending.
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_name} is not fitted yet.")

        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_).flatten()
        else:
            raise AttributeError(
                f"{self.model_name} does not expose feature_importances_ or coef_."
            )

        n = len(imp)
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(n)]

        df = pd.DataFrame({"feature": feature_names[:n], "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def plot_feature_importance(
        self,
        feature_names: list[str] | None = None,
        top_n: int = 15,
        title: str | None = None,
    ):
        """Horizontal bar chart of top-N feature importances."""
        df = self.feature_importance_df(feature_names).head(top_n)

        plt.figure(figsize=(8, max(4, top_n * 0.35)))
        plt.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
        plt.xlabel("Importance")
        plt.title(title or f"{self.model_name}: Feature Importances (top {top_n})")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_predictions(
        self,
        y_true,
        y_pred=None,
        title: str | None = None,
        n_points: int | None = 200,
    ):
        if y_pred is None:
            raise ValueError("y_pred must be provided.")

        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        if n_points is not None:
            y_true = y_true[:n_points]
            y_pred = y_pred[:n_points]

        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="True values")
        plt.plot(y_pred, label="Predictions")
        plt.title(title or f"{self.model_name}: True vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Target")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_residuals(self, X, y):
        y_pred = self.predict(X)
        residuals = np.asarray(y).reshape(-1) - np.asarray(y_pred).reshape(-1)

        plt.figure(figsize=(10, 4))
        plt.plot(residuals)
        plt.title(f"{self.model_name}: Residuals")
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.grid(True)
        plt.show()
