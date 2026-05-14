from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseRegressionModel


class RandomForestModel(BaseRegressionModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(model_name="RandomForestRegressor")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self