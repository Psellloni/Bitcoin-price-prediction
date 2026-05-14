from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor

from src.models.base_model import BaseRegressionModel


class GradientBoostingModel(BaseRegressionModel):
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        super().__init__(model_name="GradientBoostingRegressor")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self