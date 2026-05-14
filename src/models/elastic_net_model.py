from __future__ import annotations

from sklearn.linear_model import ElasticNet

from src.models.base_model import BaseRegressionModel


class ElasticNetModel(BaseRegressionModel):
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        random_state: int = 42,
    ):
        super().__init__(model_name="ElasticNet")
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self