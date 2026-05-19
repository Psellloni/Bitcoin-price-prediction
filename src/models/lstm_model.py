from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LSTMModel:
    """
    LSTM model for time-series price prediction.

    Accepts 2D tabular input (n_samples, n_features) and internally
    converts it to 3D sequences (n_samples - seq_len, seq_len, n_features)
    required by Keras LSTM layers.

    Parameters
    ----------
    seq_len       : lookback window (number of time steps per sample)
    lstm_units    : number of LSTM hidden units
    dropout_rate  : dropout fraction after LSTM layer
    learning_rate : Adam optimiser learning rate
    epochs        : maximum training epochs (early stopping may reduce this)
    batch_size    : mini-batch size
    """

    def __init__(
        self,
        seq_len: int = 20,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 30,
        batch_size: int = 32,
        random_state: int = 42,
        validation_split: float = 0.1,
        verbose: int = 1,
    ):
        self.model_name = "LSTM"
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.validation_split = validation_split
        self.verbose = verbose

        self.is_fitted = False
        self.history = None
        self.n_features: int | None = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None  # built lazily after n_features is known

    def _set_random_seed(self):
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------

    @staticmethod
    def create_sequences(
        X: np.ndarray, y: np.ndarray, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D arrays to 3D LSTM-compatible sequences.

        Parameters
        ----------
        X       : (n_samples, n_features)
        y       : (n_samples,)
        seq_len : lookback window

        Returns
        -------
        X_seq : (n_samples - seq_len, seq_len, n_features)
        y_seq : (n_samples - seq_len,)
        """
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, n_features: int) -> Sequential:
        model = Sequential(
            [
                Input(shape=(self.seq_len, n_features)),
                LSTM(self.lstm_units),
                Dropout(self.dropout_rate),
                Dense(1),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LSTMModel":
        self._set_random_seed()

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

        self.n_features = X_train.shape[1]

        # Scale features and target to [0, 1] for stable LSTM training
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train).reshape(-1)

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.seq_len)

        self.model = self._build_model(self.n_features)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            X_val_sc = self.scaler_X.transform(X_val)
            y_val_sc = self.scaler_y.transform(y_val).reshape(-1)
            X_val_seq, y_val_seq = self.create_sequences(
                X_val_sc, y_val_sc, self.seq_len
            )
            validation_data = (X_val_seq, y_val_seq)
        elif self.validation_split > 0.0:
            split_idx = int(len(X_seq) * (1.0 - self.validation_split))
            split_idx = max(split_idx, 1)
            split_idx = min(split_idx, len(X_seq) - 1) if len(X_seq) > 1 else len(X_seq)
            if 0 < split_idx < len(X_seq):
                validation_data = (X_seq[split_idx:], y_seq[split_idx:])
                X_seq = X_seq[:split_idx]
                y_seq = y_seq[:split_idx]

        self.history = self.model.fit(
            X_seq,
            y_seq,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks,
            shuffle=False,
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict prices for all positions in X that have a full lookback window.

        Returns array of length max(0, len(X) - seq_len).
        """
        if not self.is_fitted:
            raise ValueError("LSTM model is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_X.transform(X)

        # Dummy y for sequence creation (values not used in prediction)
        dummy_y = np.zeros(len(X_scaled))
        X_seq, _ = self.create_sequences(X_scaled, dummy_y, self.seq_len)

        y_scaled_pred = self.model.predict(X_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred).reshape(-1)
        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate on test data.  Aligns y to the seq_len offset automatically.
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=np.float32)[self.seq_len :]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {
            "model": self.model_name,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray | None = None,
        title: str | None = None,
        n_points: int = 200,
    ):
        if y_pred is None:
            raise ValueError("y_pred must be provided.")

        y_true = np.asarray(y_true).reshape(-1)[self.seq_len :]
        y_pred = np.asarray(y_pred).reshape(-1)

        if n_points:
            y_true = y_true[:n_points]
            y_pred = y_pred[:n_points]

        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="True values")
        plt.plot(y_pred, label="Predictions")
        plt.title(title or "LSTM: True vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self):
        if self.history is None:
            raise ValueError("No training history. Fit the model first.")

        plt.figure(figsize=(10, 4))
        plt.plot(self.history.history["loss"], label="Train loss")
        if "val_loss" in self.history.history:
            plt.plot(self.history.history["val_loss"], label="Validation loss")
        plt.title("LSTM Learning Curves")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
