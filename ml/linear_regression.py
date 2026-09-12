"""
ml/linear_regression.py
From-scratch Linear Regression using vectorized Batch Gradient Descent with NumPy.
Supports L2 Regularization (Ridge) and divergence guards.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class LinearRegression:
    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 200,
        use_regularization: bool = False,
        lambda_reg: float = 0.0
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_regularization = use_regularization
        self.lambda_reg = lambda_reg

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.history: List[Dict[str, Any]] = []
        self.is_diverged: bool = False

    def initialize_weights(self, num_features: int) -> None:
        self.weights = np.zeros(num_features, dtype=np.float64)
        self.bias = 0.0
        self.history = []
        self.is_diverged = False

    def predict_norm(self, X: np.ndarray) -> np.ndarray:
        """Predicts normalized target: y_hat = X @ w + b."""
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        return X @ self.weights + self.bias

    def compute_cost(self, y_hat: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes Mean Squared Error with optional L2 regularization on weights.
        J(w, b) = (1 / 2m) * sum((y_hat - y)^2) + (lambda / 2m) * sum(w^2)
        """
        m = len(y_true)
        if m == 0:
            return 0.0

        errors = y_hat - y_true
        cost = np.sum(errors ** 2) / (2.0 * m)

        if self.use_regularization and self.lambda_reg > 0:
            l2_penalty = (self.lambda_reg / (2.0 * m)) * np.sum(self.weights ** 2)
            cost += l2_penalty

        return float(cost)

    def train_step(self, X: np.ndarray, y: np.ndarray, epoch_num: int) -> Dict[str, Any]:
        """Performs a single vectorized batch gradient descent update step."""
        m, d = X.shape
        if self.weights is None or len(self.weights) != d:
            self.initialize_weights(d)

        y_hat = self.predict_norm(X)
        cost = self.compute_cost(y_hat, y)

        if np.isnan(cost) or np.isinf(cost) or cost > 1e10:
            self.is_diverged = True
            return {"epoch": epoch_num, "cost": float("nan"), "diverged": True}

        errors = y_hat - y

        # Vectorized gradient calculation: (1 / m) * (X^T @ errors) + (lambda / m) * w
        grad_w = (X.T @ errors) / m
        if self.use_regularization and self.lambda_reg > 0:
            grad_w += (self.lambda_reg / m) * self.weights

        grad_b = float(np.sum(errors) / m)

        # Gradient descent update
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

        step_record = {"epoch": epoch_num, "cost": float(cost), "diverged": False}
        self.history.append(step_record)
        return step_record

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Trains the model across all configured epochs."""
        m, d = X.shape
        self.initialize_weights(d)

        for epoch in range(1, self.epochs + 1):
            step = self.train_step(X, y, epoch)
            if step["diverged"]:
                break

        return self

    def predict(self, X: np.ndarray, price_stats: Dict[str, float]) -> np.ndarray:
        """Predicts in raw USD using the training price mean and standard deviation."""
        y_norm = self.predict_norm(X)
        return y_norm * price_stats["stdDev"] + price_stats["mean"]

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test_norm: np.ndarray,
        y_test_raw: np.ndarray,
        price_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Computes R², MSE, and RMSE on unseen test partition."""
        y_pred_norm = self.predict_norm(X_test)
        y_pred_raw = y_pred_norm * price_stats["stdDev"] + price_stats["mean"]

        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y_test_raw - y_pred_raw) ** 2)
        ss_tot = np.sum((y_test_raw - np.mean(y_test_raw)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        mse_raw = float(np.mean((y_test_raw - y_pred_raw) ** 2))
        rmse_raw = float(np.sqrt(mse_raw))
        mse_norm = float(np.mean((y_test_norm - y_pred_norm) ** 2))

        return {
            "r2": r2,
            "mse": mse_raw,
            "rmse": rmse_raw,
            "normalizedMSE": mse_norm,
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "history": self.history
        }
