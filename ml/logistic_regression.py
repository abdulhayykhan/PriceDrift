"""
ml/logistic_regression.py
From-scratch Logistic Regression using vectorized Batch Gradient Descent with NumPy.
Supports L2 Regularization, dynamic decision thresholds, and complete classification metrics.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class LogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 250,
        use_regularization: bool = False,
        lambda_reg: float = 0.0,
        threshold: float = 0.5
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_regularization = use_regularization
        self.lambda_reg = lambda_reg
        self.threshold = threshold

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.history: List[Dict[str, Any]] = []
        self.is_diverged: bool = False

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function using clipping."""
        clamped_z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clamped_z))

    def initialize_weights(self, num_features: int) -> None:
        self.weights = np.zeros(num_features, dtype=np.float64)
        self.bias = 0.0
        self.history = []
        self.is_diverged = False

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes probabilities: p = sigmoid(X @ w + b)."""
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict_classes(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        th = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(np.int32)

    def compute_cost(self, y_probs: np.ndarray, y_true: np.ndarray) -> float:
        """
        Binary Cross Entropy loss:
        J(w, b) = -1/m * sum(y*ln(p) + (1-y)*ln(1-p)) + (lambda / 2m) * sum(w^2)
        """
        m = len(y_true)
        if m == 0:
            return 0.0

        eps = 1e-15
        p_safe = np.clip(y_probs, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(p_safe) + (1.0 - y_true) * np.log(1.0 - p_safe))

        if self.use_regularization and self.lambda_reg > 0:
            l2 = (self.lambda_reg / (2.0 * m)) * np.sum(self.weights ** 2)
            loss += l2

        return float(loss)

    def train_step(self, X: np.ndarray, y: np.ndarray, epoch_num: int) -> Dict[str, Any]:
        """Performs a single vectorized batch gradient descent update."""
        m, d = X.shape
        if self.weights is None or len(self.weights) != d:
            self.initialize_weights(d)

        y_probs = self.predict_proba(X)
        cost = self.compute_cost(y_probs, y)

        if np.isnan(cost) or np.isinf(cost) or cost > 1e6:
            self.is_diverged = True
            return {"epoch": epoch_num, "cost": float("nan"), "diverged": True}

        errors = y_probs - y

        # Vectorized gradient calculation: (1 / m) * (X^T @ errors) + (lambda / m) * w
        grad_w = (X.T @ errors) / m
        if self.use_regularization and self.lambda_reg > 0:
            grad_w += (self.lambda_reg / m) * self.weights

        grad_b = float(np.sum(errors) / m)

        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

        step_record = {"epoch": epoch_num, "cost": float(cost), "diverged": False}
        self.history.append(step_record)
        return step_record

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Trains the model across all configured epochs."""
        m, d = X.shape
        self.initialize_weights(d)

        for epoch in range(1, self.epochs + 1):
            step = self.train_step(X, y, epoch)
            if step["diverged"]:
                break

        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Computes confusion matrix, accuracy, precision, recall, and F1 score."""
        th = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X_test)
        preds = (probs >= th).astype(np.int32)

        tp = int(np.sum((y_test == 1) & (preds == 1)))
        fp = int(np.sum((y_test == 0) & (preds == 1)))
        tn = int(np.sum((y_test == 0) & (preds == 0)))
        fn = int(np.sum((y_test == 1) & (preds == 0)))
        total = tp + fp + tn + fn

        accuracy = float((tp + tn) / total) if total > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "threshold": float(th),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusionMatrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "total": total
            },
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "history": self.history
        }
