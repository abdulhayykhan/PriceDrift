"""
ml/train.py
Orchestrates data pipeline, trains scratch Linear and Logistic regression models (both baseline
and L2-regularized variants), evaluates on test set, and exports scripts/data/housing-data.json.
"""

import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_pipeline import run_pipeline
from ml.linear_regression import LinearRegression
from ml.logistic_regression import LogisticRegression


def train_and_export(output_path: str = None) -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_path is None:
        output_path = os.path.join(base_dir, "scripts", "data", "housing-data.json")

    print("Executing ML data pipeline (leakage-free)...")
    data = run_pipeline()
    meta = data["metadata"]

    X_train = data["X_train"]
    y_price_train_norm = data["y_price_train_norm"]
    y_fast_train = data["y_fast_train"]

    X_test = data["X_test"]
    y_price_test = data["y_price_test"]
    y_price_test_norm = data["y_price_test_norm"]
    y_fast_test = data["y_fast_test"]

    print(f"Dataset split: {len(X_train)} train, {len(X_test)} test samples.")

    # 1. Train Baseline Linear Regression
    print("\n[1/4] Training Baseline Linear Regression (lr=0.05, epochs=200)...")
    lin_base = LinearRegression(learning_rate=0.05, epochs=200, use_regularization=False)
    lin_base.fit(X_train, y_price_train_norm)
    lin_eval = lin_base.evaluate(X_test, y_price_test_norm, y_price_test, meta["priceStats"])
    print(f"  -> Baseline Linear R²: {lin_eval['r2']:.4f}, RMSE: ${lin_eval['rmse']:,.0f}")

    # 2. Train Regularized Linear Regression (Ridge lambda=50)
    print("\n[2/4] Training Regularized Linear Regression (lr=0.05, epochs=200, lambda=50.0)...")
    lin_reg = LinearRegression(learning_rate=0.05, epochs=200, use_regularization=True, lambda_reg=50.0)
    lin_reg.fit(X_train, y_price_train_norm)
    lin_reg_eval = lin_reg.evaluate(X_test, y_price_test_norm, y_price_test, meta["priceStats"])
    print(f"  -> Ridge Linear R²: {lin_reg_eval['r2']:.4f}, RMSE: ${lin_reg_eval['rmse']:,.0f}")

    # 3. Train Baseline Logistic Regression
    print("\n[3/4] Training Baseline Logistic Regression (lr=0.1, epochs=250, threshold=0.5)...")
    log_base = LogisticRegression(learning_rate=0.1, epochs=250, use_regularization=False, threshold=0.5)
    log_base.fit(X_train, y_fast_train)
    log_eval = log_base.evaluate(X_test, y_fast_test, threshold=0.5)
    print(f"  -> Baseline Logistic Accuracy: {log_eval['accuracy']*100:.2f}%, F1: {log_eval['f1']:.3f}")

    # 4. Train Regularized Logistic Regression (lambda=10.0)
    print("\n[4/4] Training Regularized Logistic Regression (lr=0.1, epochs=250, lambda=10.0)...")
    log_reg = LogisticRegression(learning_rate=0.1, epochs=250, use_regularization=True, lambda_reg=10.0, threshold=0.5)
    log_reg.fit(X_train, y_fast_train)
    log_reg_eval = log_reg.evaluate(X_test, y_fast_test, threshold=0.5)
    print(f"  -> Regularized Logistic Accuracy: {log_reg_eval['accuracy']*100:.2f}%, F1: {log_reg_eval['f1']:.3f}")

    payload = {
        "metadata": meta,
        "train": {
            "X": X_train.tolist(),
            "X_raw": data["X_train_raw"].tolist(),
            "y_price": data["y_price_train"].tolist(),
            "y_price_norm": y_price_train_norm.tolist(),
            "y_fast": y_fast_train.tolist()
        },
        "test": {
            "X": X_test.tolist(),
            "X_raw": data["X_test_raw"].tolist(),
            "y_price": y_price_test.tolist(),
            "y_price_norm": y_price_test_norm.tolist(),
            "y_fast": y_fast_test.tolist()
        },
        "trainedModels": {
            "linear": {
                "weights": lin_eval["weights"],
                "bias": lin_eval["bias"],
                "history": lin_eval["history"],
                "testMetrics": {
                    "r2": lin_eval["r2"],
                    "rmse": lin_eval["rmse"],
                    "mse": lin_eval["mse"],
                    "normalizedMSE": lin_eval["normalizedMSE"]
                }
            },
            "linear_regularized": {
                "weights": lin_reg_eval["weights"],
                "bias": lin_reg_eval["bias"],
                "history": lin_reg_eval["history"],
                "testMetrics": {
                    "r2": lin_reg_eval["r2"],
                    "rmse": lin_reg_eval["rmse"],
                    "mse": lin_reg_eval["mse"],
                    "normalizedMSE": lin_reg_eval["normalizedMSE"]
                }
            },
            "logistic": {
                "weights": log_eval["weights"],
                "bias": log_eval["bias"],
                "history": log_eval["history"],
                "testMetrics": {
                    "accuracy": log_eval["accuracy"],
                    "precision": log_eval["precision"],
                    "recall": log_eval["recall"],
                    "f1": log_eval["f1"],
                    "threshold": log_eval["threshold"],
                    "confusionMatrix": log_eval["confusionMatrix"]
                }
            },
            "logistic_regularized": {
                "weights": log_reg_eval["weights"],
                "bias": log_reg_eval["bias"],
                "history": log_reg_eval["history"],
                "testMetrics": {
                    "accuracy": log_reg_eval["accuracy"],
                    "precision": log_reg_eval["precision"],
                    "recall": log_reg_eval["recall"],
                    "f1": log_reg_eval["f1"],
                    "threshold": log_reg_eval["threshold"],
                    "confusionMatrix": log_reg_eval["confusionMatrix"]
                }
            }
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"\nSuccessfully exported trained models & dataset JSON to:\n  {output_path} ({file_size_kb:.1f} KB)")
    return payload


if __name__ == "__main__":
    train_and_export()
