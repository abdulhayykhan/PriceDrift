"""
ml/tests/test_models.py
Pytest suite verifying data pipeline leakage fixes, scratch math, and model convergence.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.data_pipeline import run_pipeline
from ml.linear_regression import LinearRegression
from ml.logistic_regression import LogisticRegression


def test_sigmoid_numerical_stability():
    """Verifies sigmoid computation and clamping behavior."""
    assert np.isclose(LogisticRegression.sigmoid(np.array([0.0]))[0], 0.5)
    large_pos = LogisticRegression.sigmoid(np.array([50.0]))[0]
    large_neg = LogisticRegression.sigmoid(np.array([-50.0]))[0]
    assert large_pos > 0.999999
    assert large_neg < 0.000001
    assert not np.isnan(large_pos)
    assert not np.isnan(large_neg)


def test_data_pipeline_no_leakage():
    """Verifies that statistics are computed on training set only and applied consistently."""
    data = run_pipeline()
    meta = data["metadata"]

    assert meta["trainCount"] == 4000
    assert meta["testCount"] == 1000
    assert data["X_train"].shape == (4000, 5)
    assert data["X_test"].shape == (1000, 5)

    # Check that X_train is standardized with near 0 mean and 1 std
    train_means = np.mean(data["X_train"], axis=0)
    train_stds = np.std(data["X_train"], axis=0)
    assert np.allclose(train_means, 0.0, atol=1e-5)
    assert np.allclose(train_stds, 1.0, atol=1e-5)

    # Test set transformed by train stats should NOT have exact 0 mean or 1 std
    test_means = np.mean(data["X_test"], axis=0)
    assert not np.allclose(test_means, 0.0, atol=1e-7)

    # FastSale class presence in both splits
    assert np.sum(data["y_fast_train"] == 1) > 0
    assert np.sum(data["y_fast_train"] == 0) > 0
    assert np.sum(data["y_fast_test"] == 1) > 0
    assert np.sum(data["y_fast_test"] == 0) > 0


def test_linear_regression_convergence():
    """Tests linear regression convergence, decreasing cost, and R² performance."""
    data = run_pipeline()
    meta = data["metadata"]

    lin = LinearRegression(learning_rate=0.05, epochs=200, use_regularization=False)
    lin.fit(data["X_train"], data["y_price_train_norm"])

    assert not lin.is_diverged, "Linear regression diverged!"
    assert len(lin.history) == 200

    # Verify cost decrease across epochs
    costs = [h["cost"] for h in lin.history]
    decreases = sum(1 for i in range(1, len(costs)) if costs[i] < costs[i - 1])
    assert decreases > 190, f"Cost did not decrease consistently: {decreases}/199 steps"

    eval_res = lin.evaluate(
        data["X_test"],
        data["y_price_test_norm"],
        data["y_price_test"],
        meta["priceStats"]
    )

    assert eval_res["r2"] > 0.85, f"R² below threshold: {eval_res['r2']:.4f}"
    assert eval_res["rmse"] < 150000, f"RMSE too high: ${eval_res['rmse']:,.0f}"


def test_linear_regression_regularization_shrinkage():
    """Verifies that L2 regularization shrinks weight magnitude."""
    data = run_pipeline()

    lin_vanilla = LinearRegression(learning_rate=0.05, epochs=200, use_regularization=False)
    lin_vanilla.fit(data["X_train"], data["y_price_train_norm"])

    lin_ridge = LinearRegression(learning_rate=0.05, epochs=200, use_regularization=True, lambda_reg=50.0)
    lin_ridge.fit(data["X_train"], data["y_price_train_norm"])

    norm_vanilla = np.sum(lin_vanilla.weights ** 2)
    norm_ridge = np.sum(lin_ridge.weights ** 2)
    assert norm_ridge < norm_vanilla, f"Ridge did not shrink weights: {norm_ridge} >= {norm_vanilla}"


def test_logistic_regression_convergence():
    """Tests logistic regression convergence, confusion matrix, and accuracy."""
    data = run_pipeline()

    log_reg = LogisticRegression(learning_rate=0.1, epochs=250, use_regularization=False, threshold=0.5)
    log_reg.fit(data["X_train"], data["y_fast_train"])

    assert not log_reg.is_diverged, "Logistic regression diverged!"
    assert len(log_reg.history) == 250

    costs = [h["cost"] for h in log_reg.history]
    decreases = sum(1 for i in range(1, len(costs)) if costs[i] < costs[i - 1])
    assert decreases > 240, f"BCE cost did not decrease consistently: {decreases}/249 steps"

    eval_res = log_reg.evaluate(data["X_test"], data["y_fast_test"], threshold=0.5)
    cm = eval_res["confusionMatrix"]

    assert eval_res["accuracy"] > 0.70, f"Accuracy too low: {eval_res['accuracy']*100:.2f}%"
    assert cm["total"] == 1000
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 1000
    # Non-degenerate predictions
    assert cm["tp"] > 0 and cm["tn"] > 0


def test_logistic_threshold_tradeoff():
    """Verifies precision-recall sensitivity across decision thresholds."""
    data = run_pipeline()

    log_reg = LogisticRegression(learning_rate=0.1, epochs=250, use_regularization=False, threshold=0.5)
    log_reg.fit(data["X_train"], data["y_fast_train"])

    low_eval = log_reg.evaluate(data["X_test"], data["y_fast_test"], threshold=0.2)
    high_eval = log_reg.evaluate(data["X_test"], data["y_fast_test"], threshold=0.8)

    assert low_eval["recall"] >= high_eval["recall"], "Lower threshold should yield higher/equal recall"
    assert high_eval["precision"] >= low_eval["precision"], "Higher threshold should yield higher/equal precision"
