"""
ml/evaluate.py
Loads scripts/data/housing-data.json and prints an evaluation report of test set performance.
"""

import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_evaluation_report(json_path: str = None):
    if json_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(base_dir, "scripts", "data", "housing-data.json")

    if not os.path.exists(json_path):
        print(f"Error: dataset file not found at {json_path}. Run 'python ml/train.py' first.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    models = data["trainedModels"]
    lin = models["linear"]
    log = models["logistic"]
    cm = log["testMetrics"]["confusionMatrix"]

    print("=" * 60)
    print("      PriceDrift -- Model Evaluation Report (Test Split)     ")
    print("=" * 60)
    print(f"Total Samples: {meta['totalSamples']:,} (Train: {meta['trainCount']:,}, Test: {meta['testCount']:,})")
    print(f"Fitted Training Median PricePerRoom: ${meta['fastSaleHeuristic']['medianPricePerRoom']:,.2f}")
    print(f"Fitted Training Median House Age:   {meta['fastSaleHeuristic']['medianHouseAge']:.2f} years")

    print("\n" + "-" * 60)
    print("1. Linear Regression (House Price Prediction)")
    print("-" * 60)
    print(f"  Test R^2 (Variance Explained): {lin['testMetrics']['r2']:.4f}")
    print(f"  Test RMSE:                     ${lin['testMetrics']['rmse']:,.2f}")
    print(f"  Test MSE:                      {lin['testMetrics']['mse']:,.2f}")
    print(f"  Learned Bias (Intercept):      {lin['bias']:.4f}")
    print("  Learned Weights (Standardized):")
    for name, w in zip(meta["featureNames"], lin["weights"]):
        print(f"    - {name:<30}: {w:+.4f}")

    print("\n" + "-" * 60)
    print("2. Logistic Regression (Fast-Sale Binary Classifier)")
    print("-" * 60)
    print(f"  Decision Threshold (theta):    {log['testMetrics']['threshold']:.2f}")
    print(f"  Test Accuracy:                 {log['testMetrics']['accuracy']*100:.2f}%")
    print(f"  Test Precision:                {log['testMetrics']['precision']*100:.2f}%")
    print(f"  Test Recall:                   {log['testMetrics']['recall']*100:.2f}%")
    print(f"  Test F1 Score:                 {log['testMetrics']['f1']:.4f}")
    print("  Confusion Matrix (1,000 test samples):")
    print(f"    TP: {cm['tp']:<5} | FN: {cm['fn']:<5}")
    print(f"    FP: {cm['fp']:<5} | TN: {cm['tn']:<5}")
    print(f"    Total Test Count: {cm['total']}")
    print("=" * 60)


if __name__ == "__main__":
    print_evaluation_report()
