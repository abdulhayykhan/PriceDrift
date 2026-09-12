"""
ml/data_pipeline.py
Data ingestion, train/test splitting, training-only statistical fitting (data leakage fix),
heuristic FastSale label engineering, and z-score standardization for PriceDrift.
"""

import csv
import os
from typing import Dict, Any, Tuple
import numpy as np


def load_raw_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Loads USA_Housing.csv and extracts numeric features and price."""
    feature_names = [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Avg. Area Number of Bedrooms",
        "Area Population"
    ]

    features = []
    prices = []

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat_row = [
                float(row["Avg. Area Income"]),
                float(row["Avg. Area House Age"]),
                float(row["Avg. Area Number of Rooms"]),
                float(row["Avg. Area Number of Bedrooms"]),
                float(row["Area Population"])
            ]
            price = float(row["Price"])
            features.append(feat_row)
            prices.append(price)

    return np.array(features, dtype=np.float64), np.array(prices, dtype=np.float64), feature_names


def compute_stats(arr: np.ndarray) -> Dict[str, float]:
    """Computes mean, stdDev, min, max, and median for a 1D numpy array."""
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))
    if std_val == 0.0:
        std_val = 1.0
    return {
        "mean": mean_val,
        "stdDev": std_val,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr))
    }


def run_pipeline(csv_path: str = None, test_ratio: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """
    Executes the preprocessing pipeline strictly avoiding data leakage:
    1. Splits indices into 80% train / 20% test first.
    2. Computes feature & target standardization parameters and FastSale medians from train rows only.
    3. Transforms train and test sets using training parameters.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "dataset", "USA_Housing.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"USA_Housing.csv not found at {csv_path}")

    X_raw, y_price_raw, feature_names = load_raw_csv(csv_path)
    n_samples = X_raw.shape[0]

    # 1. Train / Test Split FIRST (Leakage Prevention)
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    split_idx = int(n_samples * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train_raw = X_raw[train_idx]
    X_test_raw = X_raw[test_idx]
    y_price_train_raw = y_price_raw[train_idx]
    y_price_test_raw = y_price_raw[test_idx]

    # 2. Fit statistics on training partition ONLY
    feature_stats = []
    means = np.zeros(X_raw.shape[1], dtype=np.float64)
    stds = np.zeros(X_raw.shape[1], dtype=np.float64)

    for j, name in enumerate(feature_names):
        col_train = X_train_raw[:, j]
        stats = compute_stats(col_train)
        stats["name"] = name
        feature_stats.append(stats)
        means[j] = stats["mean"]
        stds[j] = stats["stdDev"]

    price_stats = compute_stats(y_price_train_raw)

    # 3. FastSale label engineering using training medians ONLY
    # PricePerRoom = Price / Avg. Area Number of Rooms (feature index 2)
    # HouseAge = feature index 1
    train_ppr = y_price_train_raw / X_train_raw[:, 2]
    train_age = X_train_raw[:, 1]
    median_train_ppr = float(np.median(train_ppr))
    median_train_age = float(np.median(train_age))

    # Apply rule to train
    y_fast_train = ((train_ppr < median_train_ppr) & (train_age < median_train_age)).astype(np.int32)

    # Apply rule to test using fitted training medians
    test_ppr = y_price_test_raw / X_test_raw[:, 2]
    test_age = X_test_raw[:, 1]
    y_fast_test = ((test_ppr < median_train_ppr) & (test_age < median_train_age)).astype(np.int32)

    # 4. Standardize features and price using training parameters
    X_train_std = (X_train_raw - means) / stds
    X_test_std = (X_test_raw - means) / stds

    y_price_train_norm = (y_price_train_raw - price_stats["mean"]) / price_stats["stdDev"]
    y_price_test_norm = (y_price_test_raw - price_stats["mean"]) / price_stats["stdDev"]

    fast_train_count = int(np.sum(y_fast_train))
    fast_test_count = int(np.sum(y_fast_test))

    metadata = {
        "totalSamples": n_samples,
        "trainCount": len(train_idx),
        "testCount": len(test_idx),
        "featureNames": feature_names,
        "featureStats": feature_stats,
        "priceStats": price_stats,
        "fastSaleHeuristic": {
            "medianPricePerRoom": median_train_ppr,
            "medianHouseAge": median_train_age,
            "positiveCountTrain": fast_train_count,
            "positiveRatioTrain": float(fast_train_count / len(train_idx)),
            "positiveCountTest": fast_test_count,
            "positiveRatioTest": float(fast_test_count / len(test_idx)),
            "description": (
                f"FastSale is labeled 1 if PricePerRoom is below training median (${median_train_ppr:,.0f}) "
                f"AND House Age is below training median ({median_train_age:.2f} yrs). Synthetic heuristic proxy."
            )
        }
    }

    return {
        "X_train": X_train_std,
        "X_train_raw": X_train_raw,
        "y_price_train": y_price_train_raw,
        "y_price_train_norm": y_price_train_norm,
        "y_fast_train": y_fast_train,
        "X_test": X_test_std,
        "X_test_raw": X_test_raw,
        "y_price_test": y_price_test_raw,
        "y_price_test_norm": y_price_test_norm,
        "y_fast_test": y_fast_test,
        "metadata": metadata
    }


if __name__ == "__main__":
    data = run_pipeline()
    meta = data["metadata"]
    print(f"Data pipeline executed successfully without data leakage.")
    print(f"Train samples: {meta['trainCount']}, Test samples: {meta['testCount']}")
    print(f"Fitted median PricePerRoom: ${meta['fastSaleHeuristic']['medianPricePerRoom']:,.2f}")
    print(f"Fitted median House Age: {meta['fastSaleHeuristic']['medianHouseAge']:.2f} yrs")
    print(f"FastSale train count: {meta['fastSaleHeuristic']['positiveCountTrain']}/{meta['trainCount']} ({meta['fastSaleHeuristic']['positiveRatioTrain']*100:.1f}%)")
