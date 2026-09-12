# PriceDrift — System Architecture & Technical Specifications

This document outlines the architectural design, data pipelines, interface boundaries, and technical decisions underpinning **PriceDrift**.

---

## 1. Architectural Philosophy & Design Constraints

PriceDrift was designed under a strict set of educational, performance, and operational constraints:

1. **Zero External Machine Learning Frameworks**:
   - Neither the offline Python pipeline nor the web frontend may import `scikit-learn`, `TensorFlow`, `PyTorch`, `ml.js`, `brain.js`, or equivalent ML libraries.
   - All vector calculus, gradient descent optimizers, activation functions, loss functions, and evaluation metrics are written strictly from first principles using standard array arithmetic (NumPy in Python; typed float arrays and vanilla math in JavaScript).
2. **Decoupled Training vs. Inference Topology**:
   - **Offline Build-Time Training**: Python handles data ingestion, train/test splitting, statistical standardization fitting, batch gradient descent, and evaluation.
   - **Static Artifact Bundle**: Python outputs a deterministic JSON file (`scripts/data/housing-data.json`) containing learned parameters, test metrics, and convergence trajectories.
   - **Zero-Latency In-Browser Inference**: The client web application executes inference in pure JavaScript directly on the client thread in $< 0.1\text{ ms}$, requiring zero runtime API calls or serverless backends.
3. **100% Static Deployment (Vercel)**:
   - Deployable as static assets (HTML/CSS/JS/JSON) with no backend process, Node.js server, or Python runtime required at request time.

---

## 2. System Flow & Component Diagram

```
+----------------------------------------------------------------------------------+
|                            OFFLINE PIPELINE (Python)                            |
+----------------------------------------------------------------------------------+
|                                                                                  |
|  [ dataset/USA_Housing.csv ] (5,000 samples)                                     |
|           |                                                                      |
|           v                                                                      |
|  [ ml/data_pipeline.py ]                                                         |
|    - 80/20 Deterministic Train/Test Split (np.random.default_rng(42))            |
|    - Fit feature stats (means, stdDevs) on Train partition ONLY                  |
|    - Fit FastSale medians (PricePerRoom, HouseAge) on Train partition ONLY       |
|    - Standardize Train (4,000) and Test (1,000) using fitted Train stats         |
|           |                                                                      |
|           +---------------------------------------+                              |
|           |                                       |                              |
|           v                                       v                              |
|  [ ml/linear_regression.py ]             [ ml/logistic_regression.py ]           |
|    - Batch Gradient Descent (NumPy)        - Batch Gradient Descent (NumPy)      |
|    - MSE Loss + Ridge Regularization       - BCE Loss + L2 Regularization        |
|    - Records epoch-by-epoch loss           - Records epoch-by-epoch loss         |
|           |                                       |                              |
|           +-------------------+-------------------+                              |
|                               |                                                  |
|                               v                                                  |
|                      [ ml/train.py ]                                             |
|                        - Evaluates out-of-sample test split                      |
|                        - Generates scripts/data/housing-data.json                |
+----------------------------------------------------------------------------------+
                                        |
                            STATIC JSON ARTIFACT (1.19 MB)
                                        |
+----------------------------------------------------------------------------------+
|                           CLIENT RUNTIME (Vanilla JS)                            |
+----------------------------------------------------------------------------------+
|                                                                                  |
|                        [ scripts/main.js ] (App Bootstrap)                       |
|                               |                                                  |
|      +------------------------+------------------------+                         |
|      |                        |                        |                         |
|      v                        v                        v                         |
| [ trainingChart.js ]   [ predictionPanel.js ]    [ confusionMatrix.js ]          |
|  - HTML5 Canvas 60fps   - Standardize input       - 2x2 Heat Grid (1,000 test)   |
|  - Replays Python loss  - dot(w_lin, x) + b_lin   - Dynamic Threshold Slider     |
|    history smoothly     - sigmoid(dot(w_log, x))    (Re-evaluates instantly)     |
|                                                                                  |
+----------------------------------------------------------------------------------+
```

---

## 3. Data Contract: `scripts/data/housing-data.json` Schema

The shared contract between Python and the JavaScript frontend is defined by the following JSON structure:

```json
{
  "metadata": {
    "totalSamples": 5000,
    "trainCount": 4000,
    "testCount": 1000,
    "featureNames": [
      "Avg. Area Income",
      "Avg. Area House Age",
      "Avg. Area Number of Rooms",
      "Avg. Area Number of Bedrooms",
      "Area Population"
    ],
    "featureStats": [
      {
        "name": "Avg. Area Income",
        "mean": 68583.108984375,
        "stdDev": 10657.0673828125,
        "min": 17796.6328125,
        "max": 107701.7421875,
        "median": 68804.2890625
      }
    ],
    "priceStats": {
      "mean": 1232084.25,
      "stdDev": 353110.125,
      "min": 15938.6572265625,
      "max": 2469065.5,
      "median": 1232084.25
    },
    "fastSaleHeuristic": {
      "medianPricePerRoom": 176459.125,
      "medianHouseAge": 5.97,
      "positiveCountTrain": 1341,
      "positiveRatioTrain": 0.33525,
      "positiveCountTest": 314,
      "positiveRatioTest": 0.314,
      "description": "FastSale is labeled 1 if PricePerRoom is below training median ($176,459) AND House Age is below training median (5.97 yrs)."
    }
  },
  "train": {
    "X": [[0.12, -0.45, 0.88, -0.12, 0.54]],
    "y_price": [1250000.0],
    "y_price_norm": [0.05],
    "y_fast": [1]
  },
  "test": {
    "X": [[-0.34, 0.65, -0.22, 0.44, -0.87]],
    "y_price": [1120000.0],
    "y_price_norm": [-0.31],
    "y_fast": [0]
  },
  "trainedModels": {
    "linear": {
      "weights": [0.6470, 0.4680, 0.3421, 0.0087, 0.4263],
      "bias": -0.0000,
      "history": [{"epoch": 1, "cost": 0.4215}],
      "testMetrics": {
        "r2": 0.9148,
        "rmse": 100224.13,
        "mse": 10044876474.43,
        "normalizedMSE": 0.0805
      }
    },
    "linear_regularized": { ... },
    "logistic": {
      "weights": [-0.8524, -1.2415, 0.5214, -0.0412, -0.6312],
      "bias": -0.9124,
      "history": [{"epoch": 1, "cost": 0.6842}],
      "testMetrics": {
        "accuracy": 0.8970,
        "precision": 0.8809,
        "recall": 0.7771,
        "f1": 0.8257,
        "threshold": 0.5,
        "confusionMatrix": {
          "tp": 244,
          "fp": 33,
          "tn": 653,
          "fn": 70,
          "total": 1000
        }
      }
    },
    "logistic_regularized": { ... }
  }
}
```

---

## 4. Preventing Data Leakage

In statistical machine learning, **data leakage** occurs when information from outside the training dataset is used to train or standardize the model.

### The Leakage Bug in Naive Implementations:
Computing `mean` and `stdDev` across all $N=5,000$ samples before splitting into train/test partitions means the test set's distribution subtly influences the scaling factors of the training set.

### The Strict PriceDrift Solution:
In [`ml/data_pipeline.py`](file:///c:/Users/USER/OneDrive%20-%20Dawood%20University%20of%20Engineering%20Technology/Desktop/PriceDrift/ml/data_pipeline.py):
1. **Partition First**: Indices $\{0, \dots, N-1\}$ are randomly permuted and split into $80\%$ ($m_{\text{train}} = 4,000$) and $20\%$ ($m_{\text{test}} = 1,000$).
2. **Fit on Train Only**:
   $$\mu_j = \frac{1}{m_{\text{train}}} \sum_{i \in \text{train}} x_{i, j}, \quad \sigma_j = \sqrt{\frac{1}{m_{\text{train}}} \sum_{i \in \text{train}} (x_{i, j} - \mu_j)^2}$$
   $$\text{median}_{\text{train}}(\text{PPR}) = \text{median}\left(\left\{\frac{\text{Price}_i}{\text{Rooms}_i} \;\middle|\; i \in \text{train}\right\}\right)$$
   $$\text{median}_{\text{train}}(\text{Age}) = \text{median}\left(\left\{\text{Age}_i \;\middle|\; i \in \text{train}\right\}\right)$$
3. **Transform Both**:
   $$X_{\text{train}, j} = \frac{X_{\text{train}, j} - \mu_j}{\sigma_j}, \quad X_{\text{test}, j} = \frac{X_{\text{test}, j} - \mu_j}{\sigma_j}$$
   This guarantees that test set evaluations represent **pure out-of-sample performance**.

---

## 5. Performance & Complexity Analysis

### Time Complexity

| Component | Operation | Complexity | Runtime |
| :--- | :--- | :--- | :--- |
| **Python Training (Linear)** | Batch Gradient Descent ($E=200, m=4000, d=5$) | $\mathcal{O}(E \cdot m \cdot d)$ | $\approx 25\text{ ms}$ |
| **Python Training (Logistic)**| Batch Gradient Descent ($E=250, m=4000, d=5$) | $\mathcal{O}(E \cdot m \cdot d)$ | $\approx 40\text{ ms}$ |
| **JS Slider Inference** | Standardize + Dot Product + Sigmoid | $\mathcal{O}(d) \quad (d=5)$ | $< 0.02\text{ ms}$ |
| **JS Threshold Evaluation** | Evaluate test set confusion matrix ($m=1000$) | $\mathcal{O}(m \cdot d)$ | $\approx 0.35\text{ ms}$ |
| **Canvas Replay Animation** | Plot frame ($k$ points) | $\mathcal{O}(k)$ | $60\text{ FPS}$ ($16.6\text{ ms}$ budget) |

### Space & Network Overhead
- **Static Artifact Bundle (`housing-data.json`)**: $1.19\text{ MB}$ uncompressed, $\approx 180\text{ KB}$ gzipped across HTTP.
- **Client Heap Footprint**: $< 4\text{ MB}$ in browser memory.
- **Client Compute Footprint**: 0% CPU idle; negligible bursts during slider dragging.

---

## 6. Security, Privacy & Static Hosting

- **Zero PII & Data Confidentiality**: The USA Housing dataset is completely synthetic. All processing occurs locally on the developer machine during training, and client-side in the user's browser during prediction.
- **Content Security**: No dynamic `eval()`, no external CDN script injections, and no cross-origin API calls.
- **Vercel Cache Optimization**: Configured in [`vercel.json`](file:///c:/Users/USER/OneDrive%20-%20Dawood%20University%20of%20Engineering%20Technology/Desktop/PriceDrift/vercel.json) to serve `scripts/data/*` with immutable long-term caching (`Cache-Control: public, max-age=86400, immutable`), ensuring instantaneous subsequent loads.
