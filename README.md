# PriceDrift — Housing Analytics & Scratch ML (Python + Vanilla JS)

**PriceDrift** is a dual-model housing analytics tool that implements **Linear Regression** (for continuous sale price estimation) and **Logistic Regression** (for binary fast-sale classification) strictly from scratch with **zero ML libraries** (no scikit-learn, TensorFlow, PyTorch, etc.).

---

## Architectural Split & Responsibilities

1. **Python (`ml/`) — 100% of the Machine Learning**:
   - Data loading, train/test splitting, and label engineering.
   - **Data Leakage Fix**: All standardization parameters (feature means, stdDevs, target price mean/stdDev) and heuristic medians are fitted **strictly on the 80% training split**, then applied to transform both train and test sets.
   - Vectorized Batch Gradient Descent implemented from scratch with NumPy.
   - Computes test-set evaluation metrics ($R^2$, RMSE, MSE, Accuracy, Precision, Recall, F1, Confusion Matrix).
   - Exports the trained parameters, convergence history, and dataset to `scripts/data/housing-data.json`.
2. **JavaScript (`scripts/`) — Lightweight Frontend & In-Browser Inference Only**:
   - Zero training math runs in the browser.
   - Fetches the pre-trained `housing-data.json` static artifact.
   - Performs **instantaneous client-side inference** directly on user slider adjustments using standard dot product (`X @ w + b`) and sigmoid (`1 / (1 + exp(-z))`).
   - Replays the recorded convergence curves on an HTML5 `<canvas>` at 60 FPS.
   - Allows interactive exploration of decision thresholds ($\theta \in [0.05, 0.95]$) on the confusion matrix.
3. **Static Deployment (Vercel)**:
   - 100% static site.
   - No Python runs at request time. No backend or serverless functions required.

---

## Dataset & FastSale Heuristic Formulation

- **Dataset**: USA Housing Dataset (5,000 records).
- **Features**:
  1. `Avg. Area Income`
  2. `Avg. Area House Age`
  3. `Avg. Area Number of Rooms`
  4. `Avg. Area Number of Bedrooms`
  5. `Area Population`
- **Dropped Column**: `Address` (non-numeric).
- **Train/Test Split**: 80% Train (4,000 records) / 20% Test (1,000 records) with `np.random.default_rng(42)`.

### FastSale Proxy Heuristic (Disclosed)
Because the dataset does not contain time-on-market fields, the binary classification target `FastSale` is engineered as an economic proxy:
1. Compute $\text{PricePerRoom} = \frac{\text{Price}}{\text{Avg. Area Rooms}}$
2. Compute median $\text{PricePerRoom}$ and median $\text{Avg. Area House Age}$ **strictly from the training split**:
   - Training Median PricePerRoom: **$176,459.12**
   - Training Median House Age: **5.97 years**
3. Label:
   $$\text{FastSale} = \begin{cases} 1 & \text{if } \text{PricePerRoom} < \text{median}_{\text{train}}(\text{PricePerRoom}) \land \text{HouseAge} < \text{median}_{\text{train}}(\text{HouseAge}) \\ 0 & \text{otherwise} \end{cases}$$

---

## Mathematical Formulations (NumPy Implementation)

### 1. Linear Regression (Batch Gradient Descent)
- **Hypothesis**:
  $$\hat{y} = X w + b$$
- **Cost Function (MSE with L2 Ridge)**:
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$
- **Vectorized Gradients**:
  $$\nabla_w = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w$$
  $$\nabla_b = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

### 2. Logistic Regression (Batch Gradient Descent)
- **Sigmoid Activation**:
  $$\sigma(z) = \frac{1}{1 + e^{-\text{clamp}(z, -30, 30)}}$$
- **Hypothesis**:
  $$\hat{p} = \sigma(X w + b)$$
- **Cost Function (Binary Cross-Entropy with L2)**:
  $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \ln(\hat{p}^{(i)} + \epsilon) + (1 - y^{(i)}) \ln(1 - \hat{p}^{(i)} + \epsilon)] + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$
- **Vectorized Gradients**:
  $$\nabla_w = \frac{1}{m} X^T (\hat{p} - y) + \frac{\lambda}{m} w$$
  $$\nabla_b = \frac{1}{m} \sum_{i=1}^m (\hat{p}^{(i)} - y^{(i)})$$

---

## Model Evaluation Results (Pure Out-of-Sample Test Split)

| Model | Metric | Out-of-Sample Value |
| :--- | :--- | :--- |
| **Linear Regression** | Test $R^2$ | **0.9148** (91.48% variance explained) |
| | Test RMSE | **$100,224.13** |
| | Baseline Weights | Income: `+0.6470`, Age: `+0.4680`, Rooms: `+0.3421`, Bedrooms: `+0.0087`, Population: `+0.4263` |
| **Logistic Regression** | Test Accuracy | **89.70%** |
| | Test Precision | **88.09%** |
| | Test Recall | **77.71%** |
| | Test F1 Score | **0.8257** |
| | Confusion Matrix | $TP=244, FN=70, FP=33, TN=653$ ($N=1,000$) |

---

## File Structure

```
pricedrift/
├── index.html                           # Single-page semantic UI
├── styles/
│   └── main.css                         # Dark theme responsive stylesheet
├── scripts/
│   ├── data/
│   │   └── housing-data.json            # Produced by ml/train.py
│   ├── utils/
│   │   └── math.js                      # Trimmed: dot, sigmoid, formatters, confusionMatrix
│   ├── viz/
│   │   ├── trainingChart.js             # Canvas convergence curve animation replay
│   │   ├── confusionMatrix.js           # 2x2 confusion matrix with interactive threshold
│   │   └── coefficientsChart.js         # Feature importance bar chart
│   ├── ui/
│   │   ├── controls.js                  # Model selection & replay controls
│   │   ├── predictionPanel.js           # Feature sliders, real-time normalization, live inference
│   │   └── modal.js                     # Methodology & synthetic proxy disclosure dialog
│   └── main.js                          # App coordinator (replays history + runs live inference)
├── ml/
│   ├── requirements.txt                 # numpy>=1.26, pytest>=8.0
│   ├── data_pipeline.py                 # Leakage-free preprocessing & standardization
│   ├── linear_regression.py             # Vectorized batch gradient descent in NumPy
│   ├── logistic_regression.py           # Vectorized logistic regression in NumPy
│   ├── train.py                         # Trains models and exports housing-data.json
│   ├── evaluate.py                      # Prints test metrics evaluation report
│   └── tests/
│       └── test_models.py               # Pytest suite verifying convergence & leakage fix
├── dataset/
│   └── USA_Housing.csv                  # Kaggle USA Housing dataset
├── vercel.json                          # Static deployment configuration
├── package.json
└── README.md
```

---

## How to Run & Develop

### 1. Python ML Pipeline
Install dependencies (NumPy & Pytest):
```bash
pip install -r ml/requirements.txt
```

Run test suite:
```bash
pytest ml/tests/ -v
```

Train models and export `housing-data.json`:
```bash
python ml/train.py
```

Print evaluation metrics report:
```bash
python ml/evaluate.py
```

### 2. Local Frontend
Serve the static directory:
```bash
python -m http.server 3000
# Open http://localhost:3000
```

### 3. Deploy to Vercel
Deploy static files:
```bash
vercel --prod
```
No Python build commands or serverless functions are required in production — Vercel serves `index.html` and static assets directly.
