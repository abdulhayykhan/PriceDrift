# PriceDrift — In-Browser Housing Analytics & Scratch ML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized_Math-013243?style=for-the-badge&logo=numpy&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Vanilla_ES6-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5 Canvas](https://img.shields.io/badge/HTML5-Canvas_60FPS-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-100%25_Static_Deploy-000000?style=for-the-badge&logo=vercel&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-100%25_Pass-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

**A dual-model housing valuation and market velocity analytics tool implementing Linear Regression and Logistic Regression strictly from first principles (no scikit-learn, no TensorFlow, no ML libraries).**

[Live Web Demo](#8-developer-quickstart-retraining--testing) • [User Guide](docs/USER_GUIDE.md) • [Architecture Specs](docs/ARCHITECTURE.md) • [Mathematical Formulations](docs/MATHEMATICAL_FORMULATION.md)

</div>

---

## Table of Contents

1. [Executive Summary & Motivation](#1-executive-summary--motivation)
2. [Architectural Paradigm: Python ML + Vanilla JS Frontend](#2-architectural-paradigm-python-ml--vanilla-js-frontend)
   - [System Topology](#system-topology)
   - [Why This Hybrid Design Excels](#why-this-hybrid-design-excels)
   - [Zero-Latency Client-Side Inference](#zero-latency-client-side-inference)
   - [Pedagogical Convergence Replay on Canvas](#pedagogical-convergence-replay-on-canvas)
3. [Mathematical Foundations & Algorithms (From Scratch)](#3-mathematical-foundations--algorithms-from-scratch)
   - [Conditioning & Z-Score Standardization](#conditioning--z-score-standardization)
   - [Linear Regression via Vectorized Batch Gradient Descent](#linear-regression-via-vectorized-batch-gradient-descent)
   - [Logistic Regression via Vectorized Batch Gradient Descent](#logistic-regression-via-vectorized-batch-gradient-descent)
   - [Regularization Mechanics (Ridge / L2 Penalty)](#regularization-mechanics-ridge--l2-penalty)
4. [Dataset & Heuristic Label Engineering](#4-dataset--heuristic-label-engineering)
   - [The USA Housing Dataset](#the-usa-housing-dataset)
   - [Strict Data Leakage Prevention](#strict-data-leakage-prevention)
   - [FastSale Heuristic Formulation (Required Disclosure)](#fastsale-heuristic-formulation-required-disclosure)
5. [Empirical Evaluation & Test Set Benchmarks](#5-empirical-evaluation--test-set-benchmarks)
   - [Out-of-Sample Performance Summary](#out-of-sample-performance-summary)
   - [Baseline vs. L2 Ridge Regularization Comparison](#baseline-vs-l2-ridge-regularization-comparison)
   - [Learned Linear Coefficients & Dollar Impact](#learned-linear-coefficients--dollar-impact)
6. [Interactive Frontend Capabilities](#6-interactive-frontend-capabilities)
   - [Live HTML5 Canvas Convergence Replay](#live-html5-canvas-convergence-replay)
   - [Synchronized Feature Sliders & Archetype Presets](#synchronized-feature-sliders--archetype-presets)
   - [Dynamic Valuation & Fast-Sale Probability Outputs](#dynamic-valuation--fast-sale-probability-outputs)
   - [Interactive Confusion Matrix & Threshold Tuning](#interactive-confusion-matrix--threshold-tuning)
7. [Comprehensive Codebase Anatomy](#7-comprehensive-codebase-anatomy)
8. [Developer Quickstart, Retraining & Testing](#8-developer-quickstart-retraining--testing)
   - [Environment Setup](#environment-setup)
   - [Running the Automated Pytest Suite](#running-the-automated-pytest-suite)
   - [Retraining Models Offline](#retraining-models-offline)
   - [Launching Local Static Web Server](#launching-local-static-web-server)
9. [Static Vercel Deployment Guide](#9-static-vercel-deployment-guide)
10. [Documentation Index](#10-documentation-index)
11. [Known Limitations & Educational Disclosures](#11-known-limitations--educational-disclosures)
12. [License](#12-license)

---

## 1. Executive Summary & Motivation

In modern machine learning, developers and students frequently rely on black-box abstractions like `sklearn.linear_model.LinearRegression` or deep learning runtimes. This abstraction hides fundamental computational realities: numerical conditioning, loss landscapes, matrix calculus derivations, gradient descent stability, and the critical distinction between training-time optimization and runtime inference.

**PriceDrift** was created to demonstrate a clean, transparent, production-quality implementation of both continuous price regression and binary classification velocity modeling **built 100% from first principles**:
- **Zero Machine Learning Libraries**: Every gradient, loss function, vector dot product, and standardization transformation is hand-coded using pure array math in Python (NumPy) and vanilla JavaScript.
- **Strict Data Leakage Prevention**: Enforces mathematically sound preprocessing where all scaling parameters ($\mu, \sigma$) and heuristic medians are fitted exclusively on the training partition (80%) before transforming test samples.
- **Production-Ready Static Delivery**: Compiles the learned parameters into a portable JSON artifact, enabling instant in-browser client-side evaluation without running a backend server.

---

## 2. Architectural Paradigm: Python ML + Vanilla JS Frontend

### System Topology

PriceDrift decouples the computational lifecycle into two distinct phases:

```
[ BUILD-TIME / OFFLINE ] (Python 3.10+)
+----------------------------------------------------------------------------------+
| dataset/USA_Housing.csv (5,000 samples)                                          |
|   |                                                                              |
|   v                                                                              |
| ml/data_pipeline.py                                                              |
|   - 80/20 Train/Test Split FIRST (Seed: 42)                                      |
|   - Fit feature statistics & FastSale medians on Train rows ONLY                 |
|   - Transform Train (4,000) & Test (1,000) via Train-fitted z-score              |
|   |                                                                              |
|   +---------------------------------------+                                      |
|   |                                       |                                      |
|   v                                       v                                      |
| ml/linear_regression.py           ml/logistic_regression.py                      |
|   - Vectorized Batch GD (NumPy)     - Vectorized Batch GD (NumPy)                |
|   - MSE Loss + Ridge Penalty        - BCE Loss + L2 Penalty                      |
|   - Epoch-by-epoch loss tracking    - Epoch-by-epoch loss tracking               |
|   |                                       |                                      |
|   +-------------------+-------------------+                                      |
|                       |                                                          |
|                       v                                                          |
|               ml/train.py                                                        |
|                 - Out-of-sample evaluation (R², RMSE, Accuracy, F1, CM)          |
|                 - Exports scripts/data/housing-data.json (1.19 MB)               |
+----------------------------------------------------------------------------------+
                                        |
                      STATIC ARTIFACT BUNDLE (JSON)
                                        |
[ RUNTIME / CLIENT BROWSER ] (Vanilla ES6 HTML/CSS/JS)
+----------------------------------------------------------------------------------+
| scripts/main.js (Loads housing-data.json)                                        |
|   |                                                                              |
|   +---> scripts/viz/trainingChart.js       (HTML5 Canvas 60 FPS Replay)          |
|   +---> scripts/ui/predictionPanel.js      (Instant Client-Side Inference)       |
|   +---> scripts/viz/confusionMatrix.js     (Dynamic Threshold θ Exploration)     |
|   +---> scripts/viz/coefficientsChart.js   (Feature Importance Bar Chart)        |
+----------------------------------------------------------------------------------+
```

### Why This Hybrid Design Excels

| Strategy | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **Monolithic JS (Training in Browser)** | Interactive training controls in UI. | Heavy battery/CPU drain; slower execution; floating-point inconsistencies across browser engines. |
| **Backend API (Node/FastAPI Server)** | High-compute model hosting. | Requires active hosting server; cold-start latency; network dependency per slider adjustment. |
| **PriceDrift: Python Training + JS Inference** | **Best of both worlds**: Rigorous NumPy math offline; instant $< 0.1\text{ms}$ in-browser inference; zero hosting costs (100% static Vercel host). | Training hyperparameters cannot be tuned dynamically in-browser without re-running Python. |

### Zero-Latency Client-Side Inference
When a user moves an attribute slider in the UI (e.g., changing house age from 4.0 to 6.5 years), no API calls occur. The browser executes:
1. **Input Normalization**:
   $$x_{\text{std}, j} = \frac{x_j - \mu_{j, \text{train}}}{\sigma_{j, \text{train}}}$$
2. **Linear Regression Price Prediction**:
   $$\hat{y}_{\text{USD}} = \left( \sum_{j=1}^5 w_j x_{\text{std}, j} + b \right) \cdot \sigma_{y, \text{train}} + \mu_{y, \text{train}}$$
3. **Logistic Regression Velocity Probability**:
   $$P(\text{FastSale}) = \sigma\left( \sum_{j=1}^5 w_{\text{log}, j} x_{\text{std}, j} + b_{\text{log}} \right)$$

This entire inference pass requires only 10 multiplications, 10 additions, and 1 exponential evaluation, executing in **under 0.02 milliseconds** directly on the UI thread.

### Pedagogical Convergence Replay on Canvas
Instead of displaying a static, lifeless chart, PriceDrift includes an HTML5 Canvas animator that **replays** the exact step-by-step cost curve recorded by the Python training run at 60 frames per second. Users can pause, resume, change playback speed (1x, 2x, 5x), or skip to the end.

---

## 3. Mathematical Foundations & Algorithms (From Scratch)

Detailed mathematical proofs are available in [`docs/MATHEMATICAL_FORMULATION.md`](docs/MATHEMATICAL_FORMULATION.md). Below is the operational summary.

### Conditioning & Z-Score Standardization
Because raw housing features range from small counts (e.g., $1-8$ bedrooms) to large monetary sums (e.g., $\$80,000$ income), the unstandardized Hessian matrix $\mathbf{X}^T \mathbf{X}$ is ill-conditioned.

To sphericalize the loss contours, features and target price are standardized:
$$z_{i, j} = \frac{X_{i, j} - \mu_j}{\sigma_j}, \quad \mu_j = \frac{1}{m} \sum_{i=1}^m X_{i, j}, \quad \sigma_j = \sqrt{\frac{1}{m} \sum_{i=1}^m (X_{i, j} - \mu_j)^2}$$

This guarantees that every standardized feature column has an empirical mean of 0 and an empirical variance of 1.

---

### Linear Regression via Vectorized Batch Gradient Descent

- **Hypothesis**:
  $$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b \mathbf{1}_m$$
- **Cost Function (Mean Squared Error with Ridge)**:
  $$J(\mathbf{w}, b) = \frac{1}{2m} \|\mathbf{X} \mathbf{w} + b \mathbf{1}_m - \mathbf{y}\|_2^2 + \frac{\lambda}{2m} \|\mathbf{w}\|_2^2$$
- **Vectorized Gradients**:
  $$\nabla_{\mathbf{w}} J = \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) + \frac{\lambda}{m} \mathbf{w}$$
  $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$
- **Parameter Updates**:
  $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \nabla_{\mathbf{w}} J$$
  $$b^{(t+1)} = b^{(t)} - \alpha \frac{\partial J}{\partial b}$$

---

### Logistic Regression via Vectorized Batch Gradient Descent

- **Hypothesis (Sigmoid Activation)**:
  $$\hat{\mathbf{p}} = \sigma(\mathbf{X} \mathbf{w} + b \mathbf{1}_m), \quad \sigma(z) = \frac{1}{1 + e^{-\text{clamp}(z, -30, 30)}}$$
- **Cost Function (Binary Cross-Entropy with L2)**:
  $$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(\hat{p}^{(i)} + \epsilon) + (1 - y^{(i)}) \ln(1 - \hat{p}^{(i)} + \epsilon) \right] + \frac{\lambda}{2m} \|\mathbf{w}\|_2^2$$
- **Vectorized Gradients**:
  $$\nabla_{\mathbf{w}} J = \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{p}} - \mathbf{y}) + \frac{\lambda}{m} \mathbf{w}$$
  $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{p}^{(i)} - y^{(i)})$$

---

### Regularization Mechanics (Ridge / L2 Penalty)
Setting $\lambda > 0$ penalizes large weight magnitudes, shrinking coefficients towards zero. The parameter update becomes:
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} \left( 1 - \frac{\alpha \lambda}{m} \right) - \frac{\alpha}{m} \mathbf{X}^T \mathbf{e}$$
The term $\left(1 - \frac{\alpha \lambda}{m}\right)$ applies a deterministic weight decay step on every epoch.

---

## 4. Dataset & Heuristic Label Engineering

### The USA Housing Dataset
The project utilizes the Kaggle USA Housing dataset (`dataset/USA_Housing.csv`), consisting of **5,000 residential housing records**.

| Feature Column | Description | Type | Fitted Training Mean ($\mu$) | Fitted Training StdDev ($\sigma$) |
| :--- | :--- | :--- | :--- | :--- |
| `Avg. Area Income` | Average annual income of area residents | Continuous (USD) | \$68,583.11 | \$10,657.07 |
| `Avg. Area House Age` | Average construction age of properties | Continuous (Years) | 5.97 yrs | 0.99 yrs |
| `Avg. Area Number of Rooms` | Average total rooms per property | Continuous (Rooms) | 6.98 rooms | 1.00 rooms |
| `Avg. Area Number of Bedrooms` | Average bedrooms per property | Continuous (Bedrooms) | 3.98 beds | 1.23 beds |
| `Area Population` | Total population residing in immediate area | Continuous (Residents) | 36,155.67 | 9,926.24 |
| `Price` (Target 1) | Property sale price | Target (USD) | \$1,232,084.25 | \$353,110.13 |
| `Address` | Street address text | Text | *Dropped — non-numeric* | *Dropped* |

---

### Strict Data Leakage Prevention

> [!IMPORTANT]
> **Data Leakage Fix**: Previous iterations suffered from subtle data leakage because normalization metrics were computed across all 5,000 rows prior to splitting. In [`ml/data_pipeline.py`](ml/data_pipeline.py), records are partitioned **first** into 80% Train (4,000 rows) and 20% Test (1,000 rows). All means, standard deviations, and heuristic medians are fit **exclusively on the training rows**, then applied to transform the test set.

---

### FastSale Heuristic Formulation (Required Disclosure)

> [!WARNING]
> **Mandatory Disclosure**: The USA Housing dataset does not contain transactional time-stamps or listing durations. `FastSale` is an **engineered pedagogical domain proxy**, not observed real-world market velocity.

#### Formal Definition:
1. For every property $i$, compute its room-normalized price:
   $$\text{PricePerRoom}_i = \frac{\text{Price}_i}{\text{Avg. Area Rooms}_i}$$
2. Compute the medians strictly on the training partition:
   - Training Median PricePerRoom: **\$176,459.12**
   - Training Median House Age: **5.97 years**
3. Assign binary velocity label:
   $$\text{FastSale}_i = \begin{cases} 1 & \text{if } \text{PricePerRoom}_i < \$176,459.12 \;\land\; \text{HouseAge}_i < 5.97\text{ yrs} \\ 0 & \text{otherwise} \end{cases}$$

This creates an economically intuitive, balanced classification problem: **properties priced below the room median that are newer than average construction age sell fast** (33.5% positive in train, 31.4% positive in test).

---

## 5. Empirical Evaluation & Test Set Benchmarks

The models were evaluated on the **1,000 unseen test samples** transformed strictly using training statistics.

### Out-of-Sample Performance Summary

| Model | Task | Objective Metric | Benchmark Value | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Price Prediction | Test $R^2$ (Variance Explained) | **0.9148** (91.48%) | ✅ Converged |
| | | Test RMSE (Root Mean Squared Error) | **\$100,224.13** | ✅ Robust |
| | | Test MSE | $1.004 \times 10^{10}$ | ✅ Bounded |
| **Logistic Regression** | Fast-Sale Classifier | Test Accuracy ($\theta=0.5$) | **89.70%** | ✅ Non-Degenerate |
| | | Test Precision | **88.09%** | ✅ High Reliability |
| | | Test Recall | **77.71%** | ✅ Strong Coverage |
| | | Test $F_1$ Score | **0.8257** | ✅ Balanced |
| | | Test Confusion Matrix ($N=1,000$) | $\begin{pmatrix} TP=244 & FN=70 \\ FP=33 & TN=653 \end{pmatrix}$ | ✅ Valid |

---

### Baseline vs. L2 Ridge Regularization Comparison

| Metric | Linear Baseline ($\lambda=0$) | Linear Ridge ($\lambda=50$) | Logistic Baseline ($\lambda=0$) | Logistic L2 ($\lambda=10$) |
| :--- | :--- | :--- | :--- | :--- |
| **Test Loss / Metric** | $R^2 = 0.9148$ | $R^2 = 0.9146$ | $\text{Acc} = 89.70\%$ | $\text{Acc} = 89.80\%$ |
| **Test Error** | $\text{RMSE} = \$100,224$ | $\text{RMSE} = \$100,341$ | $F_1 = 0.8257$ | $F_1 = 0.8266$ |
| **Weight L2 Norm $\|\mathbf{w}\|_2^2$** | **0.8752** | **0.8529** (Shrunk) | **3.8941** | **3.6110** (Shrunk) |

---

### Learned Linear Coefficients & Dollar Impact

$$\text{Dollar Impact per Unit}_j = w_j \cdot \frac{\sigma_{y, \text{train}}}{\sigma_{j, \text{train}}}$$

| Feature Name | Standardized Weight ($w_j$) | Natural Unit | Empirical Impact per Unit |
| :--- | :--- | :--- | :--- |
| **Avg. Area Income** | `+0.6470` | per \$1.00 income | **+\$21.44** |
| **Avg. Area House Age** | `+0.4680` | per 1 year older | **+\$165,116.00** |
| **Avg. Area Number of Rooms**| `+0.3421` | per 1 additional room | **+\$121,080.00** |
| **Area Population** | `+0.4263` | per 1 additional resident | **+\$15.19** |
| **Avg. Area Number of Bedrooms** | `+0.0087` | per 1 additional bedroom | **+\$2,482.00** |

*Interpretation: Area income and construction age are the primary drivers of housing value in this dataset, while bedrooms contribute minimal marginal value when total room count is already accounted for.*

---

## 6. Interactive Frontend Capabilities

### Live HTML5 Canvas Convergence Replay
- Rendered via vanilla 2D Canvas API ([`scripts/viz/trainingChart.js`](scripts/viz/trainingChart.js)).
- Features high-DPI scaling (`window.devicePixelRatio`), dual series glows, gradient area fills, and animated pulse tracking.
- Interactive controls: Replay (`▶`), Pause/Resume (`⏸`), Skip to End (`⏭`), Speed Toggle (`1x`, `2x`, `5x`).

### Synchronized Feature Sliders & Archetype Presets
- Sliders and numeric number inputs are continuously bidirectionally synchronized.
- Archetype preset buttons allow instant demonstration of market dynamics:
  - **Median Home**: Exact center of dataset distributions.
  - **Newer & Spacious**: Low age, large room volume $\to$ High probability of Fast Sale.
  - **Luxury Suburban**: Ultra-high income and property size $\to$ Low fast-sale probability due to high price per room.
  - **Older Dense Urban**: High population, older construction $\to$ Normal market speed.

### Dynamic Valuation & Fast-Sale Probability Outputs
- **Continuous Valuation Card**: Formatted USD estimate, accompanied by visual markup/markdown percentage relative to the regional median.
- **Fast-Sale Classifier Card**: Real-time percentage probability bar, active decision threshold marker ($\theta$), and dynamic status badge (`⚡ FAST SALE` vs. `⏳ NORMAL / SLOW SALE`).

### Interactive Confusion Matrix & Threshold Tuning
- Renders a 2x2 grid ([`scripts/viz/confusionMatrix.js`](scripts/viz/confusionMatrix.js)) evaluating all 1,000 unseen test samples.
- Cells feature proportional heat-shading based on sample densities.
- **Dynamic Decision Threshold Slider ($\theta \in [0.05, 0.95]$)**: Dragging the slider reclassifies test samples instantly and recalculates Accuracy, Precision, Recall, and F1 in real time, making precision-recall trade-offs visually and intuitively clear.

---

## 7. Comprehensive Codebase Anatomy

```
PriceDrift/
├── dataset/
│   └── USA_Housing.csv              # 5,000-sample Kaggle USA Housing dataset
├── ml/                              # Standalone Python ML Pipeline (Build-Time)
│   ├── __init__.py                  # Python package declaration
│   ├── requirements.txt             # numpy>=1.26, pytest>=8.0
│   ├── data_pipeline.py             # Leakage-free train/test split, standardization, FastSale
│   ├── linear_regression.py         # From-scratch batch gradient descent linear model
│   ├── logistic_regression.py       # From-scratch batch gradient descent logistic model
│   ├── train.py                     # Orchestrator: trains models & exports housing-data.json
│   ├── evaluate.py                  # Standalone CLI metrics report
│   └── tests/
│       └── test_models.py           # 6-part Pytest verification suite
├── scripts/                         # Static Frontend Client Runtime (Vanilla JS)
│   ├── data/
│   │   └── housing-data.json        # Bundled static artifact (models + test data + stats)
│   ├── utils/
│   │   └── math.js                  # Lightweight inference math (dot, sigmoid, clamp, formatters)
│   ├── viz/
│   │   ├── trainingChart.js         # Canvas 60 FPS convergence curve replay engine
│   │   ├── confusionMatrix.js       # 2x2 dynamic confusion matrix & metric cards
│   │   └── coefficientsChart.js     # Learned weights & dollar-impact bar chart
│   ├── ui/
│   │   ├── controls.js              # Model selection, replay speed, and variant controls
│   │   ├── predictionPanel.js       # Sliders, z-score transform, and live inference cards
│   │   └── modal.js                 # Methodology, data disclosure, and math dialog
│   └── main.js                      # Application coordinator (loads JSON, wires DOM)
├── styles/
│   └── main.css                     # Obsidian dark theme, responsive grid layouts, glassmorphism
├── docs/                            # Deep-Dive Documentation
│   ├── USER_GUIDE.md                # Exhaustive operator and user guide
│   ├── ARCHITECTURE.md              # System design, data contracts, and complexity analysis
│   └── MATHEMATICAL_FORMULATION.md  # Pure mathematical proofs and matrix calculus derivations
├── index.html                       # Semantic, accessible single-page dashboard HTML
├── vercel.json                      # Vercel static deployment config & caching headers
├── package.json                     # Project metadata & npm helper scripts
├── .gitignore                       # Clean repository exclusions (pycache, envs, logs)
└── README.md                        # Master project documentation
```

---

## 8. Developer Quickstart, Retraining & Testing

### Environment Setup

#### Prerequisites
- **Python 3.10+** (with `pip`)
- Any modern web browser (Chrome, Firefox, Safari, Edge)

```bash
# Clone the repository
git clone https://github.com/abdulhayykhan/PriceDrift.git
cd PriceDrift

# Install Python dependencies
pip install -r ml/requirements.txt
```

---

### Running the Automated Pytest Suite
Run the test suite to verify math correctness, sigmoid stability, leakage prevention, and model convergence:
```bash
pytest ml/tests/ -v
```

Expected output:
```
ml/tests/test_models.py::test_sigmoid_numerical_stability PASSED         [ 16%]
ml/tests/test_models.py::test_data_pipeline_no_leakage PASSED            [ 33%]
ml/tests/test_models.py::test_linear_regression_convergence PASSED       [ 50%]
ml/tests/test_models.py::test_linear_regression_regularization_shrinkage PASSED [ 66%]
ml/tests/test_models.py::test_logistic_regression_convergence PASSED     [ 83%]
ml/tests/test_models.py::test_logistic_threshold_tradeoff PASSED         [100%]
============================== 6 passed in 1.26s ==============================
```

---

### Retraining Models Offline
To retrain both models from scratch and regenerate the bundled static artifact:
```bash
python ml/train.py
```
To print the out-of-sample evaluation report to the console:
```bash
python ml/evaluate.py
```

---

### Launching Local Static Web Server
Because modern browsers enforce CORS security policies on ES modules loaded via `file://`, serve the project root via any local HTTP server:

```bash
# Using Python's built-in HTTP server:
python -m http.server 3000

# Or using Node.js npx:
npx serve .
```

Open your browser and navigate to:
```
http://localhost:3000
```

---

## 9. Static Vercel Deployment Guide

PriceDrift is designed for zero-backend static hosting on **Vercel**.

### Method 1: Deploy via Vercel CLI
```bash
# Install Vercel CLI globally if not installed
npm install -g vercel

# Deploy directly from repository root
vercel --prod
```

### Method 2: Deploy via GitHub Integration
1. Push your repository to GitHub: `https://github.com/abdulhayykhan/PriceDrift`
2. Log into your [Vercel Dashboard](https://vercel.com).
3. Click **Add New...** $\to$ **Project**.
4. Import `PriceDrift`.
5. Keep default build settings:
   - **Framework Preset**: `Other`
   - **Build Command**: *Leave blank* (Static files)
   - **Output Directory**: `.` (Root directory)
6. Click **Deploy**.

[`vercel.json`](vercel.json) automatically enforces clean URLs and configures immutable HTTP cache-control headers for `housing-data.json`.

---

## 10. Documentation Index

For in-depth explanations, refer to the dedicated documents in the [`docs/`](docs/) directory:

- 📖 **[User Guide (`docs/USER_GUIDE.md`)](docs/USER_GUIDE.md)**: Exhaustive manual covering UI controls, preset interpretation, threshold tuning, and retraining.
- 📐 **[System Architecture (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md)**: Deep-dive into software design, the data leakage fix, JSON schema data contracts, and complexity analysis.
- 🔬 **[Mathematical Formulations (`docs/MATHEMATICAL_FORMULATION.md`)](docs/MATHEMATICAL_FORMULATION.md)**: Rigorous matrix calculus derivations, gradient descent proofs, sigmoid identities, and loss functions.

---

## 11. Known Limitations & Educational Disclosures

1. **Synthetic Nature of Dataset**: The Kaggle USA Housing dataset is synthetic. Features are smoother and have higher linear correlation ($R^2 > 0.91$) than messy, real-world MLS transaction data.
2. **FastSale is an Engineered Proxy**: Because the source data lacks transaction date and days-on-market fields, `FastSale` was created as an economic heuristic ($PPR < \text{median} \land \text{Age} < \text{median}$). It must be treated as an educational proxy, not verified historical liquidity.
3. **Replay Mode**: The convergence curves on the canvas chart represent deterministic replays of the Python training run, ensuring smooth cross-device performance without client CPU drain.

---

## 12. License

Distributed under the **MIT License**. See `LICENSE` for more information. Developed for educational clarity, machine learning transparency, and static web performance.
