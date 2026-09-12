# PriceDrift — Comprehensive User & Operator Guide

Welcome to the **PriceDrift** User Guide. This document provides an exhaustive, step-by-step walkthrough of how to operate, configure, interpret, and extend the PriceDrift housing valuation and market velocity analytics dashboard.

---

## Table of Contents
1. [Overview & Philosophy](#1-overview--philosophy)
2. [Interface Walkthrough](#2-interface-walkthrough)
   - [2.1 Header & Disclosure Bar](#21-header--disclosure-bar)
   - [2.2 Model Controls & Convergence Replay Panel](#22-model-controls--convergence-replay-panel)
   - [2.3 Live Canvas Convergence Replay](#23-live-canvas-convergence-replay)
   - [2.4 Interactive Property Valuation & Fast-Sale Estimator](#24-interactive-property-valuation--fast-sale-estimator)
   - [2.5 Model Insights & Diagnostic Panels](#25-model-insights--diagnostic-panels)
3. [Interpreting Predictions](#3-interpreting-predictions)
   - [Continuous Sale Price Prediction](#continuous-sale-price-prediction)
   - [Fast-Sale Probability & Binary Classification](#fast-sale-probability--binary-classification)
   - [Decision Threshold Trade-Off Analysis](#decision-threshold-trade-off-analysis)
4. [Exploring Scenario Presets](#4-exploring-scenario-presets)
5. [Model Training & Retraining Workflow (Python)](#5-model-training--retraining-workflow-python)
6. [Troubleshooting & Frequently Asked Questions](#6-troubleshooting--frequently-asked-questions)

---

## 1. Overview & Philosophy

**PriceDrift** bridges rigorous, from-scratch Machine Learning (built with NumPy in Python) with a high-performance, zero-latency static web frontend written in vanilla JavaScript (HTML5 Canvas and ES6 modules).

### Key Architectural Concepts:
- **Offline Training (Python)**: All gradient descent optimization, standardization parameter fitting, and data leakage checks occur offline prior to deployment.
- **Client-Side Real-Time Inference (JavaScript)**: Moving any slider immediately executes normalized vector dot products and sigmoid activation directly in the client's browser in less than 0.1 milliseconds. No network requests are made.
- **Pedagogical Convergence Replay**: The training convergence chart replays the exact iterative loss trajectory recorded during Python gradient descent, allowing learners and analysts to visually observe the rate of convergence without running heavy computation on mobile or battery-constrained client devices.

---

## 2. Interface Walkthrough

```
+---------------------------------------------------------------------------------------+
|  [PD] PriceDrift (Python ML + JS UI)          [⚠️ Synthetic Data Disclosure] [About & Math] |
+---------------------------------------------------------------------------------------+
|  [Model Controls]                   |  [Live Convergence Replay Canvas]               |
|  - Radio: Both | Linear | Logistic  |  - High-DPI Canvas Rendering Cost vs Epoch      |
|  - L2 Regularization Toggle         |  - Area fills, glow effects, legend, axis ticks |
|  - Replay Speed: 1x, 2x, 5x         |                                                 |
|  - Buttons: Replay, Pause, Skip     |                                                 |
+---------------------------------------------------------------------------------------+
|  [Interactive Property Valuation & Speed Estimator]                                   |
|  - Scenarios: [Median Home] [Newer & Spacious] [Luxury Suburban] [Older Dense Urban] |
|  - 5 Feature Sliders with Synchronized Numeric Inputs & Min/Avg/Max Boundaries        |
|  - Valuation Output Card: Predicted Price ($) vs. Dataset Median                      |
|  - Fast-Sale Output Card: Probability Gauge (%), Decision Threshold, Dynamic Badge    |
+---------------------------------------------------------------------------------------+
|  [Linear Regression: Learned Weights] |  [Logistic Regression: Confusion Matrix]      |
|  - Test R², RMSE ($), MSE             |  - Accuracy, Precision, Recall, F1 Metrics    |
|  - Feature Importance Bar Chart       |  - 2x2 Interactive Heat Grid (TP/FP/TN/FN)    |
|    (Standardized & Dollar Impact)     |  - Dynamic Threshold Slider (θ = 0.05 - 0.95) |
+---------------------------------------------------------------------------------------+
```

### 2.1 Header & Disclosure Bar
- **Brand Title**: Displays the application name and current architecture badge (`Python ML + JS UI`).
- **Synthetic Data Disclosure Pill (`⚠️ Synthetic Data Disclosure`)**: Clicking this pill triggers a comprehensive modal explaining that the `FastSale` target is an engineered domain heuristic proxy and not recorded historical days-on-market telemetry.
- **About & Math Button**: Opens the full documentation modal containing dataset schemas, mathematical definitions (MSE, BCE, Sigmoid, Ridge), and preprocessing details.

---

### 2.2 Model Controls & Convergence Replay Panel
Located at the top-left of the dashboard:
1. **Model Selection Tabs**:
   - **Both Models**: Replays and visualizes convergence trajectories for both Linear and Logistic Regression simultaneously.
   - **Linear (Price)**: Focuses exclusively on the continuous price estimation model (Mean Squared Error).
   - **Logistic (Fast)**: Focuses exclusively on the binary velocity classification model (Binary Cross-Entropy).
2. **L2 Regularization Toggle**:
   - Switches the active parameter set between the **Unregularized Baseline ($\lambda=0$)** and **L2 Regularized Model (Linear Ridge $\lambda=50$, Logistic $\lambda=10$)**.
   - Watch the weights and cost curves adjust dynamically upon switching.
3. **Animation Speed**:
   - Dropdown options: `1x Normal Speed` (smooth 60fps pacing), `2x Fast Speed`, or `5x High Speed`.
4. **Action Buttons**:
   - **Replay Training (`▶`)**: Restarts the canvas convergence animation from Epoch 1.
   - **Pause (`⏸`)**: Freezes the animation loop at the current epoch. Clicking again resumes replay.
   - **Skip to End (`⏭`)**: Instantly advances the chart to the final trained epoch.
   - **Clear Chart (`↺`)**: Clears the canvas viewport.

---

### 2.3 Live Canvas Convergence Replay
Located at the top-right:
- Renders an auto-scaling 2D coordinate grid with precise Y-axis (Cost / Loss) and X-axis (Epoch) labels.
- **Linear Regression Series** is rendered in glowing cyan (`#38bdf8`).
- **Logistic Regression Series** is rendered in glowing purple (`#c084fc`).
- Both curves feature dynamic gradient fills beneath the trajectory and an active pulse indicator marking the current frontier of the optimization run.

---

### 2.4 Interactive Property Valuation & Fast-Sale Estimator
This is the core interactive sandbox:
- **Preset Buttons**: Quickly inject representative archetypes into the sliders:
  - `Median Home`: Defaults all features to their exact training-split medians.
  - `Newer & Spacious`: Low house age with generous room counts and solid income.
  - `Luxury Suburban`: High-income, large room count, high-density affluent area.
  - `Older Dense Urban`: High population density, older construction, lower room counts.
- **Feature Sliders & Synchronized Numeric Inputs**:
  1. **Avg. Area Income**: Range \$20,000 – \$120,000 (step \$500).
  2. **Avg. Area House Age**: Range 1.0 – 12.0 years (step 0.1 yrs).
  3. **Avg. Area Number of Rooms**: Range 3.0 – 12.0 rooms (step 0.1).
  4. **Avg. Area Number of Bedrooms**: Range 1.0 – 8.0 bedrooms (step 0.1).
  5. **Area Population**: Range 5,000 – 75,000 residents (step 500).
- **Valuation Output Card (Linear Regression)**:
  - Displays the estimated home price in US Dollars (e.g., `$1,234,567`).
  - Displays difference relative to the training median price (\$1,232,084) with percentage markup or markdown.
- **Fast-Sale Output Card (Logistic Regression)**:
  - Displays the calculated probability $P(\text{FastSale})$ (e.g., `78%`).
  - **Dynamic Badge**: Toggles between `⚡ FAST SALE` (emerald badge) and `⏳ NORMAL / SLOW SALE` (slate badge) based on whether the probability exceeds the active decision threshold $\theta$.
  - **Visual Probability Track**: Shows the fill percentage alongside an amber vertical tick mark denoting the position of the decision threshold.

---

### 2.5 Model Insights & Diagnostic Panels
Located at the bottom:
1. **Linear Regression Weights & Coefficients**:
   - Displays **Test $R^2$** (`0.9148`), **Test RMSE** (`$100,224.13`), and **Test MSE**.
   - Displays a centered horizontal bar chart showing both the normalized weight $w_j$ and the converted dollar impact per natural unit:
     - Income: `+$21.48 per $1 income`
     - House Age: `+$165,116 per year of age`
     - Number of Rooms: `+$121,080 per room`
     - Number of Bedrooms: `+$2,482 per bedroom`
     - Population: `+$15.19 per person`
2. **Logistic Regression Confusion Matrix**:
   - Evaluated on **1,000 unseen test samples**.
   - Displays real-time **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
   - Displays a 2x2 heat-shaded grid:
     - **True Positive (TP)**: Actual Fast, Predicted Fast
     - **False Negative (FN)**: Actual Fast, Predicted Normal
     - **False Positive (FP)**: Actual Normal, Predicted Fast
     - **True Negative (TN)**: Actual Normal, Predicted Normal
   - **Interactive Threshold Slider ($\theta \in [0.05, 0.95]$)**: Dragging this slider reclassifies all 1,000 test predictions on-the-fly and updates the confusion matrix, metrics, and live inference badge simultaneously.

---

## 3. Interpreting Predictions

### Continuous Sale Price Prediction
The predicted price is calculated via:
$$\hat{y}_{\text{USD}} = \left( \sum_{j=1}^5 w_j \cdot \frac{x_j - \mu_j}{\sigma_j} + b \right) \cdot \sigma_y + \mu_y$$
Where $\mu_j, \sigma_j$ and $\mu_y, \sigma_y$ are the means and standard deviations fit strictly on the 80% training partition.

- **Economic Baseline**: If all sliders are set to their training mean values, all normalized features become 0, and the model outputs the training mean price of **\$1,232,084**.
- **Dominant Predictors**: `Avg. Area Income` ($w \approx +0.647$) and `Avg. Area House Age` ($w \approx +0.468$) have the strongest positive elasticity on price.

---

### Fast-Sale Probability & Binary Classification
The model outputs the conditional probability:
$$P(\text{FastSale} = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}_{\text{norm}} + b)$$

Properties predicted with $P \ge \theta$ are classified as fast-moving assets.

---

### Decision Threshold Trade-Off Analysis
By adjusting the threshold slider $\theta$ in the Confusion Matrix panel:
- **Lowering Threshold ($\theta \to 0.20$)**:
  - **High Recall Strategy**: Minimizes False Negatives (FN). Ensures almost every potential fast-selling property is flagged, useful for investors hunting for undervalued opportunities who don't mind vetting extra candidates.
  - **Trade-off**: Increases False Positives (FP) and lowers Precision.
- **Raising Threshold ($\theta \to 0.80$)**:
  - **High Precision Strategy**: Minimizes False Positives (FP). Only properties with overwhelming confidence are flagged as fast-sellers.
  - **Trade-off**: Increases False Negatives (FN) and lowers Recall.
- **Balanced Default ($\theta = 0.50$)**:
  - Delivers balanced performance with **89.70% Accuracy**, **88.09% Precision**, **77.71% Recall**, and an **F1 Score of 0.8257**.

---

## 4. Exploring Scenario Presets

| Scenario | Income | House Age | Rooms | Bedrooms | Population | Est. Price | $P(\text{Fast})$ | Classification ($\theta=0.5$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Median Home** | \$68,800 | 5.97 yrs | 7.0 | 4.0 | 36,200 | ~\$1,228,000 | ~28% | ⏳ Normal / Slow |
| **Newer & Spacious** | \$85,000 | 3.50 yrs | 8.5 | 4.5 | 32,000 | ~\$1,375,000 | ~88% | ⚡ Fast Sale |
| **Luxury Suburban** | \$110,000 | 5.00 yrs | 9.5 | 5.0 | 45,000 | ~\$2,175,000 | ~6% | ⏳ Normal / Slow (High PPR) |
| **Older Dense Urban** | \$45,000 | 9.00 yrs | 5.0 | 2.5 | 55,000 | ~\$760,000 | ~1% | ⏳ Normal / Slow (High Age) |

---

## 5. Model Training & Retraining Workflow (Python)

If you wish to modify hyperparameters, alter regularization penalties, or retrain the pipeline on revised data:

### Step 1: Set Up Python Virtual Environment
```bash
# Optional: Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install requirements
pip install -r ml/requirements.txt
```

### Step 2: Configure Hyperparameters in `ml/train.py`
Open [`ml/train.py`](file:///c:/Users/USER/OneDrive%20-%20Dawood%20University%20of%20Engineering%20Technology/Desktop/PriceDrift/ml/train.py) and modify:
```python
# Linear Regression parameters
lin_base = LinearRegression(learning_rate=0.05, epochs=200, use_regularization=False)

# Ridge Linear Regression parameters
lin_reg = LinearRegression(learning_rate=0.05, epochs=200, use_regularization=True, lambda_reg=50.0)

# Logistic Regression parameters
log_base = LogisticRegression(learning_rate=0.1, epochs=250, use_regularization=False, threshold=0.5)
```

### Step 3: Run the Training Pipeline
```bash
python ml/train.py
```
This automatically updates [`scripts/data/housing-data.json`](file:///c:/Users/USER/OneDrive%20-%20Dawood%20University%20of%20Engineering%20Technology/Desktop/PriceDrift/scripts/data/housing-data.json) with newly fitted weights, biases, cost trajectories, and out-of-sample test metrics.

### Step 4: Validate with Pytest
```bash
pytest ml/tests/ -v
```

### Step 5: Verify the Terminal Report
```bash
python ml/evaluate.py
```

---

## 6. Troubleshooting & Frequently Asked Questions

### Q1: Why does the prediction update immediately without a "Calculate" button?
**A**: PriceDrift implements zero-latency client-side inference. Because inference involves only evaluating a 5-element dot product ($\mathbf{w}^T \mathbf{x} + b$) and a single scalar sigmoid function, evaluation completes in less than 0.05ms per frame.

### Q2: Why are the convergence curves identical every time I click "Replay"?
**A**: The application follows **Replay Mode**. The convergence curves represent the exact, deterministic gradient descent path calculated by Python NumPy offline. This guarantees reproducibility and eliminates the risk of high-CPU battery drain on mobile devices.

### Q3: Why is the FastSale label heuristic called "synthetic"?
**A**: The Kaggle USA Housing dataset does not include transaction dates or listing duration fields. `FastSale` was engineered as a domain proxy heuristic: houses priced below the room median that are newer than median construction age sell faster in real-world liquid markets. We disclose this transparently to prevent students and users from misinterpreting this pedagogical proxy as real-world market transaction history.

### Q4: How do I test the app offline without internet access?
**A**: Simply run a local static server:
```bash
python -m http.server 3000
```
Open `http://localhost:3000` in any web browser. All fonts, styles, math scripts, and datasets are 100% self-contained with zero external CDN dependencies.
