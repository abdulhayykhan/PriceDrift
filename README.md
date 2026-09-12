# PriceDrift — Browser-Based Housing Analytics & ML from Scratch

**PriceDrift** is a client-side housing analytics tool that implements **Linear Regression** (for continuous sale price estimation) and **Logistic Regression** (for binary fast-sale classification) strictly from scratch in pure vanilla JavaScript (ES6 modules).

**Zero ML libraries. Zero external math libraries. Zero backend.**

---

## Key Features

1. **Dual Scratch ML Models**:
   - **Linear Regression**: Predicts continuous sale price via vectorized Batch Gradient Descent optimizing Mean Squared Error (MSE), with optional L2 (Ridge) regularization.
   - **Logistic Regression**: Classifies probability of a property selling fast via vectorized Batch Gradient Descent optimizing Binary Cross-Entropy (BCE), with dynamic classification threshold adjustment and optional L2 penalty.
2. **Live Training Convergence Visualization**:
   - Real-time HTML5 `<canvas>` charting cost vs. epoch at 60 FPS.
   - Uses non-blocking chunked animation loops (`requestAnimationFrame`) so the UI thread remains responsive.
   - Supports live Pause, Resume, Reset, and Instant Train modes.
3. **Interactive Inference & Synchronized Normalization**:
   - Sliders and synchronized inputs for 5 housing features.
   - Features are standardized on-the-fly using the **exact same z-score ($\mu$, $\sigma$) parameters** computed during training before feeding into the models.
   - Displays real-time estimated price (in USD) and fast-sale probability (%) with an animated threshold marker.
4. **Model Insights & Diagnostics**:
   - Interactive 2&times;2 Confusion Matrix ($TP, FP, TN, FN$) with dynamic threshold slider ($\theta \in [0.05, 0.95]$) and live metric cards (Accuracy, Precision, Recall, F1).
   - Feature Coefficients Bar Chart displaying both normalized importance weights and unstandardized dollar impact per unit of feature.
   - Automatic divergence detection: alerts user if learning rate is set too high.
5. **Synthetic Data Disclosure**:
   - In-app "About This Data" modal disclosing the Kaggle USA Housing dataset and the engineered heuristic proxy used for the `FastSale` target.

---

## Dataset & FastSale Label Engineering

- **Base Dataset**: Kaggle USA Housing dataset (5,000 samples).
- **Features**:
  1. `Avg. Area Income`
  2. `Avg. Area House Age`
  3. `Avg. Area Number of Rooms`
  4. `Avg. Area Number of Bedrooms`
  5. `Area Population`
- **Dropped**: `Address` (non-numeric).
- **Split**: 80% Train (4,000 samples) / 20% Test (1,000 samples) randomized with Mulberry32 fixed seed (42).

### FastSale Heuristic Formula (Disclosed)
Because the dataset does not have time-on-market telemetry, the `FastSale` binary target is engineered as a domain heuristic proxy:
1. Compute $\text{PricePerRoom} = \frac{\text{Price}}{\text{Avg. Area Rooms}}$
2. Compute median $\text{PricePerRoom}$ ($\approx \$176,684$) and median $\text{Avg. Area House Age}$ ($\approx 5.97$ years)
3. Label:
   $$\text{FastSale} = \begin{cases} 1 & \text{if } \text{PricePerRoom} < \text{median}(\text{PricePerRoom}) \land \text{HouseAge} < \text{median}(\text{HouseAge}) \\ 0 & \text{otherwise} \end{cases}$$
   This accounts for 1,681 properties (33.62% positive class).

---

## Mathematical Architecture

All linear algebra, loss functions, and gradients are implemented manually in `scripts/utils/math.js`.

### 1. Z-Score Standardization
$$x_{\text{norm}, j} = \frac{x_j - \mu_j}{\sigma_j}$$

### 2. Linear Regression (Batch Gradient Descent)
- **Hypothesis**:
  $$h(x) = X w + b$$
- **Cost Function (MSE with L2 Ridge)**:
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$
- **Gradients**:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w$$
  $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

### 3. Logistic Regression (Batch Gradient Descent)
- **Sigmoid Function**:
  $$\sigma(z) = \frac{1}{1 + e^{-\text{clamp}(z, -30, 30)}}$$
- **Hypothesis**:
  $$\hat{p} = \sigma(X w + b)$$
- **Cost Function (Binary Cross-Entropy with L2)**:
  $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \ln(\hat{p}^{(i)} + \epsilon) + (1 - y^{(i)}) \ln(1 - \hat{p}^{(i)} + \epsilon)] + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$
- **Gradients**:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{p} - y) + \frac{\lambda}{m} w$$
  $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{p}^{(i)} - y^{(i)})$$

---

## File Structure

```
pricedrift/
├── index.html                           # Single-page semantic HTML UI
├── vercel.json                          # Static hosting deployment configuration
├── package.json                         # Node module configuration & test scripts
├── styles/
│   └── main.css                         # Dark theme responsive stylesheet
├── scripts/
│   ├── utils/
│   │   └── math.js                      # Scratch vector/matrix math, sigmoid, losses, metrics
│   ├── data/
│   │   ├── preprocess.js                # CSV parser, z-score, split, label engineer
│   │   └── housing-data.json            # Precomputed, normalized dataset + split
│   ├── models/
│   │   ├── linearRegression.js          # Linear regression class with batch GD & L2
│   │   └── logisticRegression.js        # Logistic regression class with batch GD & L2
│   ├── viz/
│   │   ├── trainingChart.js             # Canvas cost-vs-epoch live animation
│   │   ├── confusionMatrix.js           # 2x2 confusion matrix with dynamic threshold
│   │   └── coefficientsChart.js         # Feature weights / importance bar chart
│   ├── ui/
│   │   ├── controls.js                  # Hyperparameter sliders, buttons, state management
│   │   ├── predictionPanel.js           # Feature sliders, real-time normalization, inference
│   │   └── modal.js                     # Disclosure & methodology modal dialog
│   └── main.js                          # App orchestrator
├── tests/
│   └── models.test.js                   # Node test suite verifying convergence & math
└── README.md
```

---

## Local Development & Testing

### 1. Run Automated Test Suite
Verify that gradient descent converges and achieves $R^2 > 0.85$ and Accuracy $> 75\%$:
```bash
npm test
```

### 2. Re-run Preprocessing (Optional)
If you wish to re-generate `housing-data.json` from `dataset/USA_Housing.csv`:
```bash
npm run preprocess
```

### 3. Launch the Local Web App
Because modern browsers restrict ES module fetching on `file://` URLs, serve the folder via any static web server:
```bash
# Python
python -m http.server 3000

# or Node.js npx
npx serve .
```
Then visit `http://localhost:3000` in your web browser.

---

## Deployment to Vercel

PriceDrift is a 100% static client-side web application. To deploy:

1. **Via Vercel CLI**:
   ```bash
   vercel --prod
   ```
2. **Via Git / GitHub**:
   - Push this repository to GitHub/GitLab.
   - Import the project into your Vercel Dashboard.
   - Keep default settings (Framework: Other / Static, Output Directory: `./`).
   - Click **Deploy**.
