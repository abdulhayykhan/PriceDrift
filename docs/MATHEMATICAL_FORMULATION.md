# PriceDrift — Mathematical Formulations & Optimization Derivations

This document provides a comprehensive, graduate-level reference for all mathematical models, matrix calculus derivations, loss functions, and optimization routines implemented from scratch in **PriceDrift**.

---

## Table of Contents
1. [Notation & Conventions](#1-notation--conventions)
2. [Feature Standardization (Z-Score) & Conditioning](#2-feature-standardization-z-score--conditioning)
3. [Linear Regression via Vectorized Batch Gradient Descent](#3-linear-regression-via-vectorized-batch-gradient-descent)
   - [Hypothesis Formulation](#hypothesis-formulation)
   - [Mean Squared Error (MSE) with L2 Ridge Penalty](#mean-squared-error-mse-with-l2-ridge-penalty)
   - [Matrix Calculus Gradient Derivations](#matrix-calculus-gradient-derivations)
   - [Iterative Gradient Descent vs. Normal Equation](#iterative-gradient-descent-vs-normal-equation)
4. [Logistic Regression for Binary Market Velocity](#4-logistic-regression-for-binary-market-velocity)
   - [The Sigmoid Link Function](#the-sigmoid-link-function)
   - [Maximum Likelihood Estimation & Cross-Entropy](#maximum-likelihood-estimation--cross-entropy)
   - [Binary Cross-Entropy (BCE) Gradient Derivation](#binary-cross-entropy-bce-gradient-derivation)
   - [Decision Hyperplane & Threshold Theory](#decision-hyperplane--threshold-theory)
5. [Numerical Stability & Divergence Protections](#5-numerical-stability--divergence-protections)
6. [Statistical Evaluation Metrics](#6-statistical-evaluation-metrics)

---

## 1. Notation & Conventions

- Let $m$ denote the number of training samples ($m = 4,000$).
- Let $d$ denote the number of input features ($d = 5$).
- Let $\mathbf{X} \in \mathbb{R}^{m \times d}$ denote the design matrix whose $i$-th row $\mathbf{x}^{(i)T}$ represents sample $i$.
- Let $\mathbf{y} \in \mathbb{R}^m$ denote the ground-truth target vector.
- Let $\mathbf{w} \in \mathbb{R}^d$ denote the parameter weight vector.
- Let $b \in \mathbb{R}$ denote the scalar bias (intercept).
- Let $\mathbf{1}_m \in \mathbb{R}^m$ denote a column vector of ones: $[1, 1, \dots, 1]^T$.
- Let $\alpha \in \mathbb{R}^+$ denote the gradient descent learning rate.
- Let $\lambda \in \mathbb{R}^{\ge 0}$ denote the L2 regularization coefficient.

---

## 2. Feature Standardization (Z-Score) & Conditioning

### 2.1 The Need for Standardization
The USA Housing dataset features span radically disparate physical scales:
- `Avg. Area Income`: $\sim \$68,000 \pm \$10,600$ (Order $10^4$)
- `Avg. Area House Age`: $\sim 5.97 \pm 0.99$ years (Order $10^0$)
- `Avg. Area Number of Rooms`: $\sim 6.98 \pm 1.00$ rooms (Order $10^0$)
- `Area Population`: $\sim 36,000 \pm 9,900$ residents (Order $10^4$)

Without feature scaling, the eigenvalues of the Hessian matrix $\mathbf{H} = \frac{1}{m} \mathbf{X}^T \mathbf{X}$ exhibit an extreme condition number ($\kappa(\mathbf{H}) \gg 10^8$). The loss contours become highly eccentric, elongated ellipsoids. Batch gradient descent oscillates severely along steep dimensions while making imperceptible progress along flat dimensions, requiring an impractically small learning rate ($\alpha \le 10^{-9}$) that prevents practical convergence.

### 2.2 Standardization Transformation
For each feature $j \in \{1, \dots, d\}$, the sample mean $\mu_j$ and sample standard deviation $\sigma_j$ are fit **strictly on the training split**:
$$\mu_j = \frac{1}{m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} X_{i, j}$$
$$\sigma_j = \sqrt{\frac{1}{m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} (X_{i, j} - \mu_j)^2}$$

The standardized feature $z_{i, j}$ is computed as:
$$z_{i, j} = \frac{X_{i, j} - \mu_j}{\sigma_j}$$

**Theorem (Unit Variance & Zero Mean)**:
$$\mathbb{E}[z_j] = \frac{1}{m} \sum_{i=1}^m \frac{X_{i, j} - \mu_j}{\sigma_j} = \frac{1}{\sigma_j} \left( \frac{1}{m} \sum_{i=1}^m X_{i, j} - \mu_j \right) = 0$$
$$\text{Var}(z_j) = \frac{1}{m} \sum_{i=1}^m (z_{i, j} - 0)^2 = \frac{1}{\sigma_j^2} \left( \frac{1}{m} \sum_{i=1}^m (X_{i, j} - \mu_j)^2 \right) = \frac{\sigma_j^2}{\sigma_j^2} = 1$$

This sphericalizes the Hessian loss contours ($\kappa(\mathbf{H}) \approx 1$), allowing gradient descent to descend monotonically toward the global minimum with $\alpha \in [0.01, 0.1]$.

---

## 3. Linear Regression via Vectorized Batch Gradient Descent

### Hypothesis Formulation
The model predicts continuous home prices through the linear hypothesis:
$$h_{\mathbf{w}, b}(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$
In vectorized notation across all $m$ instances simultaneously:
$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b \mathbf{1}_m$$

---

### Mean Squared Error (MSE) with L2 Ridge Penalty
The objective cost function $J(\mathbf{w}, b)$ combines the empirical Mean Squared Error with a Tikhonov (L2 Ridge) regularization penalty on the weight parameters (the intercept $b$ is deliberately left unpenalized to avoid biasing the baseline price):
$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2$$
In compact matrix norm notation:
$$J(\mathbf{w}, b) = \frac{1}{2m} \|\mathbf{X} \mathbf{w} + b \mathbf{1}_m - \mathbf{y}\|_2^2 + \frac{\lambda}{2m} \|\mathbf{w}\|_2^2$$

---

### Matrix Calculus Gradient Derivations
Let $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y} = \mathbf{X} \mathbf{w} + b \mathbf{1}_m - \mathbf{y}$ define the error residual vector ($\mathbf{e} \in \mathbb{R}^m$).

#### 1. Gradient with respect to weight vector $\mathbf{w}$:
Expanding the cost function:
$$J(\mathbf{w}, b) = \frac{1}{2m} \mathbf{e}^T \mathbf{e} + \frac{\lambda}{2m} \mathbf{w}^T \mathbf{w}$$
Applying the vector chain rule:
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \left( \frac{\partial \mathbf{e}}{\partial \mathbf{w}} \right)^T \mathbf{e} + \frac{\lambda}{m} \mathbf{w}$$
Since $\frac{\partial \mathbf{e}}{\partial \mathbf{w}} = \mathbf{X}$:
$$\nabla_{\mathbf{w}} J = \frac{1}{m} \mathbf{X}^T \left( \mathbf{X} \mathbf{w} + b \mathbf{1}_m - \mathbf{y} \right) + \frac{\lambda}{m} \mathbf{w}$$

#### 2. Gradient with respect to scalar bias $b$:
$$\frac{\partial J}{\partial b} = \frac{1}{m} \left( \frac{\partial \mathbf{e}}{\partial b} \right)^T \mathbf{e} = \frac{1}{m} \mathbf{1}_m^T \mathbf{e} = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)$$

#### 3. Parameter Update Rules:
For each iteration $t = 0, 1, \dots, E-1$:
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \nabla_{\mathbf{w}} J = \mathbf{w}^{(t)} \left(1 - \frac{\alpha \lambda}{m}\right) - \frac{\alpha}{m} \mathbf{X}^T \mathbf{e}^{(t)}$$
$$b^{(t+1)} = b^{(t)} - \alpha \frac{\partial J}{\partial b} = b^{(t)} - \frac{\alpha}{m} \sum_{i=1}^m e_i^{(t)}$$

---

### Iterative Gradient Descent vs. Normal Equation
While Linear Regression with L2 regularization admits an exact analytical closed-form solution (the Normal Equation with Ridge):
$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
PriceDrift explicitly implements **Batch Gradient Descent**:
1. **Algorithmic Transparency**: Allows live visualization of convergence dynamics (cost vs. epoch) and eigenvalue decay.
2. **Computational Scalability**: Avoiding matrix inversion $\mathcal{O}(d^3)$ allows easy scaling to arbitrarily high-dimensional feature spaces.
3. **Consistency**: Provides mathematical parity with non-linear models (such as Logistic Regression) where no closed-form solution exists.

---

## 4. Logistic Regression for Binary Market Velocity

### The Sigmoid Link Function
To model the conditional probability $P(\text{FastSale} = 1 \mid \mathbf{x}) \in (0, 1)$, we map the continuous real logit $z = \mathbf{w}^T \mathbf{x} + b$ through the logistic sigmoid function:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

#### Important Derivative Identity:
$$\frac{d\sigma}{dz} = \frac{d}{dz} (1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} (-e^{-z}) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) (1 - \sigma(z))$$

---

### Maximum Likelihood Estimation & Cross-Entropy
Assuming observations $y^{(i)} \in \{0, 1\}$ are independently conditionally distributed according to a Bernoulli distribution with parameter $p^{(i)} = \sigma(\mathbf{w}^T \mathbf{x}^{(i)} + b)$:
$$P(y^{(i)} \mid \mathbf{x}^{(i)}) = \left( p^{(i)} \right)^{y^{(i)}} \left( 1 - p^{(i)} \right)^{1 - y^{(i)}}$$

The likelihood of the parameter vector across all $m$ independent training instances is:
$$L(\mathbf{w}, b) = \prod_{i=1}^m \left( p^{(i)} \right)^{y^{(i)}} \left( 1 - p^{(i)} \right)^{1 - y^{(i)}}$$

Taking the natural logarithm yields the log-likelihood:
$$\ell(\mathbf{w}, b) = \sum_{i=1}^m \left[ y^{(i)} \ln(p^{(i)}) + (1 - y^{(i)}) \ln(1 - p^{(i)}) \right]$$

Maximizing log-likelihood is equivalent to minimizing the negative log-likelihood normalized by sample count $m$ plus the L2 regularization penalty, yielding the **Binary Cross-Entropy (BCE)** cost function:
$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(p^{(i)}) + (1 - y^{(i)}) \ln(1 - p^{(i)}) \right] + \frac{\lambda}{2m} \|\mathbf{w}\|_2^2$$

---

### Binary Cross-Entropy (BCE) Gradient Derivation
Let $z^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)} + b$, so $p^{(i)} = \sigma(z^{(i)})$.

By the chain rule for the $j$-th feature weight $w_j$:
$$\frac{\partial J}{\partial w_j} = -\frac{1}{m} \sum_{i=1}^m \left[ \frac{y^{(i)}}{p^{(i)}} \frac{\partial p^{(i)}}{\partial w_j} - \frac{1 - y^{(i)}}{1 - p^{(i)}} \frac{\partial p^{(i)}}{\partial w_j} \right] + \frac{\lambda}{m} w_j$$
Since $\frac{\partial p^{(i)}}{\partial w_j} = \frac{\partial \sigma(z^{(i)})}{\partial z^{(i)}} \frac{\partial z^{(i)}}{\partial w_j} = p^{(i)}(1 - p^{(i)}) x_j^{(i)}$:
$$\frac{\partial J}{\partial w_j} = -\frac{1}{m} \sum_{i=1}^m \left[ \frac{y^{(i)}}{p^{(i)}} - \frac{1 - y^{(i)}}{1 - p^{(i)}} \right] p^{(i)}(1 - p^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j$$
Simplifying the term in brackets:
$$\left[ \frac{y^{(i)}(1 - p^{(i)}) - (1 - y^{(i)})p^{(i)}}{p^{(i)}(1 - p^{(i)})} \right] p^{(i)}(1 - p^{(i)}) = y^{(i)} - y^{(i)}p^{(i)} - p^{(i)} + y^{(i)}p^{(i)} = y^{(i)} - p^{(i)}$$

Multiplying by the leading negative sign yields:
$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (p^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j$$

In vectorized matrix form:
$$\nabla_{\mathbf{w}} J = \frac{1}{m} \mathbf{X}^T (\mathbf{p} - \mathbf{y}) + \frac{\lambda}{m} \mathbf{w}$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (p^{(i)} - y^{(i)})$$

Notice that despite the non-linearity of the sigmoid function, the gradient vector has the **exact same structural form** as linear regression error backpropagation: $\frac{1}{m} \mathbf{X}^T \mathbf{e}$.

---

### Decision Hyperplane & Threshold Theory
Given model probability $\hat{p} = \sigma(\mathbf{w}^T \mathbf{x} + b)$ and user threshold $\theta \in (0, 1)$:
$$\hat{y} = \begin{cases} 1 & \text{if } \hat{p} \ge \theta \\ 0 & \text{if } \hat{p} < \theta \end{cases}$$

Since $\sigma(z) \ge \theta \iff z \ge \ln\left(\frac{\theta}{1 - \theta}\right) = \text{logit}(\theta)$:
$$\mathbf{w}^T \mathbf{x} + b \ge \text{logit}(\theta)$$

The decision boundary is the $(d-1)$-dimensional hyperplane in standardized feature space:
$$\mathcal{H} = \left\{ \mathbf{x} \in \mathbb{R}^d \;\middle|\; \mathbf{w}^T \mathbf{x} + \left(b - \ln\frac{\theta}{1 - \theta}\right) = 0 \right\}$$

- Setting $\theta = 0.5 \implies \text{logit}(0.5) = 0$, placing the boundary at $\mathbf{w}^T \mathbf{x} + b = 0$.
- Varying $\theta$ translates the hyperplane orthogonally to $\mathbf{w}$ without altering its orientation, directly modulating the trade-off between False Positives and False Negatives.

---

## 5. Numerical Stability & Divergence Protections

### 1. Sigmoid Overflow Prevention
Floating-point arithmetic overflow occurs in 64-bit IEEE 754 floats when evaluating $\exp(-z)$ for $z < -709$. To guarantee unconditional numerical stability:
$$\sigma_{\text{clamped}}(z) = \sigma(\text{clamp}(z, -30.0, +30.0))$$
Since $\sigma(30) = 1 - 9.35 \times 10^{-14} \approx 1$ and $\sigma(-30) = 9.35 \times 10^{-14} \approx 0$, clamping introduces negligible numerical error ($< 10^{-13}$) while eliminating `OverflowError` and `NaN`.

### 2. Logarithm of Zero Guard (BCE Loss)
When $p \to 0$ or $p \to 1$, $\ln(p)$ or $\ln(1-p)$ evaluates to $-\infty$. The implementation applies epsilon clipping:
$$p_{\text{safe}} = \text{clamp}(p, \epsilon, 1 - \epsilon), \quad \epsilon = 10^{-15}$$

### 3. Divergence Guard
During iterative gradient descent, if the step size $\alpha$ exceeds the Lipschitz constant $\frac{2}{L}$ of the gradient, the parameter updates diverge exponentially. The optimizer inspects $J^{(t)}$ at each epoch:
$$\text{If } J^{(t)} = \text{NaN} \quad \lor \quad J^{(t)} = \infty \quad \lor \quad J^{(t)} > 10^{10} \implies \text{Flag Divergence \& Halt}$$

---

## 6. Statistical Evaluation Metrics

### Linear Regression Metrics
1. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{m_{\text{test}}} \sum_{i=1}^{m_{\text{test}}} (y^{(i)} - \hat{y}^{(i)})^2$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
3. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{m_{\text{test}}} (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{m_{\text{test}}} (y^{(i)} - \bar{y}_{\text{test}})^2}$$

### Logistic Regression Metrics
Given the test set confusion matrix:

| | Predicted Positive ($\hat{y}=1$) | Predicted Negative ($\hat{y}=0$) |
| :--- | :--- | :--- |
| **Actual Positive ($y=1$)** | True Positive ($TP$) | False Negative ($FN$) |
| **Actual Negative ($y=0$)** | False Positive ($FP$) | True Negative ($TN$) |

1. **Accuracy**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}$$
2. **Precision**:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity / True Positive Rate)**:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **$F_1$ Score (Harmonic Mean)**:
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$
