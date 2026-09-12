// scripts/models/logisticRegression.js
// Logistic Regression with Batch Gradient Descent & L2 Regularization
// Implemented strictly from scratch in vanilla JavaScript.

import { matVecMul, matTransposeVecMul, sigmoid, binaryCrossEntropy, confusionMatrix } from '../utils/math.js';

export class LogisticRegression {
  /**
   * @param {Object} options
   * @param {number} [options.learningRate=0.1]
   * @param {number} [options.epochs=250]
   * @param {boolean} [options.useRegularization=false]
   * @param {number} [options.lambda=1.0] - L2 penalty
   * @param {number} [options.threshold=0.5] - Decision boundary
   */
  constructor(options = {}) {
    this.learningRate = options.learningRate ?? 0.1;
    this.epochs = options.epochs ?? 250;
    this.useRegularization = options.useRegularization ?? false;
    this.lambda = options.lambda ?? 1.0;
    this.threshold = options.threshold ?? 0.5;

    this.weights = [];
    this.bias = 0;
    this.history = []; // Array of { epoch, cost }
    this.isDiverged = false;
    this.currentEpoch = 0;
    this.isTraining = false;
  }

  /**
   * Initializes weights and bias.
   * @param {number} numFeatures 
   */
  initializeWeights(numFeatures) {
    this.weights = new Array(numFeatures).fill(0);
    this.bias = 0;
    this.history = [];
    this.isDiverged = false;
    this.currentEpoch = 0;
  }

  /**
   * Predict probability p = sigmoid(X * w + b)
   * @param {number[][]} X - Shape [m, d]
   * @returns {number[]} Array of probabilities in (0, 1)
   */
  predictProbs(X) {
    const xw = matVecMul(X, this.weights);
    const m = X.length;
    const probs = new Array(m);
    for (let i = 0; i < m; i++) {
      probs[i] = sigmoid(xw[i] + this.bias);
    }
    return probs;
  }

  /**
   * Predict binary classes based on specified or current decision threshold.
   * @param {number[][]} X 
   * @param {number} [threshold] 
   * @returns {number[]}
   */
  predictClasses(X, threshold) {
    const th = threshold !== undefined ? threshold : this.threshold;
    const probs = this.predictProbs(X);
    return probs.map(p => (p >= th ? 1 : 0));
  }

  /**
   * Computes binary cross entropy cost.
   * @param {number[]} yProbs 
   * @param {number[]} yTrue 
   * @returns {number}
   */
  computeCost(yProbs, yTrue) {
    return binaryCrossEntropy(
      yTrue,
      yProbs,
      this.weights,
      this.useRegularization ? this.lambda : 0
    );
  }

  /**
   * Executes a single epoch of batch gradient descent.
   * @param {number[][]} X - Shape [m, d]
   * @param {number[]} y - Shape [m], values 0 or 1
   * @returns {{ epoch: number, cost: number, diverged: boolean }}
   */
  trainStep(X, y) {
    const m = X.length;
    const d = X[0].length;

    if (this.weights.length !== d) {
      this.initializeWeights(d);
    }

    // 1. Hypothesis: probabilities p_hat = sigmoid(X * w + b)
    const yProbs = this.predictProbs(X);

    // 2. Compute cost
    const cost = this.computeCost(yProbs, y);

    // Divergence check
    if (isNaN(cost) || !isFinite(cost) || cost > 1e6) {
      this.isDiverged = true;
      return { epoch: this.currentEpoch, cost: NaN, diverged: true };
    }

    // 3. Error vector: p_hat - y
    const errors = new Array(m);
    let sumError = 0;
    for (let i = 0; i < m; i++) {
      const err = yProbs[i] - y[i];
      errors[i] = err;
      sumError += err;
    }

    // 4. Weight gradients: (1 / m) * X^T * errors + (lambda / m) * w
    const gradW = matTransposeVecMul(X, errors);
    const regFactor = (this.useRegularization && this.lambda > 0) ? (this.lambda / m) : 0;

    for (let j = 0; j < d; j++) {
      const g = (gradW[j] / m) + (regFactor * this.weights[j]);
      this.weights[j] -= this.learningRate * g;
    }

    // 5. Bias gradient: (1 / m) * sum(errors)
    const gradB = sumError / m;
    this.bias -= this.learningRate * gradB;

    this.currentEpoch++;
    const stepRecord = { epoch: this.currentEpoch, cost, diverged: false };
    this.history.push(stepRecord);

    return stepRecord;
  }

  /**
   * Synchronous training for all epochs.
   * @param {number[][]} X 
   * @param {number[]} y 
   */
  train(X, y) {
    const d = X[0].length;
    this.initializeWeights(d);
    for (let e = 0; e < this.epochs; e++) {
      const res = this.trainStep(X, y);
      if (res.diverged) break;
    }
  }

  /**
   * Inference for a single standardized feature vector.
   * @param {number[]} xStd 
   * @returns {{ probability: number, isFastSale: boolean, threshold: number }}
   */
  predictSample(xStd, customThreshold) {
    let z = this.bias;
    for (let j = 0; j < this.weights.length; j++) {
      z += this.weights[j] * xStd[j];
    }
    const probability = sigmoid(z);
    const th = customThreshold !== undefined ? customThreshold : this.threshold;
    return {
      probability,
      isFastSale: probability >= th,
      threshold: th
    };
  }

  /**
   * Full evaluation on test set.
   * @param {number[][]} XTest 
   * @param {number[]} yTest 
   * @param {number} [customThreshold] 
   */
  evaluate(XTest, yTest, customThreshold) {
    const th = customThreshold !== undefined ? customThreshold : this.threshold;
    const probs = this.predictProbs(XTest);
    const preds = probs.map(p => (p >= th ? 1 : 0));
    const metrics = confusionMatrix(yTest, preds);
    const cost = this.computeCost(probs, yTest);

    return {
      cost,
      threshold: th,
      ...metrics,
      weights: [...this.weights],
      bias: this.bias
    };
  }
}
