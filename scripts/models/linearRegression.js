// scripts/models/linearRegression.js
// Linear Regression with Batch Gradient Descent & L2 Regularization (Ridge)
// Implemented strictly from scratch in vanilla JavaScript.

import { matVecMul, matTransposeVecMul, rSquared, mse, rmse } from '../utils/math.js';

export class LinearRegression {
  /**
   * @param {Object} options
   * @param {number} [options.learningRate=0.05]
   * @param {number} [options.epochs=200]
   * @param {boolean} [options.useRegularization=false]
   * @param {number} [options.lambda=1.0] - L2 regularization penalty parameter
   */
  constructor(options = {}) {
    this.learningRate = options.learningRate ?? 0.05;
    this.epochs = options.epochs ?? 200;
    this.useRegularization = options.useRegularization ?? false;
    this.lambda = options.lambda ?? 1.0;

    this.weights = []; // Length d
    this.bias = 0;
    this.history = []; // Array of { epoch, cost }
    this.isDiverged = false;
    this.currentEpoch = 0;
    this.isTraining = false;
  }

  /**
   * Initializes weights and bias to zeros or small random numbers.
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
   * Forward pass: computes predictions y_hat = X * w + b.
   * @param {number[][]} X - Matrix of shape [m, d]
   * @returns {number[]} Vector of predictions of length m
   */
  predictRaw(X) {
    const xw = matVecMul(X, this.weights);
    const m = X.length;
    const yHat = new Array(m);
    for (let i = 0; i < m; i++) {
      yHat[i] = xw[i] + this.bias;
    }
    return yHat;
  }

  /**
   * Computes Mean Squared Error with optional L2 regularization.
   * J(w, b) = (1 / 2m) * sum((y_hat - y)^2) + (lambda / 2m) * sum(w^2)
   * @param {number[]} yHat 
   * @param {number[]} yTrue 
   * @returns {number}
   */
  computeCost(yHat, yTrue) {
    const m = yTrue.length;
    if (m === 0) return 0;

    let sumSqErr = 0;
    for (let i = 0; i < m; i++) {
      const err = yHat[i] - yTrue[i];
      sumSqErr += err * err;
    }

    let cost = sumSqErr / (2 * m);

    if (this.useRegularization && this.lambda > 0) {
      let l2Sum = 0;
      for (let j = 0; j < this.weights.length; j++) {
        l2Sum += this.weights[j] * this.weights[j];
      }
      cost += (this.lambda / (2 * m)) * l2Sum;
    }

    return cost;
  }

  /**
   * Executes a single epoch of vectorized batch gradient descent.
   * @param {number[][]} X - Shape [m, d]
   * @param {number[]} y - Shape [m]
   * @returns {{ epoch: number, cost: number }}
   */
  trainStep(X, y) {
    const m = X.length;
    const d = X[0].length;

    if (this.weights.length !== d) {
      this.initializeWeights(d);
    }

    // 1. Forward hypothesis: y_hat = X * w + b
    const yHat = this.predictRaw(X);

    // 2. Compute cost
    const cost = this.computeCost(yHat, y);

    // Guard against NaN or explosion
    if (isNaN(cost) || !isFinite(cost) || cost > 1e10) {
      this.isDiverged = true;
      return { epoch: this.currentEpoch, cost: NaN, diverged: true };
    }

    // 3. Compute error vector: (y_hat - y)
    const errors = new Array(m);
    let sumError = 0;
    for (let i = 0; i < m; i++) {
      const err = yHat[i] - y[i];
      errors[i] = err;
      sumError += err;
    }

    // 4. Compute gradient w.r.t weights: (1 / m) * X^T * errors + (lambda / m) * w
    const gradW = matTransposeVecMul(X, errors);
    const regFactor = (this.useRegularization && this.lambda > 0) ? (this.lambda / m) : 0;

    for (let j = 0; j < d; j++) {
      const g = (gradW[j] / m) + (regFactor * this.weights[j]);
      this.weights[j] -= this.learningRate * g;
    }

    // 5. Compute gradient w.r.t bias: (1 / m) * sum(errors)
    const gradB = sumError / m;
    this.bias -= this.learningRate * gradB;

    this.currentEpoch++;
    const stepRecord = { epoch: this.currentEpoch, cost, diverged: false };
    this.history.push(stepRecord);

    return stepRecord;
  }

  /**
   * Synchronously trains all epochs (ideal for quick tests/benchmarks).
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
   * Predict single standardized feature vector.
   * @param {number[]} xStd 
   * @returns {number} Normalized prediction
   */
  predictSample(xStd) {
    let sum = this.bias;
    for (let j = 0; j < this.weights.length; j++) {
      sum += this.weights[j] * xStd[j];
    }
    return sum;
  }

  /**
   * Full evaluation on test dataset.
   * Computes R^2, MSE, RMSE (both normalized and unscaled in original USD).
   * @param {number[][]} XTestStd 
   * @param {number[]} yTestNorm 
   * @param {number[]} yTestRaw 
   * @param {{ mean: number, stdDev: number }} targetStats 
   */
  evaluate(XTestStd, yTestNorm, yTestRaw, targetStats) {
    const yPredNorm = this.predictRaw(XTestStd);
    const r2Norm = rSquared(yTestNorm, yPredNorm);
    const mseNorm = mse(yTestNorm, yPredNorm);

    // Unscale to raw USD
    const yPredRaw = yPredNorm.map(val => val * targetStats.stdDev + targetStats.mean);
    const r2Raw = rSquared(yTestRaw, yPredRaw);
    const mseRaw = mse(yTestRaw, yPredRaw);
    const rmseRaw = rmse(yTestRaw, yPredRaw);

    return {
      r2: r2Raw,
      mse: mseRaw,
      rmse: rmseRaw,
      normalizedMSE: mseNorm,
      weights: [...this.weights],
      bias: this.bias
    };
  }
}
