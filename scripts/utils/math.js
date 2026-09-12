// scripts/utils/math.js
// Custom linear algebra, statistics, and loss calculations implemented from scratch in vanilla JS.
// NO ML libraries or external dependencies.

/**
 * Computes the dot product of two vectors of equal length.
 * @param {number[]} u 
 * @param {number[]} v 
 * @returns {number}
 */
export function dot(u, v) {
  let sum = 0;
  for (let i = 0; i < u.length; i++) {
    sum += u[i] * v[i];
  }
  return sum;
}

/**
 * Multiplies a matrix X [m x d] with a vector w [d], returning a vector of length m.
 * @param {number[][]} X 
 * @param {number[]} w 
 * @returns {number[]}
 */
export function matVecMul(X, w) {
  const m = X.length;
  const result = new Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    const row = X[i];
    for (let j = 0; j < w.length; j++) {
      sum += row[j] * w[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * Computes X^T * v where X is [m x d] and v is [m], returning a vector of length d.
 * Equivalent to sum_i(v[i] * X[i, j]) for each feature j.
 * @param {number[][]} X 
 * @param {number[]} v 
 * @returns {number[]}
 */
export function matTransposeVecMul(X, v) {
  const m = X.length;
  if (m === 0) return [];
  const d = X[0].length;
  const result = new Array(d).fill(0);

  for (let i = 0; i < m; i++) {
    const vi = v[i];
    const row = X[i];
    for (let j = 0; j < d; j++) {
      result[j] += vi * row[j];
    }
  }
  return result;
}

/**
 * Vector addition: u + v
 * @param {number[]} u 
 * @param {number[]} v 
 * @returns {number[]}
 */
export function vecAdd(u, v) {
  const res = new Array(u.length);
  for (let i = 0; i < u.length; i++) {
    res[i] = u[i] + v[i];
  }
  return res;
}

/**
 * Vector subtraction: u - v
 * @param {number[]} u 
 * @param {number[]} v 
 * @returns {number[]}
 */
export function vecSub(u, v) {
  const res = new Array(u.length);
  for (let i = 0; i < u.length; i++) {
    res[i] = u[i] - v[i];
  }
  return res;
}

/**
 * Scalar multiplication: c * v
 * @param {number} c 
 * @param {number[]} v 
 * @returns {number[]}
 */
export function scalarMul(c, v) {
  const res = new Array(v.length);
  for (let i = 0; i < v.length; i++) {
    res[i] = c * v[i];
  }
  return res;
}

/**
 * Clamps value to [min, max]
 * @param {number} val 
 * @param {number} min 
 * @param {number} max 
 * @returns {number}
 */
export function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

/**
 * Numerically stable Sigmoid function with clamping to avoid overflow/underflow.
 * @param {number} z 
 * @returns {number}
 */
export function sigmoid(z) {
  const clampedZ = clamp(z, -30, 30);
  return 1 / (1 + Math.exp(-clampedZ));
}

/**
 * Mean of an array of numbers.
 * @param {number[]} arr 
 * @returns {number}
 */
export function mean(arr) {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  return sum / arr.length;
}

/**
 * Variance of an array.
 * @param {number[]} arr 
 * @param {number} [precalculatedMean] 
 * @returns {number}
 */
export function variance(arr, precalculatedMean) {
  if (arr.length === 0) return 0;
  const mu = precalculatedMean !== undefined ? precalculatedMean : mean(arr);
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    sumSq += (arr[i] - mu) * (arr[i] - mu);
  }
  return sumSq / arr.length;
}

/**
 * Standard deviation of an array.
 * @param {number[]} arr 
 * @param {number} [precalculatedMean] 
 * @returns {number}
 */
export function stdDev(arr, precalculatedMean) {
  return Math.sqrt(variance(arr, precalculatedMean));
}

/**
 * Mean Squared Error: (1 / m) * sum((yTrue - yPred)^2)
 * @param {number[]} yTrue 
 * @param {number[]} yPred 
 * @returns {number}
 */
export function mse(yTrue, yPred) {
  const m = yTrue.length;
  if (m === 0) return 0;
  let sum = 0;
  for (let i = 0; i < m; i++) {
    const diff = yTrue[i] - yPred[i];
    sum += diff * diff;
  }
  return sum / m;
}

/**
 * Root Mean Squared Error.
 * @param {number[]} yTrue 
 * @param {number[]} yPred 
 * @returns {number}
 */
export function rmse(yTrue, yPred) {
  return Math.sqrt(mse(yTrue, yPred));
}

/**
 * R-Squared coefficient of determination: 1 - (SS_res / SS_tot).
 * @param {number[]} yTrue 
 * @param {number[]} yPred 
 * @returns {number}
 */
export function rSquared(yTrue, yPred) {
  const m = yTrue.length;
  if (m === 0) return 0;
  const yMean = mean(yTrue);
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < m; i++) {
    const res = yTrue[i] - yPred[i];
    const tot = yTrue[i] - yMean;
    ssRes += res * res;
    ssTot += tot * tot;
  }
  if (ssTot === 0) return 0;
  return 1 - (ssRes / ssTot);
}

/**
 * Binary Cross Entropy loss: -1/m * sum(y*ln(p) + (1-y)*ln(1-p)) + L2 penalty.
 * @param {number[]} yTrue 
 * @param {number[]} yProbs 
 * @param {number[]} [weights=[]] 
 * @param {number} [lambda=0] 
 * @returns {number}
 */
export function binaryCrossEntropy(yTrue, yProbs, weights = [], lambda = 0) {
  const m = yTrue.length;
  if (m === 0) return 0;
  const eps = 1e-15;
  let lossSum = 0;

  for (let i = 0; i < m; i++) {
    const y = yTrue[i];
    const p = clamp(yProbs[i], eps, 1 - eps);
    lossSum += y * Math.log(p) + (1 - y) * Math.log(1 - p);
  }

  let l2 = 0;
  if (lambda > 0 && weights.length > 0) {
    for (let j = 0; j < weights.length; j++) {
      l2 += weights[j] * weights[j];
    }
    l2 = (lambda / (2 * m)) * l2;
  }

  return (-lossSum / m) + l2;
}

/**
 * Computes confusion matrix and classification metrics on predictions.
 * @param {number[]} yTrue 
 * @param {number[]} yPred 
 * @returns {{tp: number, fp: number, tn: number, fn: number, accuracy: number, precision: number, recall: number, f1: number}}
 */
export function confusionMatrix(yTrue, yPred) {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const actual = yTrue[i];
    const pred = yPred[i];
    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 0 && pred === 0) tn++;
    else if (actual === 1 && pred === 0) fn++;
  }

  const total = tp + fp + tn + fn;
  const accuracy = total > 0 ? (tp + tn) / total : 0;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { tp, fp, tn, fn, accuracy, precision, recall, f1, total };
}

/**
 * Formats a numeric value as US Currency: e.g. $1,234,567
 * @param {number} val 
 * @returns {string}
 */
export function formatCurrency(val) {
  if (isNaN(val)) return '$0';
  return '$' + Math.round(val).toLocaleString('en-US');
}

/**
 * Formats a number with specified decimal digits.
 * @param {number} val 
 * @param {number} [decimals=2] 
 * @returns {string}
 */
export function formatNumber(val, decimals = 2) {
  if (isNaN(val)) return '0';
  return Number(val).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}
