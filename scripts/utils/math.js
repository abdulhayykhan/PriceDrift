// scripts/utils/math.js
// Lightweight inference and display utilities for PriceDrift.
// Training routines moved to Python (ml/).

/**
 * Computes inner dot product of two equal-length vectors for inference.
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
 * Clamps numeric value to [min, max].
 * @param {number} val 
 * @param {number} min 
 * @param {number} max 
 * @returns {number}
 */
export function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

/**
 * Numerically stable Sigmoid activation with [-30, 30] clamping.
 * @param {number} z 
 * @returns {number}
 */
export function sigmoid(z) {
  const clampedZ = clamp(z, -30, 30);
  return 1 / (1 + Math.exp(-clampedZ));
}

/**
 * Computes confusion matrix and metrics from predictions and ground truth.
 * Allows client-side re-evaluation when the user moves the decision threshold slider.
 * @param {number[]} yTrue 
 * @param {number[]} yPred 
 * @returns {{tp: number, fp: number, tn: number, fn: number, total: number, accuracy: number, precision: number, recall: number, f1: number}}
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
