// tests/models.test.js
// Automated verification suite for PriceDrift's custom ML implementations

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { LinearRegression } from '../scripts/models/linearRegression.js';
import { LogisticRegression } from '../scripts/models/logisticRegression.js';
import { dot, matVecMul, sigmoid, rSquared, confusionMatrix } from '../scripts/utils/math.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function assert(condition, message) {
  if (!condition) {
    console.error(`❌ FAILED: ${message}`);
    process.exit(1);
  } else {
    console.log(`✅ PASSED: ${message}`);
  }
}

function runTests() {
  console.log('========================================');
  console.log('Running PriceDrift Verification Tests...');
  console.log('========================================\n');

  // 1. Math Utility Tests
  console.log('--- Testing Math Utilities ---');
  assert(dot([1, 2, 3], [4, 5, 6]) === 32, 'Dot product computed correctly');
  assert(
    JSON.stringify(matVecMul([[1, 2], [3, 4]], [2, 3])) === JSON.stringify([8, 18]),
    'Matrix-vector multiplication computed correctly'
  );
  assert(Math.abs(sigmoid(0) - 0.5) < 1e-6, 'Sigmoid(0) equals 0.5');
  assert(sigmoid(50) > 0.999999, 'Sigmoid upper clamp stable');
  assert(sigmoid(-50) < 0.000001, 'Sigmoid lower clamp stable');

  const conf = confusionMatrix([1, 0, 1, 1, 0], [1, 0, 0, 1, 1]);
  assert(conf.tp === 2 && conf.fp === 1 && conf.tn === 1 && conf.fn === 1, 'Confusion matrix counts correct');
  assert(conf.accuracy === 0.6, 'Accuracy calculation correct');

  // Load dataset JSON
  const dataPath = path.resolve(__dirname, '../scripts/data/housing-data.json');
  assert(fs.existsSync(dataPath), 'Bundled housing-data.json exists');
  const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));

  const { train, test, metadata } = data;
  assert(train.X.length === 4000, 'Train set has 4000 samples');
  assert(test.X.length === 1000, 'Test set has 1000 samples');
  assert(metadata.featureNames.length === 5, '5 features present');

  // 2. Linear Regression Verification
  console.log('\n--- Testing Linear Regression ---');
  const linReg = new LinearRegression({
    learningRate: 0.05,
    epochs: 200,
    useRegularization: false
  });

  const d = train.X[0].length;
  linReg.initializeWeights(d);

  let prevCost = Infinity;
  let decreasingCount = 0;

  for (let e = 0; e < 200; e++) {
    const step = linReg.trainStep(train.X, train.y_price_norm);
    if (e > 0 && step.cost < prevCost) {
      decreasingCount++;
    }
    prevCost = step.cost;
  }

  assert(decreasingCount > 190, `Cost consistently decreases over training (${decreasingCount}/199 steps)`);
  assert(!linReg.isDiverged, 'Linear regression did not diverge');

  const linEval = linReg.evaluate(test.X, test.y_price_norm, test.y_price, metadata.priceStats);
  console.log(`Linear Regression Results: R² = ${linEval.r2.toFixed(4)}, RMSE = $${Math.round(linEval.rmse).toLocaleString()}`);
  assert(linEval.r2 > 0.85, `Linear regression achieves strong convergence (R² = ${linEval.r2.toFixed(4)} > 0.85)`);

  // Test L2 regularization effect on weights
  const linRegRidge = new LinearRegression({
    learningRate: 0.05,
    epochs: 200,
    useRegularization: true,
    lambda: 50.0
  });
  linRegRidge.train(train.X, train.y_price_norm);

  const l2NormVanilla = linReg.weights.reduce((sum, w) => sum + w * w, 0);
  const l2NormRidge = linRegRidge.weights.reduce((sum, w) => sum + w * w, 0);
  assert(l2NormRidge < l2NormVanilla, `L2 Regularization successfully penalizes weight magnitudes (${l2NormRidge.toFixed(3)} < ${l2NormVanilla.toFixed(3)})`);

  // 3. Logistic Regression Verification
  console.log('\n--- Testing Logistic Regression ---');
  const logReg = new LogisticRegression({
    learningRate: 0.1,
    epochs: 250,
    useRegularization: false,
    threshold: 0.5
  });

  logReg.initializeWeights(d);
  prevCost = Infinity;
  decreasingCount = 0;

  for (let e = 0; e < 250; e++) {
    const step = logReg.trainStep(train.X, train.y_fast);
    if (e > 0 && step.cost < prevCost) {
      decreasingCount++;
    }
    prevCost = step.cost;
  }

  assert(decreasingCount > 240, `BCE cost consistently decreases (${decreasingCount}/249 steps)`);
  assert(!logReg.isDiverged, 'Logistic regression did not diverge');

  const logEval = logReg.evaluate(test.X, test.y_fast, 0.5);
  console.log(`Logistic Regression (Threshold=0.5): Accuracy = ${(logEval.accuracy * 100).toFixed(2)}%, Precision = ${(logEval.precision * 100).toFixed(2)}%, Recall = ${(logEval.recall * 100).toFixed(2)}%, F1 = ${logEval.f1.toFixed(3)}`);
  console.log(`Confusion Matrix: TP=${logEval.tp}, FP=${logEval.fp}, TN=${logEval.tn}, FN=${logEval.fn}`);

  assert(logEval.accuracy > 0.70, `Logistic regression achieves expected accuracy (${(logEval.accuracy * 100).toFixed(2)}% > 70%)`);
  assert(logEval.tp + logEval.fp + logEval.tn + logEval.fn === 1000, 'Confusion matrix accounts for all 1000 test samples');
  assert(logEval.tp > 0 && logEval.tn > 0, 'Model is non-degenerate (predicts both positive and negative classes)');

  // Dynamic threshold test
  const logEvalLowTh = logReg.evaluate(test.X, test.y_fast, 0.2);
  const logEvalHighTh = logReg.evaluate(test.X, test.y_fast, 0.8);
  assert(logEvalLowTh.recall >= logEvalHighTh.recall, 'Lower decision threshold increases/maintains recall');
  assert(logEvalHighTh.precision >= logEvalLowTh.precision, 'Higher decision threshold increases/maintains precision');

  console.log('\n========================================');
  console.log('🎉 ALL AUTOMATED TESTS PASSED SUCCESSFULLY!');
  console.log('========================================\n');
}

runTests();
