// scripts/main.js
// Application bootstrap: loads pre-trained Python models, handles live inference,
// and manages training convergence animation replay.

import { dot, sigmoid, confusionMatrix, formatCurrency, formatNumber } from './utils/math.js';
import { TrainingChart } from './viz/trainingChart.js';
import { ConfusionMatrixView } from './viz/confusionMatrix.js';
import { CoefficientsChart } from './viz/coefficientsChart.js';
import { ControlsManager } from './ui/controls.js';
import { PredictionPanel } from './ui/predictionPanel.js';
import { ModalManager } from './ui/modal.js';

class App {
  constructor() {
    this.dataset = null;

    // Active model parameters
    this.activeLinear = null;
    this.activeLogistic = null;
    this.currentThreshold = 0.5;
    this.isRegularized = false;
    this.selectedModelChoice = 'both'; // 'both', 'linear', 'logistic'

    // Visualizations & UI
    this.chart = null;
    this.confusionView = null;
    this.coeffChart = null;
    this.predictionPanel = null;
    this.controls = null;
    this.modal = null;

    // Replay state
    this.isReplaying = false;
    this.isPaused = false;
    this.replaySpeed = 1.0;
    this.replayFrameId = null;
  }

  async init() {
    try {
      console.log('PriceDrift initializing...');
      await this.loadDataset();
      this.initUI();
      this.bindGlobalEvents();
      this.updateActiveModels();
      this.displayFinalState();
      console.log('PriceDrift initialized successfully.');
    } catch (err) {
      console.error('Initialization error:', err);
      const errBox = document.getElementById('global-alert');
      if (errBox) {
        errBox.textContent = `Failed to load application data: ${err.message}`;
        errBox.style.display = 'block';
      }
    }
  }

  async loadDataset() {
    const res = await fetch('./scripts/data/housing-data.json');
    if (!res.ok) {
      throw new Error(`HTTP error ${res.status} loading dataset JSON`);
    }
    this.dataset = await res.json();
    console.log(`Loaded dataset: ${this.dataset.metadata.totalSamples} total records.`);
  }

  initUI() {
    const meta = this.dataset.metadata;

    // 1. Live Training Chart
    const canvas = document.getElementById('training-canvas');
    this.chart = new TrainingChart(canvas);

    // 2. Confusion Matrix View (with dynamic client-side threshold slider)
    const confContainer = document.getElementById('confusion-matrix-container');
    this.confusionView = new ConfusionMatrixView(confContainer, {
      onThresholdChange: (th) => {
        this.currentThreshold = th;
        this.recalculateConfusionMatrix(th);
        if (this.predictionPanel) {
          this.predictionPanel.setDecisionThreshold(th);
        }
      }
    });

    // 3. Coefficients Importance Chart
    const coeffContainer = document.getElementById('coefficients-container');
    this.coeffChart = new CoefficientsChart(coeffContainer);

    // 4. Interactive Prediction Panel
    const predContainer = document.getElementById('prediction-container');
    this.predictionPanel = new PredictionPanel(predContainer, meta, {
      onPredict: () => this.runInference()
    });

    // 5. Methodology & Disclosure Modal
    const modalEl = document.getElementById('about-modal');
    this.modal = new ModalManager(modalEl, meta);

    // 6. Controls Manager (Replay Mode)
    this.controls = new ControlsManager({
      modelSelects: document.querySelectorAll('input[name="model-choice"]'),
      regToggle: document.getElementById('hp-reg-toggle'),
      regStatusLabel: document.getElementById('reg-status-label'),
      speedSelect: document.getElementById('replay-speed-select'),
      replayBtn: document.getElementById('btn-replay'),
      pauseBtn: document.getElementById('btn-pause'),
      skipBtn: document.getElementById('btn-skip'),
      resetBtn: document.getElementById('btn-reset'),
      statusBadge: document.getElementById('training-status-badge')
    }, {
      onModelSelect: (choice) => {
        this.selectedModelChoice = choice;
        this.displayFinalState();
      },
      onVariantChange: (isReg) => {
        this.isRegularized = isReg;
        this.updateActiveModels();
        this.displayFinalState();
      },
      onSpeedChange: (speed) => {
        this.replaySpeed = speed;
      },
      onReplay: () => this.startReplay(),
      onPauseResume: () => this.togglePause(),
      onSkip: () => this.skipToEnd(),
      onReset: () => this.resetReplay()
    });

    // Populate dataset summary counters in UI
    const trainCountEl = document.getElementById('stat-train-count');
    const testCountEl = document.getElementById('stat-test-count');
    if (trainCountEl) trainCountEl.textContent = meta.trainCount.toLocaleString();
    if (testCountEl) testCountEl.textContent = meta.testCount.toLocaleString();
  }

  bindGlobalEvents() {
    const aboutBtn = document.getElementById('btn-open-about');
    if (aboutBtn) {
      aboutBtn.addEventListener('click', () => this.modal.open());
    }

    const dataPill = document.getElementById('data-disclosure-pill');
    if (dataPill) {
      dataPill.addEventListener('click', () => this.modal.open());
    }
  }

  updateActiveModels() {
    const models = this.dataset.trainedModels;
    this.activeLinear = this.isRegularized ? models.linear_regularized : models.linear;
    this.activeLogistic = this.isRegularized ? models.logistic_regularized : models.logistic;
  }

  displayFinalState() {
    this.cancelReplay();

    // Render full convergence curves on chart
    const seriesList = [];
    const showLinear = this.selectedModelChoice === 'both' || this.selectedModelChoice === 'linear';
    const showLogistic = this.selectedModelChoice === 'both' || this.selectedModelChoice === 'logistic';

    if (showLinear) {
      seriesList.push({
        name: this.isRegularized ? 'Linear Reg (Ridge λ=50)' : 'Linear Reg (MSE)',
        color: 'rgb(56, 189, 248)',
        data: [...this.activeLinear.history]
      });
    }

    if (showLogistic) {
      seriesList.push({
        name: this.isRegularized ? 'Logistic Reg (λ=10)' : 'Logistic Reg (BCE)',
        color: 'rgb(168, 85, 247)',
        data: [...this.activeLogistic.history]
      });
    }

    this.chart.setSeries(seriesList);

    // Update metrics & coefficient displays
    this.updateLinearMetricsCard(this.activeLinear.testMetrics);
    this.coeffChart.render(
      this.dataset.metadata.featureNames,
      this.activeLinear.weights,
      this.activeLinear.bias,
      this.dataset.metadata.featureStats,
      this.dataset.metadata.priceStats
    );

    this.recalculateConfusionMatrix(this.currentThreshold);
    this.runInference();
    this.controls.setStatus('Python Trained', 'success');
  }

  startReplay() {
    this.cancelReplay();

    this.isReplaying = true;
    this.isPaused = false;
    this.controls.setReplayState(true, false);
    this.controls.setStatus('Replaying...', 'active');

    const showLinear = this.selectedModelChoice === 'both' || this.selectedModelChoice === 'linear';
    const showLogistic = this.selectedModelChoice === 'both' || this.selectedModelChoice === 'logistic';

    const seriesList = [];
    let linIdx = -1;
    let logIdx = -1;

    if (showLinear) {
      linIdx = seriesList.length;
      seriesList.push({
        name: this.isRegularized ? 'Linear (Ridge)' : 'Linear (MSE)',
        color: 'rgb(56, 189, 248)',
        data: []
      });
    }

    if (showLogistic) {
      logIdx = seriesList.length;
      seriesList.push({
        name: this.isRegularized ? 'Logistic (Ridge)' : 'Logistic (BCE)',
        color: 'rgb(168, 85, 247)',
        data: []
      });
    }

    this.chart.setSeries(seriesList);

    const linHist = this.activeLinear.history;
    const logHist = this.activeLogistic.history;
    const maxEpochs = Math.max(
      showLinear ? linHist.length : 0,
      showLogistic ? logHist.length : 0
    );

    let currentStep = 0;

    const step = () => {
      if (!this.isReplaying) return;

      if (this.isPaused) {
        this.replayFrameId = requestAnimationFrame(step);
        return;
      }

      // Step multiple points depending on replay speed
      const pointsPerFrame = Math.max(1, Math.round(2 * this.replaySpeed));
      for (let p = 0; p < pointsPerFrame && currentStep < maxEpochs; p++) {
        if (showLinear && currentStep < linHist.length) {
          const pt = linHist[currentStep];
          this.chart.addPoint(linIdx, pt.epoch, pt.cost);
        }
        if (showLogistic && currentStep < logHist.length) {
          const pt = logHist[currentStep];
          this.chart.addPoint(logIdx, pt.epoch, pt.cost);
        }
        currentStep++;
      }

      this.controls.setStatus(`Replay (Epoch ${currentStep}/${maxEpochs})`, 'active');

      if (currentStep < maxEpochs) {
        this.replayFrameId = requestAnimationFrame(step);
      } else {
        this.isReplaying = false;
        this.controls.setReplayState(false, false);
        this.controls.setStatus('Replay Complete', 'success');
      }
    };

    this.replayFrameId = requestAnimationFrame(step);
  }

  togglePause() {
    if (!this.isReplaying) return;
    this.isPaused = !this.isPaused;
    this.controls.setReplayState(this.isReplaying, this.isPaused);
    this.controls.setStatus(this.isPaused ? 'Paused' : 'Replaying...', this.isPaused ? 'warning' : 'active');
  }

  skipToEnd() {
    this.cancelReplay();
    this.displayFinalState();
  }

  resetReplay() {
    this.cancelReplay();
    this.chart.clear();
    this.controls.setStatus('Ready to Replay', 'idle');
  }

  cancelReplay() {
    if (this.replayFrameId) {
      cancelAnimationFrame(this.replayFrameId);
      this.replayFrameId = null;
    }
    this.isReplaying = false;
    this.isPaused = false;
    this.controls.setReplayState(false, false);
  }

  recalculateConfusionMatrix(threshold) {
    const testX = this.dataset.test.X;
    const testYFast = this.dataset.test.y_fast;
    const w = this.activeLogistic.weights;
    const b = this.activeLogistic.bias;

    // Fast client-side probability calculation for test set
    const preds = new Array(testX.length);
    for (let i = 0; i < testX.length; i++) {
      const z = dot(w, testX[i]) + b;
      const prob = sigmoid(z);
      preds[i] = prob >= threshold ? 1 : 0;
    }

    const metrics = confusionMatrix(testYFast, preds);
    this.confusionView.update({
      ...metrics,
      threshold
    });
  }

  updateLinearMetricsCard(evalRes) {
    const r2El = document.getElementById('metric-r2');
    const rmseEl = document.getElementById('metric-rmse');
    const mseEl = document.getElementById('metric-mse');

    if (!evalRes) {
      if (r2El) r2El.textContent = '--';
      if (rmseEl) rmseEl.textContent = '$ --';
      if (mseEl) mseEl.textContent = '--';
      return;
    }

    if (r2El) r2El.textContent = evalRes.r2.toFixed(4);
    if (rmseEl) rmseEl.textContent = formatCurrency(evalRes.rmse);
    if (mseEl) mseEl.textContent = formatNumber(evalRes.mse, 0);
  }

  runInference() {
    if (!this.predictionPanel || !this.activeLinear || !this.activeLogistic) return;

    const xStd = this.predictionPanel.getStandardizedFeatures();
    const meta = this.dataset.metadata;

    // 1. Linear Inference: y_norm = dot(w, xStd) + bias
    const linNormY = dot(this.activeLinear.weights, xStd) + this.activeLinear.bias;
    const predPrice = linNormY * meta.priceStats.stdDev + meta.priceStats.mean;

    // 2. Logistic Inference: p = sigmoid(dot(w, xStd) + bias)
    const logZ = dot(this.activeLogistic.weights, xStd) + this.activeLogistic.bias;
    const fastProb = sigmoid(logZ);
    const isFast = fastProb >= this.currentThreshold;

    this.predictionPanel.updateResults({
      predictedPrice: predPrice,
      fastProbability: fastProb,
      isFastSale: isFast,
      isTrained: true
    });
  }
}

// Bootstrap app when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  app.init();
});
