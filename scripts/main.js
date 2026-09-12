// scripts/main.js
// Application bootstrap: ties dataset, scratch ML models, live charts, and interactive UI together.

import { LinearRegression } from './models/linearRegression.js';
import { LogisticRegression } from './models/logisticRegression.js';
import { TrainingChart } from './viz/trainingChart.js';
import { ConfusionMatrixView } from './viz/confusionMatrix.js';
import { CoefficientsChart } from './viz/coefficientsChart.js';
import { ControlsManager } from './ui/controls.js';
import { PredictionPanel } from './ui/predictionPanel.js';
import { ModalManager } from './ui/modal.js';
import { formatCurrency, formatNumber } from './utils/math.js';

class App {
  constructor() {
    this.dataset = null;
    this.linearModel = null;
    this.logisticModel = null;

    this.chart = null;
    this.confusionView = null;
    this.coeffChart = null;
    this.predictionPanel = null;
    this.controls = null;
    this.modal = null;

    this.isTraining = false;
    this.isPaused = false;
    this.trainingAbortController = null;
    this.isModelsTrained = false;
  }

  async init() {
    try {
      console.log('PriceDrift initializing...');
      await this.loadDataset();
      this.initModels();
      this.initUI();
      this.bindGlobalEvents();
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
    // Relative path works both on local static server and on Vercel deployment
    const res = await fetch('./scripts/data/housing-data.json');
    if (!res.ok) {
      throw new Error(`HTTP error ${res.status} loading dataset JSON`);
    }
    this.dataset = await res.json();
    console.log(`Loaded dataset: ${this.dataset.metadata.totalSamples} total records.`);
  }

  initModels() {
    this.linearModel = new LinearRegression({
      learningRate: 0.05,
      epochs: 200,
      useRegularization: false
    });

    this.logisticModel = new LogisticRegression({
      learningRate: 0.1,
      epochs: 250,
      useRegularization: false,
      threshold: 0.5
    });
  }

  initUI() {
    const meta = this.dataset.metadata;

    // 1. Chart
    const canvas = document.getElementById('training-canvas');
    this.chart = new TrainingChart(canvas);

    // 2. Confusion Matrix View
    const confContainer = document.getElementById('confusion-matrix-container');
    this.confusionView = new ConfusionMatrixView(confContainer, {
      onThresholdChange: (th) => {
        this.logisticModel.threshold = th;
        if (this.isModelsTrained) {
          const evalRes = this.logisticModel.evaluate(this.dataset.test.X, this.dataset.test.y_fast, th);
          this.confusionView.update(evalRes);
        }
        if (this.predictionPanel) {
          this.predictionPanel.setDecisionThreshold(th);
        }
      }
    });

    // 3. Coefficients Chart
    const coeffContainer = document.getElementById('coefficients-container');
    this.coeffChart = new CoefficientsChart(coeffContainer);
    this.coeffChart.render([], [], 0, meta.featureStats, meta.priceStats);

    // 4. Prediction Panel
    const predContainer = document.getElementById('prediction-container');
    this.predictionPanel = new PredictionPanel(predContainer, meta, {
      onPredict: () => this.runInference()
    });

    // 5. Modal
    const modalEl = document.getElementById('about-modal');
    this.modal = new ModalManager(modalEl, meta);

    // 6. Controls
    this.controls = new ControlsManager({
      lrSlider: document.getElementById('hp-lr-slider'),
      lrInput: document.getElementById('hp-lr-num'),
      epochsSlider: document.getElementById('hp-epochs-slider'),
      epochsInput: document.getElementById('hp-epochs-num'),
      regToggle: document.getElementById('hp-reg-toggle'),
      regLambdaGroup: document.getElementById('hp-lambda-group'),
      lambdaSlider: document.getElementById('hp-lambda-slider'),
      lambdaInput: document.getElementById('hp-lambda-num'),
      trainBtn: document.getElementById('btn-train'),
      pauseBtn: document.getElementById('btn-pause'),
      resetBtn: document.getElementById('btn-reset'),
      instantBtn: document.getElementById('btn-instant'),
      statusBadge: document.getElementById('training-status-badge'),
      alertBox: document.getElementById('global-alert'),
      modelSelects: document.querySelectorAll('input[name="model-choice"]')
    }, {
      onTrain: (hp) => this.startTraining(hp, false),
      onInstantTrain: (hp) => this.startTraining(hp, true),
      onPauseResume: () => this.togglePause(),
      onReset: () => this.resetApp(),
      onModelSelect: (choice) => this.handleModelChoiceChange(choice)
    });

    // Populate dataset summary counters in UI
    const trainCountEl = document.getElementById('stat-train-count');
    const testCountEl = document.getElementById('stat-test-count');
    const totalCountEl = document.getElementById('stat-total-count');
    if (trainCountEl) trainCountEl.textContent = meta.trainCount.toLocaleString();
    if (testCountEl) testCountEl.textContent = meta.testCount.toLocaleString();
    if (totalCountEl) totalCountEl.textContent = meta.totalSamples.toLocaleString();

    this.updateLinearMetricsCard(null);
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

  handleModelChoiceChange(choice) {
    console.log('Model selection changed to:', choice);
  }

  togglePause() {
    if (!this.isTraining) return;
    this.isPaused = !this.isPaused;
    this.controls.setTrainingState(this.isTraining, this.isPaused);
    this.controls.setStatus(this.isPaused ? 'Paused' : 'Training...', this.isPaused ? 'warning' : 'active');
  }

  resetApp() {
    if (this.isTraining) {
      this.isTraining = false;
      this.isPaused = false;
    }
    this.isModelsTrained = false;
    this.initModels();
    this.chart.clear();
    this.controls.clearWarning();
    this.controls.setTrainingState(false, false);
    this.controls.setStatus('Ready to Train', 'idle');
    this.updateLinearMetricsCard(null);
    this.coeffChart.render([], [], 0, this.dataset.metadata.featureStats, this.dataset.metadata.priceStats);
    this.confusionView.initStructure();
    this.predictionPanel.updateResults({ isTrained: false });
  }

  async startTraining(hyperparams, isInstant = false) {
    if (this.isTraining) return;

    this.controls.clearWarning();
    const { learningRate, epochs, useRegularization, lambda, selectedModel } = hyperparams;

    const trainLinear = selectedModel === 'both' || selectedModel === 'linear';
    const trainLogistic = selectedModel === 'both' || selectedModel === 'logistic';

    // Configure Linear Model
    if (trainLinear) {
      this.linearModel.learningRate = learningRate;
      this.linearModel.epochs = epochs;
      this.linearModel.useRegularization = useRegularization;
      this.linearModel.lambda = lambda;
      this.linearModel.initializeWeights(this.dataset.metadata.featureNames.length);
    }

    // Configure Logistic Model
    if (trainLogistic) {
      this.logisticModel.learningRate = learningRate;
      this.logisticModel.epochs = epochs;
      this.logisticModel.useRegularization = useRegularization;
      this.logisticModel.lambda = lambda;
      this.logisticModel.initializeWeights(this.dataset.metadata.featureNames.length);
    }

    // Prepare chart series
    const seriesList = [];
    let linSeriesIdx = -1;
    let logSeriesIdx = -1;

    if (trainLinear) {
      linSeriesIdx = seriesList.length;
      seriesList.push({
        name: 'Linear Reg (MSE)',
        color: 'rgb(56, 189, 248)',
        data: []
      });
    }

    if (trainLogistic) {
      logSeriesIdx = seriesList.length;
      seriesList.push({
        name: 'Logistic Reg (BCE)',
        color: 'rgb(168, 85, 247)',
        data: []
      });
    }

    this.chart.setSeries(seriesList);

    if (isInstant) {
      this.controls.setStatus('Instant Training...', 'active');
      if (trainLinear) this.linearModel.train(this.dataset.train.X, this.dataset.train.y_price_norm);
      if (trainLogistic) this.logisticModel.train(this.dataset.train.X, this.dataset.train.y_fast);

      // Populate chart series with histories
      if (trainLinear && linSeriesIdx >= 0) {
        seriesList[linSeriesIdx].data = [...this.linearModel.history];
      }
      if (trainLogistic && logSeriesIdx >= 0) {
        seriesList[logSeriesIdx].data = [...this.logisticModel.history];
      }
      this.chart.render();
      this.onTrainingComplete();
      return;
    }

    // Chunked Asynchronous Loop for Live Animation
    this.isTraining = true;
    this.isPaused = false;
    this.controls.setTrainingState(true, false);
    this.controls.setStatus('Training...', 'active');

    const totalEpochs = epochs;
    let currentEpoch = 0;
    const epochsPerFrame = totalEpochs > 500 ? 5 : (totalEpochs > 200 ? 3 : 2);

    const step = () => {
      if (!this.isTraining) return;

      if (this.isPaused) {
        requestAnimationFrame(step);
        return;
      }

      for (let i = 0; i < epochsPerFrame && currentEpoch < totalEpochs; i++) {
        currentEpoch++;

        if (trainLinear) {
          const res = this.linearModel.trainStep(this.dataset.train.X, this.dataset.train.y_price_norm);
          if (res.diverged) {
            this.handleDivergence('Linear Regression');
            return;
          }
          this.chart.addPoint(linSeriesIdx, res.epoch, res.cost);
        }

        if (trainLogistic) {
          const res = this.logisticModel.trainStep(this.dataset.train.X, this.dataset.train.y_fast);
          if (res.diverged) {
            this.handleDivergence('Logistic Regression');
            return;
          }
          this.chart.addPoint(logSeriesIdx, res.epoch, res.cost);
        }
      }

      this.controls.setStatus(`Training (Epoch ${currentEpoch}/${totalEpochs})...`, 'active');

      if (currentEpoch < totalEpochs) {
        requestAnimationFrame(step);
      } else {
        this.onTrainingComplete();
      }
    };

    requestAnimationFrame(step);
  }

  handleDivergence(modelName) {
    this.isTraining = false;
    this.controls.setTrainingState(false, false);
    this.controls.setStatus('Diverged', 'error');
    this.controls.showWarning(
      `⚠️ ${modelName} diverged! The cost grew to infinity or NaN because the learning rate is too high. Please decrease the learning rate (e.g. try \u03B1 = 0.01 or 0.05) and click Train again.`
    );
  }

  onTrainingComplete() {
    this.isTraining = false;
    this.isPaused = false;
    this.isModelsTrained = true;
    this.controls.setTrainingState(false, false);
    this.controls.setStatus('Training Complete', 'success');

    const meta = this.dataset.metadata;
    const test = this.dataset.test;

    // Evaluate Linear Model on Test Set
    if (this.linearModel.weights.length > 0) {
      const linEval = this.linearModel.evaluate(
        test.X,
        test.y_price_norm,
        test.y_price,
        meta.priceStats
      );
      this.updateLinearMetricsCard(linEval);
      this.coeffChart.render(
        meta.featureNames,
        this.linearModel.weights,
        this.linearModel.bias,
        meta.featureStats,
        meta.priceStats
      );
    }

    // Evaluate Logistic Model on Test Set
    if (this.logisticModel.weights.length > 0) {
      const logEval = this.logisticModel.evaluate(test.X, test.y_fast, this.logisticModel.threshold);
      this.confusionView.update(logEval);
    }

    // Update Live Inference
    this.runInference();
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
    if (!this.predictionPanel) return;

    if (!this.isModelsTrained) {
      this.predictionPanel.updateResults({ isTrained: false });
      return;
    }

    const xStd = this.predictionPanel.getStandardizedFeatures();
    const meta = this.dataset.metadata;

    // Linear Prediction
    let predPrice = meta.priceStats.median;
    if (this.linearModel && this.linearModel.weights.length > 0) {
      const normY = this.linearModel.predictSample(xStd);
      // Unscale to raw USD
      predPrice = normY * meta.priceStats.stdDev + meta.priceStats.mean;
    }

    // Logistic Prediction
    let fastProb = 0;
    let isFast = false;
    if (this.logisticModel && this.logisticModel.weights.length > 0) {
      const logRes = this.logisticModel.predictSample(xStd);
      fastProb = logRes.probability;
      isFast = logRes.isFastSale;
    }

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
