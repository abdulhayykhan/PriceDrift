// scripts/ui/predictionPanel.js
// Interactive feature sliders, real-time input normalization, and live model inference output.

import { formatCurrency, formatNumber } from '../utils/math.js';

export class PredictionPanel {
  /**
   * @param {HTMLElement} container 
   * @param {Object} metadata - Dataset metadata containing featureStats and priceStats
   * @param {Object} callbacks - { onPredict: () => void }
   */
  constructor(container, metadata, callbacks = {}) {
    this.container = container;
    this.metadata = metadata;
    this.callbacks = callbacks;
    this.featureInputs = [];
    this.currentThreshold = 0.5;

    this.renderForm();
    this.bindEvents();
  }

  renderForm() {
    const stats = this.metadata.featureStats;

    let slidersHtml = '';
    stats.forEach((f, idx) => {
      // Configure slider bounds based on dataset min/max with a small buffer
      let minVal, maxVal, step, defaultVal, formatFn;

      if (f.name.includes('Income')) {
        minVal = 20000;
        maxVal = 120000;
        step = 500;
        defaultVal = Math.round(f.median / 100) * 100;
        formatFn = val => '$' + Math.round(val).toLocaleString();
      } else if (f.name.includes('Age')) {
        minVal = 1;
        maxVal = 12;
        step = 0.1;
        defaultVal = Number(f.median.toFixed(1));
        formatFn = val => val.toFixed(1) + ' yrs';
      } else if (f.name.includes('Bedrooms')) {
        minVal = 1;
        maxVal = 8;
        step = 0.1;
        defaultVal = Number(f.median.toFixed(1));
        formatFn = val => val.toFixed(1);
      } else if (f.name.includes('Rooms')) {
        minVal = 3;
        maxVal = 12;
        step = 0.1;
        defaultVal = Number(f.median.toFixed(1));
        formatFn = val => val.toFixed(1);
      } else if (f.name.includes('Population')) {
        minVal = 5000;
        maxVal = 75000;
        step = 500;
        defaultVal = Math.round(f.median / 100) * 100;
        formatFn = val => Math.round(val).toLocaleString();
      }

      slidersHtml += `
        <div class="feature-control" data-index="${idx}">
          <div class="feature-label-row">
            <label for="feature-slider-${idx}" class="feature-title">${f.name}</label>
            <span class="feature-display" id="feature-val-${idx}">${formatFn(defaultVal)}</span>
          </div>
          <div class="slider-row">
            <input 
              type="range" 
              id="feature-slider-${idx}" 
              min="${minVal}" 
              max="${maxVal}" 
              step="${step}" 
              value="${defaultVal}" 
              class="app-slider"
            >
            <input 
              type="number" 
              id="feature-num-${idx}" 
              min="${minVal}" 
              max="${maxVal}" 
              step="${step}" 
              value="${defaultVal}" 
              class="app-number-input"
            >
          </div>
          <div class="feature-hints">
            <span>Min: ${formatFn(minVal)}</span>
            <span>Avg: ${formatFn(f.mean)}</span>
            <span>Max: ${formatFn(maxVal)}</span>
          </div>
        </div>
      `;
    });

    this.container.innerHTML = `
      <div class="predict-presets-bar">
        <span class="presets-label">Quick Scenarios:</span>
        <button type="button" class="preset-btn" data-preset="median">Median Home</button>
        <button type="button" class="preset-btn" data-preset="fast-candidate">Newer & Spacious</button>
        <button type="button" class="preset-btn" data-preset="luxury">Luxury Suburban</button>
        <button type="button" class="preset-btn" data-preset="older-budget">Older Dense Urban</button>
      </div>

      <form id="prediction-form" class="prediction-inputs-grid">
        ${slidersHtml}
      </form>

      <div class="prediction-results-card">
        <div class="prediction-header">
          <h3>Interactive Model Inference</h3>
          <span class="live-indicator"><span class="live-dot"></span> Live Prediction</span>
        </div>

        <div class="inference-grid">
          <!-- Linear Regression Output -->
          <div class="inference-card linear-result-card">
            <div class="inference-sub">Linear Regression Price Prediction</div>
            <div class="inference-main-val" id="pred-price-display">$0</div>
            <div class="inference-details" id="pred-price-sub">
              <span id="pred-price-diff">--</span> vs. dataset median ($${Math.round(this.metadata.priceStats.median).toLocaleString()})
            </div>
            <div class="inference-formula-tag">
              <code>h(x) = w \u00B7 x_norm + b</code>
            </div>
          </div>

          <!-- Logistic Regression Output -->
          <div class="inference-card logistic-result-card">
            <div class="inference-sub">Logistic Regression Fast-Sale Classifier</div>
            <div class="fast-sale-badge-row">
              <span class="fast-sale-badge" id="fast-sale-badge">Evaluating...</span>
              <span class="fast-prob-val" id="fast-prob-display">--%</span>
            </div>

            <div class="prob-bar-track">
              <div class="prob-bar-fill" id="prob-bar-fill" style="width: 0%"></div>
              <div class="prob-threshold-marker" id="prob-threshold-marker" style="left: 50%" title="Decision Threshold"></div>
            </div>

            <div class="prob-bar-legend">
              <span>0% (Slow)</span>
              <span id="prob-threshold-text">Threshold: \u03B8 = 0.50</span>
              <span>100% (Fast)</span>
            </div>

            <div class="inference-formula-tag">
              <code>P(Fast) = \u03C3(w \u00B7 x_norm + b)</code>
            </div>
          </div>
        </div>
      </div>
    `;

    // Cache elements
    this.priceDisplay = this.container.querySelector('#pred-price-display');
    this.priceSub = this.container.querySelector('#pred-price-sub');
    this.priceDiff = this.container.querySelector('#pred-price-diff');
    this.badgeDisplay = this.container.querySelector('#fast-sale-badge');
    this.probDisplay = this.container.querySelector('#fast-prob-display');
    this.probFill = this.container.querySelector('#prob-bar-fill');
    this.probThresholdMarker = this.container.querySelector('#prob-threshold-marker');
    this.probThresholdText = this.container.querySelector('#prob-threshold-text');
  }

  bindEvents() {
    const stats = this.metadata.featureStats;

    stats.forEach((f, idx) => {
      const slider = this.container.querySelector(`#feature-slider-${idx}`);
      const numInput = this.container.querySelector(`#feature-num-${idx}`);
      const valDisplay = this.container.querySelector(`#feature-val-${idx}`);

      const updateVal = (val) => {
        let formatted = val.toString();
        if (f.name.includes('Income')) formatted = '$' + Math.round(val).toLocaleString();
        else if (f.name.includes('Age')) formatted = parseFloat(val).toFixed(1) + ' yrs';
        else if (f.name.includes('Population')) formatted = Math.round(val).toLocaleString();
        else formatted = parseFloat(val).toFixed(1);

        valDisplay.textContent = formatted;
        this.triggerPredict();
      };

      slider.addEventListener('input', (e) => {
        numInput.value = e.target.value;
        updateVal(parseFloat(e.target.value));
      });

      numInput.addEventListener('input', (e) => {
        slider.value = e.target.value;
        updateVal(parseFloat(e.target.value));
      });
    });

    // Preset buttons
    this.container.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.applyPreset(e.target.dataset.preset);
      });
    });
  }

  applyPreset(presetName) {
    const stats = this.metadata.featureStats;
    let values = [];

    switch (presetName) {
      case 'median':
        values = stats.map(s => s.median);
        break;
      case 'fast-candidate':
        // High income, low house age, high rooms -> prime fast seller
        values = [
          85000, // Income
          3.5,   // Age (young)
          8.5,   // Rooms (spacious)
          4.5,   // Bedrooms
          32000  // Population
        ];
        break;
      case 'luxury':
        values = [
          110000, // Very high income
          5.0,    // Age
          9.5,    // Rooms
          5.0,    // Bedrooms
          45000   // Population
        ];
        break;
      case 'older-budget':
        values = [
          45000,  // Lower income
          9.0,    // Older house
          5.0,    // Fewer rooms
          2.5,    // Bedrooms
          55000   // High density
        ];
        break;
    }

    stats.forEach((f, idx) => {
      const slider = this.container.querySelector(`#feature-slider-${idx}`);
      const numInput = this.container.querySelector(`#feature-num-${idx}`);
      const valDisplay = this.container.querySelector(`#feature-val-${idx}`);

      const v = values[idx];
      slider.value = v;
      numInput.value = v;

      let formatted = v.toString();
      if (f.name.includes('Income')) formatted = '$' + Math.round(v).toLocaleString();
      else if (f.name.includes('Age')) formatted = parseFloat(v).toFixed(1) + ' yrs';
      else if (f.name.includes('Population')) formatted = Math.round(v).toLocaleString();
      else formatted = parseFloat(v).toFixed(1);

      valDisplay.textContent = formatted;
    });

    this.triggerPredict();
  }

  /**
   * Retrieves raw input vector from sliders.
   * @returns {number[]}
   */
  getRawFeatures() {
    return this.metadata.featureStats.map((_, idx) => {
      const slider = this.container.querySelector(`#feature-slider-${idx}`);
      return parseFloat(slider.value);
    });
  }

  /**
   * Standardizes the current input vector using the exact training means and stdDevs.
   * CRITICAL REQUIREMENT: must use same parameters as training.
   * @returns {number[]}
   */
  getStandardizedFeatures() {
    const raw = this.getRawFeatures();
    return raw.map((val, idx) => {
      const stat = this.metadata.featureStats[idx];
      return (val - stat.mean) / stat.stdDev;
    });
  }

  triggerPredict() {
    if (this.callbacks.onPredict) {
      this.callbacks.onPredict();
    }
  }

  setDecisionThreshold(th) {
    this.currentThreshold = th;
    this.probThresholdMarker.style.left = `${th * 100}%`;
    this.probThresholdText.textContent = `Threshold: \u03B8 = ${th.toFixed(2)}`;
    this.triggerPredict();
  }

  /**
   * Updates inference result displays.
   * @param {{ predictedPrice: number, fastProbability: number, isFastSale: boolean, isTrained: boolean }} results 
   */
  updateResults(results) {
    const { predictedPrice, fastProbability, isFastSale, isTrained } = results;

    if (!isTrained) {
      this.priceDisplay.textContent = '$ --';
      this.priceDiff.textContent = 'Models not yet trained';
      this.badgeDisplay.textContent = 'Untrained';
      this.badgeDisplay.className = 'fast-sale-badge badge-neutral';
      this.probDisplay.textContent = '--%';
      this.probFill.style.width = '0%';
      return;
    }

    // Linear price update
    this.priceDisplay.textContent = formatCurrency(predictedPrice);
    const medianPrice = this.metadata.priceStats.median;
    const diff = predictedPrice - medianPrice;
    const diffPct = (diff / medianPrice) * 100;
    const sign = diff >= 0 ? '+' : '-';
    this.priceDiff.innerHTML = `<span class="${diff >= 0 ? 'text-pos' : 'text-neg'}">${sign}$${Math.abs(Math.round(diff)).toLocaleString()} (${sign}${Math.abs(diffPct).toFixed(1)}%)</span>`;

    // Logistic probability & badge update
    const probPct = Math.round(fastProbability * 100);
    this.probDisplay.textContent = `${probPct}%`;
    this.probFill.style.width = `${probPct}%`;

    if (isFastSale) {
      this.badgeDisplay.textContent = '⚡ FAST SALE';
      this.badgeDisplay.className = 'fast-sale-badge badge-fast';
      this.probFill.className = 'prob-bar-fill fill-fast';
    } else {
      this.badgeDisplay.textContent = '⏳ NORMAL / SLOW SALE';
      this.badgeDisplay.className = 'fast-sale-badge badge-slow';
      this.probFill.className = 'prob-bar-fill fill-slow';
    }
  }
}
