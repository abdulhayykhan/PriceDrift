// scripts/viz/confusionMatrix.js
// Renders the 2x2 confusion matrix grid and classification metrics cards.
// Supports dynamic decision threshold adjustment with live re-evaluation.

export class ConfusionMatrixView {
  /**
   * @param {HTMLElement} container 
   * @param {Object} [options]
   */
  constructor(container, options = {}) {
    this.container = container;
    this.onThresholdChange = options.onThresholdChange || null;
    this.initStructure();
  }

  initStructure() {
    this.container.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card">
          <span class="metric-label">Accuracy</span>
          <span class="metric-value" id="metric-accuracy">--%</span>
          <span class="metric-sub">(TP + TN) / Total</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Precision</span>
          <span class="metric-value" id="metric-precision">--%</span>
          <span class="metric-sub">TP / (TP + FP)</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Recall</span>
          <span class="metric-value" id="metric-recall">--%</span>
          <span class="metric-sub">TP / (TP + FN)</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">F1 Score</span>
          <span class="metric-value" id="metric-f1">--</span>
          <span class="metric-sub">Harmonic mean</span>
        </div>
      </div>

      <div class="matrix-section">
        <div class="matrix-threshold-bar">
          <div class="threshold-label-wrap">
            <label for="matrix-threshold-slider">Decision Threshold (\u03B8):</label>
            <span class="threshold-display" id="matrix-threshold-val">0.50</span>
          </div>
          <input type="range" id="matrix-threshold-slider" min="0.05" max="0.95" step="0.01" value="0.50" class="app-slider">
          <span class="threshold-hint">Adjust threshold to observe precision/recall tradeoff on the test set.</span>
        </div>

        <div class="matrix-wrapper">
          <div class="matrix-header-corner"></div>
          <div class="matrix-col-header">Predicted: Fast Sale (1)</div>
          <div class="matrix-col-header">Predicted: Normal Sale (0)</div>

          <div class="matrix-row-header">Actual: Fast (1)</div>
          <div class="matrix-cell matrix-cell-tp" id="cell-tp">
            <span class="cell-type">True Positive (TP)</span>
            <span class="cell-count" id="count-tp">-</span>
            <span class="cell-pct" id="pct-tp">-</span>
          </div>
          <div class="matrix-cell matrix-cell-fn" id="cell-fn">
            <span class="cell-type">False Negative (FN)</span>
            <span class="cell-count" id="count-fn">-</span>
            <span class="cell-pct" id="pct-fn">-</span>
          </div>

          <div class="matrix-row-header">Actual: Normal (0)</div>
          <div class="matrix-cell matrix-cell-fp" id="cell-fp">
            <span class="cell-type">False Positive (FP)</span>
            <span class="cell-count" id="count-fp">-</span>
            <span class="cell-pct" id="pct-fp">-</span>
          </div>
          <div class="matrix-cell matrix-cell-tn" id="cell-tn">
            <span class="cell-type">True Negative (TN)</span>
            <span class="cell-count" id="count-tn">-</span>
            <span class="cell-pct" id="pct-tn">-</span>
          </div>
        </div>
      </div>
    `;

    this.accuracyEl = this.container.querySelector('#metric-accuracy');
    this.precisionEl = this.container.querySelector('#metric-precision');
    this.recallEl = this.container.querySelector('#metric-recall');
    this.f1El = this.container.querySelector('#metric-f1');

    this.sliderEl = this.container.querySelector('#matrix-threshold-slider');
    this.thresholdValEl = this.container.querySelector('#matrix-threshold-val');

    this.tpCountEl = this.container.querySelector('#count-tp');
    this.tpPctEl = this.container.querySelector('#pct-tp');
    this.fnCountEl = this.container.querySelector('#count-fn');
    this.fnPctEl = this.container.querySelector('#pct-fn');
    this.fpCountEl = this.container.querySelector('#count-fp');
    this.fpPctEl = this.container.querySelector('#pct-fp');
    this.tnCountEl = this.container.querySelector('#count-tn');
    this.tnPctEl = this.container.querySelector('#pct-tn');

    this.tpCell = this.container.querySelector('#cell-tp');
    this.fnCell = this.container.querySelector('#cell-fn');
    this.fpCell = this.container.querySelector('#cell-fp');
    this.tnCell = this.container.querySelector('#cell-tn');

    this.sliderEl.addEventListener('input', (e) => {
      const th = parseFloat(e.target.value);
      this.thresholdValEl.textContent = th.toFixed(2);
      if (this.onThresholdChange) {
        this.onThresholdChange(th);
      }
    });
  }

  /**
   * Updates display with evaluated results.
   * @param {{tp: number, fp: number, tn: number, fn: number, total: number, accuracy: number, precision: number, recall: number, f1: number, threshold: number}} evalResult 
   */
  update(evalResult) {
    const { tp, fp, tn, fn, total, accuracy, precision, recall, f1, threshold } = evalResult;

    if (threshold !== undefined && Math.abs(parseFloat(this.sliderEl.value) - threshold) > 0.001) {
      this.sliderEl.value = threshold.toFixed(2);
      this.thresholdValEl.textContent = threshold.toFixed(2);
    }

    this.accuracyEl.textContent = (accuracy * 100).toFixed(1) + '%';
    this.precisionEl.textContent = (precision * 100).toFixed(1) + '%';
    this.recallEl.textContent = (recall * 100).toFixed(1) + '%';
    this.f1El.textContent = f1.toFixed(3);

    this.tpCountEl.textContent = tp.toLocaleString();
    this.fnCountEl.textContent = fn.toLocaleString();
    this.fpCountEl.textContent = fp.toLocaleString();
    this.tnCountEl.textContent = tn.toLocaleString();

    const t = total || 1;
    this.tpPctEl.textContent = `(${(tp / t * 100).toFixed(1)}%)`;
    this.fnPctEl.textContent = `(${(fn / t * 100).toFixed(1)}%)`;
    this.fpPctEl.textContent = `(${(fp / t * 100).toFixed(1)}%)`;
    this.tnPctEl.textContent = `(${(tn / t * 100).toFixed(1)}%)`;

    // Dynamic background intensity based on count proportions
    const maxVal = Math.max(tp, fp, tn, fn, 1);
    this.tpCell.style.backgroundColor = `rgba(16, 185, 129, ${0.12 + (tp / maxVal) * 0.35})`;
    this.tnCell.style.backgroundColor = `rgba(16, 185, 129, ${0.12 + (tn / maxVal) * 0.35})`;
    this.fpCell.style.backgroundColor = `rgba(239, 68, 68, ${0.12 + (fp / maxVal) * 0.35})`;
    this.fnCell.style.backgroundColor = `rgba(239, 68, 68, ${0.12 + (fn / maxVal) * 0.35})`;
  }
}
