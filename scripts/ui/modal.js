// scripts/ui/modal.js
// Discloses dataset methodology, synthetic FastSale label formulation, and scratch mathematical architecture.

export class ModalManager {
  /**
   * @param {HTMLElement} modalEl 
   * @param {Object} metadata 
   */
  constructor(modalEl, metadata) {
    this.modalEl = modalEl;
    this.metadata = metadata;
    this.initContent();
    this.bindEvents();
  }

  initContent() {
    const { fastSaleHeuristic, priceStats, totalSamples } = this.metadata;

    this.modalEl.innerHTML = `
      <div class="modal-backdrop"></div>
      <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <div class="modal-header">
          <h2 id="modal-title">About This Data & Methodology</h2>
          <button type="button" class="modal-close-btn" aria-label="Close modal">&times;</button>
        </div>

        <div class="modal-body">
          <div class="disclosure-alert">
            <div class="alert-icon">⚠️</div>
            <div class="alert-text">
              <strong>Mandatory Data Disclosure:</strong> The <code>FastSale</code> label is an engineered synthetic proxy heuristic, <em>not</em> observed ground-truth real estate market speed or actual days-on-market telemetry.
            </div>
          </div>

          <section class="modal-section">
            <h3>1. The USA Housing Dataset</h3>
            <p>
              The base dataset is the widely referenced <strong>USA Housing Dataset</strong> (${totalSamples.toLocaleString()} records). It is a synthetic dataset commonly utilized in statistical and machine learning education to demonstrate multi-variable regression without messy real-world missingness.
            </p>
            <ul>
              <li><strong>Continuous Features:</strong> Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population.</li>
              <li><strong>Target Variable for Linear Regression:</strong> <code>Price</code> (Median: $${Math.round(priceStats.median).toLocaleString()}, Standard Deviation: $${Math.round(priceStats.stdDev).toLocaleString()}).</li>
              <li><strong>Dropped Columns:</strong> <code>Address</code> is removed as it is non-numeric.</li>
            </ul>
          </section>

          <section class="modal-section">
            <h3>2. FastSale Heuristic Formulation (Required Disclosure)</h3>
            <p>
              Since the original dataset lacks transaction timestamps or time-on-market fields, <strong>PriceDrift</strong> derives a binary classification label <code>FastSale &isin; {0, 1}</code> using economic domain proxies:
            </p>
            <div class="code-block">
              <ol>
                <li>Compute <code>PricePerRoom = Price / Avg. Area Number of Rooms</code> for every house.</li>
                <li>Compute the dataset-wide median <code>PricePerRoom</code> ($${Math.round(fastSaleHeuristic.medianPricePerRoom).toLocaleString()}) and median <code>Avg. Area House Age</code> (${fastSaleHeuristic.medianHouseAge.toFixed(2)} years).</li>
                <li>Label <code>FastSale = 1</code> IF and only IF:
                  <div class="formula-callout">
                    <code>(PricePerRoom &lt; $${Math.round(fastSaleHeuristic.medianPricePerRoom).toLocaleString()}) &and; (House Age &lt; ${fastSaleHeuristic.medianHouseAge.toFixed(2)} yrs)</code>
                  </div>
                  Otherwise, label <code>FastSale = 0</code>.
                </li>
              </ol>
            </div>
            <p>
              <strong>Rationale:</strong> In residential real estate economics, properties priced below the room median and featuring newer construction consistently exhibit higher liquidity and velocity. In this dataset, this yields <strong>${fastSaleHeuristic.positiveCount.toLocaleString()}</strong> fast-sale properties (<strong>${(fastSaleHeuristic.positiveRatio * 100).toFixed(1)}%</strong> of the dataset), providing a well-balanced binary target for logistic regression.
            </p>
          </section>

          <section class="modal-section">
            <h3>3. Mathematical Framework & Implementation</h3>
            <p>
              Both machine learning models and data preprocessing pipelines are implemented <strong>100% from scratch in pure vanilla JavaScript</strong> without external mathematical libraries (no TensorFlow.js, ml.js, or math.js):
            </p>
            <ul>
              <li>
                <strong>Z-Score Feature Standardization:</strong>
                <code>x_norm = (x - &mu;) / &sigma;</code>. Gradient descent convergence requires identical feature scales across high-magnitude values (like $80k income vs 5 rooms). The identical scaling parameters are preserved and used during live user inference.
              </li>
              <li>
                <strong>Linear Regression:</strong>
                Hypothesis <code>h(x) = w &middot; x + b</code>. Optimized via manual Batch Gradient Descent against Mean Squared Error (MSE), with Ridge (L2) regularization support.
              </li>
              <li>
                <strong>Logistic Regression:</strong>
                Hypothesis <code>P(Fast) = &sigma;(w &middot; x + b)</code> where <code>&sigma;(z) = 1 / (1 + e^-z)</code>. Optimized via manual Batch Gradient Descent against Binary Cross-Entropy (BCE) with dynamic classification threshold adjustment.
              </li>
            </ul>
          </section>
        </div>

        <div class="modal-footer">
          <button type="button" class="btn btn-primary modal-close-btn">Understood & Close</button>
        </div>
      </div>
    `;
  }

  bindEvents() {
    this.modalEl.querySelectorAll('.modal-close-btn').forEach(btn => {
      btn.addEventListener('click', () => this.close());
    });

    const backdrop = this.modalEl.querySelector('.modal-backdrop');
    if (backdrop) {
      backdrop.addEventListener('click', () => this.close());
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen()) {
        this.close();
      }
    });
  }

  open() {
    this.modalEl.classList.add('modal-open');
    document.body.style.overflow = 'hidden';
  }

  close() {
    this.modalEl.classList.remove('modal-open');
    document.body.style.overflow = '';
  }

  isOpen() {
    return this.modalEl.classList.contains('modal-open');
  }
}
