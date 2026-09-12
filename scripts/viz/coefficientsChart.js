// scripts/viz/coefficientsChart.js
// Visualizes learned regression coefficients (weights) as a feature importance bar chart.
// Displays both normalized standardized weights and unstandardized impact estimates.

export class CoefficientsChart {
  /**
   * @param {HTMLElement} container 
   */
  constructor(container) {
    this.container = container;
  }

  /**
   * Renders the bar chart given feature names, standardized weights, and stats.
   * @param {string[]} featureNames 
   * @param {number[]} weights 
   * @param {number} bias 
   * @param {Array<{mean: number, stdDev: number, name: string}>} featureStats 
   * @param {{mean: number, stdDev: number}} priceStats 
   */
  render(featureNames, weights, bias, featureStats, priceStats) {
    if (!weights || weights.length === 0) {
      this.container.innerHTML = `
        <div class="empty-state">
          <p>Train the linear model to inspect learned feature coefficients.</p>
        </div>
      `;
      return;
    }

    // Find max absolute weight for scaling bars
    const maxAbs = Math.max(...weights.map(Math.abs), 0.001);

    let html = `
      <div class="coeff-container">
        <div class="coeff-header-info">
          <span class="bias-tag">Model Bias (Intercept): <strong>$${Math.round(bias * priceStats.stdDev + priceStats.mean).toLocaleString()}</strong> (standardized b = ${bias.toFixed(3)})</span>
        </div>
        <div class="coeff-bars-list">
    `;

    featureNames.forEach((name, idx) => {
      const w = weights[idx];
      const isPos = w >= 0;
      const barWidthPct = Math.min((Math.abs(w) / maxAbs) * 100, 100);
      
      // Calculate dollar impact per 1 original unit:
      // w_raw = w_norm * (std_y / std_x)
      const wDollarPerUnit = w * (priceStats.stdDev / featureStats[idx].stdDev);

      let unitLabel = '';
      if (name.includes('Income')) unitLabel = 'per $1 income';
      else if (name.includes('Age')) unitLabel = 'per year';
      else if (name.includes('Bedrooms')) unitLabel = 'per bedroom';
      else if (name.includes('Rooms')) unitLabel = 'per room';
      else if (name.includes('Population')) unitLabel = 'per person';

      const dollarSign = wDollarPerUnit >= 0 ? '+' : '-';
      const dollarStr = `${dollarSign}$${Math.abs(Math.round(wDollarPerUnit)).toLocaleString()} ${unitLabel}`;

      html += `
        <div class="coeff-row">
          <div class="coeff-meta">
            <span class="coeff-name">${name}</span>
            <span class="coeff-dollar-impact ${isPos ? 'impact-pos' : 'impact-neg'}">${dollarStr}</span>
          </div>

          <div class="coeff-bar-track">
            <div class="coeff-zero-line"></div>
            <div class="coeff-bar-fill ${isPos ? 'bar-pos' : 'bar-neg'}" style="width: ${barWidthPct / 2}%; ${isPos ? 'left: 50%;' : `right: 50%;`}">
            </div>
          </div>

          <div class="coeff-weight-val">
            <span>${w >= 0 ? '+' : ''}${w.toFixed(4)}</span>
          </div>
        </div>
      `;
    });

    html += `
        </div>
      </div>
    `;

    this.container.innerHTML = html;
  }
}
