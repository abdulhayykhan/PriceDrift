// scripts/viz/trainingChart.js
// Live HTML5 Canvas visualization of training convergence (cost/loss vs epoch).
// Implemented with vanilla Canvas 2D API, supporting high-DPI crisp rendering.

export class TrainingChart {
  /**
   * @param {HTMLCanvasElement} canvas 
   * @param {Object} [options]
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.series = []; // Array of { name, color, data: [{epoch, cost}] }
    this.options = {
      padding: { top: 30, right: 30, bottom: 45, left: 65 },
      gridColor: 'rgba(255, 255, 255, 0.08)',
      textColor: '#94a3b8',
      title: options.title || 'Training Loss vs. Epoch',
      ...options
    };

    this.setupHiDPI();
    window.addEventListener('resize', () => {
      this.setupHiDPI();
      this.render();
    });
  }

  setupHiDPI() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || 600;
    this.height = rect.height || 300;

    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.resetTransform?.();
    this.ctx.scale(dpr, dpr);
  }

  /**
   * Clears and sets data series.
   * @param {Array<{name: string, color: string, data: Array<{epoch: number, cost: number}>}>} seriesList 
   */
  setSeries(seriesList) {
    this.series = seriesList;
    this.render();
  }

  /**
   * Appends a new point to a specific series.
   * @param {number} seriesIndex 
   * @param {number} epoch 
   * @param {number} cost 
   */
  addPoint(seriesIndex, epoch, cost) {
    if (!this.series[seriesIndex]) return;
    this.series[seriesIndex].data.push({ epoch, cost });
    this.render();
  }

  clear() {
    this.series = [];
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    const pad = this.options.padding;

    // Clear background
    ctx.clearRect(0, 0, w, h);

    // Calculate domain & range
    let maxEpoch = 10;
    let minCost = Infinity;
    let maxCost = -Infinity;
    let hasData = false;

    for (const s of this.series) {
      if (s.data && s.data.length > 0) {
        hasData = true;
        for (const pt of s.data) {
          if (pt.epoch > maxEpoch) maxEpoch = pt.epoch;
          if (pt.cost < minCost) minCost = pt.cost;
          if (pt.cost > maxCost) maxCost = pt.cost;
        }
      }
    }

    if (!hasData) {
      minCost = 0;
      maxCost = 1;
    } else {
      if (minCost === maxCost) {
        minCost -= 0.1;
        maxCost += 0.1;
      } else {
        const span = maxCost - minCost;
        minCost = Math.max(0, minCost - span * 0.05);
        maxCost = maxCost + span * 0.05;
      }
    }

    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const scaleX = (epoch) => pad.left + (epoch / (maxEpoch || 1)) * plotW;
    const scaleY = (cost) => pad.top + plotH - ((cost - minCost) / ((maxCost - minCost) || 1)) * plotH;

    // 1. Draw Grid Lines & Ticks
    ctx.strokeStyle = this.options.gridColor;
    ctx.lineWidth = 1;
    ctx.font = '11px ui-monospace, SFMono-Regular, monospace';
    ctx.fillStyle = this.options.textColor;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const val = minCost + (i / yTicks) * (maxCost - minCost);
      const y = scaleY(val);

      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();

      ctx.fillText(val.toFixed(3), pad.left - 8, y);
    }

    const xTicks = 5;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= xTicks; i++) {
      const ep = Math.round((i / xTicks) * maxEpoch);
      const x = scaleX(ep);

      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();

      ctx.fillText(ep.toString(), x, pad.top + plotH + 8);
    }

    // Axis Labels
    ctx.save();
    ctx.fillStyle = '#64748b';
    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', pad.left + plotW / 2, h - 12);

    ctx.translate(18, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Cost / Loss', 0, 0);
    ctx.restore();

    // 2. Draw Series Lines & Glows
    if (!hasData) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Ready to train. Click "Train Models" to visualize convergence.', pad.left + plotW / 2, pad.top + plotH / 2);
      return;
    }

    for (const s of this.series) {
      if (!s.data || s.data.length === 0) continue;

      // Area fill under curve
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(scaleX(s.data[0].epoch), pad.top + plotH);
      for (const pt of s.data) {
        ctx.lineTo(scaleX(pt.epoch), scaleY(pt.cost));
      }
      ctx.lineTo(scaleX(s.data[s.data.length - 1].epoch), pad.top + plotH);
      ctx.closePath();

      const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
      gradient.addColorStop(0, s.color.replace('rgb', 'rgba').replace(')', ', 0.25)'));
      gradient.addColorStop(1, s.color.replace('rgb', 'rgba').replace(')', ', 0.0)'));
      ctx.fillStyle = gradient;
      ctx.fill();
      ctx.restore();

      // Main line
      ctx.save();
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2.5;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.shadowColor = s.color;
      ctx.shadowBlur = 8;

      ctx.beginPath();
      for (let i = 0; i < s.data.length; i++) {
        const pt = s.data[i];
        const x = scaleX(pt.epoch);
        const y = scaleY(pt.cost);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw end pulse dot
      const last = s.data[s.data.length - 1];
      const endX = scaleX(last.epoch);
      const endY = scaleY(last.cost);

      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(endX, endY, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(endX, endY, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // 3. Legend at top
    let legendX = pad.left;
    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.textBaseline = 'top';

    for (const s of this.series) {
      if (!s.data || s.data.length === 0) continue;
      const lastCost = s.data[s.data.length - 1].cost;

      ctx.fillStyle = s.color;
      ctx.fillRect(legendX, 8, 12, 12);

      ctx.fillStyle = '#e2e8f0';
      ctx.textAlign = 'left';
      const label = `${s.name}: ${lastCost.toFixed(4)}`;
      ctx.fillText(label, legendX + 18, 8);

      legendX += ctx.measureText(label).width + 36;
    }
  }
}
