// scripts/ui/controls.js
// Manages model selection, regularization variant toggle, and training history replay controls.

export class ControlsManager {
  /**
   * @param {Object} elements - Map of DOM elements
   * @param {Object} callbacks - Handler callbacks
   */
  constructor(elements, callbacks) {
    this.els = elements;
    this.callbacks = callbacks;
    this.bindEvents();
  }

  bindEvents() {
    // Model selection radio buttons (both, linear, logistic)
    if (this.els.modelSelects) {
      this.els.modelSelects.forEach(radio => {
        radio.addEventListener('change', (e) => {
          if (this.callbacks.onModelSelect) {
            this.callbacks.onModelSelect(e.target.value);
          }
        });
      });
    }

    // Regularization toggle (switches between baseline and regularized precomputed runs)
    if (this.els.regToggle) {
      this.els.regToggle.addEventListener('change', (e) => {
        const isReg = e.target.checked;
        if (this.els.regStatusLabel) {
          this.els.regStatusLabel.textContent = isReg
            ? 'L2 Regularized (Ridge λ=50 / Log λ=10)'
            : 'Unregularized Baseline (λ=0)';
        }
        if (this.callbacks.onVariantChange) {
          this.callbacks.onVariantChange(isReg);
        }
      });
    }

    // Replay Speed selector
    if (this.els.speedSelect) {
      this.els.speedSelect.addEventListener('change', (e) => {
        if (this.callbacks.onSpeedChange) {
          this.callbacks.onSpeedChange(parseFloat(e.target.value) || 1.0);
        }
      });
    }

    // Buttons
    if (this.els.replayBtn) {
      this.els.replayBtn.addEventListener('click', () => {
        if (this.callbacks.onReplay) this.callbacks.onReplay();
      });
    }

    if (this.els.pauseBtn) {
      this.els.pauseBtn.addEventListener('click', () => {
        if (this.callbacks.onPauseResume) this.callbacks.onPauseResume();
      });
    }

    if (this.els.skipBtn) {
      this.els.skipBtn.addEventListener('click', () => {
        if (this.callbacks.onSkip) this.callbacks.onSkip();
      });
    }

    if (this.els.resetBtn) {
      this.els.resetBtn.addEventListener('click', () => {
        if (this.callbacks.onReset) this.callbacks.onReset();
      });
    }
  }

  getState() {
    const selectedModel = document.querySelector('input[name="model-choice"]:checked')?.value || 'both';
    const isRegularized = !!this.els.regToggle?.checked;
    const speed = parseFloat(this.els.speedSelect?.value || 1.0);
    return { selectedModel, isRegularized, speed };
  }

  setReplayState(isReplaying, isPaused = false) {
    if (this.els.replayBtn) {
      this.els.replayBtn.disabled = isReplaying && !isPaused;
      this.els.replayBtn.textContent = isReplaying ? 'Replaying...' : 'Replay Training';
    }
    if (this.els.pauseBtn) {
      this.els.pauseBtn.disabled = !isReplaying;
      this.els.pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
      this.els.pauseBtn.classList.toggle('btn-warning', isPaused);
    }
    if (this.els.skipBtn) {
      this.els.skipBtn.disabled = isReplaying && !isPaused;
    }
  }

  setStatus(message, type = 'normal') {
    if (this.els.statusBadge) {
      this.els.statusBadge.textContent = message;
      this.els.statusBadge.className = `status-badge status-${type}`;
    }
  }
}
