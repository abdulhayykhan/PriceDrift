// scripts/ui/controls.js
// Manages hyperparameter controls, training buttons, pause/resume, and model selection.

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
    // Model selection radio/buttons
    if (this.els.modelSelects) {
      this.els.modelSelects.forEach(radio => {
        radio.addEventListener('change', (e) => {
          if (this.callbacks.onModelSelect) {
            this.callbacks.onModelSelect(e.target.value);
          }
        });
      });
    }

    // Learning Rate Slider <-> Number sync
    if (this.els.lrSlider && this.els.lrInput) {
      this.els.lrSlider.addEventListener('input', (e) => {
        this.els.lrInput.value = e.target.value;
      });
      this.els.lrInput.addEventListener('input', (e) => {
        this.els.lrSlider.value = e.target.value;
      });
    }

    // Epochs Slider <-> Number sync (enforce max cap 2000)
    if (this.els.epochsSlider && this.els.epochsInput) {
      this.els.epochsSlider.addEventListener('input', (e) => {
        this.els.epochsInput.value = Math.min(2000, parseInt(e.target.value) || 10);
      });
      this.els.epochsInput.addEventListener('input', (e) => {
        let val = parseInt(e.target.value) || 10;
        if (val > 2000) val = 2000;
        this.els.epochsInput.value = val;
        this.els.epochsSlider.value = val;
      });
    }

    // Regularization toggle & lambda
    if (this.els.regToggle) {
      this.els.regToggle.addEventListener('change', (e) => {
        if (this.els.regLambdaGroup) {
          this.els.regLambdaGroup.style.display = e.target.checked ? 'flex' : 'none';
        }
      });
    }

    if (this.els.lambdaSlider && this.els.lambdaInput) {
      this.els.lambdaSlider.addEventListener('input', (e) => {
        this.els.lambdaInput.value = e.target.value;
      });
      this.els.lambdaInput.addEventListener('input', (e) => {
        this.els.lambdaSlider.value = e.target.value;
      });
    }

    // Buttons
    if (this.els.trainBtn) {
      this.els.trainBtn.addEventListener('click', () => {
        if (this.callbacks.onTrain) this.callbacks.onTrain(this.getHyperparameters());
      });
    }

    if (this.els.pauseBtn) {
      this.els.pauseBtn.addEventListener('click', () => {
        if (this.callbacks.onPauseResume) this.callbacks.onPauseResume();
      });
    }

    if (this.els.resetBtn) {
      this.els.resetBtn.addEventListener('click', () => {
        if (this.callbacks.onReset) this.callbacks.onReset();
      });
    }

    if (this.els.instantBtn) {
      this.els.instantBtn.addEventListener('click', () => {
        if (this.callbacks.onInstantTrain) this.callbacks.onInstantTrain(this.getHyperparameters());
      });
    }
  }

  getHyperparameters() {
    return {
      learningRate: parseFloat(this.els.lrInput?.value || 0.05),
      epochs: Math.min(2000, parseInt(this.els.epochsInput?.value || 200, 10)),
      useRegularization: !!this.els.regToggle?.checked,
      lambda: parseFloat(this.els.lambdaInput?.value || 5.0),
      selectedModel: document.querySelector('input[name="model-choice"]:checked')?.value || 'both'
    };
  }

  setTrainingState(isTraining, isPaused = false) {
    if (this.els.trainBtn) {
      this.els.trainBtn.disabled = isTraining && !isPaused;
      this.els.trainBtn.textContent = isTraining ? 'Training In Progress...' : 'Train Models';
    }
    if (this.els.pauseBtn) {
      this.els.pauseBtn.disabled = !isTraining;
      this.els.pauseBtn.textContent = isPaused ? 'Resume Training' : 'Pause';
      this.els.pauseBtn.classList.toggle('btn-warning', isPaused);
    }
    if (this.els.instantBtn) {
      this.els.instantBtn.disabled = isTraining;
    }
    if (this.els.resetBtn) {
      this.els.resetBtn.disabled = isTraining;
    }
  }

  setStatus(message, type = 'normal') {
    if (this.els.statusBadge) {
      this.els.statusBadge.textContent = message;
      this.els.statusBadge.className = `status-badge status-${type}`;
    }
  }

  showWarning(msg) {
    if (this.els.alertBox) {
      this.els.alertBox.textContent = msg;
      this.els.alertBox.style.display = 'block';
    }
  }

  clearWarning() {
    if (this.els.alertBox) {
      this.els.alertBox.textContent = '';
      this.els.alertBox.style.display = 'none';
    }
  }
}
