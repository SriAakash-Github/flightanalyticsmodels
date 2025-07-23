/**
 * Baggage ML Prediction System - Frontend JavaScript
 * Handles form validation, user interactions, and data visualization
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeFormValidation();
    initializePredictionForms();
    initializeCharts();
    initializeTooltips();
    
    // Replace feather icons
    feather.replace();
});

/**
 * Form validation for prediction inputs
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                showValidationErrors(form);
            } else {
                // Show loading state
                showLoadingState(form);
            }
            
            form.classList.add('was-validated');
        });
        
        // Real-time validation for number inputs
        const numberInputs = form.querySelectorAll('input[type="number"]');
        numberInputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateNumberInput(input);
            });
            
            input.addEventListener('input', function() {
                clearValidationError(input);
            });
        });
        
        // Validation for select inputs
        const selectInputs = form.querySelectorAll('select');
        selectInputs.forEach(select => {
            select.addEventListener('change', function() {
                clearValidationError(select);
            });
        });
    });
}

/**
 * Enhanced prediction form interactions
 */
function initializePredictionForms() {
    // Auto-calculate derived fields where applicable
    initializeAutoCalculations();
    
    // Form reset functionality
    const resetButtons = document.querySelectorAll('[data-reset-form]');
    resetButtons.forEach(button => {
        button.addEventListener('click', function() {
            const form = button.closest('form');
            if (form) {
                resetForm(form);
            }
        });
    });
    
    // Save form data to localStorage for convenience
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            // Load saved value
            const savedValue = localStorage.getItem(`baggage_ml_${input.name}`);
            if (savedValue && !input.value) {
                input.value = savedValue;
            }
            
            // Save on change
            input.addEventListener('change', function() {
                localStorage.setItem(`baggage_ml_${input.name}`, input.value);
            });
        });
    });
}

/**
 * Initialize Chart.js visualizations
 */
function initializeCharts() {
    // Set global Chart.js defaults for dark theme
    Chart.defaults.color = '#ffffff';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
    Chart.defaults.backgroundColor = 'rgba(255, 255, 255, 0.1)';
}

/**
 * Initialize Bootstrap tooltips and popovers
 */
function initializeTooltips() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

/**
 * Auto-calculation functionality for related fields
 */
function initializeAutoCalculations() {
    // Damage prediction: Auto-suggest values based on inputs
    const damageForm = document.querySelector('form[action*="damage_prediction"]');
    if (damageForm) {
        const totalBags = damageForm.querySelector('input[name="total_bags"]');
        const avgReportTime = damageForm.querySelector('input[name="avg_report_time"]');
        
        if (totalBags && avgReportTime) {
            [totalBags, avgReportTime].forEach(input => {
                input.addEventListener('input', function() {
                    updateDamageRiskIndicator(damageForm);
                });
            });
        }
    }
    
    // Transfer prediction: Calculate buffer adequacy
    const transferForm = document.querySelector('form[action*="transfer_prediction"]');
    if (transferForm) {
        const bufferTime = transferForm.querySelector('input[name="avg_buffer_minutes"]');
        const transferTime = transferForm.querySelector('input[name="avg_transfer_time"]');
        const mct = transferForm.querySelector('input[name="avg_mct"]');
        
        if (bufferTime && transferTime && mct) {
            [bufferTime, transferTime, mct].forEach(input => {
                input.addEventListener('input', function() {
                    updateTransferRiskIndicator(transferForm);
                });
            });
        }
    }
}

/**
 * Update damage risk indicator based on inputs
 */
function updateDamageRiskIndicator(form) {
    const totalBags = parseFloat(form.querySelector('input[name="total_bags"]')?.value) || 0;
    const avgReportTime = parseFloat(form.querySelector('input[name="avg_report_time"]')?.value) || 0;
    
    let riskLevel = 'Unknown';
    let riskColor = 'secondary';
    
    if (totalBags > 0) {
        // Simple heuristic for pre-prediction risk assessment
        if (totalBags > 200 && avgReportTime > 30) {
            riskLevel = 'Higher Risk';
            riskColor = 'warning';
        } else if (totalBags > 150 || avgReportTime > 45) {
            riskLevel = 'Moderate Risk';
            riskColor = 'info';
        } else {
            riskLevel = 'Lower Risk';
            riskColor = 'success';
        }
        
        showRiskIndicator(form, riskLevel, riskColor);
    }
}

/**
 * Update transfer risk indicator based on buffer time vs requirements
 */
function updateTransferRiskIndicator(form) {
    const bufferTime = parseFloat(form.querySelector('input[name="avg_buffer_minutes"]')?.value) || 0;
    const transferTime = parseFloat(form.querySelector('input[name="avg_transfer_time"]')?.value) || 0;
    const mct = parseFloat(form.querySelector('input[name="avg_mct"]')?.value) || 0;
    
    if (bufferTime > 0 && transferTime > 0 && mct > 0) {
        const bufferAdequacy = bufferTime - transferTime;
        const mctBuffer = bufferTime - mct;
        
        let riskLevel, riskColor;
        
        if (bufferAdequacy < 10 || mctBuffer < 5) {
            riskLevel = 'Tight Connection';
            riskColor = 'danger';
        } else if (bufferAdequacy < 20 || mctBuffer < 10) {
            riskLevel = 'Moderate Buffer';
            riskColor = 'warning';
        } else {
            riskLevel = 'Comfortable Buffer';
            riskColor = 'success';
        }
        
        showRiskIndicator(form, riskLevel, riskColor);
    }
}

/**
 * Show risk indicator in form
 */
function showRiskIndicator(form, riskLevel, riskColor) {
    let indicator = form.querySelector('.risk-indicator');
    
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.className = 'risk-indicator alert mt-3';
        form.appendChild(indicator);
    }
    
    indicator.className = `risk-indicator alert alert-${riskColor} mt-3`;
    indicator.innerHTML = `
        <i data-feather="info"></i>
        <strong>Pre-assessment:</strong> ${riskLevel}
        <small class="d-block mt-1">Submit form for detailed ML prediction</small>
    `;
    
    feather.replace();
}

/**
 * Number input validation
 */
function validateNumberInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    let isValid = true;
    let message = '';
    
    if (isNaN(value)) {
        isValid = false;
        message = 'Please enter a valid number';
    } else if (!isNaN(min) && value < min) {
        isValid = false;
        message = `Value must be at least ${min}`;
    } else if (!isNaN(max) && value > max) {
        isValid = false;
        message = `Value must be at most ${max}`;
    }
    
    if (isValid) {
        input.classList.remove('is-invalid');
        clearValidationMessage(input);
    } else {
        input.classList.add('is-invalid');
        showValidationMessage(input, message);
    }
    
    return isValid;
}

/**
 * Show validation message for an input
 */
function showValidationMessage(input, message) {
    clearValidationMessage(input);
    
    const feedback = document.createElement('div');
    feedback.className = 'invalid-feedback';
    feedback.textContent = message;
    
    input.parentNode.appendChild(feedback);
}

/**
 * Clear validation message for an input
 */
function clearValidationMessage(input) {
    const feedback = input.parentNode.querySelector('.invalid-feedback');
    if (feedback) {
        feedback.remove();
    }
}

/**
 * Clear validation error styling
 */
function clearValidationError(input) {
    input.classList.remove('is-invalid');
    clearValidationMessage(input);
}

/**
 * Show validation errors for a form
 */
function showValidationErrors(form) {
    const invalidInputs = form.querySelectorAll(':invalid');
    
    invalidInputs.forEach(input => {
        if (input.type === 'number') {
            validateNumberInput(input);
        } else {
            input.classList.add('is-invalid');
        }
    });
    
    // Scroll to first error
    if (invalidInputs.length > 0) {
        invalidInputs[0].scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
        invalidInputs[0].focus();
    }
}

/**
 * Show loading state for form submission
 */
function showLoadingState(form) {
    const submitButton = form.querySelector('button[type="submit"]');
    
    if (submitButton) {
        const originalText = submitButton.innerHTML;
        submitButton.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            Processing...
        `;
        submitButton.disabled = true;
        
        // Store original state
        submitButton.dataset.originalText = originalText;
    }
}

/**
 * Reset form to initial state
 */
function resetForm(form) {
    form.reset();
    form.classList.remove('was-validated');
    
    // Clear validation messages
    const invalidInputs = form.querySelectorAll('.is-invalid');
    invalidInputs.forEach(input => {
        clearValidationError(input);
    });
    
    // Remove risk indicators
    const indicators = form.querySelectorAll('.risk-indicator');
    indicators.forEach(indicator => indicator.remove());
    
    // Clear localStorage
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        if (input.name) {
            localStorage.removeItem(`baggage_ml_${input.name}`);
        }
    });
}

/**
 * Enhanced result display animations
 */
function animateResultCards() {
    const resultCards = document.querySelectorAll('.card:has(.prediction-result)');
    
    resultCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.animation = 'slideInUp 0.5s ease-out';
        }, index * 100);
    });
}

/**
 * Copy prediction results to clipboard
 */
function copyResults(button) {
    const resultCard = button.closest('.card');
    const riskLevel = resultCard.querySelector('.display-6')?.textContent?.trim();
    const confidence = resultCard.querySelector('[data-confidence]')?.textContent?.trim();
    
    if (riskLevel && confidence) {
        const text = `Baggage ML Prediction Result:\nRisk Level: ${riskLevel}\nConfidence: ${confidence}`;
        
        navigator.clipboard.writeText(text).then(() => {
            // Show success feedback
            const originalText = button.innerHTML;
            button.innerHTML = '<i data-feather="check"></i> Copied!';
            button.classList.add('btn-success');
            
            setTimeout(() => {
                button.innerHTML = originalText;
                button.classList.remove('btn-success');
                feather.replace();
            }, 2000);
        });
    }
}

/**
 * Export prediction data (if needed)
 */
function exportPrediction(format = 'json') {
    const results = {};
    
    // Collect all prediction results on page
    const resultCards = document.querySelectorAll('[data-prediction-result]');
    resultCards.forEach(card => {
        const modelType = card.dataset.modelType;
        const riskLevel = card.querySelector('[data-risk-level]')?.textContent;
        const confidence = card.querySelector('[data-confidence]')?.textContent;
        
        if (modelType) {
            results[modelType] = {
                riskLevel,
                confidence,
                timestamp: new Date().toISOString()
            };
        }
    });
    
    if (Object.keys(results).length > 0) {
        const dataStr = JSON.stringify(results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `baggage_predictions_${Date.now()}.json`;
        link.click();
    }
}

// Global utility functions
window.BaggageML = {
    copyResults,
    exportPrediction,
    resetForm,
    animateResultCards
};

// Auto-animate results when page loads
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(animateResultCards, 500);
});
