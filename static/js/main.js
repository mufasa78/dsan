document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('preview-image');
    const resultSection = document.getElementById('result-section');
    const predictionResult = document.getElementById('prediction-result');
    const probChart = document.getElementById('prob-chart');
    const attentionChart = document.getElementById('attention-chart');
    const demoModeAlert = document.getElementById('demo-mode-alert');
    const loadingIndicator = document.getElementById('loading-indicator');
    const predictionContainer = document.getElementById('prediction-container');
    const tryAnotherBtn = document.getElementById('try-another-btn');
    
    // Check if we're in demo mode
    const isDemoMode = typeof window.demoMode !== 'undefined' ? window.demoMode : false;
    
    // Emotion icons mapping
    const emotionIcons = {
        'Neutral': 'meh',
        'Happiness': 'smile-beam',
        'Sadness': 'sad-tear',
        'Surprise': 'surprise',
        'Fear': 'grimace',
        'Disgust': 'dizzy',
        'Anger': 'angry',
        'Contempt': 'meh-rolling-eyes'
    };
    
    // Show or hide the loading indicator
    function toggleLoading(isLoading) {
        if (isLoading) {
            loadingIndicator.style.display = 'block';
            predictionContainer.style.display = 'none';
        } else {
            loadingIndicator.style.display = 'none';
            predictionContainer.style.display = 'block';
        }
    }
    
    // Reset the form and result section
    function resetForm() {
        imageInput.value = '';
        resultSection.style.display = 'none';
        probChart.src = '';
        attentionChart.src = '';
        toggleLoading(true);
        
        // Scroll back to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    // Add event listener for the "Try Another" button
    if (tryAnotherBtn) {
        tryAnotherBtn.addEventListener('click', resetForm);
    }
    
    // Preview the selected image
    imageInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });
    
    // Submit the form
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            // Show more user-friendly error with Bootstrap toast or alert
            const errorMsg = 'Please select an image file before submitting.';
            
            // Create a bootstrap alert
            const alertDiv = document.createElement('div');
            alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
            alertDiv.setAttribute('role', 'alert');
            alertDiv.innerHTML = `
                <i class="fas fa-exclamation-circle me-2"></i>
                ${errorMsg}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            
            // Insert the alert before the form
            uploadForm.parentNode.insertBefore(alertDiv, uploadForm);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                const bsAlert = new bootstrap.Alert(alertDiv);
                bsAlert.close();
            }, 5000);
            
            return;
        }
        
        // Show the result section and loading state
        resultSection.style.display = 'block';
        toggleLoading(true);
        
        // Clear previous results
        probChart.src = '';
        attentionChart.src = '';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send the request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            toggleLoading(false);
            
            if (data.error) {
                predictionResult.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i> 
                        ${data.error}
                    </div>
                `;
                return;
            }
            
            // Show demo mode alert if needed
            if (isDemoMode || data.demo_mode) {
                demoModeAlert.style.display = 'block';
            } else {
                demoModeAlert.style.display = 'none';
            }
            
            // Update the prediction result with emotion and confidence
            const iconClass = emotionIcons[data.predicted_label] || 'question';
            const emotionProb = data.probabilities[data.predicted_label] * 100;
            const confidenceLevel = emotionProb > 70 ? 'High' : 
                                   emotionProb > 40 ? 'Medium' : 'Low';
            
            const confidenceBadgeClass = emotionProb > 70 ? 'bg-success' : 
                                        emotionProb > 40 ? 'bg-warning' : 'bg-danger';
            
            predictionResult.innerHTML = `
                <div class="mb-4 text-center">
                    <i class="fas fa-face-${iconClass} fa-3x mb-3"></i>
                    <h3>${data.predicted_label}</h3>
                    <div class="badge ${confidenceBadgeClass} p-2 mt-2">
                        Confidence: ${confidenceLevel} (${emotionProb.toFixed(1)}%)
                    </div>
                </div>
            `;
            
            // Update the charts
            probChart.src = 'data:image/png;base64,' + data.plot;
            attentionChart.src = 'data:image/png;base64,' + data.attention_plot;
            
            // Add listeners to images to show loading state
            [probChart, attentionChart].forEach(img => {
                img.onload = () => {
                    img.classList.add('loaded');
                };
            });
            
            // Scroll to the result section
            resultSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            // Hide loading indicator and show error
            toggleLoading(false);
            predictionResult.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i> 
                    Error: ${error.message}
                </div>
                <p class="text-center mt-3">Please try again with a different image.</p>
            `;
        });
    });
});
