document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('preview-image');
    const resultSection = document.getElementById('result-section');
    const predictionResult = document.getElementById('prediction-result');
    const probChart = document.getElementById('prob-chart');
    const attentionChart = document.getElementById('attention-chart');
    const demoModeAlert = document.getElementById('demo-mode-alert');
    
    // Check if we're in demo mode
    const isDemoMode = typeof window.demoMode !== 'undefined' ? window.demoMode : false;
    
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
            alert('Please select an image file.');
            return;
        }
        
        // Show the result section
        resultSection.style.display = 'block';
        predictionResult.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Processing...';
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
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                predictionResult.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i> ${data.error}`;
                return;
            }
            
            // Show demo mode alert if needed
            if (isDemoMode || data.demo_mode) {
                demoModeAlert.style.display = 'block';
            } else {
                demoModeAlert.style.display = 'none';
            }
            
            // Update the prediction result
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
            
            const iconClass = emotionIcons[data.predicted_label] || 'question';
            predictionResult.innerHTML = `<i class="fas fa-face-${iconClass} me-2"></i> ${data.predicted_label}`;
            
            // Update the charts
            probChart.src = 'data:image/png;base64,' + data.plot;
            attentionChart.src = 'data:image/png;base64,' + data.attention_plot;
            
            // Scroll to the result section
            resultSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            predictionResult.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i> Error: ${error.message}`;
        });
    });
});
