import streamlit as st
import torch
import numpy as np
import time
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check if model exists
MODEL_AVAILABLE = os.path.exists('checkpoints/model_best.pth')

# Set page configuration
st.set_page_config(
    page_title="DSAN Facial Expression Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Emotion classes
FER_EMOTIONS = {
    0: 'Neutral',
    1: 'Happiness', 
    2: 'Sadness',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger'
}

# Emotion icons
EMOTION_ICONS = {
    'Neutral': '😐',
    'Happiness': '😀',
    'Sadness': '😢',
    'Surprise': '😮',
    'Fear': '😨',
    'Disgust': '🤢',
    'Anger': '😠'
}

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #4CAF50;
            text-align: center;
        }
        .sub-title {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #2196F3;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #1E1E1E;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .emotion-icon {
            font-size: 4rem;
            margin-bottom: 10px;
        }
        .info-text {
            background-color: #2C3333;
            border-left: 5px solid #4CAF50;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .chart-container {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Create demo prediction with visualization
def generate_demo_prediction():
    num_classes = 7
    emotions = {k: v for k, v in FER_EMOTIONS.items() if k < num_classes}
    
    # Generate random probabilities
    np.random.seed(int(time.time()))  # Change seed each time for variation
    probs = np.random.rand(num_classes)
    probs = probs / probs.sum()  # Normalize to sum to 1
    
    pred_idx = np.argmax(probs)
    pred_label = emotions[pred_idx]
    
    emotion_probs = {emotions[i]: float(probs[i]) for i in range(num_classes)}
    
    # Generate feature importance visualization
    feature_count = 20
    
    # Generate some realistic looking attention weights
    base_attention = np.random.rand(feature_count) * 0.3  # Base lower values
    
    # Make a few features more important
    important_indices = [3, 7, 12, 15]
    for idx in important_indices:
        base_attention[idx] = 0.5 + np.random.rand() * 0.5  # Higher values
    
    # Normalize to sum to 1
    demo_attention = base_attention / base_attention.sum()
    
    # Sort indices by importance
    sorted_indices = np.argsort(demo_attention)[::-1]  # Descending
    top_attention = demo_attention[sorted_indices[:10]]  # Top 10 features
    
    return pred_label, emotion_probs, top_attention

# Load the actual model if available (dummy function as placeholder)
def load_model():
    if MODEL_AVAILABLE:
        # In a real implementation, this would load the trained DSAN model
        try:
            from model import DSAN
            model = DSAN(num_classes=7, backbone='resnet18', pretrained=False)
            checkpoint = torch.load('checkpoints/model_best.pth', map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    return None

# Preprocess image function
def preprocess_image(image):
    # Placeholder for image preprocessing logic
    # In a real implementation, this would resize, normalize, and convert to tensor
    return image

# Run prediction on the image
def predict_expression(image, model=None):
    # If we don't have a model, use demo mode
    if model is None:
        pred_label, emotion_probs, attention_weights = generate_demo_prediction()
        return {
            'predicted_label': pred_label,
            'probabilities': emotion_probs,
            'attention_weights': attention_weights,
            'demo_mode': True
        }
    
    # Real prediction code would go here for when model is available
    # This is just a placeholder
    return generate_demo_prediction()

# Plot emotion probabilities
def plot_emotion_probs(probabilities):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by probability
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions = [item[0] for item in sorted_items]
    probs = [item[1] for item in sorted_items]
    
    # Create horizontal bar chart
    bars = ax.barh(emotions, probs, color='steelblue')
    
    # Highlight the top emotion
    bars[0].set_color('#4CAF50')
    
    # Add percentage labels
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f'{v:.1%}', va='center')
    
    # Add styling
    ax.set_xlabel('Probability')
    ax.set_title('Emotion Probabilities')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1.1)
    
    # Set background color to match Streamlit's dark theme
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    return fig

# Plot attention weights
def plot_attention_weights(attention_weights):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    x = range(len(attention_weights))
    bars = ax.bar(x, attention_weights, align='center')
    
    # Apply color gradient
    cmap = plt.cm.viridis
    for i, bar in enumerate(bars):
        bar.set_color(cmap(i/len(attention_weights)))
    
    # Add styling
    ax.set_xlabel('Top Feature Indices')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Feature Importance Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in x])
    
    # Set background color to match Streamlit's dark theme
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

# Main function
def main():
    # Load CSS
    load_css()
    
    # Header
    st.markdown('<div class="main-title">Dual Stream Attention Network (DSAN)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Facial Expression Recognition</div>', unsafe_allow_html=True)
    
    # Load model if available
    model = load_model() if MODEL_AVAILABLE else None
    
    # Show demo mode notice if model not available
    if model is None:
        st.markdown(
            '<div class="info-text">📝 <b>Demo Mode:</b> Running without a trained model. Results shown are for demonstration purposes only.</div>', 
            unsafe_allow_html=True
        )
    
    # Sidebar
    st.sidebar.title("About DSAN")
    st.sidebar.markdown("""
    **Dual Stream Attention Network** is designed for robust facial expression recognition in challenging real-world conditions.
    
    It consists of two main components:
    - **GFE-AN**: Global Feature Element-based Attention Network
    - **MFF-AN**: Multi-Feature Fusion-based Attention Network
    
    This architecture handles:
    - Facial occlusions (masks, glasses)
    - Head pose variations
    - Lighting variations
    """)
    
    st.sidebar.title("Recognized Emotions")
    emotion_list = ""
    for emotion, icon in EMOTION_ICONS.items():
        emotion_list += f"- {icon} {emotion}\n"
    st.sidebar.markdown(emotion_list)
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-title">Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image with a face", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add a predict button
            if st.button("Recognize Expression"):
                with st.spinner("Analyzing facial expression..."):
                    # Make prediction
                    result = predict_expression(image, model)
                    
                    # Show prediction results in the second column
                    with col2:
                        st.markdown('<div class="sub-title">Recognition Results</div>', unsafe_allow_html=True)
                        
                        # Display the predicted emotion with icon
                        pred_label = result['predicted_label']
                        emotion_icon = EMOTION_ICONS.get(pred_label, '❓')
                        
                        st.markdown(f'<div class="result-box">'
                                    f'<div class="emotion-icon">{emotion_icon}</div>'
                                    f'<h2>{pred_label}</h2>'
                                    f'<p>Confidence: {result["probabilities"][pred_label]:.1%}</p>'
                                    f'</div>', unsafe_allow_html=True)
                        
                        # Plot emotion probabilities
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('#### Emotion Probabilities')
                        prob_fig = plot_emotion_probs(result['probabilities'])
                        st.pyplot(prob_fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Plot attention weights
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('#### Feature Importance Analysis')
                        attention_fig = plot_attention_weights(result['attention_weights'])
                        st.pyplot(attention_fig)
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show instructions when no image is uploaded
            st.markdown(
                '<div class="info-text">'
                '👆 Upload an image with a face to analyze the facial expression.'
                '<br><br>The model recognizes 7 basic emotions: Neutral, Happiness, Sadness, Surprise, Fear, Disgust, and Anger.'
                '</div>', 
                unsafe_allow_html=True
            )
    
    # If no image uploaded, show information in the second column
    if uploaded_file is None:
        with col2:
            st.markdown('<div class="sub-title">How It Works</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-text">'
                '<h3>Dual Stream Attention Network</h3>'
                '<p>DSAN uses two attention mechanisms to effectively recognize facial expressions even with occlusions and pose variations:</p>'
                '<ol>'
                '<li><b>Global Feature Element-based Attention:</b> Selectively emphasizes informative feature elements.</li>'
                '<li><b>Multi-Feature Fusion-based Attention:</b> Extracts rich semantic information from different representation sub-spaces.</li>'
                '</ol>'
                '<p>This approach achieves state-of-the-art performance on challenging datasets like RAF-DB, FERPlus, and AffectNet.</p>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Add a sample image
            st.markdown('<div class="sub-title">Sample Results</div>', unsafe_allow_html=True)
            
            # Generate sample results
            sample_pred, sample_probs, sample_attention = generate_demo_prediction()
            sample_icon = EMOTION_ICONS.get(sample_pred, '❓')
            
            st.markdown(f'<div class="result-box">'
                        f'<div class="emotion-icon">{sample_icon}</div>'
                        f'<h2>{sample_pred} (Sample)</h2>'
                        f'<p>This is an example of how the results will be displayed.</p>'
                        f'</div>', unsafe_allow_html=True)
            
            # Plot sample emotion probabilities
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('#### Sample Emotion Probabilities')
            sample_prob_fig = plot_emotion_probs(sample_probs)
            st.pyplot(sample_prob_fig)
            st.markdown('</div>', unsafe_allow_html=True)

    # Add information about switching to Flask interface
    st.markdown("---")
    st.markdown("**Note:** This is the Streamlit interface. You can also use the Flask web interface at port 5000.")
    st.markdown("Streamlit provides an alternative, more interactive way to use the facial expression recognition system.")

if __name__ == "__main__":
    main()