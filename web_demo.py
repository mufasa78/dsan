import os
import torch
import numpy as np
import io
import base64
from PIL import Image
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    from model import DSAN
    from utils import preprocess_image, FER_EMOTIONS
    MODEL_AVAILABLE = True
except Exception as e:
    logging.error(f"Error importing model: {e}")
    MODEL_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "facial_expression_recognition")

# Global variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 7  # Default to 7 classes
checkpoint_path = 'checkpoints/best_model_resnet18_7classes.pth'

def load_model(checkpoint_path, backbone='resnet18', num_classes=7):
    """
    Load DSAN model from checkpoint
    Args:
        checkpoint_path: path to checkpoint
        backbone: backbone network
        num_classes: number of emotion classes
    Returns:
        loaded model
    """
    if not MODEL_AVAILABLE:
        logging.warning("Model modules not available, skipping model loading")
        return None
        
    model = DSAN(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logging.warning(f"Checkpoint {checkpoint_path} not found. Using untrained model.")
    
    model = model.to(device)
    model.eval()
    return model

def predict_expression(image):
    """
    Predict facial expression from an image
    Args:
        image: PIL Image
    Returns:
        prediction results dict
    """
    global model
    
    # Check if model is available
    if not MODEL_AVAILABLE:
        # Create a demo response with random predictions
        emotions = {k: v for k, v in FER_EMOTIONS.items() if k < num_classes}
        
        # Generate random probabilities
        np.random.seed(42)  # For reproducibility
        probs = np.random.rand(num_classes)
        probs = probs / probs.sum()  # Normalize to sum to 1
        
        pred_idx = np.argmax(probs)
        pred_label = emotions[pred_idx]
        
        emotion_probs = {emotions[i]: float(probs[i]) for i in range(num_classes)}
        
        # Generate visualization
        fig = plt.figure(figsize=(10, 5))
        
        # Bar chart of emotion probabilities
        emotion_names = [emotions[i] for i in range(num_classes)]
        
        # Sort by probability
        indices = np.argsort(probs)[::-1]
        sorted_emotions = [emotion_names[i] for i in indices]
        sorted_probs = probs[indices]
        
        bars = plt.barh(range(num_classes), sorted_probs, align='center')
        plt.yticks(range(num_classes), sorted_emotions)
        plt.xlabel('Probability')
        plt.title('Emotion Probabilities (Demo Mode)')
        
        # Highlight the predicted class
        for i, idx in enumerate(indices):
            if idx == pred_idx:
                bars[i].set_color('red')
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # Convert plot to base64 string
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Create a demo attention visualization
        fig = plt.figure(figsize=(10, 4))
        demo_attention = np.random.rand(10)
        plt.bar(range(len(demo_attention)), demo_attention, align='center')
        plt.xlabel('Feature Index')
        plt.ylabel('Attention Weight')
        plt.title('Demo Attention Weights (Not Real Data)')
        plt.tight_layout()
        
        # Save attention plot to bytes
        att_buf = io.BytesIO()
        plt.savefig(att_buf, format='png')
        plt.close(fig)
        att_buf.seek(0)
        
        # Convert attention plot to base64 string
        att_img_str = base64.b64encode(att_buf.getvalue()).decode('utf-8')
        
        return {
            'predicted_label': pred_label,
            'probabilities': emotion_probs,
            'plot': img_str,
            'attention_plot': att_img_str,
            'demo_mode': True
        }
    
    # Ensure model is loaded
    if model is None:
        model = load_model(checkpoint_path, num_classes=num_classes)
        
    if model is None:
        return {
            'error': 'Model could not be loaded. Please check the logs for details.'
        }
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs, attention_weights = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, pred_idx = torch.max(outputs, 1)
        pred_idx = pred_idx.item()
    
    # Get emotion labels
    emotions = {k: v for k, v in FER_EMOTIONS.items() if k < num_classes}
    pred_label = emotions[pred_idx]
    
    # Get all probabilities
    probs = probabilities.cpu().numpy()
    emotion_probs = {emotions[i]: float(probs[i]) for i in range(num_classes)}
    
    # Generate visualization
    fig = plt.figure(figsize=(10, 5))
    
    # Bar chart of emotion probabilities
    emotion_names = [emotions[i] for i in range(num_classes)]
    
    # Sort by probability
    indices = np.argsort(probs)[::-1]
    sorted_emotions = [emotion_names[i] for i in indices]
    sorted_probs = probs[indices]
    
    bars = plt.barh(range(num_classes), sorted_probs, align='center')
    plt.yticks(range(num_classes), sorted_emotions)
    plt.xlabel('Probability')
    plt.title('Emotion Probabilities')
    
    # Highlight the predicted class
    for i, idx in enumerate(indices):
        if idx == pred_idx:
            bars[i].set_color('red')
    
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Convert plot to base64 string
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Get attention weights for top 10%
    attention = attention_weights.cpu().numpy()[0]
    num_to_show = max(int(len(attention) * 0.1), 1)
    top_indices = np.argsort(attention)[-num_to_show:]
    top_attention = attention[top_indices]
    
    # Create another visualization for attention weights
    fig = plt.figure(figsize=(10, 4))
    plt.bar(range(num_to_show), top_attention, align='center')
    plt.xlabel('Feature Index')
    plt.ylabel('Attention Weight')
    plt.title('Top 10% Attention Weights (Sparse Attention)')
    plt.tight_layout()
    
    # Save attention plot to bytes
    att_buf = io.BytesIO()
    plt.savefig(att_buf, format='png')
    plt.close(fig)
    att_buf.seek(0)
    
    # Convert attention plot to base64 string
    att_img_str = base64.b64encode(att_buf.getvalue()).decode('utf-8')
    
    return {
        'predicted_label': pred_label,
        'probabilities': emotion_probs,
        'plot': img_str,
        'attention_plot': att_img_str
    }

@app.route('/')
def index():
    return render_template('index.html', demo_mode=not MODEL_AVAILABLE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read image
        img = Image.open(file.stream).convert('RGB')
        
        # Predict expression
        result = predict_expression(img)
        
        return jsonify(result)
    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({'error': str(e)})

@app.route('/demo')
def demo():
    return render_template('index.html', demo_mode=not MODEL_AVAILABLE)

if __name__ == '__main__':
    # Load model at startup only if available
    if MODEL_AVAILABLE:
        try:
            model = load_model(checkpoint_path, num_classes=num_classes)
        except Exception as e:
            logging.exception(f"Error loading model: {e}")
    else:
        logging.warning("Running in demo mode without the actual model")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
