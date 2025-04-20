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

from streamlit_app import generate_demo_prediction

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
FER_MODEL_PATH = 'checkpoints/fer_model_best.pth'
RAFDB_MODEL_PATH = 'checkpoints/rafdb_model_best.pth'
MODEL_AVAILABLE = os.path.exists(FER_MODEL_PATH) or os.path.exists(RAFDB_MODEL_PATH)
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 7  # Default to 7 classes

def load_model(checkpoint_path='', backbone='resnet18', num_classes=7):
    """
    Load DSAN models from checkpoints
    Returns:
        list of tuples (model_name, model) or None if no models available
    """
    if not MODEL_AVAILABLE:
        logging.warning("No models available, skipping model loading")
        return None
        
    models = []
    
    try:
        # Try loading FER model
        if os.path.exists(FER_MODEL_PATH):
            fer_model = DSAN(num_classes=num_classes, backbone=backbone, pretrained=False)
            checkpoint = torch.load(FER_MODEL_PATH, map_location=device)
            fer_model.load_state_dict(checkpoint['model'])
            fer_model = fer_model.to(device)
            fer_model.eval()
            models.append(('FER', fer_model))
            logging.info("FER model loaded successfully")
            
        # Try loading RAF-DB model
        if os.path.exists(RAFDB_MODEL_PATH):
            rafdb_model = DSAN(num_classes=num_classes, backbone=backbone, pretrained=False)
            checkpoint = torch.load(RAFDB_MODEL_PATH, map_location=device)
            rafdb_model.load_state_dict(checkpoint['model'])
            rafdb_model = rafdb_model.to(device)
            rafdb_model.eval()
            models.append(('RAF-DB', rafdb_model))
            logging.info("RAF-DB model loaded successfully")
            
        return models if models else None
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return None

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
        plt.xlabel('概率')
        plt.title('表情概率分布（演示模式）')
        
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
        plt.xlabel('特征索引')
        plt.ylabel('注意力权重')
        plt.title('注意力权重分析（演示数据）')
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
        model = load_model(num_classes=num_classes)
        
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
    
    all_probabilities = []
    all_attention_weights = []
    
    # Get predictions from each model
    with torch.no_grad():
        for model_name, model_instance in model:
            outputs, attention = model_instance(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_probabilities.append(probabilities[0].cpu().numpy())
            all_attention_weights.append(attention.cpu().numpy())
    
    # Average predictions from all models
    avg_probabilities = np.mean(all_probabilities, axis=0)
    avg_attention_weights = np.mean(all_attention_weights, axis=0)
    
    # Get predicted label
    pred_idx = np.argmax(avg_probabilities)
    pred_label = FER_EMOTIONS[pred_idx]
    
    # Create probability dictionary
    emotion_probs = {FER_EMOTIONS[i]: float(avg_probabilities[i]) 
                    for i in range(len(FER_EMOTIONS))}
    
    # Prepare visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # Plot emotion probabilities
    plt.bar(emotion_probs.keys(), emotion_probs.values())
    plt.xticks(rotation=45)
    plt.title('表情概率分布')
    
    plt.subplot(1, 2, 2)
    # Plot attention weights
    plt.bar(range(10), avg_attention_weights.flatten()[:10])
    plt.title('前10个注意力权重')
    plt.xlabel('特征索引')
    plt.ylabel('权重值')
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    
    return {
        'predicted_label': pred_label,
        'probabilities': emotion_probs,
        'attention_weights': avg_attention_weights.flatten()[:10].tolist(),
        'plot': plot_url,
        'demo_mode': False
    }

@app.route('/')
def index():
    return render_template('index.html', demo_mode=not MODEL_AVAILABLE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess_image(image)
        
        if model is None:
            return jsonify(generate_demo_prediction())
            
        all_probabilities = []
        all_attention_weights = []
        
        # Get predictions from each model
        with torch.no_grad():
            for model_name, model_instance in model:
                outputs, attention = model_instance(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                all_probabilities.append(probabilities[0].cpu().numpy())
                all_attention_weights.append(attention.cpu().numpy())
        
        # Average predictions from all models
        avg_probabilities = np.mean(all_probabilities, axis=0)
        avg_attention_weights = np.mean(all_attention_weights, axis=0)
        
        # Get predicted label
        pred_idx = np.argmax(avg_probabilities)
        pred_label = FER_EMOTIONS[pred_idx]
        
        # Create probability dictionary
        emotion_probs = {FER_EMOTIONS[i]: float(avg_probabilities[i]) 
                        for i in range(len(FER_EMOTIONS))}
        
        # Prepare visualization
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        # Plot emotion probabilities
        plt.bar(emotion_probs.keys(), emotion_probs.values())
        plt.xticks(rotation=45)
        plt.title('表情概率分布')
        
        plt.subplot(1, 2, 2)
        # Plot attention weights
        plt.bar(range(10), avg_attention_weights.flatten()[:10])
        plt.title('前10个注意力权重')
        plt.xlabel('特征索引')
        plt.ylabel('权重值')
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        
        return jsonify({
            'predicted_label': pred_label,
            'probabilities': emotion_probs,
            'attention_weights': avg_attention_weights.flatten()[:10].tolist(),
            'plot': plot_url,
            'demo_mode': False
        })
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

@app.route('/demo')
def demo():
    return render_template('index.html', demo_mode=not MODEL_AVAILABLE)

if __name__ == '__main__':
    # Load model at startup only if available
    if MODEL_AVAILABLE:
        try:
            model = load_model(num_classes=num_classes)
        except Exception as e:
            logging.exception(f"Error loading model: {e}")
    else:
        logging.warning("Running in demo mode without the actual model")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
