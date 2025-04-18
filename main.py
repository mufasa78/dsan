import os
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from web_demo import app
    WEB_DEMO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing web_demo: {e}")
    WEB_DEMO_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description='DSAN for Facial Expression Recognition')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_resnet18_7classes.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of expression classes (7 or 8)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone network')
    parser.add_argument('--mode', type=str, default='web',
                        choices=['web', 'train', 'evaluate', 'visualize'],
                        help='Operation mode')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Directory with image data (for train/evaluate)')
    parser.add_argument('--train_label', type=str, default='',
                        help='Train label file (for train)')
    parser.add_argument('--val_label', type=str, default='',
                        help='Validation label file (for train/evaluate)')
    parser.add_argument('--image_path', type=str, default='',
                        help='Path to input image (for visualize)')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode without a trained model')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == 'web':
        if not WEB_DEMO_AVAILABLE:
            logger.error("Cannot run web demo: web_demo module could not be imported.")
            logger.info("Creating a simple demo Flask app instead.")
            from flask import Flask, render_template, request, jsonify
            
            demo_app = Flask(__name__)
            demo_app.secret_key = os.environ.get("SESSION_SECRET", "facial_expression_recognition")
            
            @demo_app.route('/')
            def index():
                return render_template('index.html', demo_mode=True)
            
            @demo_app.route('/predict', methods=['POST'])
            def predict():
                import io
                import base64
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Demo emotions
                emotions = {
                    0: 'Neutral',
                    1: 'Happiness', 
                    2: 'Sadness',
                    3: 'Surprise',
                    4: 'Fear',
                    5: 'Disgust',
                    6: 'Anger',
                }
                
                # Generate random probabilities
                np.random.seed(42)  # For reproducibility
                probs = np.random.rand(7)
                probs = probs / probs.sum()  # Normalize to sum to 1
                
                pred_idx = np.argmax(probs)
                pred_label = emotions[pred_idx]
                
                emotion_probs = {emotions[i]: float(probs[i]) for i in range(7)}
                
                # Generate visualization
                fig = plt.figure(figsize=(10, 5))
                
                # Bar chart of emotion probabilities
                emotion_names = [emotions[i] for i in range(7)]
                
                # Sort by probability
                indices = np.argsort(probs)[::-1]
                sorted_emotions = [emotion_names[i] for i in indices]
                sorted_probs = probs[indices]
                
                bars = plt.barh(range(7), sorted_probs, align='center')
                plt.yticks(range(7), sorted_emotions)
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
                
                # Create a more informative visualization
                fig = plt.figure(figsize=(10, 4))
                feature_count = 20
                
                # Generate some realistic looking attention weights with a few dominant features
                np.random.seed(42)  # For reproducibility
                base_attention = np.random.rand(feature_count) * 0.3  # Base lower values
                
                # Make a few features more important
                important_indices = [3, 7, 12, 15]
                for idx in important_indices:
                    base_attention[idx] = 0.5 + np.random.rand() * 0.5  # Higher values
                
                # Normalize to sum to 1
                demo_attention = base_attention / base_attention.sum()
                
                # Sort indices by importance
                sorted_indices = np.argsort(demo_attention)[::-1]  # Descending
                sorted_attention = demo_attention[sorted_indices]
                
                # Show only top 10 features
                top_n = 10
                bars = plt.bar(range(top_n), sorted_attention[:top_n], align='center')
                
                # Different colors for different importance levels
                cmap = plt.cm.get_cmap('viridis')
                for i, bar in enumerate(bars):
                    bar.set_color(cmap(i/top_n))
                
                plt.xticks(range(top_n), range(1, top_n+1))
                plt.xlabel('Top Feature Indices')
                plt.ylabel('Attention Weight')
                plt.title('Feature Importance (Demonstration)')
                plt.tight_layout()
                
                # Save attention plot to bytes
                att_buf = io.BytesIO()
                plt.savefig(att_buf, format='png')
                plt.close(fig)
                att_buf.seek(0)
                
                # Convert attention plot to base64 string
                att_img_str = base64.b64encode(att_buf.getvalue()).decode('utf-8')
                
                return jsonify({
                    'predicted_label': pred_label,
                    'probabilities': emotion_probs,
                    'plot': img_str,
                    'attention_plot': att_img_str,
                    'demo_mode': True
                })
            
            demo_app.run(host='0.0.0.0', port=5000)
            return
        
        try:
            # Set global variables in web_demo
            import web_demo
            web_demo.checkpoint_path = args.checkpoint
            web_demo.num_classes = args.num_classes
            
            # Run web app
            app.run(host='0.0.0.0', port=5000)
        except Exception as e:
            logger.exception(f"Error running web demo: {e}")
            sys.exit(1)
    
    elif args.mode == 'train':
        if not args.data_dir or not args.train_label or not args.val_label:
            logger.error("data_dir, train_label, and val_label are required for training.")
            return
        
        try:
            # Import train module and run training
            from train import main as train_main
            sys.argv = [
                'train.py',
                '--data_dir', args.data_dir,
                '--train_label', args.train_label,
                '--val_label', args.val_label,
                '--num_classes', str(args.num_classes),
                '--backbone', args.backbone,
                '--checkpoint', args.checkpoint if os.path.exists(args.checkpoint) else ''
            ]
            train_main()
        except Exception as e:
            logger.exception(f"Error during training: {e}")
            sys.exit(1)
    
    elif args.mode == 'evaluate':
        if not args.data_dir or not args.val_label:
            logger.error("data_dir and val_label are required for evaluation.")
            return
        
        try:
            # Import evaluate module and run evaluation
            from evaluate import main as evaluate_main
            sys.argv = [
                'evaluate.py',
                '--data_dir', args.data_dir,
                '--val_label', args.val_label,
                '--num_classes', str(args.num_classes),
                '--backbone', args.backbone,
                '--checkpoint', args.checkpoint
            ]
            evaluate_main()
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            sys.exit(1)
    
    elif args.mode == 'visualize':
        if not args.image_path:
            logger.error("image_path is required for visualization.")
            return
        
        try:
            # Import visualization module and run visualization
            from visualization import main as visualize_main
            sys.argv = [
                'visualization.py',
                '--image_path', args.image_path,
                '--num_classes', str(args.num_classes),
                '--backbone', args.backbone,
                '--checkpoint', args.checkpoint
            ]
            visualize_main()
        except Exception as e:
            logger.exception(f"Error during visualization: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()
