import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap

from model import DSAN
from utils import preprocess_image, FER_EMOTIONS

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize DSAN attention for Facial Expression Recognition')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of expression classes (7 or 8)')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50'], help='Backbone network')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    return parser.parse_args()

def visualize_attention(model, image_path, num_classes, output_path):
    """
    Visualize the attention weights of the DSAN model
    Args:
        model: DSAN model
        image_path: path to input image
        num_classes: number of expression classes
        output_path: path to save visualization
    """
    # Set model to evaluation mode
    model.eval()
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    # Get original image for overlay
    orig_image = Image.open(image_path).convert('RGB')
    orig_image = orig_image.resize((224, 224), Image.LANCZOS)
    orig_image_np = np.array(orig_image)
    
    # Get device
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs, attention_weights = model(image_tensor)
        _, pred_idx = torch.max(outputs, 1)
        pred_idx = pred_idx.item()
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get emotion labels
    emotions = {k: v for k, v in FER_EMOTIONS.items() if k < num_classes}
    pred_label = emotions[pred_idx]
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(12, 5))
    
    # 1. Original image with prediction
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(orig_image_np)
    ax1.set_title(f'Predicted: {pred_label}')
    ax1.axis('off')
    
    # 2. Bar chart of emotion probabilities
    ax2 = fig.add_subplot(1, 2, 2)
    emotion_names = [emotions[i] for i in range(num_classes)]
    probs = probabilities.cpu().numpy()
    
    # Sort by probability
    indices = np.argsort(probs)[::-1]
    sorted_emotions = [emotion_names[i] for i in indices]
    sorted_probs = probs[indices]
    
    bars = ax2.barh(range(num_classes), sorted_probs, align='center')
    ax2.set_yticks(range(num_classes))
    ax2.set_yticklabels(sorted_emotions)
    ax2.set_xlabel('Probability')
    ax2.set_title('Emotion Probabilities')
    
    # Highlight the predicted class
    for i, idx in enumerate(indices):
        if idx == pred_idx:
            bars[i].set_color('red')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Create a separate visualization for attention weights
    attention_vis_path = output_path.replace('.png', '_attention.png')
    
    # Get attention weights and reshape for visualization
    attention = attention_weights.cpu().numpy()[0]
    
    # Sort attention weights and get top 10% for sparse visualization
    num_to_show = max(int(len(attention) * 0.1), 1)
    top_indices = np.argsort(attention)[-num_to_show:]
    
    # Create figure for attention weights
    fig = plt.figure(figsize=(10, 4))
    
    # Bar plot of top attention weights
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(num_to_show), attention[top_indices], align='center')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Top 10% Attention Weights (Sparse Attention)')
    
    plt.tight_layout()
    plt.savefig(attention_vis_path)
    plt.close()
    
    print(f"Visualizations saved to {output_path} and {attention_vis_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = DSAN(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=False  # No need for pretrained weights when loading checkpoint
    )
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} does not exist!")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Get output path
    image_name = os.path.basename(args.image_path)
    output_path = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_viz.png")
    
    # Visualize attention
    visualize_attention(model, args.image_path, args.num_classes, output_path)

if __name__ == '__main__':
    main()
