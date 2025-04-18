import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

from model import DSAN
from utils import get_dataloaders, visualize_confusion_matrix, print_classification_report

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DSAN for Facial Expression Recognition')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with image data')
    parser.add_argument('--val_label', type=str, required=True, help='Validation label file')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of expression classes (7 or 8)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50'], help='Backbone network')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint to evaluate')
    
    return parser.parse_args()

def evaluate(model, val_loader, device):
    """
    Evaluate the model
    Args:
        model: DSAN model
        val_loader: validation data loader
        device: device to use (cuda/cpu)
    Returns:
        y_true, y_pred, accuracy
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Evaluating')
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    return y_true, y_pred, accuracy

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loader
    _, val_loader = get_dataloaders(
        args.data_dir,
        args.val_label,  # Dummy parameter, not used
        args.val_label,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    print(f"Evaluation dataset size: {len(val_loader.dataset)}")
    
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
    
    # Evaluate model
    y_true, y_pred, accuracy = evaluate(model, val_loader, device)
    
    # Print results
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Visualize confusion matrix
    visualize_confusion_matrix(y_true, y_pred, num_classes=args.num_classes)
    
    # Print classification report
    report = print_classification_report(y_true, y_pred, num_classes=args.num_classes)
    
    # Save report to file
    with open('classification_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    
    print("Evaluation results saved to classification_report.txt and confusion_matrix.png")

if __name__ == '__main__':
    main()
