import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import numpy as np
from tqdm import tqdm

from model import DSAN, FeatureRecalibrationLoss
from utils import get_dataloaders, save_checkpoint, load_checkpoint, plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(description='Train DSAN for Facial Expression Recognition')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with image data')
    parser.add_argument('--train_label', type=str, required=True, help='Train label file')
    parser.add_argument('--val_label', type=str, required=True, help='Validation label file')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of expression classes (7 or 8)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50'], help='Backbone network')
    parser.add_argument('--lambda_val', type=float, default=0.1, help='Lambda for feature recalibration loss')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    return parser.parse_args()

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    Args:
        model: DSAN model
        train_loader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: device to use (cuda/cpu)
    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        gfe_outputs, mff_outputs, attention_weights = model(inputs)
        
        # Calculate loss for GFE-AN branch
        _, gfe_features, _ = model.gfe_an(model.feature_extractor(inputs))
        gfe_loss, gfe_ce_loss, gfe_center_loss, gfe_l1_loss = criterion(
            gfe_outputs, gfe_features, targets, attention_weights
        )
        
        # Calculate loss for MFF-AN branch
        _, mff_features = model.mff_an(model.feature_extractor(inputs))
        mff_loss = nn.CrossEntropyLoss()(mff_outputs, targets)
        
        # Combined loss
        loss = gfe_loss + mff_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Calculate accuracy for both branches and average
        _, gfe_preds = torch.max(gfe_outputs, 1)
        _, mff_preds = torch.max(mff_outputs, 1)
        
        # Combine predictions (simple averaging for now)
        combined_outputs = (gfe_outputs + mff_outputs) / 2
        _, preds = torch.max(combined_outputs, 1)
        
        total += targets.size(0)
        correct += (preds == targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    Args:
        model: DSAN model
        val_loader: validation data loader
        criterion: loss function
        device: device to use (cuda/cpu)
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs, attention_weights = model(inputs)
            
            # Calculate loss
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        args.train_label,
        args.val_label,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = DSAN(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=True
    )
    model = model.to(device)
    
    # Define loss function
    criterion = FeatureRecalibrationLoss(lambda_val=args.lambda_val)
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(args.checkpoint, model, optimizer)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    # Training loop
    num_epochs = args.epochs
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, f"best_model_{args.backbone}_{args.num_classes}classes.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, save_path)
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, save_path)
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
