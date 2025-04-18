import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define constants - These are accessible even if torch is not available
FER_EMOTIONS = {
    0: 'Neutral',
    1: 'Happiness', 
    2: 'Sadness',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'  # Only in some datasets like AffectNet-8
}

# Try to import torch-related modules
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing torch and related modules: {e}")
    TORCH_AVAILABLE = False
    
    # Define minimal transforms function for fallback
    class MockTransforms:
        @staticmethod
        def Compose(transforms_list):
            return lambda x: x  # Do nothing transform
            
    transforms = MockTransforms()

def get_transforms(phase):
    """
    Get transforms for data augmentation and normalization
    Args:
        phase (str): 'train' or 'val'/'test'
    Returns:
        transforms object
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class FERDataset(Dataset):
    """
    Dataset class for Facial Expression Recognition datasets
    Compatible with RAF-DB, FERPlus, and AffectNet
    """
    def __init__(self, data_dir, label_file, phase='train', num_classes=7, transform=None):
        """
        Args:
            data_dir (str): directory with images
            label_file (str): file with image names and labels
            phase (str): 'train' or 'val'/'test'
            num_classes (int): number of emotion classes (7 or 8)
            transform: transformation to apply to images
        """
        self.data_dir = data_dir
        self.phase = phase
        self.num_classes = num_classes
        self.transform = transform if transform else get_transforms(phase)
        
        # Read labels
        self.image_paths = []
        self.labels = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    label = int(parts[1])
                    
                    # Skip labels that are outside the number of classes
                    if label < num_classes:
                        image_path = os.path.join(data_dir, image_name)
                        if os.path.exists(image_path):
                            self.image_paths.append(image_path)
                            self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(data_dir, label_file_train, label_file_val, batch_size=32, num_classes=7, num_workers=4):
    """
    Create dataloaders for training and validation
    Args:
        data_dir (str): directory with images
        label_file_train (str): file with training image names and labels
        label_file_val (str): file with validation image names and labels
        batch_size (int): batch size
        num_classes (int): number of emotion classes (7 or 8)
        num_workers (int): number of workers for data loading
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = FERDataset(
        data_dir=data_dir,
        label_file=label_file_train,
        phase='train',
        num_classes=num_classes,
        transform=get_transforms('train')
    )
    
    val_dataset = FERDataset(
        data_dir=data_dir,
        label_file=label_file_val,
        phase='val',
        num_classes=num_classes,
        transform=get_transforms('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def save_checkpoint(state, filename='checkpoint.pth'):
    """
    Save model checkpoint
    Args:
        state (dict): state to save (contains model, optimizer, epoch, etc.)
        filename (str): filename to save to
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None):
    """
    Load model checkpoint
    Args:
        filename (str): filename to load from
        model: model to load weights into
        optimizer: optimizer to load state into
    Returns:
        epoch (int): epoch of the checkpoint
    """
    if not os.path.exists(filename):
        return 0
    
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded checkpoint from {filename} (epoch {checkpoint['epoch']})")
    
    return checkpoint['epoch']

def visualize_confusion_matrix(y_true, y_pred, num_classes=7, figsize=(10, 8)):
    """
    Create and visualize confusion matrix
    Args:
        y_true (numpy.ndarray): ground truth labels
        y_pred (numpy.ndarray): predicted labels
        num_classes (int): number of emotion classes
        figsize (tuple): figure size
    """
    # Get emotion labels based on the number of classes
    emotions = {k: v for k, v in FER_EMOTIONS.items() if k < num_classes}
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[emotions[i] for i in range(num_classes)],
                yticklabels=[emotions[i] for i in range(num_classes)])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('confusion_matrix.png')
    plt.close()

def print_classification_report(y_true, y_pred, num_classes=7):
    """
    Print classification report with precision, recall, f1-score
    Args:
        y_true (numpy.ndarray): ground truth labels
        y_pred (numpy.ndarray): predicted labels
        num_classes (int): number of emotion classes
    """
    # Get emotion labels based on the number of classes
    emotions = {k: v for k, v in FER_EMOTIONS.items() if k < num_classes}
    target_names = [emotions[i] for i in range(num_classes)]
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    
    return report

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, figsize=(12, 5)):
    """
    Plot training and validation loss and accuracy curves
    Args:
        train_losses (list): training losses for each epoch
        val_losses (list): validation losses for each epoch
        train_accs (list): training accuracies for each epoch
        val_accs (list): validation accuracies for each epoch
        figsize (tuple): figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('training_curves.png')
    plt.close()

def preprocess_image(image_path):
    """
    Preprocess a single image for inference
    Args:
        image_path (str): path to image
    Returns:
        tensor: preprocessed image tensor
    """
    transform = get_transforms('val')
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension
