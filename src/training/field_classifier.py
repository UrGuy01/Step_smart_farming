import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FieldImageDataset(Dataset):
    """Dataset for loading field images"""
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def create_data_loaders(dataset_dir, batch_size=32, image_size=224):
    """Create data loaders for training, validation, and testing"""
    # Define transformations for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transformations for validation/testing (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FieldImageDataset(
        os.path.join(dataset_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = FieldImageDataset(
        os.path.join(dataset_dir, 'val'),
        transform=val_test_transform
    )
    
    test_dataset = FieldImageDataset(
        os.path.join(dataset_dir, 'test'),
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def create_model(num_classes, model_type='basic'):
    """Create a model for field image classification"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'resnet':
        # Use ResNet18 as base model
        model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Replace the last fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_type == 'mobilenet':
        # Use MobileNetV2 as base model
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Replace the last classifier layer
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    else:  # basic model
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    model = model.to(device)
    return model

def train_epoch(model, data_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_model(model, data_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, epochs=20, output_dir='models/field_classifier'):
    """Train the model"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Track best validation accuracy
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    print(f"Training on {device}")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    return best_model_path

def evaluate_model(model, test_loader, classes, output_dir):
    """Evaluate the model on the test set"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Initialize variables for metrics
    all_preds = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Calculate accuracy
    accuracy = sum([all_preds[i] == all_labels[i] for i in range(len(all_preds))]) / len(all_preds)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save evaluation metrics
    metrics = {
        'accuracy': float(accuracy),
        'classification_report': report
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train field image classifier')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing the processed dataset')
    parser.add_argument('--output_dir', type=str, default='models/field_classifier',
                        help='Directory to save model and results')
    parser.add_argument('--model_type', type=str, default='basic',
                        choices=['basic', 'resnet', 'mobilenet'],
                        help='Type of model architecture to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size to resize images to (square)')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Print training parameters
    print("=== Field Classifier Training ===")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {output_dir}")
    print("")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, classes = create_data_loaders(
        args.dataset_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Print class information
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(num_classes, model_type=args.model_type)
    
    # Train model
    print("\nTraining model...")
    best_model_path = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        output_dir=output_dir
    )
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_model = create_model(num_classes, model_type=args.model_type)
    best_model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(best_model, test_loader, classes, output_dir)
    
    print(f"\nTraining complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main() 