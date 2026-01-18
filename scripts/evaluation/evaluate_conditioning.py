"""
Judge System for Evaluating Synthetic Image Conditioning.

Trains classifier/regressor models on real data, then evaluates how well
synthetic images match their conditioning labels (laterality and age).

Usage:
    python evaluate_conditioning.py \
        --real-csv data/metadata_cleaned.csv \
        --real-img-dir data/images_256_cleaned \
        --synth-csv data/synthetic_10kSamples/synthetic_manifest.csv \
        --synth-img-dir data/synthetic_10kSamples \
        --output-dir evaluation/conditioning_results
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import argparse
from PIL import Image
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
import matplotlib.pyplot as plt


# Argument Parsing

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate synthetic image conditioning accuracy.')
    
    # Data paths
    parser.add_argument('--real-csv', type=str, required=True, help='Path to real data metadata CSV')
    parser.add_argument('--real-img-dir', type=str, required=True, help='Directory containing real images')
    parser.add_argument('--synth-csv', type=str, required=True, help='Path to synthetic data manifest CSV')
    parser.add_argument('--synth-img-dir', type=str, required=True, help='Directory containing synthetic images')
    parser.add_argument('--output-dir', type=str, default='evaluation/conditioning_results', help='Directory to save results')
    
    # Training params
    parser.add_argument('--train-samples', type=int, default=5000, help='Number of real samples to train judges on')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Evaluation options
    parser.add_argument('--eval-only', action='store_true', help='Skip training, load existing models')
    parser.add_argument('--save-models', action='store_true', help='Save trained judge models')
    
    return parser.parse_args()


# Dataset

class JudgeDataset(Dataset):
    """Dataset for training/evaluating laterality and age judges."""
    
    def __init__(self, csv_path, img_dir, mode='laterality', filename_col='file_name'):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mode = mode
        self.filename_col = filename_col
        
        self.lat_map = {'L': 0, 'R': 1}
        
        # Standard ImageNet normalization for pretrained ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row[self.filename_col]
        img_path = os.path.join(self.img_dir, filename)
        
        # Load image
        try:
            img = Image.open(img_path)
        except Exception:
            # Return black image if file missing
            img = Image.new('L', (224, 224), 0)
        
        img = self.transform(img)
        
        # Get label based on mode
        if self.mode == 'laterality':
            label = self.lat_map.get(row['laterality'], 0)
            return img, torch.tensor(label, dtype=torch.long)
        else:  # age
            label = float(row['age'])
            return img, torch.tensor(label, dtype=torch.float32)


# Model Creation

def create_judge_model(mode, device):
    """Create a ResNet18 model for laterality classification or age regression."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if mode == 'laterality':
        model.fc = nn.Linear(512, 2)  # Binary classification
    else:  # age
        model.fc = nn.Linear(512, 1)  # Regression
    
    return model.to(device)


def get_criterion(mode):
    """Get the appropriate loss function for the mode."""
    if mode == 'laterality':
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()


# Training

def train_judge(model, train_loader, criterion, optimizer, epochs, device, mode):
    """Train a judge model."""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track accuracy for classification
            if mode == 'laterality':
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        
        if mode == 'laterality':
            acc = correct / total
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        else:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    return model


def train_laterality_judge(args, device):
    """Train the laterality classification judge."""
    print("\nTraining Laterality Judge on real data...")
    
    # Create dataset and loader
    dataset = JudgeDataset(args.real_csv, args.real_img_dir, mode='laterality')
    
    # Use subset for training
    train_size = min(args.train_samples, len(dataset))
    train_subset, _ = torch.utils.data.random_split(
        dataset, 
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_subset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    print(f"  Training samples: {train_size}")
    
    # Create model, criterion, optimizer
    model = create_judge_model('laterality', device)
    criterion = get_criterion('laterality')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    model = train_judge(model, train_loader, criterion, optimizer, args.epochs, device, 'laterality')
    
    return model


def train_age_judge(args, device):
    """Train the age regression judge."""
    print("\nTraining Age Judge on real data...")
    
    # Create dataset and loader
    dataset = JudgeDataset(args.real_csv, args.real_img_dir, mode='age')
    
    # Use subset for training
    train_size = min(args.train_samples, len(dataset))
    train_subset, _ = torch.utils.data.random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    print(f"  Training samples: {train_size}")
    
    # Create model, criterion, optimizer
    model = create_judge_model('age', device)
    criterion = get_criterion('age')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    model = train_judge(model, train_loader, criterion, optimizer, args.epochs, device, 'age')
    
    return model


# Evaluation

def evaluate_laterality(model, args, device):
    """Evaluate laterality accuracy on synthetic images."""
    print("\nEvaluating Laterality on synthetic data...")
    
    # Determine filename column (could be 'file_name' or 'synthetic_file')
    df = pd.read_csv(args.synth_csv)
    filename_col = 'synthetic_file' if 'synthetic_file' in df.columns else 'file_name'
    
    dataset = JudgeDataset(args.synth_csv, args.synth_img_dir, mode='laterality', filename_col=filename_col)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    
    # Per-class accuracy
    left_acc = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    right_acc = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'left_accuracy': left_acc,
        'right_accuracy': right_acc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return results


def evaluate_age(model, args, device):
    """Evaluate age correlation on synthetic images."""
    print("\nEvaluating Age on synthetic data...")
    
    # Determine filename column
    df = pd.read_csv(args.synth_csv)
    filename_col = 'synthetic_file' if 'synthetic_file' in df.columns else 'file_name'
    
    dataset = JudgeDataset(args.synth_csv, args.synth_img_dir, mode='age', filename_col=filename_col)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze().cpu().numpy()
            
            all_preds.extend(outputs)
            all_targets.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    correlation = np.corrcoef(all_targets, all_preds)[0, 1]
    r2 = r2_score(all_targets, all_preds)
    mae = np.mean(np.abs(all_targets - all_preds))
    
    results = {
        'correlation': correlation,
        'r2_score': r2,
        'mae': mae,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return results


# Reporting

def print_laterality_results(results):
    """Print laterality evaluation results."""
    print("\n" + "=" * 50)
    print("LATERALITY RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Left Eye Accuracy: {results['left_accuracy']*100:.2f}%")
    print(f"Right Eye Accuracy: {results['right_accuracy']*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted L  Predicted R")
    print(f"  Actual L    {results['confusion_matrix'][0,0]:>10}  {results['confusion_matrix'][0,1]:>10}")
    print(f"  Actual R    {results['confusion_matrix'][1,0]:>10}  {results['confusion_matrix'][1,1]:>10}")
    # print("\nTarget: > 90% accuracy indicates good laterality control")
    print("=" * 50)


def print_age_results(results):
    """Print age evaluation results."""
    print("\n" + "=" * 50)
    print("AGE RESULTS")
    print("=" * 50)
    print(f"Correlation (R): {results['correlation']:.4f}")
    print(f"R-squared:       {results['r2_score']:.4f}")
    print(f"Mean Abs Error:  {results['mae']:.2f} years")
    # print("\nTarget: R > 0.5 or R^2 > 0.25 indicates good age control")
    print("=" * 50)


def save_age_plot(results, output_path):
    """Save scatter plot of predicted vs actual age."""
    plt.figure(figsize=(8, 8))
    plt.scatter(results['targets'], results['predictions'], alpha=0.5, s=10, c='blue')
    
    # Perfect match line
    min_val = min(results['targets'].min(), results['predictions'].min())
    max_val = max(results['targets'].max(), results['predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
    
    plt.xlabel('Target Age (Conditioning Label)')
    plt.ylabel('Predicted Age (From Image)')
    plt.title(f'Age Control Check (R={results["correlation"]:.3f}, R^2={results["r2_score"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved age plot: {output_path}")


def save_results_summary(lat_results, age_results, output_path):
    """Save results summary to text file."""
    with open(output_path, 'w') as f:
        f.write("CONDITIONING EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("LATERALITY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Accuracy: {lat_results['accuracy']*100:.2f}%\n")
        f.write(f"Left Eye Accuracy: {lat_results['left_accuracy']*100:.2f}%\n")
        f.write(f"Right Eye Accuracy: {lat_results['right_accuracy']*100:.2f}%\n\n")
        
        f.write("AGE\n")
        f.write("-" * 30 + "\n")
        f.write(f"Correlation (R): {age_results['correlation']:.4f}\n")
        f.write(f"R-squared: {age_results['r2_score']:.4f}\n")
        f.write(f"Mean Absolute Error: {age_results['mae']:.2f} years\n")
    
    print(f"Saved results summary: {output_path}")


# Main

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train judges
    lat_model = train_laterality_judge(args, device)
    age_model = train_age_judge(args, device)
    
    # Save models if requested
    if args.save_models:
        torch.save(lat_model.state_dict(), os.path.join(args.output_dir, 'laterality_judge.pt'))
        torch.save(age_model.state_dict(), os.path.join(args.output_dir, 'age_judge.pt'))
        print(f"Saved models to {args.output_dir}")
    
    # Evaluate on synthetic data
    lat_results = evaluate_laterality(lat_model, args, device)
    age_results = evaluate_age(age_model, args, device)
    
    # Print results
    print_laterality_results(lat_results)
    print_age_results(age_results)
    
    # Save outputs
    save_age_plot(age_results, os.path.join(args.output_dir, 'age_correlation_plot.png'))
    save_results_summary(lat_results, age_results, os.path.join(args.output_dir, 'results_summary.txt'))


if __name__ == '__main__':
    main()
