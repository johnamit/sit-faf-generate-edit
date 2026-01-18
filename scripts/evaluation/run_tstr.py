"""
Run TSTR evaluation for medical image classification using ResNet50.
- Train on synthetic data and test on real data for SiT and StyleGAN2ADA
- Train and test on real data for upper bound comparison
- Handles dynamic class mapping based on available genes in datasets
- Outputs accuracy and classification report to specified directory
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import argparse
import json
import time
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    # experienment args
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment')
    
    # training args
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--train_mode', type=str, default='synthetic', choices=['real', 'synthetic'])
    
    # testing args
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the testing CSV file')
    parser.add_argument('--test_img_dir', type=str, required=True, help='Directory with testing images')
    parser.add_argument('--test_mode', type=str, default='real', choices=['real']) # TSTR only tests on real data
    
    # other args
    parser.add_argument('--mapping_json', type=str, required=True, help='Path to the JSON file with label mappings')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing')
    
    return parser.parse_args()

    
class MedicalDataset(Dataset):
    def __init__(self, csv_file, img_dir, class_map, transform=None, mode='real'):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        self.class_map = class_map
            
        # Filter to only include genes that are in the class mapping
        self.df = self.df[self.df['gene'].isin(self.class_map.keys())].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gene = row['gene']
        
        # handling filenames
        if self.mode == 'synthetic':
            # Use the synthetic_file column which contains the actual filename
            fname = row['synthetic_file']
        else:
            fname = row['file_name']
            
        img_path = os.path.join(self.img_dir, fname)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Image not found: {img_path}") from e
        
        label = self.class_map[gene]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

def get_resnet(num_classes, device):
    model = models.resnet50(pretrained=True)
    
    # replace the final layer for our specific number of genes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)
    
    # Transformations (Imagenet normalization)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Basic augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets and Dataloaders
    print(f"Experiment: {args.experiment_name}")
    print("Preparing datasets and dataloaders...")
    
    # Load mapping JSON to get valid gene classes
    with open(args.mapping_json, 'r') as f:
        full_mapping = json.load(f)
    
    # Read CSVs to find which genes are actually present in train and test data
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    
    # Get genes that exist in both train and test sets AND are in the mapping
    train_genes = set(train_df['gene'].unique())
    test_genes = set(test_df['gene'].unique())
    valid_genes = (train_genes & test_genes) & set(full_mapping.keys())
    
    # Create a new mapping with only the genes that are present, with consecutive indices
    sorted_genes = sorted(valid_genes)
    class_map = {gene: idx for idx, gene in enumerate(sorted_genes)}
    num_classes = len(class_map)
    
    print(f"Using {num_classes} classes (out of {len(full_mapping)} in mapping JSON)")
    print(f"Classes: {sorted_genes}")
    
    # Check for genes present in data but missing from the other set
    train_only = train_genes - test_genes
    test_only = test_genes - train_genes
    if train_only:
        print(f"Warning: {len(train_only)} genes in train but not test: {train_only}")
    if test_only:
        print(f"Warning: {len(test_only)} genes in test but not train: {test_only}")
    
    train_dataset = MedicalDataset(args.train_csv, args.train_img_dir, class_map, transform=train_transform, mode=args.train_mode)
    test_dataset = MedicalDataset(args.test_csv, args.test_img_dir, class_map, transform=test_transform, mode=args.test_mode)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True) # pin_memory for speed
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    
    # Model, Loss, Optimizer
    model = get_resnet(num_classes, device)
    criterion = nn.CrossEntropyLoss() # suitable for multi-class classification
    optimiser = optim.Adam(model.parameters(), lr=1e-4) # Adam optimizer with a small lr for fine-tuning
    
    # Training loop
    print(f"Training on {len(train_dataset)} samples (mode: {args.train_mode})")
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimiser.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()
            _, preds = torch.max(preds, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Train Loss: {epoch_loss:.2f}, Train Acc: {epoch_acc*100:.2f}%")
        
        training_time = time.time() - start_time
        print(f"Training finished in {training_time//60:.0f}m {training_time%60:.0f}s")
        
    
    # Evaluation
    print(f"Testing on {len(test_dataset)} samples (mode: {args.test_mode})")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            _, preds = torch.max(preds, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    
    # Report results
    final_acc = accuracy_score(all_labels, all_preds)
    clf_report = classification_report(all_labels, all_preds, target_names=sorted_genes, zero_division=0) # avoid division by zero
    
    print(f"\n {args.experiment_name} Final Test Accuracy: {final_acc*100:.2f}%")
    
    # save results
    results_path = os.path.join(args.outdir, f"{args.experiment_name}_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Training Set: {args.train_img_dir} (mode: {args.train_mode})\n")
        f.write(f"Testing Set: {args.test_img_dir} (mode: {args.test_mode})\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Classes: {sorted_genes}\n\n")
        f.write(f"Final Test Accuracy: {final_acc*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(clf_report)
        
    print(f"Report of {args.experiment_name} saved to {results_path}")
    
if __name__ == '__main__':
    main()