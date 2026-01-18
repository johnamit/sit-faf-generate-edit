"""
SiT/dataset.py:
    - The data pipeline for loading retinal images and associated metadata.
    - Updated with Synchronized Flipping (Image + Label) to fix laterality bug.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import json
from PIL import Image
import argparse
from torchvision import transforms
import random
import torchvision.transforms.functional as TF

class RetinalDataset(Dataset):
    def __init__(self, csv_file, img_dir, mapping_file, transform=None, do_augment=True):
        """
        Args:
            csv_file (str): Path to the metadata.csv
            img_dir (str): Path to the folder with images
            mapping_file (str): Path to the existing class_mapping.json
            transform (callable, optional): Transform to be applied on a sample.
            do_augment (bool): If True, applies random horizontal flips with synchronized label swapping.
        """
        self.csv_path = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.do_augment = do_augment
        
        # 1. Load the dataframe
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            raise RuntimeError(f"Error loading CSV file {csv_file}: {e}")
        
        # 2. Load Gene Mappings from JSON
        try:
            with open(mapping_file, 'r') as f:
                self.gene_to_idx = json.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Cant find mapping file at {mapping_file}: {e}")
        
        self.idx_to_gene = {val: key for key, val in self.gene_to_idx.items()}
        self.unique_genes = list(self.gene_to_idx.keys())
        
        # 3. Handle laterality
        self.lat_to_idx = {
            'L': 0, 'Left': 0, 'left': 0,
            'R': 1, 'Right': 1, 'right': 1
        }
        
        # Log augmentation status
        print(f"[Dataset] Loaded {len(self.df)} images, {len(self.gene_to_idx)} genes")
        print(f"[Dataset] Synchronized augmentation: {'ENABLED' if do_augment else 'DISABLED'}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image loading
        img_name = row['file_name']
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Error loading image {img_path}: {e}")
            
        # Process attributes
        
        # 1. Gene
        gene_raw = str(row['gene'])
        gene_idx = self.gene_to_idx.get(gene_raw, 0)
            
        # 2. Laterality (get original label BEFORE any augmentation)
        lat_raw = str(row['laterality'])
        lat_idx = self.lat_to_idx.get(lat_raw, 0)
        
        # 3. Age
        age_raw = row['age']
        if pd.isna(age_raw):
            age_norm = 0.5
        else:
            try:
                age_norm = float(age_raw) / 100.0
                age_norm = max(0.0, min(1.0, age_norm))
            except:
                age_norm = 0.5
        
        # ============================================================
        # SYNCHRONIZED AUGMENTATION: Flip image AND label together
        # This teaches the model that laterality = spatial orientation
        # ============================================================
        if self.do_augment and random.random() > 0.5:
            image = TF.hflip(image)    # Flip the pixels
            lat_idx = 1 - lat_idx      # Flip the label: L(0)↔R(1)
        
        # Apply standard transforms (Resize, ToTensor, Normalize)
        # NOTE: These must NOT include RandomHorizontalFlip!
        if self.transform:
            image = self.transform(image)
        
        return image, (gene_idx, lat_idx, age_norm)
    
    def get_num_genes(self):
        """Return the total number of classes in the JSON"""
        return len(self.unique_genes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RetinalDataset')
    parser.add_argument("--csv", type=str, default="data/metadata.csv")
    parser.add_argument("--img_dir", type=str, default="data/images")
    parser.add_argument("--mapping", type=str, default="data/class_mapping.json")
    args = parser.parse_args()
    
    print(f"Testing Dataset...\nCSV: {args.csv}\nImages: {args.img_dir}\nMap: {args.mapping}")
    print("-" * 40)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = RetinalDataset(args.csv, args.img_dir, args.mapping, transform=transform, do_augment=True)
    
    print(f"\nTotal Images: {len(dataset)}")
    print(f"Gene Count: {dataset.get_num_genes()}")
    
    # Test synchronized flipping
    print("\n--- Testing Synchronized Flip ---")
    random.seed(42)
    img1, (g1, l1, a1) = dataset[0]
    random.seed(42)
    img2, (g2, l2, a2) = dataset[0]
    print(f"Same seed, same result: lat={l1} == {l2}: {l1 == l2}")
    
    random.seed(123)
    img3, (g3, l3, a3) = dataset[0]
    print(f"Different seed may flip: lat changed from {l1} to {l3}")

