import sys
import os
import argparse
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

STYLEGAN_REPO_PATH = "../stylegan2-ada-pytorch" 
sys.path.append(STYLEGAN_REPO_PATH)

import dnnlib
import legacy


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--network', type=str, required=True, help='Path to your trained StyleGAN .pkl file')
    parser.add_argument('--csv', type=str, required=True, help='Path to the 10k blueprint (eval_10k.csv)')
    parser.add_argument('--outdir', type=str, required=True, help='Output folder for images')
    parser.add_argument('--mapping', type=str, default='data/class_mapping.json', help='JSON mapping Gene Name -> Class Index')
    parser.add_argument('--trunc', type=float, default=1.0, help='Truncation psi (1.0 = diversity, 0.5 = quality)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda')
    
    # 1. Setup
    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, "stylegan_manifest.csv")
    
    print(f"--- StyleGAN 10k Generator ---")
    print(f"Network: {args.network}")
    print(f"Blueprint: {args.csv}")
    
    # 2. Load Model
    print("Loading network...")
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    # 3. Load Data & Mapping
    df = pd.read_csv(args.csv)
    with open(args.mapping, 'r') as f:
        gene_mapping = json.load(f)
        
    print(f"Loaded {len(df)} targets from CSV.")
    
    manifest_rows = []
    
    # 4. Generation Loop
    # We iterate through the CSV to ensure 1-to-1 matching
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating syntheye images"):
        
        # --- A. Prepare Label (Conditioning) ---
        gene_name = row['gene']
        real_filename = row['file_name']
        
        # Verify Gene exists in mapping
        if gene_name not in gene_mapping:
            print(f"Warning: Gene '{gene_name}' not found in mapping. Skipping row {i}.")
            continue
            
        class_idx = gene_mapping[gene_name]
        
        # Create One-Hot Label Tensor
        # Shape: [Batch_Size, Num_Classes]
        label = torch.zeros([1, G.c_dim], device=device)
        label[0, class_idx] = 1
        
        # --- B. Prepare Noise (z) ---
        # We generate a random Z. This implicitly determines Age/Laterality randomly.
        z = torch.randn([1, G.z_dim], device=device)
        
        # --- C. Generate ---
        # truncation_psi=1 ensures we get the full diversity of the model
        # noise_mode='const' keeps texture layers deterministic per Z
        img = G(z, label, truncation_psi=args.trunc, noise_mode='const')
        
        # --- D. Post-Process & Save ---
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img[0].cpu().numpy()
        
        # Handle Grayscale vs RGB
        if img_np.shape[-1] == 1:
            mode = 'L'
            img_np = img_np.squeeze(-1)
        else:
            mode = 'RGB'
            
        # Filename: Match the CSV blueprint exactly
        # e.g., "0000_ABCA4_L_55.png"
        # Even though Age/Lat are random in the image, we keep the filename 
        # so the Folder Structure matches your SiT folder for easy sorting.
        fname = f"{i:04d}_{gene_name}_{row['laterality']}_{int(row['age'])}.png"
        save_path = os.path.join(args.outdir, fname)
        Image.fromarray(img_np, mode).save(save_path)
        
        # Log to Manifest
        manifest_rows.append({
            'synthetic_path': fname,              # Relative path is usually better
            'real_source_path': real_filename,    # The real image from blueprint
            'gene': gene_name,
            'age_requested': row['age'],          # What we wanted (from CSV)
            'lat_requested': row['laterality'],   # What we wanted (from CSV)
            'class_idx': class_idx
        })

# 6. Save CSV
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Saved Manifest to {manifest_path}")

if __name__ == "__main__":
    main()