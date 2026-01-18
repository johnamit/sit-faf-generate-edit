"""
Cleans the metadata CSV and copies the corresponding images.
1. Filters the input CSV to keep only genes present in class_mapping.json.
2. Saves the cleaned CSV.
3. Copies the valid images from source directory to a new clean directory.
"""

import json
import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing the full dataset.')
    parser.add_argument('--class_mapping_json', type=str, required=True, help='Path to the JSON file containing class mapping.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the cleaned CSV file.')
    parser.add_argument('--source_images_dir', type=str, required=True, help='Directory containing original images')
    parser.add_argument('--output_images_dir', type=str, required=True, help='Directory to save cleaned images')
    return parser.parse_args()

def filter_metadata(input_csv, class_mapping_json, output_csv):
    """Filters metadata CSV to keep only valid genes."""
    # load the dataset
    df = pd.read_csv(input_csv)
    print(f"Original dataset shape: {df.shape}")
    
    # load the class mapping
    with open(class_mapping_json, 'r') as f:
        class_mapping = json.load(f)
    valid_genes = set(class_mapping.keys())
    print(f"Number of valid genes from class mapping: {len(valid_genes)}")
    
    # filter the dataset to keep only valid genes
    if 'gene' not in df.columns:
        raise KeyError("The input CSV does not contain a 'gene' column.")
    
    filtered_df = df[df['gene'].isin(valid_genes)].copy()
    print(f"Cleaned dataset shape: {filtered_df.shape}")
    
    # save the cleaned dataset
    filtered_df.to_csv(output_csv)
    print(f"Cleaned dataset saved to: {output_csv}")
    
    return filtered_df


def copy_images(df, source_dir, dest_dir):
    """Copies images corresponding to the cleaned metadata."""
    os.makedirs(dest_dir, exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    
    print(f"Copying images from {source_dir} to {dest_dir}...")
    for filename in tqdm(df['file_name'], desc='Copying cleaned images'):
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(dest_dir, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            missing_count += 1
    
    print(f"\n--- Summary ---")
    print(f"Successfully copied: {copied_count}")
    print(f"Missing files: {missing_count}")
    print(f"Total processed: {len(df)}")
    
def main():
    args = parse_args()
    
    # Filter metadata
    cleaned_df = filter_metadata(args.input_csv, args.class_mapping_json, args.output_csv)
    
    # Copy corresponding images
    if cleaned_df is not None:
        copy_images(cleaned_df, args.source_images_dir, args.output_images_dir)
        
if __name__ == "__main__":
    main()