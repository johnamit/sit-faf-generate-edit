#!/usr/bin/env python3
"""
Create metadata CSV for 256x256 images
Preserves all columns from metadata_cleaned.csv and updates file paths to point to images_256_cleaned
"""

import pandas as pd
import os
from pathlib import Path

def create_metadata_256():
    # Define paths
    base_dir = Path("/mnt/data/ajohn/faf_flow_edit")
    metadata_path = base_dir / "data" / "metadata_cleaned.csv"
    images_256_dir = base_dir / "data" / "images_256_cleaned"
    output_path = base_dir / "data" / "metadata_cleaned_256.csv"
    
    print(f"Reading metadata from: {metadata_path}")
    # Read the original metadata
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} rows from metadata_cleaned.csv")
    
    # Get list of all images in images_256_cleaned directory
    print(f"\nScanning images in: {images_256_dir}")
    available_images = set()
    if images_256_dir.exists():
        for img_file in images_256_dir.iterdir():
            if img_file.is_file():
                available_images.add(img_file.name)
    
    print(f"Found {len(available_images)} images in images_256_cleaned")
    
    # Filter dataframe to only include rows where the file_name exists in images_256_cleaned
    print("\nMatching rows to available 256 images by file_name...")
    df_matched = df[df['file_name'].isin(available_images)].copy()
    
    print(f"Matched {len(df_matched)} rows (out of {len(df)} total)")
    
    if len(df_matched) == 0:
        print("\nWARNING: No matching images found!")
        print("Sample file_names from metadata:")
        print(df['file_name'].head(10).tolist())
        print("\nSample images from directory:")
        print(list(available_images)[:10])
        return
    
    # Update the file_path column to point to images_256_cleaned (full absolute path)
    print("\nUpdating file paths...")
    df_matched['file_path'] = df_matched['file_name'].apply(
        lambda x: str(images_256_dir / x)
    )
    
    # Update file_path_original if it exists
    if 'file_path_original' in df_matched.columns:
        df_matched['file_path_original'] = df_matched['file_name'].apply(
            lambda x: str(images_256_dir / x)
        )
    
    # Save the new metadata CSV
    print(f"\nSaving metadata to: {output_path}")
    df_matched.to_csv(output_path, index=False)
    
    print(f"\n✓ Success! Created {output_path}")
    print(f"  - Total rows: {len(df_matched)}")
    print(f"  - Total columns: {len(df_matched.columns)}")
    print(f"\nColumn names preserved:")
    for i, col in enumerate(df_matched.columns[:10]):
        print(f"  {i+1}. {col}")
    if len(df_matched.columns) > 10:
        print(f"  ... and {len(df_matched.columns) - 10} more columns")
    
    # Show sample of the data
    print("\nSample rows:")
    print(df_matched[['file_name', 'file_path', 'pat', 'sdb', 'age', 'gene']].head())
    
    # Show count by gene if available
    if 'gene' in df_matched.columns:
        print("\nImage count by gene:")
        print(df_matched['gene'].value_counts().head(10))

if __name__ == "__main__":
    create_metadata_256()
