"""
Makes sample dataset and image directory from a larger dataset:
1. Loads the master metadata CSV.
2. Filters for valid rows (Age/Gene/Laterality present).
3. Randomly samples N entries (reproducible via random_state).
4. Saves the sample metadata CSV.
5. Copies the corresponding images to a target directory.
"""

import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input master CSV file')
    parser.add_argument('--source_img_dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the sample CSV file')
    parser.add_argument('--output_img_dir', type=str, required=True, help='Directory to save copied sample images')
    parser.add_argument('--n_samples', type=int, required=True, help='Number of samples to select for the sample dataset')
    return parser.parse_args()

def create_sample_csv(input_csv, output_csv, n_samples):
    """Read input CSV, filter valid rows, sample N entries, and save to output CSV."""
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Filter for usable rows
    df_clean = df.dropna(subset=['gene', 'age', 'laterality', 'file_name'])
    print(f"Total usable samples: {len(df_clean)}")
    
    # make sure there are enough samples
    if n_samples > len(df_clean):
        print(f"Warning: Requested {n_samples} samples but only {len(df_clean)} available. Taking all.")
        n_samples = len(df_clean)
        
    # Randomly sample N entries
    df_sample = df_clean.sample(n=n_samples, random_state=42)
    
    # report statistics
    print(f"\nGenerated sample stats:")
    print(f"- Total Samples: {len(df_sample)}")
    print(f"- Left Eyes:  {len(df_sample[df_sample['laterality'] == 'L'])} ({len(df_sample[df_sample['laterality'] == 'L']) / len(df_sample) * 100:.1f}%)")
    print(f"- Right Eyes: {len(df_sample[df_sample['laterality'] == 'R'])} ({len(df_sample[df_sample['laterality'] == 'R']) / len(df_sample) * 100:.1f}%)")
    print(f"- Unique Genes: {df_sample['gene'].nunique()}")
    print(f"- Age: Mean={df_sample['age'].mean():.1f}, Min={df_sample['age'].min()}, Max={df_sample['age'].max()}")
    
    # save the sample CSV
    df_sample.to_csv(output_csv, index=False)
    print(f"\nSample CSV saved to {output_csv}")
    return df_sample

def copy_images(df, source_dir, dest_dir):
    """Copies images listed in the dataframe from source_dir to dest_dir"""
    os.makedirs(dest_dir, exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying files"):
        filename = row['file_name']
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            missing_count += 1
            
    print(f"Successfully copied: {copied_count}")
    print(f"Missing files:       {missing_count}")
    print(f"Target Directory:    {dest_dir}")
    
def main():
    args = parse_args()
    
    # create sample CSV
    df_sample = create_sample_csv(args.input_csv, args.output_csv, args.n_samples)
    
    # copy corresponding images
    copy_images(df_sample, args.source_img_dir, args.output_img_dir)
    
if __name__ == '__main__':
    main()