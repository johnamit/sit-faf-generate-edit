"""
scripts/custom_utils/gather_data.py:
    - An ETL pipeline
    1. Reads original CSV source
    2. Copies valid images to a flat folder (./data/images)
    3. Renames images to ensure unique filenames
    4. Creates the metadata CSV with updated filenames and original paths. Used for the project.
"""


import pandas as pd
import shutil
import os
from tqdm import tqdm

# Config
source_csv = "data/nnunet_faf_v0_dataset_v2.csv"
target_dir = "./data"
images_dir = os.path.join(target_dir, "images")
metadata_pth = os.path.join(target_dir, "metadata.csv")

# create directories
os.makedirs(images_dir, exist_ok=True)

# load the original data
print(f"Loading data from {source_csv}")
df = pd.read_csv(source_csv)

new_rows = []
skipped_count = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    src_path = row['file_path_original']
    
    # set unique filenames. Pattern = pat_sdb_filename
    unique_filename = f"{row['pat']}_{row['sdb']}_{row['file_name']}"
    dest_path = os.path.join(images_dir, unique_filename)
    
    try:
        # copy the file if it exists
        if os.path.exists(src_path):
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
            
            # clone the row and update the filename. Convert the row to a dictionary so we can modify it easily
            new_row = row.to_dict()
            new_row['file_name'] = unique_filename
            new_row['original_path_backup'] = src_path
            new_rows.append(new_row)
        else:
            # skip the row if source file is missing
            skipped_count += 1
    
    except Exception as e:
        print(f"Error processing file {src_path}: {e}")
        skipped_count += 1

# save the full-fidelity csv
final_df = pd.DataFrame(new_rows)
final_df.to_csv(metadata_pth, index=False)

print("-"*40)
print(f"Processed {len(final_df)} images")
print(f"Skipped {skipped_count} images due to missing files")
print(f"Images saved to {images_dir}")
print(f"Metadata saved to {metadata_pth}")