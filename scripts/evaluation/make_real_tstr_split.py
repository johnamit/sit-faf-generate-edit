"""
Create a held-out test split from the real cleaned 256x256 images
- Excludes all samples used in the 10k training subset.
- Saves the test split to a new CSV file.
"""

import pandas as pd
import os

# Load the full metadata CSV for cleaned 256x256 images
df_full = pd.read_csv("/mnt/data/ajohn/faf_flow_edit/data/images_256_cleaned/metadata_cleaned_256.csv")

# load 10k subset used for training
df_train = pd.read_csv("/mnt/data/ajohn/faf_flow_edit/data/eval_10kSamples_256res/real_10kSamples_256res/metadata_cleaned_10kSamples_256.csv")

# Print basic stats
print(f"Total Real Images: {len(df_full)}")
print(f"Total Training Images: {len(df_train)}")

# create remainder set for testing. Exclude all training samples from full set.
train_filenames = set(df_train['file_name'].values)
df_test = df_full[~df_full['file_name'].isin(train_filenames)] # exclude training samples

# save test set to csv
print(f"Held-out Testing Images: {len(df_test)}")
df_test.to_csv("/mnt/data/ajohn/faf_flow_edit/data/images_256_cleaned/real_cleaned_256_testset.csv", index=False)
print("Saved test set to /mnt/data/ajohn/faf_flow_edit/data/images_256_cleaned/real_cleaned_256_testset.csv")

