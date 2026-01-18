"""Resize all images in a folder to given size (e.g. --size 256 for 256x256) and save to destination folder."""

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Source folder of real images')
    parser.add_argument('--dest', type=str, required=True, help='Destination folder for resized images')
    parser.add_argument('--size', type=int, default=256, help='Size to resize images to (size x size)')
    return parser.parse_args()

def resize_worker(args):
    """Function to resize a single image."""
    src_path, dest_dir, size = args
    
    try:
        dest_path = dest_dir / src_path.name
        
        # Skip if exists
        if dest_path.exists():
            return
        
        with Image.open(src_path) as img:
            if img.mode in ['P', '1']:
                img = img.convert('RGB') 
            
            if hasattr(Image, 'Resampling'):  # Pillow >= 9.1.0
                resample_method = Image.Resampling.LANCZOS
            else:  # Pillow < 9.1.0 fallback
                resample_method = Image.LANCZOS
            
            # Resize with high-quality downsampling
            img = img.resize((size, size), resample_method)
            
            # Save (PIL automatically keeps the mode)
            img.save(dest_path)
    except Exception as e:
        print(f"Error resizing {src_path}: {e}")
        
def main():
    args = parse_args()
    
    # create destination directory
    args.dest.mkdir(parents=True, exist_ok=True)
    
    # look for extensions
    print(f"Scanning {args.src} for images...")
    exts = ['*.png', '*.jpg', '*.jpeg']
    
    # collect files recursively
    files = []
    for ext in exts:
        files.extend(list(args.src.rglob(f"*{ext}")))
        files.extend(list(args.src.rglob(f"*{ext.upper()}")))
        
    # remove duplicates
    files = list(set(files))
    
    print(f"Found {len(files)} images. Resizing to {args.size}x{args.size}...")
    
    # multiprocessing pool
    worker_args = [(f, args.dest, args.size) for f in files]
    
    # run in parallel on all available CPUs
    if files:
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(resize_worker, worker_args), total=len(worker_args), desc="Resizing images"))
        print(f"Resized images saved to {args.dest}")
    else:
        print("No images found to resize.")


if __name__ == '__main__':
    main()