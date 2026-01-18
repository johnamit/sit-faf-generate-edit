"""
Generates a large-scale synthetic dataset by sampling demographic distributions 
(Age, Gene, Laterality) from a real metadata CSV.
"""

import torch
import pandas as pd
import argparse
import json
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from models import SiT_models
from transport import create_transport, Sampler

def parse_args():
    parser = argparse.ArgumentParser()
    
    # paths
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to real metadata CSV")
    parser.add_argument("--mapping-file", type=str, default="data/class_mapping.json", help="Path to class mapping JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated images")
    
    # Generation Config
    parser.add_argument("--num-samples", type=int, default=10000, help="Total number of images to generate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def setup_device(device, seed):
    """Setup device and random seed"""
    torch.set_grad_enabled(False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    print(f"Device set to {device} with seed {seed}")
    return torch.device(device)


def load_model_components(ckpt_path, device):
    """Load and returns the SiT model, VAE and transport sampler"""
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Recover training args for model config
    if "args" not in checkpoint:
        raise ValueError("Checkpoint missing training arguments.")
    train_args = checkpoint["args"]
    
    # SiT Model
    latent_size = train_args.image_size // 8
    model = SiT_models[train_args.model](
        input_size=latent_size,
        num_classes=train_args.num_classes
    ).to(device)
    
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
        print("Loaded EMA weights for SiT model")
    else:
        model.load_state_dict(checkpoint["model"])
        print("Loaded standard weights for SiT model")
    model.eval()
    
    # VAE
    vae_type = getattr(train_args, 'vae', 'ema')
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
    print("Loaded VAE model")
    
    # Transport Sampler
    transport = create_transport(
        getattr(train_args, 'path_type', 'Linear'),
        getattr(train_args, 'prediction', 'velocity'),
        getattr(train_args, 'loss_weight', None),
        getattr(train_args, 'train_eps', None),
        getattr(train_args, 'sample_eps', None)
    )
    sampler = Sampler(transport)
    print("Initialized transport sampler")
    
    return model, vae, sampler, train_args, latent_size


def prepare_dataset(csv_path, mapping_path, num_samples, seed):
    """Load real metadata and sample a subset for generation"""
    print(f"Preparing dataset sample ({num_samples} samples) from {csv_path}")
    
    df = pd.read_csv(csv_path)
    with open(mapping_path, 'r') as f:
        gene_to_idx = json.load(f)
        
    # filter for genes present in mapping
    valid_genes = set(gene_to_idx.keys())
    df_filtered = df[df['gene'].isin(valid_genes)]
    
    # Sample distribution
    subset = df_filtered.sample(n=num_samples, replace=True, random_state=seed)
    print(f"Sampled {len(subset)} entries from real metadata")
    
    return subset, gene_to_idx


def run_generation_loop(model, vae, sampler, subset, gene_to_idx, args, latent_size, device, train_args):
    """Main generation loop"""
    os.makedirs(args.output_dir, exist_ok=True)
    lat_map = {'L': 0, 'R': 1}
    
    records = subset.to_dict('records')
    total_batches = (len(records) + args.batch_size - 1) // args.batch_size
    generated_records = []
    
    print(f"Generating {args.num_samples} images in {total_batches} batches...")
    
    for batch_start in tqdm(range(0, len(records), args.batch_size), desc="Batches"):
        batch_records = records[batch_start : batch_start + args.batch_size]
        batch_size = len(batch_records)
        
        # Prepare conditioning vectors
        b_genes, b_lats, b_ages = [], [], []
        
        for row in batch_records:
            b_genes.append(gene_to_idx[row['gene']])
            b_lats.append(lat_map.get(row['laterality'], 0))
            
            try:
                age_val = float(row['age'])
                age_norm = max(0.0, min(1.0, age_val / 100.0))
            except:
                age_norm = 0.5
            b_ages.append(age_norm)
            
        # Convert to tensors
        z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        fixed_genes = torch.tensor(b_genes, device=device)
        fixed_lats = torch.tensor(b_lats, device=device)
        fixed_ages = torch.tensor(b_ages, device=device)
        
        # CFG setup
        if args.cfg_scale > 1.0:
            z = torch.cat([z, z], 0)
            
            # null embeddings
            gene_null = torch.tensor([train_args.num_classes] * batch_size, device=device)
            lat_null = torch.tensor([2] * batch_size, device=device)
            age_null = torch.tensor([-1.0] * batch_size, device=device)
            
            fixed_genes_all = torch.cat([fixed_genes, gene_null], 0)
            fixed_lats_all = torch.cat([fixed_lats, lat_null], 0)
            fixed_ages_all = torch.cat([fixed_ages, age_null], 0)
            
            model_kwargs = dict(
                genes=fixed_genes_all, 
                lats=fixed_lats_all, 
                ages=fixed_ages_all, 
                cfg_scale=args.cfg_scale
            )
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(genes=fixed_genes, lats=fixed_lats, ages=fixed_ages)
            model_fn = model.forward
            
        
        # Inference
        samples = sampler.sample_ode()(z, model_fn, **model_kwargs)[-1]
        
        if args.cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)
            
        # Decode with VAE
        pixels = vae.decode(samples / 0.18215).sample
        pixels = torch.clamp(127.5 * pixels + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Saving images and records
        for j, img_arr in enumerate(pixels):
            img = Image.fromarray(img_arr)
            rec = batch_records[j]
            fname = f"gen_{batch_start + j}_{rec['gene']}_{rec['laterality']}_Age{rec['age']}.png"
            img.save(os.path.join(args.output_dir, fname))
            
            # Record manifest
            rec_copy = rec.copy()
            rec_copy['file_name'] = fname
            generated_records.append(rec_copy)
            
    return generated_records



def main():
    # setup arguments and device
    args = parse_args()
    device = setup_device(args.device, args.seed)
    
    # load model components
    model, vae, sampler, train_args, latent_size = load_model_components(args.ckpt, device)
    
    # prepare dataset
    subset, gene_to_idx = prepare_dataset(args.data_path, args.mapping_file, args.num_samples, args.seed)
    
    # generate images
    records = run_generation_loop(model, vae, sampler, subset, gene_to_idx, args, latent_size, device, train_args)
    
    # Save manifest
    manifest_path = os.path.join(args.output_dir, "synthetic_manifest.csv")
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    
    print(f"Generation complete")
    print(f"- Images saved to: {args.output_dir}")
    print(f"- Manifest saved to: {manifest_path}")
    
if __name__ == "__main__":
    main()
    
    