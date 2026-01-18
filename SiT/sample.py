# SiT/sample.py

import torch
import argparse
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import json
import os
import sys

def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the Checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    # We need to recover the training arguments to know the model size/depth
    # If using EMA, the args are usually stored in the checkpoint
    if "args" in checkpoint:
        train_args = checkpoint["args"]
        print(f"Loaded config from checkpoint: Model={train_args.model}, Size={train_args.image_size}")
    else:
        # Fallback if args aren't in checkpoint (rare for this repo structure)
        print("Warning: Args not found in checkpoint. Using command line defaults.")
        train_args = args

    # 2. Load the Class Mapping (to convert "ABCA4" -> 0)
    with open(args.mapping_file, 'r') as f:
        gene_to_idx = json.load(f)
    
    # 3. Process Inputs (Human String -> Tensor ID)
    
    # Gene Processing
    if args.gene not in gene_to_idx:
        print(f"Error: Gene '{args.gene}' not found in mapping file.")
        print(f"Available genes: {list(gene_to_idx.keys())}")
        return
    gene_idx = gene_to_idx[args.gene]
    
    # Laterality Processing
    lat_map = {'L': 0, 'Left': 0, 'left': 0, 'R': 1, 'Right': 1, 'right': 1}
    lat_idx = lat_map.get(args.laterality, 0) # Default to 0 if unknown
    
    # Age Processing (Normalize 0-100 -> 0.0-1.0)
    age_norm = float(args.age) / 100.0
    age_norm = max(0.0, min(1.0, age_norm))

    print(f"Generating Condition: Gene={args.gene}({gene_idx}), Eye={args.laterality}({lat_idx}), Age={args.age}({age_norm:.2f})")

    # 4. Initialize Model
    # Note: latent_size is image_size // 8 for Stable Diffusion VAE
    latent_size = train_args.image_size // 8
    
    # Initialize the architecture (Must match what we defined in models.py)
    model = SiT_models[train_args.model](
        input_size=latent_size,
        num_classes=train_args.num_classes
    ).to(device)
    
    # Load weights (Prefer EMA weights for better visual quality)
    if "ema" in checkpoint:
        print("Loading EMA weights...")
        model.load_state_dict(checkpoint["ema"])
    else:
        print("Loading standard weights...")
        model.load_state_dict(checkpoint["model"])
        
    model.eval()

    # 5. Initialize VAE and Transport (Sampling Physics)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{train_args.vae}").to(device)
    
    transport = create_transport(
        train_args.path_type,
        train_args.prediction,
        train_args.loss_weight,
        train_args.train_eps,
        train_args.sample_eps
    )
    sampler = Sampler(transport)

    # 6. Create Sampling Noise
    # We generate 'n' samples of the SAME condition to see variety
    n = args.num_samples
    z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # 7. Prepare Conditions Tensors
    # We repeat the single condition N times
    fixed_genes = torch.tensor([gene_idx] * n, device=device)
    fixed_lats = torch.tensor([lat_idx] * n, device=device)
    fixed_ages = torch.tensor([age_norm] * n, device=device)

    # Setup Classifier-Free Guidance (CFG) inputs
    if args.cfg_scale > 1.0:
        # Duplicate noise for [Conditional, Unconditional] batching
        z = torch.cat([z, z], 0)
        
        # Create Null Tokens (must match training logic)
        gene_null = torch.tensor([train_args.num_classes] * n, device=device)
        lat_null = torch.tensor([2] * n, device=device)
        age_null = torch.tensor([-1.0] * n, device=device)
        
        # Concatenate: [Real_Condition, Null_Condition]
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

    # 8. Run Sampling Loop (The Diffusion Process)
    print("Sampling (this may take a moment)...")
    # Using ODE solver by default as it is generally faster/cleaner for simple sampling
    samples = sampler.sample_ode()(z, model_fn, **model_kwargs)[-1]
    
    # Drop the null samples
    if args.cfg_scale > 1.0:
        samples, _ = samples.chunk(2, dim=0)

    # 9. Decode and Save
    print("Decoding latents to pixels...")
    samples = vae.decode(samples / 0.18215).sample
    
    # Denormalize (-1,1) -> (0, 255)
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    os.makedirs(args.output_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        img = Image.fromarray(sample)
        filename = f"{args.gene}_{args.laterality}_Age{args.age}_{i:02d}.png"
        out_path = os.path.join(args.output_dir, filename)
        img.save(out_path)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt file)")
    parser.add_argument("--mapping-file", type=str, default="../data/class_mapping.json")
    parser.add_argument("--output-dir", type=str, default="samples")
    
    # Patient Attributes to simulate
    parser.add_argument("--gene", type=str, default="ABCA4", help="Gene symbol (e.g. ABCA4)")
    parser.add_argument("--laterality", type=str, default="R", help="L or R")
    parser.add_argument("--age", type=int, default=55, help="Patient age in years")
    
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--num-samples", type=int, default=1, help="How many variations to generate")
    
    args = parser.parse_args()
    main(args)