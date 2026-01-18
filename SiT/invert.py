"""Inversion Script to:
1. Invert a real image to its latent representation (via invert command)
2. Edit the latent representation (via edit command) and generate the edited image
"""

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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # suppress future warnings from libraries

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands: invert or edit')
    
    # ----- INVERT ARGUMENTS -----
    # Invert command - core functionality
    invert_parser = subparsers.add_parser('invert', help='Invert a real image to latent representation')
    invert_parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt file)")
    invert_parser.add_argument("--input-image", type=str, required=True, help="Path to input image")
    invert_parser.add_argument("--output-dir", type=str, default="inversions", help="Output directory")
    invert_parser.add_argument("--mapping-file", type=str, default="../data/class_mapping.json")
    invert_parser.add_argument("--verify", action="store_true", help="Reconstruct from inverted noise to verify quality")
    invert_parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    
    # Invert command - conditioning attributes
    invert_parser.add_argument("--gene", type=str, required=True, help="Gene symbol of the input image")
    invert_parser.add_argument("--laterality", type=str, required=True, help="L or R")
    invert_parser.add_argument("--age", type=int, required=True, help="Patient age in years")
    
    # Invert command - ODE solver parameters
    invert_parser.add_argument("--sampling-method", type=str, default="dopri5", help="ODE solver (dopri5, euler, heun)")
    invert_parser.add_argument("--num-steps", type=int, default=50, help="Number of ODE steps")
    invert_parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    invert_parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    
    # ----- EDIT ARGUMENTS -----
    # Edit command - core functionality
    edit_parser = subparsers.add_parser('edit', help='Edit using inverted noise')
    edit_parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt file)")
    edit_parser.add_argument("--noise-file", type=str, required=True, help="Path to inverted_noise.pt")
    edit_parser.add_argument("--output-dir", type=str, default="edits", help="Output directory")
    edit_parser.add_argument("--mapping-file", type=str, default="../data/class_mapping.json")
    edit_parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    
    # Edit command - target attributes
    edit_parser.add_argument("--target-gene", type=str, default=None, help="New gene (or None to keep)")
    edit_parser.add_argument("--target-laterality", type=str, default=None, help="New laterality (or None)")
    edit_parser.add_argument("--target-age", type=int, default=None, help="New age (or None)")
    
    # Edit command - ODE solver parameters
    edit_parser.add_argument("--sampling-method", type=str, default="dopri5")
    edit_parser.add_argument("--num-steps", type=int, default=50)
    edit_parser.add_argument("--atol", type=float, default=1e-6)
    edit_parser.add_argument("--rtol", type=float, default=1e-3)
    
    return parser.parse_args()


def invert_func(args):
    """Inversion: Given a real image, invert to latent noise representation"""
    with torch.inference_mode():
        # --- 1. Device, Mapping and Checkpoint Setup ---
        device = torch.device(args.device if torch.cuda.is_available() else "cpu") # set device
        
        # load the checkpoint
        print(f"Loading checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        train_args = checkpoint["args"] if "args" in checkpoint else args # recover training args. fallback to current args
            
        # load class mapping for genes
        with open(args.mapping_file, 'r') as f:
            gene_to_idx = json.load(f)
        
        
        # --- 2. Process Inputs ---
        # Gene Processing
        if args.gene.upper() not in gene_to_idx: 
            print(f"ERROR: Gene {args.gene} not found in {args.mapping_file}")
            print(f"Available genes: {list(gene_to_idx.keys())}")
            return
        gene_idx = gene_to_idx[args.gene]
        
        # Laterality Processing
        lat_map = {'L': 0, 'R': 1}
        lat_idx = lat_map.get(args.laterality.upper(), 0)
        
        # Age Processing (Normalize 0-100 -> 0.0-1.0)
        age_norm = float(args.age) / 100.0
        age_norm = max(0.0, min(1.0, age_norm))  # clamp between 0 and 1
        
        print(f"Conditon: Gene={args.gene}({gene_idx}), Eye={args.laterality}({lat_idx}), Age={args.age}({age_norm:.2f})")
        
        
        # --- 3. Initialise Model ---
        latent_size = train_args.image_size // 8  # VAE downsamples by factor of 8
        
        model  = SiT_models[train_args.model](
            input_size = latent_size,
            num_classes = train_args.num_classes
        ).to(device)
        
        # load model weights
        if "ema" in checkpoint:
            print("Loading EMA weights...")
            model.load_state_dict(checkpoint["ema"]) # load EMA weights if available
        else:
            print("Loading standard weights...")
            model.load_state_dict(checkpoint["model"]) # load standard weights as fallback
            
        model.eval() # set to eval mode
        
        
        # --- 4. Initialise VAE and Transport ---
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{train_args.vae}").to(device)
        
        # Create transport and sampler
        transport = create_transport(
            train_args.path_type,
            train_args.prediction,
            train_args.loss_weight,
            train_args.train_eps,
            train_args.sample_eps
        )
        sampler = Sampler(transport)
        
        
        # --- 5. Load & Encode Real Images to Latent Space ---
        print(f"Loading image: {args.input_image}")
        img = Image.open(args.input_image).convert("RGB") # load image
        
        # Resize to models expected size
        img = img.resize((train_args.image_size, train_args.image_size), Image.BICUBIC) # BICUBIC for downsampling
        
        # Convert to tensor and normalize to [-1,1]
        x = np.array(img).astype(np.float32) / 127.5 - 1.0 
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Encode to latent space
        z_data = vae.encode(x).latent_dist.sample().mul_(0.18215) # scale factor from SD VAE
        print(f"Encoded to latent shape: {z_data.shape}")
        
        
        # --- 6. Prepare Conditions Tensor ---
        model_kwargs = dict(
            genes = torch.tensor([gene_idx], device=device),
            lats = torch.tensor([lat_idx], device=device),
            ages = torch.tensor([age_norm], device=device)
        )
        
        
        # --- 7. Run ODE Backward (Inversion: Data @ t=1 -> Noise @ t=0)
        # Flow Matching: standard direction is Noise(t=0) -> Data(t=1)
        # For inversion, we go Data(t=1) -> Noise(t=0) using reverse=True
        print("Inverting (Data -> Noise)...")
        
        # Get the inverse ODE sampler
        inverse_ode_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=True  # reverse flag for inversion
        )
        
        # Run inversion: z_data (image latent) -> z_noise (inverted noise)
        z_noise = inverse_ode_fn(z_data, model.forward, **model_kwargs)[-1]
        
        print(f"Inverted noise shape: {z_noise.shape}")
        print(f"Inverted noise stats: mean={z_noise.mean().item():.4f}, std={z_noise.std().item():.4f}")
        
        # --- 8. Save Inverted Noise ---
        os.makedirs(args.output_dir, exist_ok=True)
        
        noise_save_path = os.path.join(args.output_dir, "inverted_noise.pt")
        torch.save({
            'noise': z_noise,
            'gene': args.gene,
            'gene_idx': gene_idx,
            'laterality': args.laterality,
            'lat_idx': lat_idx,
            'age': args.age,
            'age_norm': age_norm,
            'input_image': args.input_image,
        }, noise_save_path)
        print(f"SUCCESS: Inverted noise saved to {noise_save_path}")
        
        
        # --- 9. Optional Verification: Reconstruct Image from Inverted Noise ---
        if args.verify:
            print("Verifying (Noise -> Data reconstruction)...")
            
            # Forward ODE: Noise(t=0) -> Data(t=1)
            forward_ode_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=False
            )
            
            z_recon = forward_ode_fn(z_noise, model.forward, **model_kwargs)[-1] # reconstructed latent

            # Decode reconstructed latent to image
            recon_img = vae.decode(z_recon / 0.18215).sample # decode from latent space
            recon_img = torch.clamp(127.5 * recon_img + 128.0, 0, 255) # de-normalize to [0,255]
            recon_img = recon_img.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0] # convert to HWC (HxWxC)
            
            # Save reconstructed image
            recon_save_path = os.path.join(args.output_dir, "reconstruction.png")
            Image.fromarray(recon_img).save(recon_save_path)
            print(f"SUCCESS: Reconstruction saved to {recon_save_path}")
            
            # Save original image for comparison
            original_save_path = os.path.join(args.output_dir, "original.png")
            img.save(original_save_path)
            print(f"SUCCESS: Original saved to {original_save_path}")
            
            # Compute reconstruction error
            latent_mse = torch.mean((z_data - z_recon) ** 2).item()
            print(f"Latent reconstruction MSE: {latent_mse:.6f}")
            

def edit_func(args): 
    """Editing: Given inverted noise, and target attributes, generate edited image"""
    with torch.inference_mode():
        # --- 1. Device, Mapping and Checkpoint Setup ---
        device = torch.device(args.device if torch.cuda.is_available() else "cpu") # set device
        
        # load the checkpoint
        print(f"Loading checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        train_args = checkpoint["args"] if "args" in checkpoint else args # recover training args. fallback to current args
        
        # Load inverted noise
        print(f"Loading inverted noise: {args.noise_file}")
        noise_data = torch.load(args.noise_file, map_location=device)
        z_noise = noise_data['noise']
        
        print(f"Original conditions: Gene={noise_data['gene']}, Lat={noise_data['laterality']}, Age={noise_data['age']}")
        
        # load class mapping for genes
        with open(args.mapping_file, 'r') as f:
            gene_to_idx = json.load(f)
            
        
        # --- 2. Prepare Target Conditions ---
        # Gene Processing
        if args.target_gene:
            if args.target_gene.upper() not in gene_to_idx:
                print(f"ERROR: Gene '{args.target_gene}' not found.")
                return
            target_gene_idx = gene_to_idx[args.target_gene]
        else:
            target_gene_idx = noise_data['gene_idx']
            
        # Laterality Processing
        lat_map = {'L': 0, 'R': 1}
        if args.target_laterality:
            target_lat_idx = lat_map.get(args.target_laterality.upper(), noise_data['lat_idx'])
        else:
            target_lat_idx = noise_data['lat_idx']
            
        # Age Processing
        if args.target_age is not None:
            target_age_norm = float(args.target_age) / 100.0
            target_age_norm = max(0.0, min(1.0, target_age_norm))  # clamp between 0 and 1
        else:
            target_age_norm = noise_data['age_norm']
            
        print(f"Target conditions: Gene={args.target_gene or noise_data['gene']}({target_gene_idx}), "
          f"Lat={args.target_laterality or noise_data['laterality']}({target_lat_idx}), "
          f"Age={args.target_age or noise_data['age']}({target_age_norm:.2f})")
        
        
        # --- 3. Initialise Model ---
        latent_size = train_args.image_size // 8  # VAE downsamples by factor of 8
        model  = SiT_models[train_args.model](
            input_size = latent_size,
            num_classes = train_args.num_classes
        ).to(device)
        
        if "ema" in checkpoint:
            print("Loading EMA weights...")
            model.load_state_dict(checkpoint["ema"]) # load EMA weights if available
        else:
            print("Loading standard weights...")
            model.load_state_dict(checkpoint["model"]) # load standard weights as fallback
        model.eval() # set to eval mode
        
        
        # --- 4. Initialise VAE and Transport ---
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{train_args.vae}").to(device)
        
        # Create transport and sampler
        transport = create_transport(
            train_args.path_type,
            train_args.prediction,
            train_args.loss_weight,
            train_args.train_eps,
            train_args.sample_eps
        )
        sampler = Sampler(transport)
        
        
        # --- 5. Prepare Conditions Tensor ---
        model_kwargs = dict(
            genes = torch.tensor([target_gene_idx], device=device),
            lats = torch.tensor([target_lat_idx], device=device),
            ages = torch.tensor([target_age_norm], device=device)
        )
        
        
        # --- 6. Run ODE Forward (Noise @ t=0 -> Data @ t=1) ---
        print("Generating edited image (Noise -> Data)...")
        forward_ode_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=False
        )
        
        z_edited = forward_ode_fn(z_noise, model.forward, **model_kwargs)[-1] # edited latent
        
        # Decode edited latent to image
        edited_img = vae.decode(z_edited / 0.18215).sample # decode from latent space
        edited_img = torch.clamp(127.5 * edited_img + 128.0, 0, 255) # de-normalize to [0,255]
        edited_img = edited_img.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0] # convert to HWC (HxWxC)
        
        
        # --- 7. Save Edited Image ---
        os.makedirs(args.output_dir, exist_ok=True)
        gene_str = args.target_gene or noise_data['gene']
        lat_str = args.target_laterality or noise_data['laterality']
        age_str = args.target_age if args.target_age is not None else noise_data['age']
        
        filename = f"edited_{gene_str}_{lat_str}_Age{age_str}.png" # custom filename with attributes
        save_path = os.path.join(args.output_dir, filename) 
        Image.fromarray(edited_img).save(save_path) # save image
        print(f"SUCCESS: Edited image saved to {save_path}")
        
def main():
    args = parse_args()
    
    if args.command == 'invert':
        invert_func(args)
    elif args.command == 'edit':
        edit_func(args)
    else:
        print("ERROR: No command specified. Use 'invert' or 'edit'.")
        
if __name__ == "__main__":
    main()