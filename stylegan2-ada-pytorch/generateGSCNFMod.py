# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Modified to also save W/WS/Z and a manifest for synthetic data.

import os
import re
import csv
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def append_manifest_row(manifest_path: str, header: list, row: list):
    is_new = not os.path.exists(manifest_path)
    with open(manifest_path, 'a', newline='') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Create output directories
    img_dir = ensure_dir(os.path.join(outdir, 'images'))
    w_dir = ensure_dir(os.path.join(outdir, 'w'))
    manifest_path = os.path.join(outdir, 'manifest_generated.csv')
    header = ['gen_id','seed','img_path','w_npz_path','gene_class_idx','z_dim','w_dim','num_ws','trunc']

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            
            # handle greyscale or rgb
            img_array = img[0].cpu().numpy()
            
            if img_array.shape[-1] == 1: #greyscale
                img_array = img_array.squeeze(-1)
                mode = 'L'
            elif img_array.shape[-1] == 3:
                mode = 'RGB'
            else:
                raise ValueError(f'Unexpected channel size: {img_array.shape[-1]}')

            PIL.Image.fromarray(img_array, mode).save(f'{img_dir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    z_dim, w_dim, num_ws = G.z_dim, G.w_dim, G.num_ws

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx+1, len(seeds)))
        
        # Sample z, map to ws with truncation, synthesize image
        rnd = np.random.RandomState(seed)
        z = torch.from_numpy(rnd.randn(1, G.z_dim)).to(device)
        ws = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=None)
        img = G.synthesis(ws, noise_mode=noise_mode)
        
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # handle greyscale or rgb
        img_array = img[0].cpu().numpy()
        
        if img_array.shape[-1] == 1: #greyscale
            img_array = img_array.squeeze(-1)
            mode = 'L'
        elif img_array.shape[-1] == 3:
            mode = 'RGB'
        else:
            raise ValueError(f'Unexpected channel size: {img_array.shape[-1]}')

        # Save image
        gen_id = f"{seed:08d}"
        filename = f'seed{seed:04d}_latClass_{class_idx if class_idx is not None else 0}.png'
        img_path = os.path.join(img_dir, filename)
        PIL.Image.fromarray(img_array, mode).save(img_path)

        # Save codes (w = first layer; ws = all layers; also keep z)
        w_single = ws[:,0,:].detach().cpu().numpy()[0]   # (w_dim,)
        ws_full = ws.detach().cpu().numpy()[0]          # (num_ws, w_dim)
        z_np = z.detach().cpu().numpy()[0]              # (z_dim,)
        npz_path = os.path.join(w_dir, f"{gen_id}.npz")
        np.savez(npz_path, w=w_single, ws=ws_full, z=z_np)

        # Append manifest row
        row = [gen_id, seed, img_path, npz_path, (class_idx or 0), z_dim, w_dim, num_ws, truncation_psi]
        append_manifest_row(manifest_path, header, row)

    print(f"\nDone.\n- Images:   {img_dir}\n- W codes:  {w_dir}\n- Manifest: {manifest_path}")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
