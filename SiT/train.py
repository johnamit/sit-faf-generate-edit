# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from dataset import RetinalDataset # import our custom dataset

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        # New: Modified wandb for this project
        entity = os.environ.get("ENTITY", None)
        project = os.environ.get("PROJECT", "retinal-sim")
        if args.wandb:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Initialise the dataset first to know 'numclasses# (num_genes)
    # NOTE: RandomHorizontalFlip REMOVED - handled by dataset with label sync
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),  # REMOVED - causes laterality bug!
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = RetinalDataset(
        csv_file=args.data_path,
        img_dir=args.img_dir,
        mapping_file=args.mapping_file,
        transform=transform,
        do_augment=True  # Enable synchronized flipping
    )
    
    # Override num_classes with actual gene count from data
    args.num_classes = dataset.get_num_genes()
    logger.info(f"Dataset contains {len(dataset):,} images and {args.num_classes} gene classes.")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )    
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes # arg.num_classes now reflects num_genes
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    
    # Load model/EMA weights BEFORE moving model to device and creating optimizer
    checkpoint = None
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        logger.info(f"Loaded model weights from checkpoint: {ckpt_path}")
    
    requires_grad(ema, False)
    
    # Move model to device and wrap in DDP BEFORE creating optimizer
    model = DDP(model.to(device), device_ids=[device])
    
    # Setup optimizer with GPU model parameters
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    # Load optimizer state AFTER optimizer is created with GPU params
    if checkpoint is not None:
        opt.load_state_dict(checkpoint["opt"])
        # Move optimizer state tensors to the correct device
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        logger.info(f"Resumed optimizer state from checkpoint")
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # New: setup static sampling noise
    # Create fixed noise and labels for visualizing progress
    use_cfg = args.cfg_scale > 1.0
    
    # create random conditioning for the fixed sampling batch
    # Genes: Random IDs from 0 to num_classes-1
    fixed_genes = torch.randint(args.num_classes, size=(local_batch_size,), device=device)
    # Lats: Random IDs 0 or 1
    fixed_lats = torch.randint(2, size=(local_batch_size,), device=device)
    # Ages: Random floats 0.0 to 1.0
    fixed_ages = torch.rand(local_batch_size, device=device)
    
    n = local_batch_size
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)
    
    # Setup classifier-free guidance for sampling
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        
        # Null tokens for CFG
        # Gene null = num_classes
        gene_null = torch.tensor([args.num_classes] * n, device=device)
        # Lat null = 2
        lat_null = torch.tensor([2] * n, device=device)
        # Age null = -1.0
        age_null = torch.tensor([-1.0] * n, device=device)
        
        # Concatenate [Conditioned, Unconditioned] for each attribute
        fixed_genes_all = torch.cat([fixed_genes, gene_null], 0)
        fixed_lats_all = torch.cat([fixed_lats, lat_null], 0)
        fixed_ages_all = torch.cat([fixed_ages, age_null], 0)
        
        # Arguments to pass to model_fn
        sample_model_kwargs = dict(
            genes=fixed_genes_all,
            lats=fixed_lats_all,
            ages=fixed_ages_all,
            cfg_scale=args.cfg_scale
        )
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(genes=fixed_genes, lats=fixed_lats, ages=fixed_ages)
        model_fn = ema.forward


    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        
        # New: unpacking training loop
        for x, (genes, lats, ages) in loader:
            x = x.to(device)
            # Unpack and move attributes to GPU
            genes = genes.to(device)
            lats = lats.to(device)
            ages = ages.to(device).float() # Ensure float32
            
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            # Pass dictionary of conditioning arguments to transport loss
            model_kwargs = dict(genes=genes, lats=lats, ages=ages)
            
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            # Generate Samples
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                sample_fn = transport_sampler.sample_ode() # default to ode sampling
                
                # Use our pre-defined 'sample_model_kwargs' which contains (Gene, Lat, Age)
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                dist.barrier()

                if use_cfg: # remove null samples
                    samples, _ = samples.chunk(2, dim=0)
                
                samples = vae.decode(samples / 0.18215).sample
                out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                dist.all_gather_into_tensor(out_samples, samples)
                
                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logging.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # Updated args for this project
    parser.add_argument("--data-path", type=str, required=True, help="Path to metadata.csv")
    parser.add_argument("--img-dir", type=str, required=True, help="Path to images/ folder")
    parser.add_argument("--mapping-file", type=str, default="class_mapping.json", help="Path to mapping json")
    
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
