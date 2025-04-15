#!/usr/bin/env python
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the format_for_paper function from fem/plots
sys.path.append('.')  # Ensure the current directory is in the path
from fem.plots import format_for_paper
from models.denoiser import KappaDenoiser

# Import the dataloader and noise generator from noisy_kappa_dataset
from dataprep.noisy_kappa_dataset import (
    create_noisy_kappa_dataloader,
    generate_noise_for_clean_sample
)


def visualize_denoising(clean_sample, snr_levels, denoiser, device, save_path=None):
    """
    Visualize clean, noisy, and denoised kappa fields.
    
    Parameters:
      clean_sample : Tensor of shape (H, W)
      snr_levels   : List of SNR values in dB to generate noise with
      denoiser     : Denoiser model
      device       : Torch device
      save_path    : Optional path to save the visualization
    """
    # Apply paper formatting
    format_for_paper()
    
    # Create figure with 3 columns (clean, noisy, denoised) for each SNR level
    n_rows = len(snr_levels)
    fig, axes = plt.subplots(n_rows, 3, figsize=(5, n_rows * 1.5))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, 3)
    
    # Set common colorbar range based on clean sample
    vmin = clean_sample.min().item()
    vmax = clean_sample.max().item()
    
    # Plot each SNR level
    for i, snr_db in enumerate(snr_levels):
        # Generate noise for this SNR level
        noise = generate_noise_for_clean_sample(
            clean_sample.unsqueeze(0),  # Add batch dimension
            snr_db_min=snr_db,
            snr_db_max=snr_db,
            device=device
        )
        
        # Create noisy sample
        noisy_sample = clean_sample + noise.squeeze()
        
        # Get denoised sample
        with torch.no_grad():
            denoised_sample = denoiser(noisy_sample.unsqueeze(0).unsqueeze(0)).squeeze()
        
        # Plot clean sample (only in first row)
        if i == 0:
            im0 = axes[i, 0].imshow(clean_sample.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(r'Clean $\kappa$', fontsize=8)
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        else:
            # Just show clean image without title in other rows
            im0 = axes[i, 0].imshow(clean_sample.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
            
        # Plot noisy sample
        im1 = axes[i, 1].imshow(noisy_sample.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Noisy ({snr_db:.1f} dB)', fontsize=8)
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Plot denoised sample
        im2 = axes[i, 2].imshow(denoised_sample.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title('Denoised', fontsize=8)
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    # Remove axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        # Create the directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Kappa Field Denoising')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to .npy file OR WandB artifact identifier for clean kappa fields')
    parser.add_argument('--denoiser_path', type=str, default='models/kappa_denoiser_stoch.pt',
                       help='Path to trained denoiser model')
    parser.add_argument('--reshape_nx', type=int, default=64,
                       help='Reshape dimension in x')
    parser.add_argument('--reshape_ny', type=int, default=64,
                       help='Reshape dimension in y')
    parser.add_argument('--snr_min', type=float, default=0.0,
                       help='Minimum SNR in dB range')
    parser.add_argument('--snr_max', type=float, default=20.0,
                       help='Maximum SNR in dB range')
    parser.add_argument('--snr_steps', type=int, default=3,
                       help='Number of SNR steps to visualize')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of the sample to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for computation')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the visualization')
    parser.add_argument('--is_artifact', action='store_true',
                       help='Set to True if data_path is a WandB artifact identifier')
    parser.add_argument('--wandb_project', type=str, default='DIPDE',
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='ECE689AdvDL',
                       help='WandB entity (team) name')
    parser.add_argument('--artifact_filename', type=str, default='kappa_fine_samples.npy',
                       help='Filename to load from within the artifact (if using artifact)')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Initialize WandB run if using artifact
    wandb_run = None
    if args.is_artifact:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            job_type='visualization'
        )
    
    # Create a dataloader (we only need one batch)
    dataloader = create_noisy_kappa_dataloader(
        data_path=args.data_path,
        batch_size=1,  # We only need one sample
        snr_db_min=args.snr_min,
        snr_db_max=args.snr_max,
        reshape_size=(args.reshape_ny, args.reshape_nx),
        device=args.device,
        is_artifact=args.is_artifact,
        wandb_run=wandb_run,
        artifact_filename=args.artifact_filename,
        num_augmentations=0  # We don't need augmentations from the dataloader
    )
    
    # Get a batch
    for batch in dataloader:
        noisy, clean = batch
        break  # We only need one batch
    
    # Extract the sample at the specified index (or first sample if out of range)
    sample_idx = min(args.sample_idx, len(clean) - 1)
    clean_sample = clean[sample_idx].squeeze()
    
    # Load denoiser model
    denoiser = KappaDenoiser()
    denoiser.load_state_dict(torch.load(args.denoiser_path, map_location=device))
    denoiser.to(device)
    denoiser.eval()
    
    # Generate SNR levels within the specified range
    if args.snr_steps > 1:
        snr_levels = np.linspace(args.snr_min, args.snr_max, args.snr_steps)
    else:
        snr_levels = [args.snr_min]
    
    # Visualize the denoising capabilities
    visualize_denoising(clean_sample, snr_levels, denoiser, device, args.save_path)
    
    # If using WandB, log the visualization
    if wandb_run is not None:
        import wandb
        wandb.log({"denoising_visualization": wandb.Image(plt.gcf())})
        wandb_run.finish()


if __name__ == "__main__":
    main() 