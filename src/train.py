import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb  # Add import for wandb

# Import our dataloader creation function from the NoisyKappaFieldDataset module
from dataprep.noisy_kappa_dataset import create_noisy_kappa_dataloader


# -----------------------------------------------------------------------------
# KappaDenoiser model architecture
# -----------------------------------------------------------------------------
from models.denoiser import KappaDenoiser


# -----------------------------------------------------------------------------
# Helper: Stochastically generate noise for a given clean tensor.
# -----------------------------------------------------------------------------
def generate_noise(clean, snr_db_min, snr_db_max, device):
    """
    Generate noise for each sample based on its signal power and a random SNR.

    Parameters:
      clean      : Tensor of shape (B, H, W) or (B, 1, H, W)
      snr_db_min : Minimum SNR in dB
      snr_db_max : Maximum SNR in dB
      device     : Torch device

    Returns:
      noise      : Tensor of the same shape as clean
    """
    if clean.dim() == 3:
        # Compute the average squared pixel value per sample (shape: [B, 1, 1])
        signal_power = torch.mean(clean**2, dim=(1, 2), keepdim=True)
    elif clean.dim() == 4:
        signal_power = torch.mean(clean**2, dim=(2, 3), keepdim=True)
    else:
        raise ValueError("Unexpected tensor shape for clean")
    # Sample a random SNR (in dB) per sample from the uniform range.

    clean = clean.to(device) if clean.device != device else clean

    snr_db = (
        torch.rand(signal_power.shape, device=device) * (snr_db_max - snr_db_min)
        + snr_db_min
    )
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(clean) * torch.sqrt(noise_power)
    return noise, snr_db


# -----------------------------------------------------------------------------
# Visualization function for denoising performance
# -----------------------------------------------------------------------------
def visualize_denoising(model, clean_sample, snr_db_min, snr_db_max, device):
    """
    Given a clean sample, generate a noisy version and compare to the denoised output.
    """
    model.eval()
    with torch.no_grad():
        # Ensure sample has a batch dimension: (1, H, W)
        if clean_sample.dim() == 2:
            clean_sample = clean_sample.unsqueeze(0)
        noise, snr_db = generate_noise(clean_sample, snr_db_min, snr_db_max, device)
        noisy_sample = clean_sample + noise
        input_tensor = noisy_sample.unsqueeze(1)  # Convert to (1, 1, H, W)
        output = model(input_tensor).squeeze().cpu().numpy()
        clean_img = clean_sample.squeeze().cpu().numpy()
        noisy_img = noisy_sample.squeeze().cpu().numpy()
        snr_db = snr_db.cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(clean_img, cmap="viridis")
        axes[0].set_title("Clean Field")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(noisy_img, cmap="viridis")
        axes[1].set_title(f"Noisy Field (SNR: {snr_db:.2f} dB)")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(output, cmap="viridis")
        axes[2].set_title("Denoised Field")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Main training loop using stochastic noise augmentation
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train Kappa Denoiser with Stochastic Noise Augmentation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .npy file OR WandB artifact identifier for clean kappa fields",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent dimension of the autoencoder",
    )
    parser.add_argument(
        "--snr_db_min", type=float, default=3.0, help="Minimum SNR in dB"
    )
    parser.add_argument(
        "--snr_db_max", type=float, default=3.0, help="Maximum SNR in dB"
    )
    parser.add_argument(
        "--reshape_nx", type=int, default=32, help="Reshape dimension in x"
    )
    parser.add_argument(
        "--reshape_ny", type=int, default=32, help="Reshape dimension in y"
    )
    parser.add_argument(
        "--augment_noise",
        type=int,
        default=1,
        help=(
            "Number of stochastic noise augmentations per sample "
            "per mini-batch. If >1, the clean field is used to generate "
            "multiple noisy variants, whose losses are averaged."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="kappa_denoiser_stoch.pt",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=0,
        help="Number of additional noise augmentations per clean sample",
    )
    parser.add_argument(
        "--is_artifact",
        action="store_true",
        help="Set to True if data_path is a WandB artifact identifier",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="DIPDE",
        help="WandB project name for this run",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="ECE689AdvDL",
        help="WandB entity (team) name",
    )
    parser.add_argument(
        "--artifact_filename",
        type=str,
        default="kappa_fine_samples.npy",
        help="Filename to load from within the artifact (if using artifact)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Initialize WandB run if using artifact
    wandb_run = None
    if args.is_artifact:
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, config=vars(args)
        )

    # Create a DataLoader using the NoisyKappaFieldDataset with artifact support
    dataloader = create_noisy_kappa_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        snr_db_min=args.snr_db_min,
        snr_db_max=args.snr_db_max,
        reshape_size=(args.reshape_nx, args.reshape_ny),
        # resize_shape=(args.reshape_nx, args.reshape_ny),
        num_workers=0,
        device=args.device,
        seed=42,
        is_artifact=args.is_artifact,
        wandb_run=wandb_run,
        artifact_filename=args.artifact_filename,
        num_augmentations=args.num_augmentations,
    )

    # Initialize the model, criterion, and optimizer.
    model = KappaDenoiser(latent_dim=args.latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    losses = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress_bar:
            # Handle different batch structures based on augmentation
            if args.num_augmentations > 0:
                # With augmentations, the batch is (original_noisy, clean, augmented_noisy)
                original_noisy, clean, augmented_noisy_list = batch
                clean = clean.to(device)

                optimizer.zero_grad()
                aggregate_loss = 0.0

                original_noisy = original_noisy.to(device)
                outputs = model(original_noisy)
                loss = criterion(outputs, clean)
                aggregate_loss += loss

                # Process each augmented noisy variant
                for aug_noisy in augmented_noisy_list:
                    aug_noisy = aug_noisy.to(device)
                    if aug_noisy.dim() == 3:
                        aug_noisy = aug_noisy.unsqueeze(1)
                    outputs = model(aug_noisy)
                    loss = criterion(outputs, clean)
                    aggregate_loss += loss

                # Average the loss across all variants
                loss = aggregate_loss / (1 + len(augmented_noisy_list))
            else:
                # Without augmentations, the batch is just (noisy, clean)
                noisy, clean = batch
                noisy = noisy.to(device)
                clean = clean.to(device)

                optimizer.zero_grad()
                if noisy.dim() == 3:
                    noisy = noisy.unsqueeze(1)
                loss = criterion(model(noisy), clean.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Log to WandB if it's initialized
            if wandb_run is not None:
                wandb.log({"batch_loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}, Avg Loss: {avg_loss:.6f}")

        # Log to WandB if it's initialized
        if wandb_run is not None:
            wandb.log({"epoch": epoch, "avg_loss": avg_loss})

        # visualize denoising every 10 epochs.
        if (epoch + 1) % 10 == 0:
            sample_clean = clean[0]
            visualize_denoising(
                model, sample_clean, args.snr_db_min, args.snr_db_max, device
            )
            # Save visualization to WandB if initialized
            if wandb_run is not None:
                wandb.log({"denoising_visualization": wandb.Image(plt.gcf())})

    # Plot the training loss curve.
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()

    # Log loss curve to WandB if initialized
    if wandb_run is not None:
        wandb.log({"loss_curve": wandb.Image(plt.gcf())})

    # Save the model.
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

    # Save model to WandB if initialized
    if wandb_run is not None:
        wandb.save(args.save_path)
        wandb_run.finish()


if __name__ == "__main__":
    main()
