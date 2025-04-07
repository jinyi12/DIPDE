"""
Noisy Kappa Field Dataset

This module provides dataset and dataloader implementations for generating
noisy versions of kappa fields on the fly during training. The noise levels
are controlled through SNR specifications.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Union, List, Callable
import logging
import wandb

import matplotlib.pyplot as plt
import argparse
import torchvision.transforms.functional as TF


class NoisyKappaFieldDataset(Dataset):
    """
    Dataset for kappa fields with on-the-fly noise generation.

    This dataset loads clean kappa fields and generates noisy versions
    during training, with noise levels determined by specified SNR ranges.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        snr_db_min: float = 10.0,
        snr_db_max: float = 30.0,
        reshape_size: Optional[Tuple[int, int]] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        transforms: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        data_path : Union[str, Path]
            Path to the .npy file containing kappa fields
        snr_db_min : float
            Minimum signal-to-noise ratio in decibels
        snr_db_max : float
            Maximum signal-to-noise ratio in decibels
        reshape_size : Optional[Tuple[int, int]]
            If provided, reshape each field to this size (ny, nx)
        device : str
            Device to store the data on ('cpu' or 'cuda')
        seed : Optional[int]
            Random seed for reproducibility
        transforms : Optional[Callable]
            Transformations to apply to clean and noisy fields
            (e.g., resizing, normalization)
        """
        self.data_path = Path(data_path)
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.reshape_size = reshape_size
        self.device = device
        self.transforms = transforms

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load and preprocess the data
        self._load_and_preprocess_data()

        self.logger.info(
            f"Initialized dataset with {len(self)} samples, "
            f"SNR range: [{snr_db_min}, {snr_db_max}] dB",
            f"Transform: {self.transforms}",
        )

    def _load_and_preprocess_data(self):
        """Load and preprocess the kappa fields."""
        try:
            # Load the data
            kappa_fields = np.load(self.data_path)

            self.logger.info(f"Loaded kappa fields with shape: {kappa_fields.shape}")

            # Print shape for debugging
            self.logger.info(f"Loaded kappa fields with shape: {kappa_fields.shape}")

            # Determine the shape and properly reshape
            if len(kappa_fields.shape) == 2 and not kappa_fields.shape[0] > 1:
                # Single sample with shape (nx, ny)
                # Expand to (1, nx, ny)
                kappa_fields = kappa_fields[np.newaxis, :, :]
                self.logger.info(f"Reshaped single sample to {kappa_fields.shape}")
            elif len(kappa_fields.shape) == 3:
                self.logger.info(f"Already in the right format: {kappa_fields.shape}")
                # Multiple samples with shape (n_samples, nx, ny)
                # Already in the right format
                pass
            elif len(kappa_fields.shape) == 2 and kappa_fields.shape[0] > 1:
                # Multiple flattened samples with shape (n_samples, ny*nx)
                if self.reshape_size is None:
                    raise ValueError("reshape_size must be provided for flattened data")
                nx, ny = self.reshape_size
                kappa_fields = kappa_fields.reshape(kappa_fields.shape[0], nx, ny)
                self.logger.info(f"Reshaped flattened samples to {kappa_fields.shape}")

            # Convert to torch tensor
            self.clean_fields = torch.tensor(
                kappa_fields, dtype=torch.float32, device=self.device
            )  # shape (n_samples, nx, ny)

            # Compute signal power across the spatial dimensions (nx, ny)
            # # This ensures we get one power value per sample
            # self.signal_powers = torch.var(
            #     self.clean_fields.reshape(len(self.clean_fields), -1), dim=1
            # )

            # resize if reshape_size is provided
            if self.reshape_size is not None:
                self.clean_fields = TF.resize(
                    self.clean_fields,
                    self.reshape_size,
                    interpolation=TF.InterpolationMode.BILINEAR,
                )

            # normalize globally to [0, 1]
            self.clean_fields = (self.clean_fields - self.clean_fields.min()) / (
                self.clean_fields.max() - self.clean_fields.min()
            )

            # compute signal power across the spatial dimensions (nx, ny)
            self.signal_powers = torch.mean(self.clean_fields**2, dim=(1, 2))

            self.logger.info(
                f"Preprocessed data shape: {self.clean_fields.shape}, "
                f"Signal powers shape: {self.signal_powers.shape}"
            )

        except Exception as e:
            raise RuntimeError(f"Error loading data from {self.data_path}: {str(e)}")

    def _generate_noise(
        self, signal_power: torch.Tensor, shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Generate noise for a given signal power and shape.

        Parameters
        ----------
        signal_power : torch.Tensor
            Power of the signal (scalar)
        shape : Tuple[int, ...]
            Shape of the noise tensor to generate

        Returns
        -------
        torch.Tensor
            Generated noise with same shape as input
        """
        # Random SNR in dB
        snr_db = (
            torch.rand(1, device=self.device) * (self.snr_db_max - self.snr_db_min)
            + self.snr_db_min
        )

        # Convert to linear scale
        snr = 10 ** (snr_db / 10)

        # Calculate noise power
        noise_power = signal_power / snr

        # Generate noise with same shape as input
        noise = torch.randn(shape, device=self.device) * torch.sqrt(noise_power)

        return noise

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.clean_fields)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of (noisy, clean) samples.

        Parameters
        ----------
        idx : int
            Index of the sample to get

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of (noisy_sample, clean_sample) with shape (1, H, W)
        """
        clean_sample = self.clean_fields[idx]

        # Generate noise with the same shape as the clean sample
        noise = self._generate_noise(self.signal_powers[idx], clean_sample.shape)

        # Add noise to create noisy sample
        noisy_sample = clean_sample + noise

        # Ensure the sample has a channel dimension (i.e. shape (1, H, W))
        if clean_sample.dim() == 2:
            clean_sample = clean_sample.unsqueeze(0)  # Add channel dimension
            noisy_sample = noisy_sample.unsqueeze(0)  # Add channel dimension

        # If transforms are provided, apply them (assumes data is (C, H, W))
        if self.transforms is not None:
            # Apply transforms
            clean_sample = self.transforms(clean_sample)
            noisy_sample = self.transforms(noisy_sample)

            # Update signal power if needed after transformation
            if clean_sample.shape != self.clean_fields[idx].shape:
                # Recalculate signal power for the transformed sample
                self.signal_powers[idx] = torch.mean(clean_sample**2, dim=(1, 2))

        return noisy_sample, clean_sample


# function for generating noise for a given clean sample with specific SNR range
def generate_noise_for_clean_sample(
    clean_sample: torch.Tensor,
    snr_db_min: float,
    snr_db_max: float,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate noise for a clean sample based on a randomly sampled SNR in the given range.

    Parameters
    ----------
    clean_sample : torch.Tensor
        Clean kappa field sample (2D or 3D tensor)
    snr_db_min : float
        Minimum signal-to-noise ratio in decibels
    snr_db_max : float
        Maximum signal-to-noise ratio in decibels
    device : str
        Device to generate noise on

    Returns
    -------
    torch.Tensor
        Generated noise with same shape as clean_sample
    """
    # Calculate mean squared value (signal power)
    if len(clean_sample.shape) == 2:  # Single 2D field
        signal_power = torch.mean(clean_sample**2)
    else:  # Batched input
        # Keep dimensions for proper broadcasting
        signal_power = torch.mean(
            clean_sample**2, dim=tuple(range(1, len(clean_sample.shape)))
        )
        signal_power = signal_power.view([-1] + [1] * (len(clean_sample.shape) - 1))

    # Generate random SNR in dB and convert to linear scale
    snr_db = torch.rand(1, device=device) * (snr_db_max - snr_db_min) + snr_db_min
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise standard deviation
    noise_std = torch.sqrt(signal_power / snr_linear)

    # Generate noise with same shape as input
    noise = torch.randn_like(clean_sample, device=device) * noise_std

    return noise


def noisy_kappa_collate_with_augmentation(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    num_augmentations: int = 1,
    snr_db_min: float = 10.0,
    snr_db_max: float = 30.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Custom collate function that generates multiple noisy versions of each clean sample.

    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, torch.Tensor]]
        List of (noisy, clean) pairs from the dataset
    num_augmentations : int
        Number of additional noise augmentations to generate (beyond the one provided by dataset)
    snr_db_min : float
        Minimum signal-to-noise ratio in decibels for augmentations
    snr_db_max : float
        Maximum signal-to-noise ratio in decibels for augmentations
    device : str
        Device to generate noise on

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]
        Tuple containing:
        - Original noisy samples from dataset (B, H, W)
        - Clean samples (B, H, W)
        - List of 'num_augmentations' additional noisy tensors, each (B, H, W)
    """
    # Extract noisy and clean samples from the batch
    noisy_samples, clean_samples = zip(*batch)

    # Stack into batches
    noisy_batch = torch.stack(noisy_samples)
    clean_batch = torch.stack(clean_samples)

    # Generate additional noise augmentations
    augmented_noisy_batches = []
    for i in range(num_augmentations):
        # Generate new noise for each clean sample in the batch
        noise = generate_noise_for_clean_sample(
            clean_batch, snr_db_min, snr_db_max, device
        )
        # Add noise to clean samples to create new noisy samples
        augmented_noisy = clean_batch + noise
        augmented_noisy_batches.append(augmented_noisy)

    return noisy_batch, clean_batch, augmented_noisy_batches


# Create a function to generate resize transform
def create_resize_transform(resize_shape: Tuple[int, int]) -> Callable:
    """
    Create a transform function for resizing kappa fields.

    Parameters
    ----------
    resize_shape : Tuple[int, int]
        Target shape for resizing (ny, nx)

    Returns
    -------
    Callable
        Transform function that resizes a tensor to the specified shape
    """

    def resize_transform(x: torch.Tensor) -> torch.Tensor:
        # Assuming x has shape (C, H, W) - add batch dimension for TF.resize
        x_resized = TF.resize(
            x.unsqueeze(0),  # Add batch dim: (1, C, H, W)
            size=resize_shape,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        ).squeeze(0)  # Remove batch dim: (C, H, W)
        return x_resized

    return resize_transform


def create_noisy_kappa_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 128,
    snr_db_min: float = 10.0,
    snr_db_max: float = 30.0,
    reshape_size: Optional[Tuple[int, int]] = None,
    resize_shape: Optional[Tuple[int, int]] = None,
    num_workers: int = 0,
    device: str = "cpu",
    seed: Optional[int] = None,
    is_artifact: bool = False,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    artifact_filename: str = "kappa_fine_samples.npy",
    num_augmentations: int = 0,
    transforms: Optional[Callable] = None,
) -> DataLoader:
    """
    Create a DataLoader for the noisy kappa field dataset with optional noise augmentation
    and resizing functionality.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the .npy file OR WandB artifact identifier
    batch_size : int
        Batch size for the dataloader
    snr_db_min : float
        Minimum signal-to-noise ratio in decibels
    snr_db_max : float
        Maximum signal-to-noise ratio in decibels
    reshape_size : Optional[Tuple[int, int]]
        If provided, reshape each field to this size (ny, nx) during loading
        (primarily for flattened data)
    resize_shape : Optional[Tuple[int, int]]
        If provided, resize the loaded fields to this shape (ny, nx)
        (for downsampling/upsampling)
    num_workers : int
        Number of worker processes for data loading
    device : str
        Device to store the data on ('cpu' or 'cuda')
    seed : Optional[int]
        Random seed for reproducibility
    is_artifact : bool
        Set to True if data_path is a WandB artifact identifier
    wandb_run : Optional[wandb.sdk.wandb_run.Run]
        The active WandB run instance (required if is_artifact is True)
    artifact_filename : str
        The specific filename to load from within the artifact (defaults to 'kappa_fine_samples.npy')
    num_augmentations : int
        Number of additional noise augmentations to generate for each clean sample (default: 0)
        if num_augmentations > 0, the batch will be (original_noisy, clean, augmented_noisy)
        if num_augmentations == 0, the batch will be (noisy, clean)
        if num_augmentations > 1, the batch will be (original_noisy, clean, augmented_noisy_1, augmented_noisy_2, ..., augmented_noisy_num_augmentations)
    transforms : Optional[Callable]
        Custom transform function to apply to the data
        (if provided, overrides resize_shape)

    Returns
    -------
    DataLoader
        DataLoader for the noisy kappa field dataset
    """
    actual_data_path = data_path
    if is_artifact:
        if wandb_run is None:
            raise ValueError("wandb_run must be provided when is_artifact is True")
        print(f"Using artifact: {data_path}")
        # Download the artifact
        artifact = wandb_run.use_artifact(data_path, type="dataset")
        # Get the directory where artifact files are downloaded
        artifact_dir = Path(artifact.download())
        # Construct the path to the specific file within the artifact directory
        actual_data_path = artifact_dir / artifact_filename
        print(f"Loading data from artifact file: {actual_data_path}")
    else:
        print(f"Loading data from local path: {actual_data_path}")

    # Determine which transforms to use (prioritize explicit transforms)
    dataset_transforms = transforms
    print(f"Dataset transforms: {dataset_transforms}")
    print(f"Resize shape: {resize_shape}")
    if dataset_transforms is None and resize_shape is not None:
        # Create a resize transform if resize_shape is provided
        dataset_transforms = create_resize_transform(resize_shape)
        print(f"Using resize transform: {dataset_transforms}")

    dataset = NoisyKappaFieldDataset(
        data_path=actual_data_path,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        reshape_size=reshape_size,
        device=device,
        seed=seed,
        transforms=dataset_transforms,
    )

    # Use custom collate function if augmentation is requested
    if num_augmentations > 0:

        def collate_function(batch):
            return noisy_kappa_collate_with_augmentation(
                batch,
                num_augmentations=num_augmentations,
                snr_db_min=snr_db_min,
                snr_db_max=snr_db_max,
                device=device,
            )

        collate_fn = collate_function
    else:
        collate_fn = None  # Use PyTorch's default collate

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader


# main test - update to demonstrate augmentation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input",
        type=str,
        default="ECE689AdvDL/DIPDE/kappa_field_pair-32x32-to-256x256:latest",
        help="Path to local .npy file OR WandB artifact identifier (e.g., 'entity/project/name:version')",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="artifact",
        choices=["artifact", "local"],
        help="Specify whether the data_input is a 'local' path or a 'artifact' identifier",
    )
    parser.add_argument(
        "--artifact_filename",
        type=str,
        default="kappa_fine_samples.npy",
        help="Filename to load from within the artifact (if using artifact input_type)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--snr_db_min", type=float, default=10.0)
    parser.add_argument("--snr_db_max", type=float, default=30.0)
    parser.add_argument(
        "--reshape_ny", type=int, default=None, help="Number of points in y direction"
    )
    parser.add_argument(
        "--reshape_nx", type=int, default=None, help="Number of points in x direction"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
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
        "--num_augmentations",
        type=int,
        default=0,
        help="Number of additional noise augmentations per clean sample",
    )

    args = parser.parse_args()

    # Initialize WandB run for this script (needed for using artifacts)
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, job_type="tests"
    )
    # Log config
    wandb.config.update(args)

    # Handle reshape_size based on individual dimensions
    reshape_size = None
    if args.reshape_ny is not None and args.reshape_nx is not None:
        reshape_size = (args.reshape_ny, args.reshape_nx)

    # Determine if input is an artifact
    is_artifact_input = args.input_type == "artifact"

    # Create dataloader with augmentation
    dataloader = create_noisy_kappa_dataloader(
        data_path=args.data_input,
        batch_size=args.batch_size,
        snr_db_min=args.snr_db_min,
        snr_db_max=args.snr_db_max,
        reshape_size=reshape_size,
        device=args.device,
        seed=args.seed,
        is_artifact=is_artifact_input,
        wandb_run=run,
        artifact_filename=args.artifact_filename,
        num_augmentations=args.num_augmentations,
    )

    # Handle the new data structure when using augmentations
    def plot_sample_pairs(original_noisy, clean, augmented_noisy=None):
        """Plot original and augmented noisy samples with the clean sample."""
        import matplotlib.pyplot as plt

        if augmented_noisy is None or not augmented_noisy:
            # Original functionality without augmentation
            print("Plotting first sample without augmentation")

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Process clean sample (squeeze extra dimensions if needed)
            if len(clean.shape) > 2:
                clean = clean.squeeze()
            im0 = axes[0].imshow(clean.cpu().numpy(), cmap="viridis")
            axes[0].set_title("Clean Field")
            plt.colorbar(im0, ax=axes[0])

            # Process original noisy sample (squeeze extra dimensions if needed)
            if len(original_noisy.shape) > 2:
                original_noisy = original_noisy.squeeze()
            im1 = axes[1].imshow(original_noisy.cpu().numpy(), cmap="viridis")
            axes[1].set_title("Noisy Field")
            plt.colorbar(im1, ax=axes[1])

            plt.tight_layout()
            plt.show()

            return
        else:
            # In case augmented_noisy is provided, plot clean, original noisy, and each augmentation
            num_variants = 1 + len(augmented_noisy)  # original + each augmentation
            fig, axes = plt.subplots(
                1, num_variants + 1, figsize=(4 * (num_variants + 1), 4)
            )

            # Plot clean sample
            if len(clean.shape) > 2:
                clean = clean.squeeze()
            im0 = axes[0].imshow(clean.cpu().numpy(), cmap="viridis")
            axes[0].set_title("Clean Field")
            plt.colorbar(im0, ax=axes[0])

            # Plot original noisy field
            if len(original_noisy.shape) > 2:
                original_noisy = original_noisy.squeeze()
            im1 = axes[1].imshow(original_noisy.cpu().numpy(), cmap="viridis")
            axes[1].set_title("Original Noisy")
            plt.colorbar(im1, ax=axes[1])

            # Plot each augmented noisy variant
            for i, aug in enumerate(augmented_noisy):
                if len(aug.shape) > 2:
                    aug = aug.squeeze()
                im = axes[i + 2].imshow(aug.cpu().numpy(), cmap="viridis")
                axes[i + 2].set_title(f"Augmented #{i + 1}")
                plt.colorbar(im, ax=axes[i + 2])

            plt.tight_layout()
            plt.savefig("noisy_kappa_sample.png")
            plt.show()

    # Get a batch and plot
    batch = next(iter(dataloader))
    if args.num_augmentations > 0:
        # Unpack the batch including augmentations
        original_noisy, clean, augmented_noisy = batch
        print(f"Plotting first sample with {args.num_augmentations} augmentations")
        plot_sample_pairs(
            original_noisy[0], clean[0], [aug[0] for aug in augmented_noisy]
        )
    else:
        # Original format without augmentation
        noisy_batch, clean_batch = batch
        print("Plotting first sample without augmentation")
        plot_sample_pairs(noisy_batch[0], clean_batch[0])

    # Finish the WandB run for this script
    run.finish()
