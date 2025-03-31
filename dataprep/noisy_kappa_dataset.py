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
from typing import Optional, Tuple, Union
import logging

import matplotlib.pyplot as plt
import argparse


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
        """
        self.data_path = Path(data_path)
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.reshape_size = reshape_size
        self.device = device

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
            f"SNR range: [{snr_db_min}, {snr_db_max}] dB"
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
                # Single sample with shape (ny, nx)
                # Expand to (1, ny, nx)
                kappa_fields = kappa_fields[np.newaxis, :, :]
                self.logger.info(f"Reshaped single sample to {kappa_fields.shape}")
            elif len(kappa_fields.shape) == 3:
                self.logger.info(f"Already in the right format: {kappa_fields.shape}")
                # Multiple samples with shape (n_samples, ny, nx)
                # Already in the right format
                pass
            elif len(kappa_fields.shape) == 2 and kappa_fields.shape[0] > 1:
                # Multiple flattened samples with shape (n_samples, ny*nx)
                if self.reshape_size is None:
                    raise ValueError("reshape_size must be provided for flattened data")
                ny, nx = self.reshape_size
                kappa_fields = kappa_fields.reshape(kappa_fields.shape[0], ny, nx)
                self.logger.info(f"Reshaped flattened samples to {kappa_fields.shape}")

            # Convert to torch tensor
            self.clean_fields = torch.tensor(
                kappa_fields, dtype=torch.float32, device=self.device
            )

            # Compute signal power across the spatial dimensions (ny, nx)
            # This ensures we get one power value per sample
            self.signal_powers = torch.var(
                self.clean_fields.reshape(len(self.clean_fields), -1), dim=1
            )

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
            Tuple of (noisy_sample, clean_sample)
        """
        clean_sample = self.clean_fields[idx]

        # Generate noise with the same shape as the clean sample
        noise = self._generate_noise(self.signal_powers[idx], clean_sample.shape)

        # Add noise to create noisy sample
        noisy_sample = clean_sample + noise

        return noisy_sample, clean_sample


def create_noisy_kappa_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 128,
    snr_db_min: float = 10.0,
    snr_db_max: float = 30.0,
    reshape_size: Optional[Tuple[int, int]] = None,
    num_workers: int = 0,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for the noisy kappa field dataset.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the .npy file containing kappa fields
    batch_size : int
        Batch size for the dataloader
    snr_db_min : float
        Minimum signal-to-noise ratio in decibels
    snr_db_max : float
        Maximum signal-to-noise ratio in decibels
    reshape_size : Optional[Tuple[int, int]]
        If provided, reshape each field to this size (ny, nx)
    num_workers : int
        Number of worker processes for data loading
    device : str
        Device to store the data on ('cpu' or 'cuda')
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    DataLoader
        DataLoader for the noisy kappa field dataset
    """
    dataset = NoisyKappaFieldDataset(
        data_path=data_path,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        reshape_size=reshape_size,
        device=device,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    return dataloader


# main test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data1/jy384/research/Data/DIPDE/kappa_fine_samples.npy",
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
    args = parser.parse_args()

    # Handle reshape_size based on individual dimensions
    reshape_size = None
    if args.reshape_ny is not None and args.reshape_nx is not None:
        reshape_size = (args.reshape_ny, args.reshape_nx)

    # Create dataloader
    dataloader = create_noisy_kappa_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        snr_db_min=args.snr_db_min,
        snr_db_max=args.snr_db_max,
        reshape_size=reshape_size,
        device=args.device,
        seed=args.seed,
    )

    # the dataset and dataloader usage
    def plot_sample_pair(
        noisy: torch.Tensor, clean: torch.Tensor, snr_db_range: Tuple[float, float]
    ):
        """Plot a pair of noisy and clean samples."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Ensure tensors are 2D
        if len(clean.shape) > 2:
            clean = clean.squeeze()
        if len(noisy.shape) > 2:
            noisy = noisy.squeeze()

        # Calculate actual SNR
        noise = noisy - clean
        signal_power = torch.var(clean).item()
        noise_power = torch.var(noise).item()
        actual_snr_db = 10 * np.log10(signal_power / noise_power)

        im1 = ax1.imshow(clean.cpu().numpy())
        ax1.set_title("Clean Field")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(noisy.cpu().numpy())
        ax2.set_title(f"Noisy Field (SNR: {actual_snr_db:.1f} dB)")
        plt.colorbar(im2, ax=ax2)

        plt.suptitle(f"SNR Range: [{snr_db_range[0]}, {snr_db_range[1]}] dB")
        plt.tight_layout()
        plt.savefig(f"noisy_kappa_sample_{snr_db_range[0]}_{snr_db_range[1]}.png")
        plt.close()

    # Get a batch and plot first sample
    noisy_batch, clean_batch = next(iter(dataloader))
    print("Plotting first sample")
    plot_sample_pair(noisy_batch[0], clean_batch[0], (args.snr_db_min, args.snr_db_max))
