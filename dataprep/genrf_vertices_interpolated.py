#!/usr/bin/env python3
"""
Random Field Interpolator

This script loads non-Gaussian random field samples generated on a coarse grid
and interpolates them onto a finer grid for improved resolution. It uses PyTorch
to perform the upscaling.

Computational savings are achieved by generating the random fields at coarse
resolution first, then interpolating to the desired fine resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb  # Import wandb
import torch
import torch.nn.functional as F


def load_coarse_data(input_folder: str) -> tuple:
    """
    Load coarse grid data from the specified folder.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing coarse grid data

    Returns
    -------
    tuple
        Loaded kappa samples and related information
    """
    input_path = Path(input_folder)

    # Load non-Gaussian (conductivity) samples
    kappa_samples = np.load(input_path / "kappa_samples.npy")

    # Optional: Load Gaussian samples if needed
    # gaussian_samples = np.load(input_path / 'gaussian_field_samples.npy')

    print(
        f"Loaded {kappa_samples.shape[0]} samples with shape {kappa_samples.shape[1]}"
    )

    return kappa_samples


def create_fine_mesh(lx: float, ly: float, nx_fine: int, ny_fine: int) -> tuple:
    """
    Create a fine mesh grid.

    Parameters
    ----------
    lx : float
        Length of domain in x-direction
    ly : float
        Length of domain in y-direction
    nx_fine : int
        Number of points in x-direction (fine grid)
    ny_fine : int
        Number of points in y-direction (fine grid)

    Returns
    -------
    tuple
        X, Y mesh coordinates and flattened coordinates for fine grid
    """
    # Create fine mesh grid
    x_fine = np.linspace(0, lx, nx_fine)
    y_fine = np.linspace(0, ly, ny_fine)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # Stack coordinates for interpolation
    xy_fine_coords = np.column_stack((X_fine.flatten(), Y_fine.flatten()))

    return X_fine, Y_fine, xy_fine_coords, x_fine, y_fine


def recreate_coarse_mesh(lx: float, ly: float, nx_coarse: int, ny_coarse: int) -> tuple:
    """
    Recreate the coarse mesh grid to match the original generation.

    Parameters
    ----------
    lx : float
        Length of domain in x-direction
    ly : float
        Length of domain in y-direction
    nx_coarse : int
        Number of points in x-direction (coarse grid)
    ny_coarse : int
        Number of points in y-direction (coarse grid)

    Returns
    -------
    tuple
        X, Y mesh coordinates and flattened coordinates for coarse grid
    """
    # Create coarse mesh grid (must match the original generation)
    x_coarse = np.linspace(0, lx, nx_coarse)
    y_coarse = np.linspace(0, ly, ny_coarse)
    X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

    # Stack coordinates for interpolation
    xy_coarse_coords = np.column_stack((X_coarse.flatten(), Y_coarse.flatten()))

    return X_coarse, Y_coarse, xy_coarse_coords, x_coarse, y_coarse


def interpolate_samples_torch(
    kappa_samples: np.ndarray,
    nx_coarse: int,
    ny_coarse: int,
    nx_fine: int,
    ny_fine: int,
    lx: float,
    ly: float,
    method: str = "bicubic",
) -> np.ndarray:
    """
    Interpolate samples from coarse grid to fine grid using PyTorch.

    Parameters
    ----------
    kappa_samples : np.ndarray
        Non-Gaussian random field samples on coarse grid
    nx_coarse, ny_coarse : int
        Number of points in coarse grid
    nx_fine, ny_fine : int
        Number of points in fine grid
    lx, ly : float
        Domain dimensions
    method : str, optional
        Interpolation method ('bilinear', 'bicubic'), by default 'bicubic'

    Returns
    -------
    np.ndarray
        Interpolated samples on fine grid
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples = kappa_samples.shape[0]

    # Reshape all samples at once into a batch of 2D grids
    # [n_samples, nx_coarse, ny_coarse]
    samples_reshaped = kappa_samples.reshape(n_samples, nx_coarse, ny_coarse)

    # Convert to PyTorch tensor
    samples_tensor = torch.tensor(samples_reshaped, dtype=torch.float32, device=device)

    # Add channel dimension required by grid_sample/interpolate
    # [n_samples, 1, nx_coarse, ny_coarse]
    samples_tensor = samples_tensor.unsqueeze(1)

    # Use torch.nn.functional.interpolate for the upsampling
    # This handles the whole batch at once
    mode = "bicubic" if method == "cubic" else "bilinear"
    samples_fine = F.interpolate(
        samples_tensor,
        size=(nx_fine, ny_fine),
        mode=mode,
        align_corners=True,
    )

    # Convert back to numpy and reshape to desired output format
    # Remove the channel dimension and convert to numpy
    kappa_fine_samples = samples_fine.squeeze(1).cpu().numpy()

    return kappa_fine_samples


def convert_nodal_to_element(kappa_nodal, mesh=None):
    """
    Convert nodal kappa values to element values.

    Args:
        kappa_nodal: Tensor of shape [n_samples, 1, H, W] or numpy array
        mesh: Optional Mesh instance for reference

    Returns:
        Element-based kappa values
    """
    if isinstance(kappa_nodal, torch.Tensor):
        # PyTorch fast path using average pooling
        kappa_element = F.avg_pool2d(kappa_nodal, kernel_size=2, stride=1)
        return kappa_element
    else:
        # NumPy implementation [n_samples, n_x, n_y] into [n_samples, numel_x, numel_y]
        N = kappa_nodal.shape[1] - 1  # or appropriate dimension
        kappa_element = (
            kappa_nodal[:, :N, :N]
            + kappa_nodal[:, 1:, :N]
            + kappa_nodal[:, :N, 1:]
            + kappa_nodal[:, 1:, 1:]
        ) / 4.0
        return kappa_element


def save_interpolated_data(kappa_fine_samples: np.ndarray, output_folder: str) -> Path:
    """
    Save interpolated data to disk and return the file path.

    Parameters
    ----------
    kappa_fine_samples : np.ndarray
        Interpolated samples on fine grid
    output_folder : str
        Path to save the interpolated data

    Returns
    -------
    Path
        Path to the saved numpy file
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Define save path
    save_file_path = output_path / "kappa_fine_samples.npy"

    # Save interpolated samples
    np.save(save_file_path, kappa_fine_samples)

    print(
        f"Saved interpolated samples with shape {kappa_fine_samples.shape} to {save_file_path}"
    )
    return save_file_path  # Return the path


def plot_comparison(
    kappa_samples: np.ndarray,
    kappa_fine_samples: np.ndarray,
    nx_coarse: int,
    ny_coarse: int,
    nx_fine: int,
    ny_fine: int,
    output_folder: str,
    lx: float,
    ly: float,
) -> None:
    """
    Plot comparison between coarse and fine grid for a sample.

    Parameters
    ----------
    kappa_samples : np.ndarray
        Coarse grid samples
    kappa_fine_samples : np.ndarray
        Fine grid interpolated samples (element-based)
    nx_coarse, ny_coarse : int
        Coarse grid dimensions
    nx_fine, ny_fine : int
        Fine grid dimensions (nodal)
    output_folder : str
        Folder to save plots
    lx, ly : float
        Domain dimensions
    """
    output_path = Path(output_folder)

    # Select first sample for visualization
    sample_idx = 0

    # Reshape samples to 2D grids
    coarse_sample = kappa_samples[sample_idx].reshape(ny_coarse, nx_coarse)

    # For the fine sample, we're now using element-based values (nx_fine-1, ny_fine-1)
    num_elements_x = nx_fine - 1
    num_elements_y = ny_fine - 1
    fine_sample = kappa_fine_samples[sample_idx].reshape(num_elements_y, num_elements_x)

    # Create meshgrids
    x_coarse = np.linspace(0, lx, nx_coarse)
    y_coarse = np.linspace(0, ly, ny_coarse)

    # For elements, use cell centers for plotting
    x_fine_elem = np.linspace(0, lx, num_elements_x + 1)  # Cell edges
    y_fine_elem = np.linspace(0, ly, num_elements_y + 1)  # Cell edges

    # Create meshgrids
    X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

    # Plot comparison
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1]}
    )

    # Coarse grid
    im1 = axes[0].pcolormesh(
        X_coarse, Y_coarse, coarse_sample, shading="auto", cmap="viridis"
    )
    axes[0].set_title(f"Coarse Grid ({nx_coarse}x{ny_coarse})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Fine grid - use pcolormesh which accepts cell edges
    im2 = axes[1].pcolormesh(
        x_fine_elem, y_fine_elem, fine_sample, shading="auto", cmap="viridis"
    )
    axes[1].set_title(f"Fine Grid Elements ({num_elements_x}x{num_elements_y})")
    axes[1].set_xlabel("X")

    # Add colorbars
    plt.colorbar(im1, ax=axes[0], label="κ (W/m·K)")
    plt.colorbar(im2, ax=axes[1], label="κ (W/m·K)")

    plt.tight_layout()
    plt.savefig(output_path / "grid_comparison.png", dpi=300)
    plt.close(fig)

    # Create a difference plot - for element-based comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    # Interpolate coarse sample to fine element grid for direct comparison
    # First, get the cell centers for both coarse and fine grids
    x_coarse_centers = (x_coarse[:-1] + x_coarse[1:]) / 2
    y_coarse_centers = (y_coarse[:-1] + y_coarse[1:]) / 2
    x_fine_centers = (x_fine_elem[:-1] + x_fine_elem[1:]) / 2
    y_fine_centers = (y_fine_elem[:-1] + y_fine_elem[1:]) / 2

    X_coarse_centers, Y_coarse_centers = np.meshgrid(x_coarse_centers, y_coarse_centers)
    X_fine_centers, Y_fine_centers = np.meshgrid(x_fine_centers, y_fine_centers)

    # Reshape coarse sample for element-wise representation if needed
    if nx_coarse > x_coarse_centers.size:
        # Convert nodal to element for coarse sample too (if necessary)
        coarse_element = (
            coarse_sample[:-1, :-1]
            + coarse_sample[1:, :-1]
            + coarse_sample[:-1, 1:]
            + coarse_sample[1:, 1:]
        ) / 4.0
    else:
        coarse_element = coarse_sample

    # Use griddata for interpolation from coarse element centers to fine element centers
    coarse_values = coarse_element.flatten()
    X_coarse_flat, Y_coarse_flat = (
        X_coarse_centers.flatten(),
        Y_coarse_centers.flatten(),
    )

    interp_coarse = griddata(
        (X_coarse_flat, Y_coarse_flat),
        coarse_values,
        (X_fine_centers, Y_fine_centers),
        method="cubic",
    )

    # Calculate absolute difference
    diff = np.abs(fine_sample - interp_coarse)

    # Plot difference
    im = ax.pcolormesh(x_fine_elem, y_fine_elem, diff, shading="auto", cmap="hot")
    ax.set_title("Absolute Difference (Element Values)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.colorbar(im, ax=ax, label="Absolute Difference")
    plt.tight_layout()
    plt.savefig(output_path / "grid_difference.png", dpi=300)
    plt.close(fig)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interpolate random field data from coarse to fine grid."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="randomfield",
        help="Input folder with coarse grid data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="randomfield_fine",
        help="Output folder for fine grid data",
    )
    parser.add_argument(
        "--nx_coarse",
        type=int,
        default=32,
        help="Number of coarse grid points in x-direction",
    )
    parser.add_argument(
        "--ny_coarse",
        type=int,
        default=32,
        help="Number of coarse grid points in y-direction",
    )
    parser.add_argument(
        "--nx_fine",
        type=int,
        default=128,
        help="Number of fine grid points in x-direction",
    )
    parser.add_argument(
        "--ny_fine",
        type=int,
        default=128,
        help="Number of fine grid points in y-direction",
    )
    parser.add_argument(
        "--lx", type=float, default=1.0, help="Length of domain in x-direction"
    )
    parser.add_argument(
        "--ly", type=float, default=1.0, help="Length of domain in y-direction"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cubic",
        help="Interpolation method: linear, cubic, etc.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset type: train, val, or test",
    )

    return parser.parse_args()


def main():
    """Main function to run the interpolation process."""
    # Parse arguments
    try:
        args = parse_arguments()
        input_folder = args.input
        output_folder = args.output
        nx_coarse = args.nx_coarse
        ny_coarse = args.ny_coarse
        nx_fine = args.nx_fine
        ny_fine = args.ny_fine
        lx = args.lx
        ly = args.ly
        method = args.method
        dataset_type = args.dataset_type
    except:
        # Default parameters if parsing fails
        input_folder = "randomfield"
        output_folder = "randomfield_fine"
        nx_coarse = 32
        ny_coarse = 32
        nx_fine = 128
        ny_fine = 128
        lx = 1.0
        ly = 1.0
        method = "cubic"
        dataset_type = "train"

    # Initialize WandB
    run = wandb.init(
        project="DIPDE",
        entity="ECE689AdvDL",
        job_type="data_upload",
        tags=["upload", dataset_type],  # Add dataset type as a tag
        config={
            "input_folder": input_folder,
            "output_folder": output_folder,
            "nx_coarse": nx_coarse,
            "ny_coarse": ny_coarse,
            "nx_fine": nx_fine,
            "ny_fine": ny_fine,
            "lx": lx,
            "ly": ly,
            "method": method,
            "dataset_type": dataset_type,
        },
    )

    print(
        f"Interpolating from {nx_coarse}x{ny_coarse} grid to {nx_fine}x{ny_fine} grid"
    )

    # Load coarse data
    kappa_samples = load_coarse_data(input_folder)

    # Interpolate samples
    kappa_fine_samples = interpolate_samples_torch(
        kappa_samples, nx_coarse, ny_coarse, nx_fine, ny_fine, lx, ly, method
    )

    # Convert nodal to element
    kappa_fine_samples = convert_nodal_to_element(kappa_fine_samples)
    if isinstance(kappa_fine_samples, torch.Tensor):
        # [n_samples, 1, numel_x, numel_y]
        numel_x = kappa_fine_samples.shape[2]
        numel_y = kappa_fine_samples.shape[3]
        print(f"Interpolated fine grid has {numel_x}x{numel_y} elements")
    else:
        # [n_samples, numel_x, numel_y]
        numel_x = kappa_fine_samples.shape[1]
        numel_y = kappa_fine_samples.shape[2]
        print(f"Interpolated fine grid has {numel_x}x{numel_y} elements")
    # Save interpolated data and get the path
    saved_file_path = save_interpolated_data(kappa_fine_samples, output_folder)

    # Create and log artifact with dataset type
    artifact_name = (
        f"kappa_field_pair-{nx_coarse}x{ny_coarse}N-{numel_x}x{numel_y}E-{dataset_type}"
    )

    artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description=f"{dataset_type.capitalize()} dataset: Coarse ({nx_coarse}x{ny_coarse}) Nodal and interpolated fine ({numel_x}x{numel_y}) Elemental random field samples.",
        metadata=wandb.config.as_dict(),
    )
    # add the saved file (Fine Samples)
    artifact.add_file(
        str(saved_file_path), name="kappa_fine_samples.npy"
    )  # Optionally specify name in artifact
    # add the coarse grid samples
    artifact.add_file(
        str(Path(input_folder) / "kappa_samples.npy"), name="kappa_coarse_samples.npy"
    )  # Optionally specify name
    print(f"Added fine and coarse grid samples to artifact {artifact.name}.")

    run.log_artifact(artifact)
    print(f"Logged artifact {artifact.name} to WandB.")

    # Plot comparison
    plot_comparison(
        kappa_samples,
        kappa_fine_samples,
        nx_coarse,
        ny_coarse,
        nx_fine,
        ny_fine,
        output_folder,
        lx,
        ly,
    )

    print("Interpolation completed successfully!")
    run.finish()  # Finish the wandb run


if __name__ == "__main__":
    main()
