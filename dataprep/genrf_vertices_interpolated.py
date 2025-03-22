#!/usr/bin/env python3
"""
Random Field Interpolator

This script loads non-Gaussian random field samples generated on a coarse grid
and interpolates them onto a finer grid for improved resolution. It uses
scipy's interpolation functions to perform the upscaling efficiently.

This approach allows computational savings by generating the random fields
at coarse resolution first, then interpolating to the desired fine resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
import os
from pathlib import Path
import argparse
from tqdm import tqdm


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
    kappa_samples = np.load(input_path / 'kappa_samples.npy')
    
    # Optional: Load Gaussian samples if needed
    # gaussian_samples = np.load(input_path / 'gaussian_field_samples.npy')
    
    print(f"Loaded {kappa_samples.shape[0]} samples with shape {kappa_samples.shape[1]}")
    
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


def interpolate_samples(kappa_samples: np.ndarray, 
                        nx_coarse: int, ny_coarse: int,
                        nx_fine: int, ny_fine: int,
                        lx: float, ly: float,
                        method: str = 'cubic') -> np.ndarray:
    """
    Interpolate samples from coarse grid to fine grid.
    
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
        Interpolation method ('linear', 'cubic', etc.), by default 'cubic'
        
    Returns
    -------
    np.ndarray
        Interpolated samples on fine grid
    """
    n_samples = kappa_samples.shape[0]
    
    # Recreate coarse mesh
    X_coarse, Y_coarse, xy_coarse_coords, x_coarse, y_coarse = recreate_coarse_mesh(
        lx, ly, nx_coarse, ny_coarse)
    
    # Create fine mesh
    X_fine, Y_fine, xy_fine_coords, x_fine, y_fine = create_fine_mesh(
        lx, ly, nx_fine, ny_fine)
    
    # Initialize array for interpolated samples
    kappa_fine_samples = np.zeros((n_samples, nx_fine * ny_fine))
    
    # Interpolate each sample
    for i in tqdm(range(n_samples), desc="Interpolating samples"):
        # Reshape the sample back to 2D grid
        sample_2d = kappa_samples[i].reshape(ny_coarse, nx_coarse)
        
        # Use RegularGridInterpolator for faster interpolation on structured grids
        interp_func = RegularGridInterpolator(
            (y_coarse, x_coarse), 
            sample_2d, 
            method=method,
            bounds_error=False,
            fill_value=None
        )
        
        # Points for interpolation must be (y, x) to match the order used in RegularGridInterpolator
        points = np.column_stack((Y_fine.flatten(), X_fine.flatten()))
        
        # Perform interpolation
        kappa_fine_samples[i] = interp_func(points)
    
    return kappa_fine_samples


def save_interpolated_data(kappa_fine_samples: np.ndarray, output_folder: str) -> None:
    """
    Save interpolated data to disk.
    
    Parameters
    ----------
    kappa_fine_samples : np.ndarray
        Interpolated samples on fine grid
    output_folder : str
        Path to save the interpolated data
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Save interpolated samples
    np.save(output_path / 'kappa_fine_samples.npy', kappa_fine_samples)
    
    print(f"Saved interpolated samples with shape {kappa_fine_samples.shape}")


def plot_comparison(kappa_samples: np.ndarray, kappa_fine_samples: np.ndarray,
                    nx_coarse: int, ny_coarse: int, 
                    nx_fine: int, ny_fine: int,
                    output_folder: str,
                    lx: float, ly: float) -> None:
    """
    Plot comparison between coarse and fine grid for a sample.
    
    Parameters
    ----------
    kappa_samples : np.ndarray
        Coarse grid samples
    kappa_fine_samples : np.ndarray
        Fine grid interpolated samples
    nx_coarse, ny_coarse : int
        Coarse grid dimensions
    nx_fine, ny_fine : int
        Fine grid dimensions
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
    fine_sample = kappa_fine_samples[sample_idx].reshape(ny_fine, ny_fine)
    
    # Create meshgrids
    x_coarse = np.linspace(0, lx, nx_coarse)
    y_coarse = np.linspace(0, ly, ny_coarse)
    x_fine = np.linspace(0, lx, nx_fine)
    y_fine = np.linspace(0, ly, ny_fine)
    
    X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), 
                             gridspec_kw={'width_ratios': [1, 1]})
    
    # Coarse grid
    im1 = axes[0].pcolormesh(X_coarse, Y_coarse, coarse_sample, 
                            shading='auto', cmap='viridis')
    axes[0].set_title(f'Coarse Grid ({nx_coarse}x{ny_coarse})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Fine grid
    im2 = axes[1].pcolormesh(X_fine, Y_fine, fine_sample, 
                            shading='auto', cmap='viridis')
    axes[1].set_title(f'Fine Grid ({nx_fine}x{ny_fine})')
    axes[1].set_xlabel('X')
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0], label='κ (W/m·K)')
    plt.colorbar(im2, ax=axes[1], label='κ (W/m·K)')
    
    plt.tight_layout()
    plt.savefig(output_path / 'grid_comparison.png', dpi=300)
    plt.close(fig)
    
    # Also create a difference plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Interpolate coarse sample to fine grid for direct comparison
    X_coarse_flat, Y_coarse_flat = X_coarse.flatten(), Y_coarse.flatten()
    coarse_values = coarse_sample.flatten()
    
    # Use griddata for interpolation to match fine grid points
    interp_coarse = griddata(
        (X_coarse_flat, Y_coarse_flat), 
        coarse_values, 
        (X_fine, Y_fine), 
        method='cubic'
    )
    
    # Calculate absolute difference
    diff = np.abs(fine_sample - interp_coarse)
    
    # Plot difference
    im = ax.pcolormesh(X_fine, Y_fine, diff, 
                      shading='auto', cmap='hot')
    ax.set_title('Absolute Difference')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.colorbar(im, ax=ax, label='Absolute Difference')
    plt.tight_layout()
    plt.savefig(output_path / 'grid_difference.png', dpi=300)
    plt.close(fig)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Interpolate random field data from coarse to fine grid.')
    
    parser.add_argument('--input', type=str, default='randomfield',
                       help='Input folder with coarse grid data')
    parser.add_argument('--output', type=str, default='randomfield_fine',
                       help='Output folder for fine grid data')
    parser.add_argument('--nx_coarse', type=int, default=32,
                       help='Number of coarse grid points in x-direction')
    parser.add_argument('--ny_coarse', type=int, default=32,
                       help='Number of coarse grid points in y-direction')
    parser.add_argument('--nx_fine', type=int, default=128,
                       help='Number of fine grid points in x-direction')
    parser.add_argument('--ny_fine', type=int, default=128,
                       help='Number of fine grid points in y-direction')
    parser.add_argument('--lx', type=float, default=1.0,
                       help='Length of domain in x-direction')
    parser.add_argument('--ly', type=float, default=1.0,
                       help='Length of domain in y-direction')
    parser.add_argument('--method', type=str, default='cubic',
                       help='Interpolation method: linear, cubic, etc.')
    
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
    except:
        # Default parameters if parsing fails
        input_folder = 'randomfield'
        output_folder = 'randomfield_fine'
        nx_coarse = 32
        ny_coarse = 32
        nx_fine = 128
        ny_fine = 128
        lx = 1.0
        ly = 1.0
        method = 'cubic'
    
    print(f"Interpolating from {nx_coarse}x{ny_coarse} grid to {nx_fine}x{ny_fine} grid")
    
    # Load coarse data
    kappa_samples = load_coarse_data(input_folder)
    
    # Interpolate samples
    kappa_fine_samples = interpolate_samples(
        kappa_samples, nx_coarse, ny_coarse, nx_fine, ny_fine, lx, ly, method
    )
    
    # Save interpolated data
    save_interpolated_data(kappa_fine_samples, output_folder)
    
    # Plot comparison
    plot_comparison(
        kappa_samples, kappa_fine_samples, 
        nx_coarse, ny_coarse, nx_fine, ny_fine,
        output_folder, lx, ly
    )
    
    print("Interpolation completed successfully!")


if __name__ == '__main__':
    main()