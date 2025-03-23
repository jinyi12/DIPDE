#!/usr/bin/env python3
"""
Non-Gaussian Random Field Generator using Karhunen-Loève Expansion

This script generates non-Gaussian random fields using the Karhunen-Loève (KL) expansion
method. It creates a mesh grid, calculates a covariance matrix with an exponential kernel,
computes eigenvalues and eigenvectors, and then uses these to generate random field samples.
The Gaussian random field is then transformed into a non-Gaussian random field using gamma
inverse transformation.

Author: Jin Yi Yong
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import csv
import os
from pathlib import Path
from typing import Tuple, Optional
import argparse
from tqdm import tqdm  # For progress bars


def create_mesh(
    lx: float, ly: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D mesh grid.

    Parameters
    ----------
    lx : float
        Length of domain in x-direction
    ly : float
        Length of domain in y-direction
    nx : int
        Number of points in x-direction
    ny : int
        Number of points in y-direction

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        X, Y mesh coordinates and flattened coordinates (nx*ny, 2)
    """
    # Create mesh grid
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)

    # Stack coordinates for distance calculations
    xy_coords = np.column_stack((X.flatten(), Y.flatten()))

    return X, Y, xy_coords


def plot_mesh(X: np.ndarray, Y: np.ndarray, output_path: Path) -> None:
    """
    Plot and save mesh grid.

    Parameters
    ----------
    X : np.ndarray
        X coordinates of mesh
    Y : np.ndarray
        Y coordinates of mesh
    output_path : Path
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(X, Y, "k.", markersize=4)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_title("Mesh Grid")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.savefig(output_path / "mesh.png", dpi=300)
    plt.close(fig)


def compute_covariance_matrix(
    coords: np.ndarray, correlation_length: float, sigma: float = 1.0
) -> np.ndarray:
    """
    Compute covariance matrix using exponential kernel.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of shape (n_points, 2)
    correlation_length : float
        Correlation length of the random field
    sigma : float, optional
        Standard deviation, by default 1.0

    Returns
    -------
    np.ndarray
        Covariance matrix
    """
    # More efficient distance computation using scipy
    distances = squareform(pdist(coords, "euclidean"))

    # Exponential kernel
    cov_matrix = sigma * np.exp(-distances / correlation_length)

    return cov_matrix


def compute_kl_decomposition(
    cov_matrix: np.ndarray, error_threshold: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute and truncate Karhunen-Loève decomposition.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    error_threshold : float, optional
        Error threshold for truncation, by default 1e-3

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        Truncated eigenvalues, eigenvectors, and number of retained components
    """
    # Compute eigenvalues and eigenvectors using scipy's more stable algorithm
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = np.abs(eig_vals[idx])  # Ensure positive eigenvalues
    eig_vecs = eig_vecs[:, idx]

    # Calculate error function
    error_func = 1 - (np.cumsum(eig_vals) / np.sum(eig_vals))

    # Find truncation point
    n_truncate = np.argwhere(error_func <= error_threshold)[0][0] + 1

    return eig_vals[:n_truncate], eig_vecs[:, :n_truncate], n_truncate


def plot_eigenvalues(
    eig_vals: np.ndarray, error_func: np.ndarray, output_path: Path, n_points: int
) -> None:
    """
    Plot eigenvalue decay and error function.

    Parameters
    ----------
    eig_vals : np.ndarray
        Eigenvalues
    error_func : np.ndarray
        Error function
    output_path : Path
        Path to save plots
    n_points : int
        Number of points in the mesh
    """
    # Plot error function
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(error_func, "k-", linewidth=2)
    ax.set_xlim(0, n_points)
    ax.set_ylim(0, 1)
    ax.set_title("Error Function: Eigenvalues")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Error Function")
    fig.savefig(output_path / "eigenvalues_decay.png", dpi=300)
    plt.close(fig)

    # Plot error function on log scale
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(error_func, "k-", linewidth=2)
    ax.set_title("Error Function: Eigenvalues (Log Scale)")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Error Function (Log Scale)")
    fig.savefig(output_path / "eigenvalues_decay_semilogy.png", dpi=300)
    plt.close(fig)

    # Plot eigenvalues on log scale
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(eig_vals, "k-", linewidth=2)
    ax.set_title("Eigenvalues (Log Scale)")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalues (Log Scale)")
    fig.savefig(output_path / "eigenvalues_semilogy.png", dpi=300)
    plt.close(fig)


def generate_random_fields(
    n_samples: int,
    eig_vals: np.ndarray,
    eig_vecs: np.ndarray,
    mu_target: float,
    sigma_target: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gaussian and non-Gaussian (conductivity) random fields.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    eig_vals : np.ndarray
        Truncated eigenvalues
    eig_vecs : np.ndarray
        Truncated eigenvectors
    mu_target : float
        Target mean for the non-Gaussian (conductivity) field
    sigma_target : float
        Target standard deviation for the non-Gaussian field

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Gaussian field samples and non-Gaussian (conductivity) field samples (kappa_samples)
    """
    n_points = eig_vecs.shape[0]
    n_kl = eig_vals.shape[0]

    # Parameters for standard normal
    mu_gauss = 0
    sigma_gauss = 1

    # Parameters for target gamma distribution
    beta = 1 / np.power(sigma_target, 2)
    alpha = mu_target * np.power(sigma_target, 2)

    # Initialize arrays for field samples
    gaussian_samples = np.zeros((n_samples, n_points))
    kappa_samples = np.zeros((n_samples, n_points))

    # Create sqrt of eigenvalue diagonal matrix once
    sqrt_eig_vals_diag = np.diag(np.sqrt(eig_vals))

    # Generate samples with progress bar
    for j in tqdm(range(n_samples), desc="Generating random fields"):
        # Generate standard normal random variables
        xi = np.random.normal(mu_gauss, sigma_gauss, n_kl)

        # Compute KL expansion to obtain Gaussian field realization
        gaussian = mu_gauss + sigma_gauss * eig_vecs @ sqrt_eig_vals_diag @ xi

        # Non-linear transformation to obtain non-Gaussian (conductivity) field
        # Apply the probability integral transform: Standard normal CDF -> gamma PPF
        z_normcdf = stats.norm.cdf(gaussian, mu_gauss, sigma_gauss)
        kappa = stats.gamma.ppf(z_normcdf, beta, scale=alpha)

        # Store samples
        gaussian_samples[j, :] = gaussian
        kappa_samples[j, :] = kappa

    return gaussian_samples, kappa_samples


def save_samples(
    gaussian_samples: np.ndarray, kappa_samples: np.ndarray, output_path: Path
) -> None:
    """
    Save random field samples to disk.

    Parameters
    ----------
    gaussian_samples : np.ndarray
        Gaussian field samples obtained from the KL expansion.
    kappa_samples : np.ndarray
        Non-Gaussian (conductivity) field samples after transformation.
    output_path : Path
        Path to save the files.
    """
    # Save Gaussian samples
    np.save(output_path / "gaussian_field_samples.npy", gaussian_samples)

    # Save non-Gaussian (conductivity) samples
    np.save(output_path / "kappa_samples.npy", kappa_samples)


def plot_convergence(
    kappa_samples: np.ndarray,
    cov_matrix: np.ndarray,
    output_path: Path,
    n_samples: int,
    nx: int,
) -> None:
    """
    Plot convergence diagnostics for the generated samples.

    Parameters
    ----------
    kappa_samples : np.ndarray
        Non-Gaussian field samples
    cov_matrix : np.ndarray
        Theoretical covariance matrix
    output_path : Path
        Path to save the plots
    n_samples : int
        Number of samples
    nx : int
        Number of points in x-direction
    """
    # Calculate correlation coefficient matrix
    corr_samp = np.corrcoef(kappa_samples, rowvar=False)

    # Calculate mean of samples
    mean_samp = np.mean(kappa_samples, axis=0)

    # Plot mean
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_samp, "k-", linewidth=2)
    ax.set_title(f"Mean of Samples\nn_samples = {n_samples}")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Mean")
    fig.savefig(output_path / "mean_samples.png", dpi=300)
    plt.close(fig)

    # Plot correlation function
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(cov_matrix[0, :nx], "k*-", linewidth=2, label="Theoretical")
    ax.plot(corr_samp[0, :nx], "r*-", linewidth=2, label="Estimated")
    ax.legend()
    ax.set_title(f"Correlation Convergence of Node 1\nn_samples = {n_samples}")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Correlation")
    fig.savefig(output_path / "corr_convergence.png", dpi=300)
    plt.close(fig)


def generate_random_field(
    n_samples: int,
    lx: float,
    ly: float,
    correlation_length: float,
    nx: int,
    ny: int,
    output_folder: str,
    mu_target: float,
    sigma_target: float,
    error_threshold: float,
) -> None:
    """
    Generate non-Gaussian random fields using KL expansion.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    lx : float
        Length of domain in x-direction
    ly : float
        Length of domain in y-direction
    correlation_length : float
        Correlation length of the random field
    nx : int
        Number of points in x-direction
    ny : int
        Number of points in y-direction
    output_folder : str
        Path to save outputs
    mu_target : float
        Target mean for non-Gaussian field
    sigma_target : float
        Target standard deviation for non-Gaussian field
    """
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Create mesh
    X, Y, xy_coords = create_mesh(lx, ly, nx, ny)
    plot_mesh(X, Y, output_path)

    print("Calculating covariance matrix...")
    cov_matrix = compute_covariance_matrix(xy_coords, correlation_length)

    print("Computing KL decomposition...")
    eig_vals, eig_vecs, n_truncate = compute_kl_decomposition(
        cov_matrix, error_threshold=error_threshold
    )
    print(f"Truncated to {n_truncate} KL components")

    # Calculate error function for plotting
    full_eig_vals = np.abs(np.linalg.eigvalsh(cov_matrix))
    full_eig_vals = np.sort(full_eig_vals)[::-1]
    error_func = 1 - (np.cumsum(full_eig_vals) / np.sum(full_eig_vals))

    # Plot eigenvalue information
    plot_eigenvalues(full_eig_vals, error_func, output_path, nx * ny)

    # Generate random fields
    print("Generating random fields...")
    gaussian_samples, kappa_samples = generate_random_fields(
        n_samples, eig_vals, eig_vecs, mu_target, sigma_target
    )

    # Save samples
    print("Saving samples to npy...")
    save_samples(gaussian_samples, kappa_samples, output_path)

    # Plot convergence diagnostics
    print("Plotting convergence diagnostics...")
    plot_convergence(kappa_samples, cov_matrix, output_path, n_samples, nx)

    print("Simulation completed successfully!")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate non-Gaussian random fields using KL expansion."
    )

    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--lx", type=float, default=1.0, help="Length of domain in x-direction"
    )
    parser.add_argument(
        "--ly", type=float, default=1.0, help="Length of domain in y-direction"
    )
    parser.add_argument("--lc", type=float, default=0.2, help="Correlation length")
    parser.add_argument(
        "--nx", type=int, default=32, help="Number of points in x-direction"
    )
    parser.add_argument(
        "--ny", type=int, default=32, help="Number of points in y-direction"
    )
    parser.add_argument(
        "--mu", type=float, default=2.7, help="Target mean for non-Gaussian field"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="Target standard deviation for non-Gaussian field",
    )
    parser.add_argument(
        "--output", type=str, default="randomfield", help="Output folder path"
    )
    parser.add_argument(
        "--error_threshold",
        type=float,
        default=1e-3,
        help="Error threshold for KL decomposition",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Default parameters
    default_n_samples = 10000
    default_lx = 1.0
    default_ly = 1.0
    default_lc = default_lx / 5  # correlation length
    default_nx = 32
    default_ny = 32
    default_mu_target = 2.7  # conductivity of 17 W/mK
    default_sigma_target = 0.3
    default_output_folder = "randomfield"
    default_error_threshold = 1e-3
    # Parse command line arguments if provided
    try:
        args = parse_arguments()
        n_samples = args.n_samples
        lx = args.lx
        ly = args.ly
        lc = args.lc
        nx = args.nx
        ny = args.ny
        mu_target = args.mu
        sigma_target = args.sigma
        output_folder = args.output
        error_threshold = args.error_threshold
    except:
        # Use defaults if argument parsing fails
        n_samples = default_n_samples
        lx = default_lx
        ly = default_ly
        lc = default_lc
        nx = default_nx
        ny = default_ny
        mu_target = default_mu_target
        sigma_target = default_sigma_target
        output_folder = default_output_folder
        error_threshold = default_error_threshold

    # compute correlation length, correlation length must be >= 4 * discretization length, here we use 8 * discretization length
    lc = 8 * lx / nx

    # Generate random field
    generate_random_field(
        n_samples,
        lx,
        ly,
        lc,
        nx,
        ny,
        output_folder,
        mu_target,
        sigma_target,
        error_threshold,
    )
