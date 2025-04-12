"""
Generate a dataset of u_kappa solutions from existing kappa fields.

This script loads kappa fields, resizes them to 64x64, and solves the forward problem
with sin(πx)sin(πy) forcing and zero Dirichlet boundary conditions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
from tqdm import tqdm

from fem.inverse_problem import FEMProblem
from dataprep.noisy_kappa_dataset import NoisyKappaFieldDataset


def solve_forward_problem(kappa, resolution=64, plot_example=False, example_idx=None):
    """
    Solve the forward problem for a given kappa field.

    Args:
        kappa: The conductivity field (element-wise values)
        resolution: The mesh resolution (number of elements per side)
        plot_example: Whether to plot the solution and kappa field
        example_idx: Index of the example (for plot titles)

    Returns:
        u_kappa: The solution of the forward problem
    """
    # Create FEM problem with specified resolution
    fem_problem = FEMProblem(num_pixels=resolution, rtol=1e-8)

    # Define zero Dirichlet boundary conditions
    u_d = np.zeros(fem_problem.num_nodes)

    # Get node coordinates for forcing function
    x_coords = fem_problem.mesh.coordinates[0, :]
    y_coords = fem_problem.mesh.coordinates[1, :]

    # Define forcing function f(x,y) = sin(πx)sin(πy)
    # This naturally creates zero values at the boundary (x=0, x=1, y=0, y=1)
    f = np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords)

    # Set parameters for the forward problem
    fem_problem.set_parameters(f=f, u_d=u_d)

    # Solve the forward problem with the provided kappa field
    u_kappa = fem_problem.forward(kappa)

    # Optionally plot the results
    if plot_example:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Instead of using triangulation, reconstruct the structured grid.
        num_nodes_x = fem_problem.mesh.num_elements_x + 1
        num_nodes_y = fem_problem.mesh.num_elements_y + 1

        # Reshape nodal coordinates into a grid.
        x_nodes = fem_problem.mesh.coordinates[0, :].reshape(num_nodes_y, num_nodes_x)
        y_nodes = fem_problem.mesh.coordinates[1, :].reshape(num_nodes_y, num_nodes_x)

        # kappa is defined per element. Reshape it to a grid of shape (num_elements_y, num_elements_x)
        kappa_reshaped = kappa.reshape(
            fem_problem.mesh.num_elements_y, fem_problem.mesh.num_elements_x
        )

        # pcolormesh expects coordinates for cell boundaries, which coincide with x_nodes, y_nodes.
        kappa_plot = ax1.pcolormesh(x_nodes, y_nodes, kappa_reshaped, cmap="viridis")
        ax1.set_title(
            f"Kappa Field (Example {example_idx})"
            if example_idx is not None
            else "Kappa Field"
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(kappa_plot, ax=ax1)

        # Plot solution u_kappa using the mesh plot method from the FEM mesh.
        title = (
            f"Solution u_kappa (Example {example_idx})"
            if example_idx is not None
            else "Solution u_kappa"
        )
        fem_problem.mesh.plot(u_kappa, title=title)
        plt.tight_layout()

        # Create output directory if it doesn't exist
        os.makedirs("figures", exist_ok=True)
        plt.savefig(
            f"figures/example_{example_idx}_ukappa.png"
            if example_idx is not None
            else "figures/example_ukappa.png"
        )
        plt.close()

    return u_kappa


def generate_ukappa_dataset(
    data_path,
    output_path,
    resolution=64,
    num_samples=None,
    plot_examples=5,
    use_wandb=False,
    dataset_type="train",
    seed=None,
):
    """
    Generate a dataset of u_kappa solutions from kappa fields.

    Args:
        data_path: Path to the kappa fields data
        output_path: Path to save the u_kappa dataset
        resolution: Resolution to resize the kappa fields to
        num_samples: Number of samples to process (None for all)
        plot_examples: Number of examples to plot
        use_wandb: Whether to use wandb to download data
        dataset_type: Type of dataset to generate
        seed: Random seed for reproducibility

    Returns:
        None
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")
    else:
        print("Using system entropy for random seed")

    # Setup wandb if needed
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="DIPDE",
            entity="ECE689AdvDL",
            job_type="data_preparation",
        )
        artifact = wandb.use_artifact(
            f"ECE689AdvDL/DIPDE/kappa_field_pair-32x32N-255x255E-{dataset_type}:latest"
        )
        # Get the directory where artifact files are downloaded
        artifact_dir = Path(artifact.download())
        # Construct the path to the specific file within the artifact directory
        data_path = artifact_dir / "kappa_fine_samples.npy"

    # Load the kappa fields and resize them
    print(f"Loading kappa fields from {data_path}")
    dataset = NoisyKappaFieldDataset(
        data_path=data_path,
        reshape_size=(resolution, resolution),
    )

    # Get the clean kappa fields (without noise),
    kappa_fields = dataset.clean_fields.numpy()  # We don't want noise for this dataset

    # Limit number of samples if specified
    if num_samples is not None and num_samples < len(kappa_fields):
        kappa_fields = kappa_fields[:num_samples]

    print(f"Processing {len(kappa_fields)} kappa fields")

    # Prepare array to store u_kappa solutions
    u_kappa_dataset = []

    # Process each kappa field
    start_time = time.time()
    for i, kappa in enumerate(tqdm(kappa_fields)):
        # Check if we should plot this example
        plot_this_example = i < plot_examples

        # Solve the forward problem
        u_kappa = solve_forward_problem(
            kappa, resolution=resolution, plot_example=plot_this_example, example_idx=i
        )

        # Store the solution
        u_kappa_dataset.append(u_kappa)

    # Convert to numpy array
    u_kappa_dataset = np.array(u_kappa_dataset)

    # Save the dataset
    print(f"Saving dataset to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, u_kappa_dataset)

    # Report statistics
    end_time = time.time()
    print(f"Generated {len(u_kappa_dataset)} u_kappa solutions")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Dataset shape: {u_kappa_dataset.shape}")
    print(f"Min value: {u_kappa_dataset.min()}")
    print(f"Max value: {u_kappa_dataset.max()}")
    print(f"Mean value: {u_kappa_dataset.mean()}")
    print(f"Std value: {u_kappa_dataset.std()}")

    # Optionally log to wandb
    if use_wandb:
        wandb.log(
            {
                "dataset_size": len(u_kappa_dataset),
                "min_value": u_kappa_dataset.min(),
                "max_value": u_kappa_dataset.max(),
                "mean_value": u_kappa_dataset.mean(),
                "std_value": u_kappa_dataset.std(),
            }
        )

        # Create an artifact
        artifact = wandb.Artifact(
            name=f"ukappa_dataset_{resolution}x{resolution}-{dataset_type}",
            type="dataset",
            description=f"Dataset of u_kappa solutions from {resolution}x{resolution} kappa fields",
        )
        artifact.add_file(output_path)
        wandb_run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate u_kappa dataset from kappa fields"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/kappa_fine_samples.npy",
        help="Path to the kappa fields data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/ukappa_dataset.npy",
        help="Path to save the u_kappa dataset",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Resolution (number of elements per side) to resize the kappa fields to",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (None for all)",
    )
    parser.add_argument(
        "--plot_examples", type=int, default=5, help="Number of examples to plot"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to use wandb to download data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default is None (use system entropy).",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset type: train, val, or test",
    )

    args = parser.parse_args()

    generate_ukappa_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        resolution=args.resolution,
        num_samples=args.num_samples,
        plot_examples=args.plot_examples,
        use_wandb=args.use_wandb,
        seed=args.seed,
        dataset_type=args.dataset_type,
    )
