"""
UKappa Inverse Problem Solver

This script solves inverse problems for the ukappa dataset using a denoiser regularizer.
It downloads the ukappa dataset from wandb, adds noise to create synthetic measurements,
and solves the inverse problem to recover the conductivity field.

Usage:
    python -m src.ukappa_inverse_problem --noise 0.01 --lambda_reg 0.1 --num_samples 3
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import torch
from pathlib import Path
import wandb
from tqdm import tqdm
from scipy.optimize import minimize

from fem.inverse_problem import FEMProblem
from models.denoiser import KappaDenoiser
from fem.regularizers import (
    DenoiserRegularizer,
    ValueRegularizer,
    GradientRegularizer,
    TotalVariationRegularizer,
)
from dataprep.noisy_kappa_dataset import NoisyKappaFieldDataset


class ProximalOperator:
    """
    Proximal operator for the inverse problem optimization.

    This class implements the proximal objective and gradient calculation
    for the alternating minimization approach.
    """

    def __init__(self, kappa_intermediate, fem_problem, step_size=0.01):
        self.kappa_intermediate = kappa_intermediate
        self.fem_problem = fem_problem
        self.step_size = step_size

    def proximal_objective(self, kappa):
        """Calculate the proximal objective function value"""
        solution_misfit = self.fem_problem.objective(kappa)
        kappa_misfit = 0.5 * np.sum((kappa - self.kappa_intermediate) ** 2)
        return kappa_misfit + self.step_size * solution_misfit

    def proximal_gradient(self, kappa):
        """Calculate the gradient of the proximal objective function"""
        misfit_grad = self.fem_problem.gradient(kappa)
        kappa_misfit_grad = kappa - self.kappa_intermediate
        return kappa_misfit_grad + self.step_size * misfit_grad


def solve_inverse_problem_for_ukappa(
    ukappa,
    true_kappa,
    resolution=64,
    noise_level=0.01,
    initial_lambda_reg=0.1,
    initial_step_size=1,
    denoiser_path=None,
    norm_min=None,
    norm_max=None,
    plot_results=True,
    max_iterations=100,
    output_dir="figures/ukappa_inverse",
    idx=0,
    regularizer_type="denoiser",
    p_norm=2,
):
    """
    Solve the inverse problem for a given ukappa solution.

    Args:
        ukappa: The true solution field, shape (resolution, resolution)
        true_kappa: The true conductivity field (for comparison), shape (resolution, resolution)
        resolution: Mesh resolution
        noise_level: Level of noise to add to the measurements
        lambda_reg: Regularization strength
        denoiser_path: Path to the denoiser model
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        plot_results: Whether to generate plots
        max_iterations: Maximum number of outer iterations
        output_dir: Directory to save results
        regularizer_type: Type of regularizer to use (denoiser, value, gradient, tv)
        p_norm: p value for Lp norm when using value or gradient regularizer

    Returns:
        dict: Results including errors and recovered fields
    """
    # Create FEM problem
    fem_problem = FEMProblem(num_pixels=resolution, rtol=1e-10)

    # Get node coordinates
    x_coords = fem_problem.mesh.coordinates[0, :]
    y_coords = fem_problem.mesh.coordinates[1, :]

    # Define zero Dirichlet boundary conditions
    u_d = np.zeros(fem_problem.num_nodes)

    # Define forcing function f(x,y) = sin(πx)sin(πy)
    f = np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords)

    # Add noise to create synthetic measurements
    if noise_level > 0:
        noise = noise_level * np.linalg.norm(ukappa) * np.random.randn(ukappa.size)
        u_meas = ukappa + noise
        # Ensure boundary conditions remain exact
        u_meas[fem_problem.dirichlet_nodes] = u_d[fem_problem.dirichlet_nodes]
    else:
        u_meas = ukappa.copy()

    # Set the measured data as the target for inversion
    fem_problem.set_parameters(f=f, u_d=u_d, uhat=u_meas)

    # Create the appropriate regularizer based on type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if regularizer_type == "denoiser":
        # Load the denoiser model
        denoiser = KappaDenoiser()
        denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
        denoiser.to(device)
        denoiser.eval()

        # Create the denoiser regularizer
        regularizer = DenoiserRegularizer(
            denoiser=denoiser,
            lambda_reg=initial_lambda_reg,
            device=device,
            norm_min=norm_min,
            norm_max=norm_max,
        )
        print(f"Using denoiser regularization with λ={initial_lambda_reg}")
    elif regularizer_type == "value":
        regularizer = ValueRegularizer(lambda_reg=initial_lambda_reg, p=p_norm)
        print(f"Using value regularization with λ={initial_lambda_reg}, p={p_norm}")
    elif regularizer_type == "gradient":
        # Need dx and dy for gradient regularizer
        dx = fem_problem.dx
        dy = fem_problem.dy
        regularizer = GradientRegularizer(
            lambda_reg=initial_lambda_reg, dx=dx, dy=dy, p=p_norm
        )
        print(f"Using gradient regularization with λ={initial_lambda_reg}, p={p_norm}")
    elif regularizer_type == "tv":
        # Need dx and dy for total variation regularizer
        dx = fem_problem.dx
        dy = fem_problem.dy
        regularizer = TotalVariationRegularizer(
            lambda_reg=initial_lambda_reg, dx=dx, dy=dy
        )
        print(f"Using total variation regularization with λ={initial_lambda_reg}")
    else:
        raise ValueError(f"Unknown regularizer type: {regularizer_type}")

    # reshape 2D fields to 1D arrays
    true_kappa = true_kappa.reshape(-1)
    ukappa = ukappa.reshape(-1)

    # Create an initial guess for kappa using random normal noise
    kappa0 = np.random.randn(true_kappa.size)
    kappa0 = np.maximum(kappa0, 0.1)  # ensure positivity

    # Set up iteration tracking
    iterations = []
    objectives = []
    kappa_errors = []

    def callback(xk):
        """Callback function for the optimizer to track progress"""
        iter_num = len(iterations)
        iterations.append(iter_num)

        kappa_current = xk
        kappa_misfit = 0.5 * np.sum((kappa_current - intermediate_kappa) ** 2)
        solution_misfit = fem_problem.objective(kappa_current)

        objective = kappa_misfit + step_size * solution_misfit

        error = np.linalg.norm(kappa_current - true_kappa) / np.linalg.norm(true_kappa)

        print(
            f"Iteration {iter_num}: objective = {objective:.6e}, "
            f"relative error in kappa = {error:.6e}"
        )
        objectives.append(objective)
        kappa_errors.append(error)

    # Optimization parameters
    options = {
        "maxiter": 5,
        "disp": True,
        "return_all": True,
        "c1": 0.001,
        "c2": 0.9,
    }

    # Solve the inverse problem using the approach from mms.py
    print("\nSolving inverse problem...")
    start_time = time.perf_counter()

    # Outer loop (alternating minimization)
    k_max = max_iterations
    intermediate_kappa = kappa0.copy()
    prev_kappa = intermediate_kappa.copy()
    min_step_size = 1e-4
    step_size = initial_step_size
    relative_improvement_threshold = 1e-2  # Threshold for slow relative improvement
    # lambda_reg = initial_lambda_reg
    conv_tol = 1e-6

    # we will half lambda_reg every 10 iterations
    lambda_reg = initial_lambda_reg
    # Step size adaptation parameters
    previous_objective = float("inf")
    objective_history = []
    patience = 5
    consecutive_increases = 0

    for k in range(k_max):
        # Apply regularization gradient step (without momentum)
        reg_gradient = regularizer.gradient(intermediate_kappa)
        intermediate_kappa = intermediate_kappa - step_size * reg_gradient
        # ensure positivity
        intermediate_kappa = np.maximum(intermediate_kappa, 1e-6)

        # Create and solve the proximal problem
        proximal_operator = ProximalOperator(intermediate_kappa, fem_problem, step_size)
        result = minimize(
            proximal_operator.proximal_objective,
            intermediate_kappa,
            method="BFGS",
            jac=proximal_operator.proximal_gradient,
            callback=callback,
            options=options,
            bounds=[(1e-6, None) for _ in range(intermediate_kappa.size)],
        )

        # Update kappa
        prev_kappa = intermediate_kappa.copy()
        intermediate_kappa = result.x

        # Get current objective value
        current_objective = objectives[-1] if objectives else float("inf")
        objective_history.append(current_objective)

        # Annealing for lambda_reg, minimum lambda_reg is 0.01
        if (k + 1) % 10 == 0:
            lambda_reg = max(lambda_reg * 0.5, 0.1)
            regularizer.update_lambda(lambda_reg)
            print(f"Reducing lambda to {lambda_reg:.6e}")

        # Refined adaptive step size scheduler

        if previous_objective != float("inf"):
            relative_improvement = (
                previous_objective - current_objective
            ) / previous_objective
        else:
            relative_improvement = 0.0

        if current_objective >= previous_objective:
            consecutive_increases += 1
            if consecutive_increases >= patience:
                step_size = max(step_size * 0.5, min_step_size)
                print(
                    f"Reducing step size to {step_size:.6e} after {consecutive_increases} consecutive increases"
                )
                consecutive_increases = 0
        else:
            consecutive_increases = 0
            if k > 10:
                # If the relative improvement is very small, increase step size more aggressively.
                if relative_improvement < relative_improvement_threshold:
                    step_size = min(step_size * 1.5, initial_step_size)
                    print(
                        f"Slow improvement detected (relative improvement = {relative_improvement:.2e}): Increasing step size aggressively to {step_size:.6e}"
                    )
                # Otherwise, if improvement is decent, increase by a moderate factor.
                elif (
                    previous_objective - current_objective
                ) > 0.1 * previous_objective:
                    step_size = min(step_size * 1.2, initial_step_size)
                    print(f"Increasing step size to {step_size:.6e}")

        previous_objective = current_objective
        print(f"Outer iteration {k + 1}, step_size = {step_size:.6e}")

        # Add this function to create proper visualizations for wandb
        def create_field_visualization(field_data, title, cmap="viridis"):
            """Create a matplotlib figure with colorbar for wandb logging"""
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(field_data, cmap=cmap)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            return fig

        # log recovered kappa and solution u every 10 iterations
        if (k + 1) % 10 == 0:
            kappa_recovered = intermediate_kappa.reshape(resolution, resolution)
            u_recovered = fem_problem.forward(kappa_recovered)

            true_kappa_reshaped = true_kappa.reshape(resolution, resolution)
            ukappa_reshaped = ukappa.reshape(resolution + 1, resolution + 1)
            u_recovered_reshaped = u_recovered.reshape(resolution + 1, resolution + 1)

            # Create proper visualizations with matplotlib
            kappa_recovered_fig = create_field_visualization(
                kappa_recovered, "Recovered Kappa Field"
            )
            u_recovered_fig = create_field_visualization(
                u_recovered_reshaped, "Recovered Solution Field"
            )
            true_kappa_fig = create_field_visualization(
                true_kappa_reshaped, "True Kappa Field"
            )
            ukappa_fig = create_field_visualization(
                ukappa_reshaped, "True Solution Field"
            )

            # Log the figures to wandb
            wandb.log(
                {
                    f"kappa_recovered_{idx}": wandb.Image(kappa_recovered_fig),
                    f"u_recovered_{idx}": wandb.Image(u_recovered_fig),
                    f"true_kappa_{idx}": wandb.Image(true_kappa_fig),
                    f"ukappa_{idx}": wandb.Image(ukappa_fig),
                }
            )

            # Close the figures to avoid memory leaks
            plt.close(kappa_recovered_fig)
            plt.close(u_recovered_fig)
            plt.close(true_kappa_fig)
            plt.close(ukappa_fig)

        if (
            np.linalg.norm(intermediate_kappa - prev_kappa) / np.linalg.norm(prev_kappa)
            < conv_tol
        ):
            print(f"Convergence achieved at iteration {k}")
            break

    # Get the recovered conductivity
    kappa_recovered = intermediate_kappa

    end_time = time.perf_counter()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")

    # Compute final error
    rel_error = np.linalg.norm(kappa_recovered - true_kappa) / np.linalg.norm(
        true_kappa
    )
    print(f"Final relative L2 error in kappa: {rel_error:.6e}")

    # Compute forward solution with recovered kappa
    u_recovered = fem_problem.forward(kappa_recovered)
    u_error_rel = np.linalg.norm(u_recovered - ukappa) / np.linalg.norm(ukappa)
    u_error_l2 = np.linalg.norm(u_recovered - ukappa)
    u_error_l2_meas = np.linalg.norm(u_recovered - u_meas)
    print(f"Relative L2 error in solution u: {u_error_rel:.6e}")
    print(f"L2 error in solution u: {u_error_l2:.6e}")
    print(f"L2 error in solution u w/ noise: {u_error_l2_meas:.6e}")

    # Plot results if requested
    if plot_results:
        os.makedirs(output_dir, exist_ok=True)

        # Get element coordinates for plotting
        elem_coords = fem_problem.mesh.get_element_coordinates(
            np.arange(fem_problem.num_elements)
        )
        elem_coords = np.array(elem_coords)
        x_elem = np.mean(elem_coords[0], axis=0)
        y_elem = np.mean(elem_coords[1], axis=0)

        # Plot kappa fields
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # True kappa
        im1 = axes[0].tricontourf(x_elem, y_elem, true_kappa, 20, cmap="viridis")
        axes[0].set_title("True Kappa Field")
        fig.colorbar(im1, ax=axes[0])

        # Initial guess
        im2 = axes[1].tricontourf(x_elem, y_elem, kappa0, 20, cmap="viridis")
        axes[1].set_title("Initial Guess")
        fig.colorbar(im2, ax=axes[1])

        # Recovered kappa
        im3 = axes[2].tricontourf(x_elem, y_elem, kappa_recovered, 20, cmap="viridis")
        axes[2].set_title(f"Recovered Kappa (Error: {rel_error:.2%})")
        fig.colorbar(im3, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/kappa_comparison.png")

        # Plot solutions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original (clean) solution
        title = "True Solution"
        im1 = fem_problem.mesh.plot(ukappa, title=title)
        axes[0].set_title(title)
        fig.colorbar(im1, ax=axes[0])

        # Noisy measurements
        title = f"Measurements (Noise: {noise_level:.2%})"
        im2 = fem_problem.mesh.plot(u_meas, title=title)
        axes[1].set_title(title)
        fig.colorbar(im2, ax=axes[1])

        # Recovered solution
        title = f"Recovered Solution (Error: {u_error_rel:.2%})"
        im3 = fem_problem.mesh.plot(u_recovered, title=title)
        axes[2].set_title(title)
        fig.colorbar(im3, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/solution_comparison.png")

        # Plot convergence
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].semilogy(iterations, objectives, "o-")
        axes[0].grid(True)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Objective Value")
        axes[0].set_title("Convergence of Objective Function")

        axes[1].semilogy(iterations, kappa_errors, "o-")
        axes[1].grid(True)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Relative Error in κ")
        axes[1].set_title("Error in Recovered Conductivity")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/convergence.png")

    # Return results
    return {
        "kappa_true": true_kappa,
        "kappa_recovered": kappa_recovered,
        "kappa_error": rel_error,
        "ukappa_true": ukappa,
        "ukappa_measured": u_meas,
        "ukappa_recovered": u_recovered,
        "ukappa_error": u_error_rel,
        "iterations": len(iterations),
        "objectives": objectives,
        "kappa_errors": kappa_errors,
    }


def main():
    """Main function to parse arguments and run the inverse problem solver"""
    parser = argparse.ArgumentParser(
        description="Solve inverse problem for ukappa dataset"
    )
    parser.add_argument(
        "--noise", type=float, default=0.01, help="Noise level as fraction of signal"
    )
    parser.add_argument("--resolution", type=int, default=64, help="Mesh resolution")
    parser.add_argument(
        "--initial_lambda_reg", type=float, default=0.1, help="Regularization strength"
    )
    parser.add_argument(
        "--denoiser_path",
        type=str,
        default="/home/jy384/projects/DIPDE/models/kappa_denoiser_stoch.pt",
        help="Path to denoiser model",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/ukappa_inverse",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset type to use",
    )
    parser.add_argument(
        "--initial_step_size",
        type=float,
        default=1,
        help="Initial step size for the outer loop",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Maximum number of outer iterations",
    )
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--regularizer",
        type=str,
        default="denoiser",
        choices=["denoiser", "value", "gradient", "tv"],
        help="Type of regularizer to use",
    )
    parser.add_argument(
        "--p_norm",
        type=float,
        default=2.0,
        help="p value for Lp norm in value/gradient regularizers",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Initialize wandb
    wandb_run = wandb.init(
        project="DIPDE",
        entity="ECE689AdvDL",
        config=vars(args),
        job_type="inverse_problem",
    )

    # Download ukappa dataset from wandb
    print(
        f"Downloading ukappa_{args.resolution}x{args.resolution}-{args.dataset_type} dataset..."
    )
    ukappa_artifact = wandb.use_artifact(
        f"ECE689AdvDL/DIPDE/ukappa_dataset_{args.resolution}x{args.resolution}-{args.dataset_type}:latest"
    )
    ukappa_dir = Path(ukappa_artifact.download())
    ukappa_path = list(ukappa_dir.glob("*.npy"))[0]
    ukappa_dataset = np.load(ukappa_path)

    # Download kappa fields for ground truth
    print(f"Downloading test kappa field dataset...")
    kappa_artifact_test = wandb.use_artifact(
        f"ECE689AdvDL/DIPDE/kappa_field_pair-32x32N-255x255E-test:latest"
    )
    kappa_dir_test = Path(kappa_artifact_test.download())
    kappa_path_test = kappa_dir_test / "kappa_fine_samples.npy"

    # Download kappa fields used for training to get normalization statistics
    print(f"Downloading train kappa field dataset...")
    kappa_artifact_train = wandb.use_artifact(
        f"ECE689AdvDL/DIPDE/kappa_field_pair-32x32N-255x255E-train:latest"
    )
    kappa_dir_train = Path(kappa_artifact_train.download())
    kappa_path_train = kappa_dir_train / "kappa_fine_samples.npy"

    # Load and preprocess kappa dataset for the correct resolution
    dataset_train = NoisyKappaFieldDataset(
        data_path=kappa_path_train,
        reshape_size=(args.resolution, args.resolution),
    )

    # Load and preprocess kappa dataset for the correct resolution
    dataset_test = NoisyKappaFieldDataset(
        data_path=kappa_path_test,
        reshape_size=(args.resolution, args.resolution),
    )

    kappa_fields_train = dataset_train.clean_fields.numpy()
    kappa_fields_test = dataset_test.clean_fields.numpy()

    # Get normalization statistics for the denoiser
    norm_min = kappa_fields_train.min().item()
    norm_max = kappa_fields_train.max().item()
    print(f"Normalization range: [{norm_min}, {norm_max}]")

    # Ensure we dont request more samples than available
    assert args.num_samples <= len(ukappa_dataset), (
        f"Requested {args.num_samples} samples, but only {len(ukappa_dataset)} "
        f"available"
    )

    # Limit to number of samples requested
    num_samples = min(args.num_samples, len(ukappa_dataset))
    indices = np.random.choice(len(ukappa_dataset), num_samples, replace=False)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Track overall results
    all_results = []

    # Process each sample
    for i, idx in enumerate(tqdm(indices, desc="Processing samples")):
        print(
            f"\n{'=' * 80}\nProcessing sample {i + 1}/{num_samples} (index {idx})\n{'=' * 80}"
        )

        # Get ukappa and corresponding true kappa
        ukappa = ukappa_dataset[idx]
        true_kappa = kappa_fields_test[idx]

        # save figures of the true kappa and ukappa
        true_kappa_reshaped = true_kappa.reshape(args.resolution, args.resolution)
        ukappa_reshaped = ukappa.reshape(args.resolution + 1, args.resolution + 1)
        plt.imsave(
            os.path.join(args.output_dir, f"true_kappa_{idx}.png"), true_kappa_reshaped
        )
        plt.imsave(os.path.join(args.output_dir, f"ukappa_{idx}.png"), ukappa_reshaped)

        # Sample-specific output directory
        sample_dir = os.path.join(args.output_dir, f"sample_{idx}")
        os.makedirs(sample_dir, exist_ok=True)

        # Solve the inverse problem
        results = solve_inverse_problem_for_ukappa(
            ukappa=ukappa,
            true_kappa=true_kappa,
            resolution=args.resolution,
            noise_level=args.noise,
            initial_lambda_reg=args.initial_lambda_reg,
            initial_step_size=args.initial_step_size,
            denoiser_path=args.denoiser_path,
            norm_min=norm_min,
            norm_max=norm_max,
            plot_results=not args.no_plot,
            max_iterations=args.max_iterations,
            output_dir=sample_dir,
            idx=idx,
            regularizer_type=args.regularizer,
            p_norm=args.p_norm,
        )

        # Save results
        np.savez(
            os.path.join(sample_dir, "results.npz"),
            kappa_true=results["kappa_true"],
            kappa_recovered=results["kappa_recovered"],
            ukappa_true=results["ukappa_true"],
            ukappa_measured=results["ukappa_measured"],
            ukappa_recovered=results["ukappa_recovered"],
        )

        # Log to wandb
        wandb.log(
            {
                f"sample_{idx}/kappa_error": results["kappa_error"],
                f"sample_{idx}/ukappa_error": results["ukappa_error"],
                f"sample_{idx}/iterations": results["iterations"],
            }
        )

        # Store summary results
        all_results.append(
            {
                "index": idx,
                "kappa_error": results["kappa_error"],
                "ukappa_error": results["ukappa_error"],
                "iterations": results["iterations"],
            }
        )

    # Compute and log summary statistics
    kappa_errors = [r["kappa_error"] for r in all_results]
    ukappa_errors = [r["ukappa_error"] for r in all_results]
    iterations = [r["iterations"] for r in all_results]

    summary = {
        "mean_kappa_error": np.mean(kappa_errors),
        "std_kappa_error": np.std(kappa_errors),
        "mean_ukappa_error": np.mean(ukappa_errors),
        "std_ukappa_error": np.std(ukappa_errors),
        "mean_iterations": np.mean(iterations),
        "noise_level": args.noise,
        "initial_lambda_reg": args.initial_lambda_reg,
        "initial_step_size": args.initial_step_size,
        "num_samples": num_samples,
        "regularizer_type": args.regularizer,
        "p_norm": args.p_norm if args.regularizer in ["value", "gradient"] else None,
    }

    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    wandb.log(summary)

    # Save summary to file
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("Summary Statistics:\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"\nInverse problem solution completed for {num_samples} samples.")
    wandb.finish()


if __name__ == "__main__":
    main()
