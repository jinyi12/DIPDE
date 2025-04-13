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
import json
from typing import List, Dict, Union, Optional, Tuple
from scipy import stats

from fem.inverse_problem import FEMProblem
from models.denoiser import KappaDenoiser
from fem.regularizers import (
    Regularizer,
    DenoiserRegularizer,
    ValueRegularizer,
    GradientRegularizer,
    TotalVariationRegularizer,
    CompositeRegularizer,
)
from dataprep.noisy_kappa_dataset import NoisyKappaFieldDataset
from dataprep.genrf_vertices_coarse import generate_random_fields


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
        solution_misfit = 100* self.fem_problem.objective(kappa)
        kappa_misfit = 0.5 * np.sum((kappa - self.kappa_intermediate) ** 2)
        return kappa_misfit + self.step_size * solution_misfit

    def proximal_gradient(self, kappa):
        """Calculate the gradient of the proximal objective function"""
        misfit_grad = self.fem_problem.gradient(kappa)
        kappa_misfit_grad = kappa - self.kappa_intermediate
        return kappa_misfit_grad + self.step_size * misfit_grad


def solve_inverse_problem_for_ukappa(
    ukappa,
    ukappa_normalized,
    true_kappa,
    resolution=64,
    noise_level=0.01,
    initial_lambda_reg=1,
    initial_step_size=1,
    denoiser_path=None,
    norm_min=None,
    norm_max=None,
    plot_results=True,
    max_iterations=100,
    output_dir="figures/ukappa_inverse",
    idx=0,
    regularizer_type="denoiser",
    regularizer_weights=None,
    p_norm=2,
    epsilon=1e-6,
    lambda_min_factor=0.1,  # Start lambda_reg at this fraction of initial_lambda_reg
    lambda_schedule_iterations=50,  # Iterations over which lambda increases
):
    """
    Solve the inverse problem for a given ukappa solution.

    Args:
        ukappa: The true solution field, shape (resolution, resolution)
        true_kappa: The true conductivity field (for comparison), shape (resolution, resolution)
        resolution: Mesh resolution
        noise_level: Level of noise to add to the measurements
        initial_lambda_reg: Regularization strength
        initial_step_size: Initial step size for optimization
        denoiser_path: Path to the denoiser model
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        plot_results: Whether to generate plots
        max_iterations: Maximum number of outer iterations
        output_dir: Directory to save results
        idx: Index of the current sample
        regularizer_type: Type of regularizer(s) to use - can be a single type or a comma-separated list
        regularizer_weights: Optional weights for composite regularizer, comma-separated floats
        p_norm: p value for Lp norm when using value or gradient regularizer
        epsilon: Smoothing parameter for TV regularizer
        lambda_min_factor: Initial lambda_reg as a fraction of initial_lambda_reg
        lambda_schedule_iterations: Number of iterations over which lambda increases

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
        u_meas_normalized = ukappa_normalized + noise
        # Ensure boundary conditions remain exact
        u_meas[fem_problem.dirichlet_nodes] = u_d[fem_problem.dirichlet_nodes]
        u_meas_normalized[fem_problem.dirichlet_nodes] = u_d[fem_problem.dirichlet_nodes]
    else:
        u_meas = ukappa.copy()
        u_meas_normalized = ukappa_normalized.copy()
    # Set the measured data as the target for inversion
    fem_problem.set_parameters(f=f, u_d=u_d, uhat=u_meas)

    # Calculate initial lambda_reg based on minimum factor
    min_lambda_reg = initial_lambda_reg * lambda_min_factor

    # Create the appropriate regularizer based on type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Process regularizer types and weights
    regularizer_types = regularizer_type.split(',')
    regularizer_types = [r.strip() for r in regularizer_types]
    
    # Process weights if provided
    if regularizer_weights:
        weights = [float(w.strip()) for w in regularizer_weights.split(',')]
        if len(weights) != len(regularizer_types):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of regularizers ({len(regularizer_types)})")
    else:
        # Equal weights if not specified
        weights = [1.0] * len(regularizer_types)
    
    # Initialize list to hold individual regularizers
    regularizers = []
    
    # Create each individual regularizer
    for reg_type in regularizer_types:
        if reg_type == "denoiser":
            # Load the denoiser model
            denoiser = KappaDenoiser()
            denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
            denoiser.to(device)
            denoiser.eval()

            # Create the denoiser regularizer
            reg = DenoiserRegularizer(
                denoiser=denoiser,
                lambda_reg=initial_lambda_reg,  # Will be scaled by CompositeRegularizer
                device=device,
                norm_min=norm_min,
                norm_max=norm_max,
            )
            print(f"Added denoiser regularization")
            
        elif reg_type == "value":
            reg = ValueRegularizer(lambda_reg=initial_lambda_reg, p=p_norm)
            print(f"Added value regularization with p={p_norm}")
            
        elif reg_type == "gradient":
            # Need dx and dy for gradient regularizer
            dx = fem_problem.dx
            dy = fem_problem.dy
            reg = GradientRegularizer(
                lambda_reg=initial_lambda_reg, dx=dx, dy=dy, p=p_norm
            )
            print(f"Added gradient regularization with p={p_norm}")
            
        elif reg_type == "tv":
            # Need dx and dy for total variation regularizer
            dx = fem_problem.dx
            dy = fem_problem.dy
            reg = TotalVariationRegularizer(
                lambda_reg=initial_lambda_reg, dx=dx, dy=dy, epsilon=epsilon
            )
            print(f"Added total variation regularization")
            
        else:
            raise ValueError(f"Unknown regularizer type: {reg_type}")
            
        regularizers.append(reg)
    
    # Create the composite regularizer if multiple types are specified
    if len(regularizers) > 1:
        regularizer = CompositeRegularizer(
            regularizers=regularizers,
            weights=weights,
            global_lambda_reg=initial_lambda_reg
        )
        print(f"Created composite regularizer with weights {weights} and initial λ={initial_lambda_reg}")
    else:
        # Single regularizer - apply initial_lambda_reg directly
        regularizer = regularizers[0]
        regularizer.update_lambda(initial_lambda_reg)
        print(f"Using single {regularizer_types[0]} regularization with λ={initial_lambda_reg}")

    # reshape 2D fields to 1D arrays
    true_kappa = true_kappa.reshape(-1)
    ukappa = ukappa.reshape(-1)
    

    # Create an initial guess for kappa using KL expansion with FEM mesh
    kappa0 = generate_kappa_initialization(
        fem_problem=fem_problem,
        mu_target=2.0,  # some guess for the dataset mean
        sigma_target=0.2,  # some guess for the dataset std
        correlation_length=None,  # Auto-calculate based on mesh size
        error_threshold=1e-3
    )
    
    # # convert kappa0 to the range of the dataset
    # kappa0 = (kappa0 - kappa0.min()) / (kappa0.max() - kappa0.min()) * (norm_max - norm_min) + norm_min

    # kappa0 = np.random.randn(true_kappa.size)
    # kappa0 = np.maximum(kappa0, 0.1)

    # Set up iteration tracking
    iterations = []
    objectives = []
    kappa_errors = []
    reg_values = []  # Track regularization values

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
        "maxiter": 3,
        "disp": True,
        # "return_all": True,
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
    conv_tol = 1e-5

    # Additional convergence criteria parameters
    obj_conv_tol = 1e-4   # Tolerance for relative change in objective function
    grad_norm_tol = 1e-4  # Tolerance for gradient norm
    reg_conv_tol = 1e-4   # Tolerance for relative change in regularization value
    window_size = 10       # Window size for moving average calculations
    
    # Tracking variables for convergence criteria
    prev_objective = float('inf')
    prev_reg_value = float('inf')
    objective_rel_changes = []  # Track relative changes in objective
    gradient_norms = []        # Track gradient norms
    reg_rel_changes = []       # Track relative changes in regularization value

    # Step size adaptation parameters
    previous_objective = float("inf")
    objective_history = []  # Store last few objectives
    history_window = 3  # Number of iterations to consider for trends
    patience = 30  # Iterations of increase before decreasing step size
    consecutive_increases = 0
    decrease_factor = 0.5  # Factor to decrease step size
    increase_factor = 1.1  # Factor to increase step size
    max_step_size_multiplier = 1.0  # Max step size relative to initial
    max_step_size = initial_step_size * max_step_size_multiplier

    # Regularization tracking parameters
    best_reg_value = float("inf")  # Best regularization value seen so far
    best_kappa_by_reg = None  # Kappa that gave the best regularization value

    for k in range(k_max):
        # Update lambda_reg according to the schedule (linear decrease)
        if k < lambda_schedule_iterations:
            # Linear interpolation: from initial_lambda_reg down to min_lambda_reg
            progress = k / lambda_schedule_iterations  # 0 to 1
            current_lambda = initial_lambda_reg - progress * (
                initial_lambda_reg - min_lambda_reg
            )
        else:
            # After scheduled iterations, use the minimum lambda
            current_lambda = min_lambda_reg

        # Update regularizer with current lambda
        regularizer.update_lambda(current_lambda)

        # Apply regularization gradient step (without momentum)
        reg_gradient = regularizer.gradient(intermediate_kappa)
        intermediate_kappa = intermediate_kappa - step_size * reg_gradient

        # Calculate regularization value for tracking and early stopping
        # This calculation happens before the proximal step to evaluate the current kappa
        current_reg_value = regularizer(intermediate_kappa)
        reg_values.append(current_reg_value)

        # Calculate relative change in regularization value for convergence check
        reg_rel_change = abs(current_reg_value - prev_reg_value) / (abs(prev_reg_value) + 1e-10)
        reg_rel_changes.append(reg_rel_change)
        if len(reg_rel_changes) > window_size:
            reg_rel_changes.pop(0)
        prev_reg_value = current_reg_value

        # Create and solve the proximal problem
        proximal_operator = ProximalOperator(intermediate_kappa, fem_problem, step_size)
        result = minimize(
            proximal_operator.proximal_objective,
            intermediate_kappa,
            method="L-BFGS-B",
            jac=proximal_operator.proximal_gradient,
            callback=callback,
            options=options,
            bounds=[(1e-3, None) for _ in range(intermediate_kappa.size)],
        )

        # Update kappa
        prev_kappa = intermediate_kappa.copy()
        intermediate_kappa = result.x

        # Get current objective value (use the last one from the callback)
        current_objective = objectives[-1] if objectives else float("inf")

        # Calculate relative change in objective for convergence check
        obj_rel_change = abs(current_objective - prev_objective) / (abs(prev_objective) + 1e-10)
        objective_rel_changes.append(obj_rel_change)
        if len(objective_rel_changes) > window_size:
            objective_rel_changes.pop(0)
        prev_objective = current_objective

        # Calculate gradient norm for gradient-based convergence check
        current_gradient_norm = np.linalg.norm(proximal_operator.proximal_gradient(intermediate_kappa))
        gradient_norms.append(current_gradient_norm)
        if len(gradient_norms) > window_size:
            gradient_norms.pop(0)

        # Update progress reporting
        print(
            f"Outer iteration {k + 1}, λ = {current_lambda:.4e}, step_size = {step_size:.4e}, "
            f"objective = {current_objective:.4e}, reg_value = {current_reg_value:.4e}"
        )

        # function to create proper visualizations for wandb
        def create_field_visualization(field_data, title, cmap="viridis", vmin=None, vmax=None):
            """Create a matplotlib figure with colorbar for wandb logging"""
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(field_data, cmap=cmap, vmin=vmin, vmax=vmax)
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
            ukappa_normalized_reshaped = ukappa_normalized.reshape(resolution + 1, resolution + 1)
            
            # Calculate common color scales
            kappa_vmin = min(np.min(kappa_recovered), np.min(true_kappa_reshaped))
            kappa_vmax = max(np.max(kappa_recovered), np.max(true_kappa_reshaped))
            u_vmin = min(np.min(u_recovered_reshaped), np.min(ukappa_reshaped))
            u_vmax = max(np.max(u_recovered_reshaped), np.max(ukappa_reshaped))
            
            # Create proper visualizations with matplotlib
            kappa_recovered_fig = create_field_visualization(
                kappa_recovered, "Recovered Kappa Field", vmin=kappa_vmin, vmax=kappa_vmax
            )
            u_recovered_fig = create_field_visualization(
                u_recovered_reshaped, "Recovered Solution Field", vmin=u_vmin, vmax=u_vmax
            )
            true_kappa_fig = create_field_visualization(
                true_kappa_reshaped, "True Kappa Field", vmin=kappa_vmin, vmax=kappa_vmax
            )
            ukappa_fig = create_field_visualization(
                ukappa_reshaped, "True Solution Field", vmin=u_vmin, vmax=u_vmax
            )

            # Log the figures to wandb
            wandb.log(
                {
                    f"kappa_recovered_{idx}": wandb.Image(kappa_recovered_fig),
                    f"u_recovered_{idx}": wandb.Image(u_recovered_fig),
                    f"true_kappa_{idx}": wandb.Image(true_kappa_fig),
                    f"ukappa_{idx}": wandb.Image(ukappa_fig),
                },
                step=k
            )

            # Close the figures to avoid memory leaks
            plt.close(kappa_recovered_fig)
            plt.close(u_recovered_fig)
            plt.close(true_kappa_fig)
            plt.close(ukappa_fig)

        # Calculate average metrics over the window for smoother convergence detection
        avg_obj_rel_change = np.mean(objective_rel_changes) if objective_rel_changes else float('inf')
        avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else float('inf')
        avg_reg_rel_change = np.mean(reg_rel_changes) if reg_rel_changes else float('inf')
        
        # Original criterion: relative change in kappa
        kappa_rel_change = np.linalg.norm(intermediate_kappa - prev_kappa) / (np.linalg.norm(prev_kappa) + 1e-10)
        
        # Log all convergence metrics
        print(
            f"Convergence metrics: kappa_change = {kappa_rel_change:.4e}, "
            f"obj_change = {avg_obj_rel_change:.4e}, grad_norm = {avg_gradient_norm:.4e}, "
            f"reg_change = {avg_reg_rel_change:.4e}"
        )
        
        # Composite convergence check
        conv_criteria_met = (
            (kappa_rel_change < conv_tol) or  # Original criterion
            (avg_obj_rel_change < obj_conv_tol and k > window_size) or  # Objective stabilized
            (avg_gradient_norm < grad_norm_tol and k > window_size) or  # Gradient small
            (avg_reg_rel_change < reg_conv_tol and k > window_size * 2)  # Regularization stabilized
        )
        
        if conv_criteria_met:
            print(f"Convergence achieved at iteration {k}")
            if kappa_rel_change < conv_tol:
                print("  Converged based on change in kappa")
            if avg_obj_rel_change < obj_conv_tol and k > window_size:
                print("  Converged based on objective function stability")
            if avg_gradient_norm < grad_norm_tol and k > window_size:
                print("  Converged based on gradient norm")
            if avg_reg_rel_change < reg_conv_tol and k > window_size * 2:
                print("  Converged based on regularization value stability")
            break

    # Get the recovered conductivity (use the one with the best regularization value)
    if best_kappa_by_reg is not None:
        kappa_recovered = best_kappa_by_reg
        print("Using kappa with best regularization value for final result")
    else:
        kappa_recovered = intermediate_kappa
        print(
            "Using final kappa for result (no improvement in regularization was found)"
        )

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
        "reg_values": reg_values,
        "best_reg_value": best_reg_value,
    }


def generate_kappa_initialization(fem_problem, mu_target=2.7, sigma_target=0.3, correlation_length=None, error_threshold=1e-3):
    """
    Generate a non-Gaussian random field using KL expansion for kappa initialization.
    Uses the mesh coordinates from the FEM problem.
    
    Parameters:
    -----------
    fem_problem : FEMProblem
        The FEM problem with mesh coordinates
    mu_target : float
        Target mean for the conductivity field (default: 2.7)
    sigma_target : float
        Target standard deviation for the conductivity field (default: 0.3)
    correlation_length : float
        Correlation length for the random field. If None, calculated as 8*dx
    error_threshold : float
        Error threshold for truncating the KL expansion
        
    Returns:
    --------
    kappa : np.ndarray
        Random field initialization for kappa
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Get element coordinates for random field generation
    elem_coords = fem_problem.mesh.get_element_coordinates(np.arange(fem_problem.num_elements))
    elem_coords = np.array(elem_coords)
    
    # Calculate centroids of elements
    x_elem = np.mean(elem_coords[0], axis=0)
    y_elem = np.mean(elem_coords[1], axis=0)
    
    # Create coordinates array for distance calculation
    xy_coords = np.column_stack((x_elem, y_elem))
    
    # Set correlation length based on mesh size if not provided
    if correlation_length is None:
        correlation_length = 8 * fem_problem.dx
    
    print(f"Using correlation length of {correlation_length:.4f} for kappa initialization")
    
    # Calculate covariance matrix using exponential kernel
    distances = squareform(pdist(xy_coords, "euclidean"))
    cov_matrix = np.exp(-distances / correlation_length)
    
    # Compute KL decomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = np.abs(eig_vals[idx])  # Ensure positive eigenvalues
    eig_vecs = eig_vecs[:, idx]
    
    # Calculate error function and truncate
    error_func = 1 - (np.cumsum(eig_vals) / np.sum(eig_vals))
    n_truncate = np.argwhere(error_func <= error_threshold)[0][0] + 1
    
    print(f"Truncated KL expansion to {n_truncate} components")
    
    # Use only truncated eigenvalues and eigenvectors
    eig_vals = eig_vals[:n_truncate]
    eig_vecs = eig_vecs[:, :n_truncate]
    
    # Parameters for standard normal
    mu_gauss = 0
    sigma_gauss = 1
    
    # Parameters for target gamma distribution
    beta = 1 / np.power(sigma_target, 2)
    alpha = mu_target * np.power(sigma_target, 2)
    
    # Create sqrt of eigenvalue diagonal matrix
    sqrt_eig_vals = np.sqrt(eig_vals)
    
    # Generate standard normal random variables
    xi = np.random.normal(mu_gauss, sigma_gauss, n_truncate)
    
    # Compute KL expansion to obtain Gaussian field realization
    gaussian_field = mu_gauss + sigma_gauss * eig_vecs @ (sqrt_eig_vals * xi)
    
    # Transform to non-Gaussian field using gamma distribution
    # Apply the probability integral transform: Standard normal CDF -> gamma PPF
    z_normcdf = stats.norm.cdf(gaussian_field, mu_gauss, sigma_gauss)
    kappa = stats.gamma.ppf(z_normcdf, beta, scale=alpha)
    
    # Ensure strict positivity
    kappa = np.maximum(kappa, 1e-6)
    
    return kappa


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
        "--base_output_dir",
        type=str,
        default="results/ukappa_inverse",
        help="Base directory for saving run outputs",
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
        help="Type of regularizer(s) to use, comma-separated for multiple (denoiser,tv,value,gradient)",
    )
    parser.add_argument(
        "--regularizer_weights",
        type=str,
        default=None,
        help="Comma-separated weights for regularizers (must match number of regularizers)",
    )
    parser.add_argument(
        "--p_norm",
        type=float,
        default=2.0,
        help="p value for Lp norm in value/gradient regularizers",
    )

    parser.add_argument(
        "--lambda_min_factor",
        type=float,
        default=0.1,
        help="Initial lambda_reg as fraction of target value",
    )

    parser.add_argument(
        "--lambda_schedule_iterations",
        type=int,
        default=50,
        help="Number of iterations over which lambda increases",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Smoothing parameter for TV regularizer",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Initialize wandb FIRST to get the run ID
    wandb_run = wandb.init(
        project="DIPDE",
        entity="ECE689AdvDL",
        config=vars(args),
        job_type="inverse_problem",
    )

    # --- Create structured output directories ---
    run_output_dir = os.path.join(args.base_output_dir, wandb_run.id)
    data_dir = os.path.join(run_output_dir, "data")
    figures_dir = os.path.join(run_output_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Saving results to: {run_output_dir}")

    # Save configuration
    config_path = os.path.join(run_output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved configuration to: {config_path}")
    # -----------------------------------------

    # Download ukappa dataset from wandb
    print(
        f"Downloading ukappa_dataset_{args.resolution}x{args.resolution}-{args.dataset_type} dataset..."
    )
    ukappa_artifact = wandb.use_artifact(
        f"ECE689AdvDL/DIPDE/ukappa_dataset_{args.resolution}x{args.resolution}-{args.dataset_type}:latest"
    )
    ukappa_dir = Path(ukappa_artifact.download())
    ukappa_path = list(ukappa_dir.glob("*dataset.npy"))[0]
    ukappa_dataset = np.load(ukappa_path)
    
    print(f"Downloading ukappa_dataset_normalized_{args.resolution}x{args.resolution}-{args.dataset_type} dataset...")
    ukappa_artifact_normalized = wandb.use_artifact(
        f"ECE689AdvDL/DIPDE/ukappa_dataset_normalized_{args.resolution}x{args.resolution}-{args.dataset_type}:latest"
    )
    ukappa_dir_normalized = Path(ukappa_artifact_normalized.download())
    ukappa_path_normalized = list(ukappa_dir_normalized.glob("*dataset_normalized.npy"))[0]
    ukappa_dataset_normalized = np.load(ukappa_path_normalized)

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

    # Get clean kappa fields, note !Unnormalized!
    kappa_fields_train = dataset_train.clean_fields_original.numpy()
    kappa_fields_test = dataset_test.clean_fields_original.numpy()

    # Get normalization statistics for the denoiser
    norm_min = dataset_train.norm_min
    norm_max = dataset_train.norm_max
    print(f"Range of training kappa fields: [{kappa_fields_train.min()}, {kappa_fields_train.max()}]")

    # Ensure we dont request more samples than available
    assert args.num_samples <= len(ukappa_dataset), (
        f"Requested {args.num_samples} samples, but only {len(ukappa_dataset)} "
        f"available"
    )

    # Limit to number of samples requested
    num_samples = min(args.num_samples, len(ukappa_dataset))
    if args.dataset_type == "test":
        total_available = len(kappa_fields_test)
    elif args.dataset_type == "val":
        # Assuming val uses the same logic or adjust if needed
        total_available = len(
            ukappa_dataset
        )  # Placeholder if val kappa isn't loaded separately
    else:  # train
        total_available = len(kappa_fields_train)

    indices = np.random.choice(total_available, num_samples, replace=False)

    # Track overall results and detailed results per sample
    all_summary_results = []
    consolidated_npz_data = {}

    # Process each sample
    for i, idx in enumerate(tqdm(indices, desc="Processing samples")):
        print(
            f"\n{'=' * 80}\nProcessing sample {i + 1}/{num_samples} (index {idx})\n{'=' * 80}"
        )

        # Get ukappa and corresponding true kappa (adapt based on dataset type if needed)
        ukappa = ukappa_dataset[idx]
        ukappa_normalized = ukappa_dataset_normalized[idx]
        # Assuming test set kappa is the primary target for inversion comparison
        true_kappa = kappa_fields_test[idx]

        # Sample-specific figure directory within the main run's figures dir
        sample_figure_dir = os.path.join(figures_dir, f"sample_{idx}")
        # No need to create - solve_inverse_problem_for_ukappa will handle if plot_results is True

        # Solve the inverse problem
        # Pass the sample-specific figure directory
        results = solve_inverse_problem_for_ukappa(
            ukappa=ukappa,
            ukappa_normalized=ukappa_normalized,
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
            output_dir=sample_figure_dir,  # Use the dedicated figure dir
            idx=idx,
            regularizer_type=args.regularizer,
            regularizer_weights=args.regularizer_weights,
            p_norm=args.p_norm,
            lambda_min_factor=args.lambda_min_factor,
            lambda_schedule_iterations=args.lambda_schedule_iterations,
            epsilon=args.epsilon,
        )

        # Store results for the consolidated npz file
        # Use str(idx) as key because np.savez requires string keys
        consolidated_npz_data[str(idx)] = {
            "kappa_true": results["kappa_true"],
            "kappa_recovered": results["kappa_recovered"],
            "ukappa_true": results["ukappa_true"],
            "ukappa_measured": results["ukappa_measured"],
            "ukappa_recovered": results["ukappa_recovered"],
            # Add other numerical arrays if needed, e.g., objectives, errors per iteration
            "objectives": results["objectives"],
            "kappa_errors_iter": results["kappa_errors"],
            "reg_values": results["reg_values"],
        }

        # Log individual sample metrics to wandb
        wandb.log(
            {
                f"sample_{idx}/kappa_error_final": results["kappa_error"],
                f"sample_{idx}/ukappa_error_final": results["ukappa_error"],
                f"sample_{idx}/iterations": results["iterations"],
            }
        )

        # Store summary results
        all_summary_results.append(
            {
                "index": idx,
                "kappa_error": results["kappa_error"],
                "ukappa_error": results["ukappa_error"],
                "iterations": results["iterations"],
            }
        )

    # --- Save consolidated numerical data ---
    npz_path = os.path.join(data_dir, "results.npz")
    np.savez(npz_path, **consolidated_npz_data)
    print(f"Saved consolidated numerical results to: {npz_path}")
    
    # also add the npz file to wandb
    consolidated_results_artifact = wandb.Artifact(
        "consolidated_results",
        type="dataset",
        description="Consolidated results from inverse problem",
    )
    consolidated_results_artifact.add_file(npz_path)
    wandb.log_artifact(consolidated_results_artifact)
    # --------------------------------------

    # Compute and log summary statistics
    kappa_errors = [r["kappa_error"] for r in all_summary_results]
    ukappa_errors = [r["ukappa_error"] for r in all_summary_results]
    iterations = [r["iterations"] for r in all_summary_results]

    summary = {
        "mean_kappa_error": np.mean(kappa_errors),
        "std_kappa_error": np.std(kappa_errors),
        "min_kappa_error": np.min(kappa_errors),
        "max_kappa_error": np.max(kappa_errors),
        "mean_ukappa_error": np.mean(ukappa_errors),
        "std_ukappa_error": np.std(ukappa_errors),
        "min_ukappa_error": np.min(ukappa_errors),
        "max_ukappa_error": np.max(ukappa_errors),
        "mean_iterations": np.mean(iterations),
        "std_iterations": np.std(iterations),
        "noise_level": args.noise,
        "initial_lambda_reg": args.initial_lambda_reg,
        "initial_step_size": args.initial_step_size,
        "num_samples_processed": num_samples,
        "regularizer_type": args.regularizer,
        "regularizer_weights": args.regularizer_weights,
        "p_norm": args.p_norm if "value" in args.regularizer or "gradient" in args.regularizer else None,
        "max_iterations_allowed": args.max_iterations,
        "lambda_min_factor": args.lambda_min_factor,
        "lambda_schedule_iterations": args.lambda_schedule_iterations,
    }

    print("\nSummary Statistics:")
    for key, value in summary.items():
        # Format floats nicely for printing
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Log summary to wandb (as final summary)
    wandb.log(summary)

    # Save summary to file
    summary_path = os.path.join(run_output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Run Configuration:\n")
        json.dump(vars(args), f, indent=4)
        f.write("\n\nSummary Statistics:\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved summary statistics to: {summary_path}")

    print(f"\nInverse problem solution completed for {num_samples} samples.")
    wandb.finish()


if __name__ == "__main__":
    main()
