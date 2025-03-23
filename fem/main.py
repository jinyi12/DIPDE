"""
Main driver script for FEM solver demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple

from .mesh import Mesh
from .fem_solver import FEMSolver
from .utils import (
    convergence_study,
    plot_error_comparison,
    visualize_solution_3d,
    linear_solve_stats,
)


def main():
    """
    Example: Solve the Poisson equation -Δu = f on a rectangular domain with homogeneous Dirichlet BCs.

    The exact solution is u(x,y) = sin(πx)sin(πy), which gives f(x,y) = 2π²sin(πx)sin(πy).

    This example demonstrates:
    1. Creating a mesh
    2. Setting up and solving the FEM problem
    3. Computing errors
    4. Visualizing results
    5. Performing a convergence study
    """

    # Define the problem
    def forcing_function(x, y):
        """Source term f(x,y) = 2π²sin(πx)sin(πy)"""
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_solution(x, y):
        """Exact solution u(x,y) = sin(πx)sin(πy)"""
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_gradient(x, y):
        """Exact gradient of the solution [∂u/∂x, ∂u/∂y]"""
        return np.array(
            [
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
            ]
        )

    # Domain
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    print("=== 2D Finite Element Method Demo ===")
    print("\nSolving: -Δu = 2π²sin(πx)sin(πy) on [0,1]×[0,1] with u=0 on boundary")
    print("Exact solution: u(x,y) = sin(πx)sin(πy)")

    # Create a mesh with 16x16 elements
    print("\nCreating mesh with 16x16 elements...")
    mesh = Mesh(x_min, x_max, y_min, y_max, 16, 16)

    # Create a solver with quadrature order 2
    print("Setting up solver with quadrature order 2...")
    solver = FEMSolver(mesh, quadrature_order=2)

    # Assemble the system
    print("Assembling system...")
    stiffness_matrix, load_vector = solver.assemble(forcing_function)

    # Print statistics about the linear system
    print("\nLinear system statistics:")
    stats = linear_solve_stats(stiffness_matrix, load_vector)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Solve the problem
    print("\nSolving linear system...")
    solution = solver.solve(forcing_function)
    print(
        f"Solution computed with {mesh.num_nodes} nodes and {mesh.num_equations} unknowns"
    )

    # Compute errors
    print("\nComputing errors...")
    l2_error, h1_error = solver.evaluate_error(exact_solution, exact_gradient)
    print(f"L2 Error: {l2_error:.6e}")
    print(f"H1 Error: {h1_error:.6e}")

    # Plot the solution and error
    print("\nPlotting solution and error...")
    plot_error_comparison(mesh, solution, exact_solution)

    # 3D visualization
    print("Creating 3D visualization...")
    visualize_solution_3d(mesh, solution, title="FEM Solution")

    # Perform convergence study
    print("\nPerforming convergence study...")
    results = convergence_study(
        x_min,
        x_max,
        y_min,
        y_max,
        element_counts=[4, 8, 16, 32],
        force_function=forcing_function,
        exact_solution=exact_solution,
        exact_gradient=exact_gradient,
        quadrature_order=2,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
