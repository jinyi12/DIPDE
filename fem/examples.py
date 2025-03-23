"""
Example problems for the 2D FEM solver.
"""

import numpy as np
from typing import Callable, Tuple

from .mesh import Mesh
from .fem_solver import FEMSolver
from .utils import plot_error_comparison, visualize_solution_3d


def poisson_example():
    """
    Example 1: Standard Poisson equation with sin product solution.

    Problem: -Δu = 2π²sin(πx)sin(πy) on [0,1]×[0,1]
    Boundary conditions: u = 0 on boundary
    Exact solution: u(x,y) = sin(πx)sin(πy)
    """

    # Define the problem
    def forcing_function(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_gradient(x, y):
        return np.array(
            [
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
            ]
        )

    # Create mesh and solve
    mesh = Mesh(0.0, 1.0, 0.0, 1.0, 16, 16)
    solver = FEMSolver(mesh)
    solution = solver.solve(forcing_function)

    # Compute error
    l2_error, h1_error = solver.evaluate_error(exact_solution, exact_gradient)
    print(f"Poisson example - L2 Error: {l2_error:.6e}, H1 Error: {h1_error:.6e}")

    # Visualize
    plot_error_comparison(mesh, solution, exact_solution)

    return mesh, solution


def anisotropic_diffusion_example():
    """
    Example 2: Anisotropic diffusion.

    Problem: -∇·(D∇u) = f on [0,1]×[0,1]
    where D = [[10, 0], [0, 1]] (anisotropic diffusion tensor)
    Boundary conditions: u = 0 on boundary
    Manufactured solution: u(x,y) = sin(πx)sin(πy)
    """
    # Define anisotropic diffusion tensor
    diffusion_tensor = np.array([[10.0, 0.0], [0.0, 1.0]])

    def forcing_function(x, y):
        # Adjusted for anisotropic diffusion
        return 10 * np.pi**2 * np.sin(np.pi * x) * np.sin(
            np.pi * y
        ) + 1 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_gradient(x, y):
        return np.array(
            [
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
            ]
        )

    # Create mesh and solve
    mesh = Mesh(0.0, 1.0, 0.0, 1.0, 16, 16)
    solver = FEMSolver(mesh)
    solution = solver.solve(forcing_function, diffusion_tensor)

    # Compute error
    l2_error, h1_error = solver.evaluate_error(exact_solution, exact_gradient)
    print(f"Anisotropic diffusion - L2 Error: {l2_error:.6e}, H1 Error: {h1_error:.6e}")

    # Visualize
    plot_error_comparison(mesh, solution, exact_solution)

    return mesh, solution


def gaussian_source_example():
    """
    Example 3: Poisson equation with Gaussian source.

    Problem: -Δu = f on [0,1]×[0,1]
    where f is a Gaussian source centered at (0.5, 0.5)
    Boundary conditions: u = 0 on boundary
    No exact solution available
    """

    # Define the Gaussian source
    def gaussian_source(x, y, sigma=0.1):
        return np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * sigma**2))

    def forcing_function(x, y):
        return gaussian_source(x, y)

    # Create mesh and solve
    mesh = Mesh(0.0, 1.0, 0.0, 1.0, 32, 32)
    solver = FEMSolver(mesh)
    solution = solver.solve(forcing_function)

    # Visualize
    mesh.plot(solution, title="Solution with Gaussian Source")
    visualize_solution_3d(mesh, solution, title="Solution with Gaussian Source")

    return mesh, solution


def rectangular_domain_example():
    """
    Example 4: Solving on a rectangular domain with different aspect ratio.

    Problem: -Δu = 1 on [0,2]×[0,1]
    Boundary conditions: u = 0 on boundary
    """

    def forcing_function(x, y):
        return 1.0  # Constant source term

    # Create mesh with 2:1 aspect ratio
    mesh = Mesh(0.0, 2.0, 0.0, 1.0, 40, 20)
    solver = FEMSolver(mesh)
    solution = solver.solve(forcing_function)

    # Visualize
    mesh.plot(solution, title="Solution on Rectangular Domain")
    visualize_solution_3d(mesh, solution, title="Solution on Rectangular Domain")

    return mesh, solution


def run_all_examples():
    """Run all example problems."""
    print("=== Example 1: Standard Poisson Equation ===")
    poisson_example()

    print("\n=== Example 2: Anisotropic Diffusion ===")
    anisotropic_diffusion_example()

    print("\n=== Example 3: Gaussian Source ===")
    gaussian_source_example()

    print("\n=== Example 4: Rectangular Domain ===")
    rectangular_domain_example()


if __name__ == "__main__":
    run_all_examples()
