"""
Utility functions for FEM analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple, Dict, Optional
import numpy.typing as npt

from .mesh import Mesh
from .fem_solver import FEMSolver


def convergence_study(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    element_counts: List[int],
    force_function: Callable[[float, float], float],
    exact_solution: Callable[[float, float], float],
    exact_gradient: Callable[[float, float], npt.NDArray[np.float64]],
    quadrature_order: int = 2,
) -> Dict[str, List[float]]:
    """
    Perform a convergence study for the FEM solver.

    Args:
        x_min: Left boundary of the domain
        x_max: Right boundary of the domain
        y_min: Bottom boundary of the domain
        y_max: Top boundary of the domain
        element_counts: List of element counts in each direction to test
        force_function: Source term function f(x,y)
        exact_solution: Exact solution function u(x,y)
        exact_gradient: Exact gradient function [du/dx, du/dy](x,y)
        quadrature_order: Order of the quadrature rule

    Returns:
        Dictionary containing h_values, l2_errors, h1_errors, l2_rates, and h1_rates
    """
    h_values = []
    l2_errors = []
    h1_errors = []
    solutions = []

    for n in element_counts:
        # Create mesh with n elements in each direction
        mesh = Mesh(x_min, x_max, y_min, y_max, n, n)

        # Maximum element size
        h = max(mesh.dx, mesh.dy)
        h_values.append(h)

        # Solve the problem
        solver = FEMSolver(mesh, quadrature_order)
        solution = solver.solve(force_function)
        solutions.append(solution)

        # Compute errors
        l2_error, h1_error = solver.evaluate_error(exact_solution, exact_gradient)
        l2_errors.append(l2_error)
        h1_errors.append(h1_error)

        print(
            f"Elements: {n}x{n}, h = {h:.6f}, L2 error = {l2_error:.6e}, H1 error = {h1_error:.6e}"
        )

    # Compute convergence rates
    l2_rates = [
        np.log(l2_errors[i] / l2_errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
        for i in range(len(h_values) - 1)
    ]
    h1_rates = [
        np.log(h1_errors[i] / h1_errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
        for i in range(len(h_values) - 1)
    ]

    # Plot convergence
    plt.figure(figsize=(10, 8))
    plt.loglog(
        h_values, l2_errors, "o-", label=f"L2 Error (Rate ≈ {np.mean(l2_rates):.2f})"
    )
    plt.loglog(
        h_values, h1_errors, "s-", label=f"H1 Error (Rate ≈ {np.mean(h1_rates):.2f})"
    )

    # Reference lines
    ref_h = np.array([h_values[0], h_values[-1]])
    plt.loglog(ref_h, ref_h**2 * l2_errors[0] / h_values[0] ** 2, "k--", label="O(h²)")
    plt.loglog(ref_h, ref_h * h1_errors[0] / h_values[0], "k-.", label="O(h)")

    plt.xlabel("Element Size (h)")
    plt.ylabel("Error")
    plt.title("FEM Convergence Study")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    # Print convergence rates
    print("\nConvergence Rates:")
    for i in range(len(l2_rates)):
        print(
            f"Refinement {i + 1}: L2 rate = {l2_rates[i]:.2f}, H1 rate = {h1_rates[i]:.2f}"
        )
    print(
        f"Average: L2 rate = {np.mean(l2_rates):.2f}, H1 rate = {np.mean(h1_rates):.2f}"
    )

    return {
        "h_values": h_values,
        "l2_errors": l2_errors,
        "h1_errors": h1_errors,
        "l2_rates": l2_rates,
        "h1_rates": h1_rates,
        "solutions": solutions,
    }


def compute_exact_solution_at_nodes(
    mesh: Mesh, exact_solution: Callable[[float, float], float]
) -> npt.NDArray[np.float64]:
    """
    Compute the exact solution at all mesh nodes.

    Args:
        mesh: The finite element mesh
        exact_solution: Exact solution function u(x,y)

    Returns:
        Array containing exact solution values at mesh nodes
    """
    exact_values = np.zeros(mesh.num_nodes)

    for i in range(mesh.num_nodes):
        x, y = mesh.coordinates[0, i], mesh.coordinates[1, i]
        exact_values[i] = exact_solution(x, y)

    return exact_values


def plot_error_comparison(
    mesh: Mesh,
    fem_solution: npt.NDArray[np.float64],
    exact_solution: Callable[[float, float], float],
) -> None:
    """
    Plot FEM solution, exact solution, and error.

    Args:
        mesh: The finite element mesh
        fem_solution: FEM solution at mesh nodes
        exact_solution: Exact solution function u(x,y)
    """
    # Compute exact solution at nodes
    exact_values = compute_exact_solution_at_nodes(mesh, exact_solution)

    # Compute error
    error_values = np.abs(fem_solution - exact_values)

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot FEM solution
    im1 = axes[0].tricontourf(
        mesh.coordinates[0, :], mesh.coordinates[1, :], fem_solution, 20, cmap="viridis"
    )
    plt.colorbar(im1, ax=axes[0], label="Value")
    axes[0].set_title("FEM Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # Plot exact solution
    im2 = axes[1].tricontourf(
        mesh.coordinates[0, :], mesh.coordinates[1, :], exact_values, 20, cmap="viridis"
    )
    plt.colorbar(im2, ax=axes[1], label="Value")
    axes[1].set_title("Exact Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    # Plot error
    im3 = axes[2].tricontourf(
        mesh.coordinates[0, :], mesh.coordinates[1, :], error_values, 20, cmap="plasma"
    )
    plt.colorbar(im3, ax=axes[2], label="Error")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")

    # Add mesh overlay
    for e in range(mesh.num_elements):
        nodes = mesh.connectivity[:, e]
        # Close the loop
        nodes = np.append(nodes, nodes[0])
        for ax in axes:
            ax.plot(
                mesh.coordinates[0, nodes], mesh.coordinates[1, nodes], "k-", lw=0.5
            )

    plt.tight_layout()
    plt.show()


def visualize_solution_3d(
    mesh: Mesh, solution: npt.NDArray[np.float64], title: str = "FEM Solution"
) -> None:
    """
    Create a 3D visualization of the solution.

    Args:
        mesh: The finite element mesh
        solution: Solution values at mesh nodes
        title: Plot title
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create a triangulation for the 3D surface plot
    from matplotlib.tri import Triangulation

    triangles = []

    for e in range(mesh.num_elements):
        nodes = mesh.connectivity[:, e]
        # Add two triangles for each quadrilateral element
        triangles.append([nodes[0], nodes[1], nodes[2]])
        triangles.append([nodes[0], nodes[2], nodes[3]])

    tri = Triangulation(mesh.coordinates[0, :], mesh.coordinates[1, :], triangles)

    # Plot the surface
    surf = ax.plot_trisurf(
        tri, solution, cmap="viridis", linewidth=0.2, antialiased=True, edgecolor="gray"
    )

    fig.colorbar(surf, shrink=0.5, aspect=5, label="Solution Value")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    ax.set_title(title)

    plt.savefig(f"{title}.png")

    plt.show()


def compute_condition_number(stiffness_matrix: npt.NDArray[np.float64]) -> float:
    """
    Compute the condition number of the stiffness matrix.

    Args:
        stiffness_matrix: Global stiffness matrix

    Returns:
        Condition number
    """
    return np.linalg.cond(stiffness_matrix)


def linear_solve_stats(
    stiffness_matrix: npt.NDArray[np.float64], load_vector: npt.NDArray[np.float64]
) -> Dict[str, float]:
    """
    Provide statistics about the linear system.

    Args:
        stiffness_matrix: Global stiffness matrix
        load_vector: Global load vector

    Returns:
        Dictionary containing statistics about the system
    """
    stats = {}

    # Matrix size
    stats["matrix_size"] = stiffness_matrix.shape[0]

    # Condition number
    stats["condition_number"] = compute_condition_number(stiffness_matrix)

    # Matrix properties
    stats["matrix_norm"] = np.linalg.norm(stiffness_matrix)
    stats["matrix_trace"] = np.trace(stiffness_matrix)
    stats["matrix_determinant"] = (
        np.linalg.det(stiffness_matrix)
        if stiffness_matrix.shape[0] <= 10
        else "Too large to compute"
    )

    # Load vector properties
    stats["load_vector_norm"] = np.linalg.norm(load_vector)
    stats["load_vector_min"] = np.min(load_vector)
    stats["load_vector_max"] = np.max(load_vector)

    # Sparsity
    non_zeros = np.count_nonzero(stiffness_matrix)
    total_elements = stiffness_matrix.size
    stats["sparsity_ratio"] = 1.0 - (non_zeros / total_elements)

    return stats
