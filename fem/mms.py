"""
Method of Manufactured Solutions (MMS) for the 2D Poisson equation with variable conductivity.

This script verifies the FEM implementation by solving:
-∇·(κ(x)∇u(x)) = f(x)

The method of manufactured solutions works as follows:
1. Choose an exact solution u(x,y) a priori
2. Choose a conductivity field κ(x,y)
3. Derive the forcing function f(x,y) by substituting u and κ into the PDE
4. Apply the exact solution as Dirichlet boundary conditions
5. Solve the forward problem with the derived forcing and BC
6. Compare the numerical solution with the exact solution to verify accuracy

For this test case:
- Exact solution: u(x,y) = x^2 + y^2
- Conductivity field: κ(x,y) = 1 + x^2 + y^2
- Forcing function (derived analytically): f(x,y) = 4 + 8(x^2 + y^2)
- Domain: Unit square [0,1]^2

The derivation of f is as follows:
- ∇u = (2x, 2y)
- κ∇u = (2x(1+x^2+y^2), 2y(1+x^2+y^2))
- ∇·(κ∇u) = ∂/∂x(2x(1+x^2+y^2)) + ∂/∂y(2y(1+x^2+y^2))
          = 2(1+x^2+y^2) + 2x·2x + 2(1+x^2+y^2) + 2y·2y
          = 4 + 4x^2 + 2y^2 + 4x^2 + 4y^2
          = 4 + 8x^2 + 8y^2 = 4 + 8(x^2+y^2)
- f = -∇·(κ∇u) = -(4 + 8(x^2+y^2))
"""

import numpy as np
import matplotlib.pyplot as plt
from fem.inverse_problem import FEMProblem
import os
from scipy.optimize import minimize
import time
import argparse

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)


def run_manufactured_solution_test(plot_results=True):
    """
    Run the method of manufactured solutions test.

    Args:
        plot_results: Whether to plot and save figures

    Returns:
        tuple: (fem_problem, u_exact, u_computed, kappa, error)
    """
    # Step 1: Create FEM problem on unit square mesh
    num_pixels = 8  # Number of elements in each direction
    fem_problem = FEMProblem(num_pixels=num_pixels, rtol=1e-10)

    # Step 2: Define exact solution and boundary conditions
    # Get node coordinates (shape: (2, num_nodes))
    x_coords = fem_problem.mesh.coordinates[0, :]  # x-coordinates of all nodes
    y_coords = fem_problem.mesh.coordinates[1, :]  # y-coordinates of all nodes

    print(f"x_coords.shape: {x_coords.shape}")
    print(f"y_coords.shape: {y_coords.shape}")
    print(f"x_coords: {x_coords}")
    print(f"y_coords: {y_coords}")

    # Manufactured solution at all nodes: u(x,y) = x^2 + y^2
    u_exact = x_coords**2 + y_coords**2

    # Set up Dirichlet boundary conditions
    # Note on boundary condition handling:
    # 1. In fem/inverse_problem.py, the FEMProblem class automatically identifies
    #    boundary nodes when initialized with apply_dirichlet_on="boundary"
    # 2. The boundary nodes are stored in fem_problem.dirichlet_nodes
    # 3. A boolean mask fem_problem.dirichlet_nodes_bool is created to identify these nodes
    # 4. When calling make_stiffness_and_bcs(), the Dirichlet values are extracted
    #    using u_d * self.dirichlet_nodes_bool
    # 5. The solver only solves for internal nodes, and boundary values are set directly from u_d
    # Therefore, we only need to set values in u_d for the boundary nodes
    u_d = np.zeros_like(u_exact)
    u_d[fem_problem.dirichlet_nodes] = u_exact[fem_problem.dirichlet_nodes]

    # Verify that boundary nodes include all edges of unit square
    boundary_x = x_coords[fem_problem.dirichlet_nodes]
    boundary_y = y_coords[fem_problem.dirichlet_nodes]
    if plot_results:
        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords, y_coords, s=10, alpha=0.5, label="All Nodes")
        plt.scatter(boundary_x, boundary_y, s=30, c="red", label="Boundary Nodes")
        plt.grid(True)
        plt.axis("equal")
        plt.title("Mesh with Boundary Nodes Highlighted")
        plt.legend()
        plt.savefig("figures/boundary_nodes.png")

    # Step 3: Define the forcing term at nodes
    # f(x,y) = -(4 + 8(x^2 + y^2))
    f = -(4 + 8 * (x_coords**2 + y_coords**2))

    # Step 4: Define the conductivity field κ(x,y) = 1 + x^2 + y^2
    # Important note on kappa field:
    # In the FEM implementation, conductivity κ is defined per element, not per node.
    # Each element has a constant κ value, which we evaluate at the element centroid.
    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.num_elements)
    )
    elem_coords = np.array(
        elem_coords
    )  # Convert to tuple of arrays to a single array, shape (2, 4, num_elements), quadrilateral elements, num_elements = num_pixels**2

    # print(f"elem_coords: {elem_coords}")
    # print(f"elem_coords.shape: {elem_coords.shape}")

    # Calculate element centroids (average of the 4 nodes' coordinates)
    x_elem = np.mean(elem_coords[0], axis=0)  # Mean x-coordinate for each element
    y_elem = np.mean(elem_coords[1], axis=0)  # Mean y-coordinate for each element

    print(f"x_elem.shape: {x_elem.shape}")
    print(f"y_elem.shape: {y_elem.shape}")

    # Define kappa at element centroids
    kappa = 1 + x_elem**2 + y_elem**2

    # Step 5: Set parameters in the FEM problem
    fem_problem.set_parameters(f=f, u_d=u_d, uhat=u_exact)

    # Step 6: Solve forward problem with the defined kappa
    u_computed = fem_problem.forward(kappa)

    # Compute error between manufactured and computed solutions
    error = np.linalg.norm(u_computed - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.6e}")

    if plot_results:
        # Visualize the results
        # Plot exact solution
        fem_problem.mesh.plot(u_exact, title="Exact_Solution")

        # Plot computed solution
        fem_problem.mesh.plot(u_computed, title="Computed_Solution")

        # Plot absolute error
        error_plot = np.abs(u_computed - u_exact)
        fem_problem.mesh.plot(error_plot, title="Absolute_Error")

        # Plot kappa field for reference
        plt.figure(figsize=(10, 8))
        plt.tricontourf(x_elem, y_elem, kappa, 20, cmap="viridis")
        plt.colorbar(label="Conductivity κ")
        plt.title("Conductivity Field κ(x,y) = 1 + x² + y²")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("figures/kappa_field.png")

    return fem_problem, u_exact, u_computed, kappa, error


def run_convergence_study():
    """
    Run a mesh convergence study for the manufactured solution.

    Returns:
        tuple: (resolutions, errors)
    """
    # Test with different mesh resolutions to verify convergence
    resolutions = [8, 16, 32, 64, 128]
    errors = []

    for res in resolutions:
        print(f"\nRunning with resolution {res}x{res}...")
        # Create FEM problem with this resolution
        fem_prob = FEMProblem(num_pixels=res, rtol=1e-10)

        # Get node coordinates
        x = fem_prob.mesh.coordinates[0, :]
        y = fem_prob.mesh.coordinates[1, :]

        # Define exact solution, BCs, and forcing
        u_exact_res = x**2 + y**2
        u_d_res = np.zeros_like(u_exact_res)
        u_d_res[fem_prob.dirichlet_nodes] = u_exact_res[fem_prob.dirichlet_nodes]
        f_res = -(4 + 8 * (x**2 + y**2))

        # Get element coordinates and compute kappa
        ec = fem_prob.mesh.get_element_coordinates(np.arange(fem_prob.num_elements))
        ec = np.array(ec)
        x_e = np.mean(ec[0], axis=0)
        y_e = np.mean(ec[1], axis=0)
        kappa_res = 1 + x_e**2 + y_e**2

        # Solve and compute error
        fem_prob.set_parameters(f=f_res, u_d=u_d_res, uhat=u_exact_res)
        u_comp_res = fem_prob.forward(kappa_res)
        err = np.linalg.norm(u_comp_res - u_exact_res) / np.linalg.norm(u_exact_res)
        errors.append(err)
        print(f"Resolution {res}x{res}: Relative L2 error = {err:.6e}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.loglog(resolutions, errors, "o-", linewidth=2, markersize=8)
    plt.grid(True, which="both", ls="--")
    plt.xlabel("Mesh Resolution (N×N)")
    plt.ylabel("Relative L2 Error")
    plt.title("Convergence of Method of Manufactured Solutions")

    # Compute and plot theoretical second-order convergence rate
    ref_res = resolutions[0]
    ref_err = errors[0]
    second_order = [ref_err * (ref_res / res) ** 2 for res in resolutions]
    plt.loglog(resolutions, second_order, "--", label="Second-Order (O(h²))")
    plt.legend()

    plt.savefig("figures/mms_convergence.png")
    plt.show()

    return resolutions, errors


def run_inverse_problem_test(
    noise_level=0.01, resolution=64, max_iter=100, plot_results=True
):
    """
    Run an inverse problem test using the method of manufactured solutions.

    This test attempts to recover a known conductivity field from synthetic measurements:
    1. Define a true conductivity field κ(x,y) = 1 + x^2 + y^2
    2. Generate synthetic measurements u_true by solving the forward problem
    3. Add optional measurement noise to create u_meas
    4. Start with an initial guess κ₀ (constant or random field)
    5. Solve the inverse problem to recover κ from u_meas
    6. Compare the recovered κ with the true κ

    Args:
        noise_level: Standard deviation of Gaussian noise to add to measurements (relative to u_true)
        resolution: Number of elements in each direction (resolution × resolution mesh)
        plot_results: Whether to plot and save figures

    Returns:
        tuple: (fem_problem, kappa_true, kappa_recovered, error)
    """
    print("\n" + "=" * 80)
    print("INVERSE PROBLEM TEST WITH METHOD OF MANUFACTURED SOLUTIONS")
    print("=" * 80)

    # Step 1: Create FEM problem on unit square mesh
    num_pixels = resolution  # Number of elements in each direction
    fem_problem = FEMProblem(num_pixels=num_pixels, rtol=1e-10)

    # Get coordinates
    x_coords = fem_problem.mesh.coordinates[0, :]
    y_coords = fem_problem.mesh.coordinates[1, :]

    # Define Dirichlet boundary conditions: u(x,y) = x^2 + y^2 on boundary
    u_d = np.zeros(fem_problem.num_nodes)
    u_d[fem_problem.dirichlet_nodes] = (x_coords**2 + y_coords**2)[
        fem_problem.dirichlet_nodes
    ]

    # check B.Cs are satisfied
    u_d_check = u_d[fem_problem.dirichlet_nodes]
    u_exact_check = (x_coords**2 + y_coords**2)[fem_problem.dirichlet_nodes]
    print(
        f"norm of u_d_check - u_exact_check: {np.linalg.norm(u_d_check - u_exact_check)}"
    )

    # Define forcing term: f(x,y) = 4 + 8(x^2 + y^2)
    f = -(4 + 8 * (x_coords**2 + y_coords**2))

    # Get element coordinates for defining kappa
    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.num_elements)
    )
    elem_coords = np.array(elem_coords)
    x_elem = np.mean(elem_coords[0], axis=0)
    y_elem = np.mean(elem_coords[1], axis=0)

    # Define the true conductivity field: κ(x,y) = 1 + x^2 + y^2
    kappa_true = 1 + x_elem**2 + y_elem**2

    # Generate synthetic measurements by solving the forward problem
    fem_problem.set_parameters(f=f, u_d=u_d)
    u_true = fem_problem.forward(kappa_true)

    # Add noise to create synthetic measurements
    if noise_level > 0:
        np.random.seed(42)  # For reproducibility
        noise = noise_level * np.linalg.norm(u_true) * np.random.randn(u_true.size)
        u_meas = u_true + noise
        # Ensure boundary conditions remain exact
        u_meas[fem_problem.dirichlet_nodes] = u_d[fem_problem.dirichlet_nodes]
    else:
        u_meas = u_true.copy()

    # Set the measured data as the target for inversion
    fem_problem.set_parameters(f=f, u_d=u_d, uhat=u_meas)

    # Create an initial guess for kappa
    # kappa0 = np.full_like(kappa_true, 5)
    # kappa0 += 10 * np.random.randn(kappa0.size)
    # kappa0[kappa0 < 0] = 0  # Ensure positivity
    kappa0 = kappa_true.copy()
    kappa0 += 1 * np.random.randn(kappa0.size)
    kappa0[kappa0 < 0] = 0  # Ensure positivity

    print(f"Initial guess: constant kappa = 2.0")
    print(f"True kappa range: [{kappa_true.min():.2f}, {kappa_true.max():.2f}]")
    print(f"Noise level: {noise_level:.2%} of signal")

    # Set up callback to monitor progress
    iterations = []
    objectives = []
    kappa_errors = []

    def callback(xk):
        iter_num = len(iterations)
        iterations.append(iter_num)

        # Current kappa
        kappa_current = xk

        # Objective value (misfit)
        objective = fem_problem.objective(kappa_current)
        objectives.append(objective)

        # Error in kappa
        error = np.linalg.norm(kappa_current - kappa_true) / np.linalg.norm(kappa_true)
        kappa_errors.append(error)

        print(
            f"Iteration {iter_num}: objective = {objective:.6e}, relative error in kappa = {error:.6e}"
        )
        return False  # Continue optimization

    # Configure optimization
    options = {
        "maxiter": 1000,  # Increase maximum iterations if needed.
        # "factr": 1e2,  # Lower factr to tighten the stopping criterion.
        "gtol": 1e-12,  # Optional: adjust the projected gradient tolerance if necessary.
        "ftol": 1e-12,  # Optional: adjust the function tolerance if necessary.
        # "norm": 2,
        "return_all": True,
        "disp": True,
    }

    # Solve the inverse problem
    print("\nSolving inverse problem...")
    start_time = time.perf_counter()

    result = minimize(
        fem_problem.objective,
        kappa0.copy(),
        method="L-BFGS-B",
        jac=fem_problem.gradient,
        callback=callback,
        options=options,
        bounds=[(0.1, None) for _ in range(kappa_true.size)],  # Ensure positivity
    )

    end_time = time.perf_counter()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")

    # Get the recovered conductivity
    kappa_recovered = result.x

    # Compute final error
    rel_error = np.linalg.norm(kappa_recovered - kappa_true) / np.linalg.norm(
        kappa_true
    )
    print(f"Final relative L2 error in kappa: {rel_error:.6e}")

    # Compute forward solution with recovered kappa
    u_recovered = fem_problem.forward(kappa_recovered)
    u_error = np.linalg.norm(u_recovered - u_true) / np.linalg.norm(u_true)
    print(f"Relative L2 error in solution u: {u_error:.6e}")

    if plot_results:
        # Plot the true kappa, initial guess, and recovered kappa
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

        # True kappa
        im1 = axes[0].tricontourf(x_elem, y_elem, kappa_true, 20, cmap="viridis")
        axes[0].set_title("True κ(x,y) = 1 + x² + y²")
        fig.colorbar(im1, ax=axes[0])

        # Initial guess
        im2 = axes[1].tricontourf(x_elem, y_elem, kappa0, 20, cmap="viridis")
        axes[1].set_title("Initial Guess κ₀(x,y) = 2")
        fig.colorbar(im2, ax=axes[1])

        # Recovered kappa
        im3 = axes[2].tricontourf(x_elem, y_elem, kappa_recovered, 20, cmap="viridis")
        axes[2].set_title(f"Recovered κ (Error: {rel_error:.2%})")
        fig.colorbar(im3, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        plt.savefig("figures/inverse_problem_kappa.png")

        # Plot the solutions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

        # True solution
        fem_problem.mesh.plot(u_true, title="True Solution")

        # Measured data (with noise)
        fem_problem.mesh.plot(u_meas, title=f"Measurements (Noise: {noise_level:.2%})")

        # Recovered solution
        fem_problem.mesh.plot(
            u_recovered, title=f"Recovered Solution (Error: {u_error:.2%})"
        )

        # Plot convergence
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.semilogy(iterations, objectives, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value (Misfit)")
        plt.title("Convergence of Objective Function")

        plt.subplot(1, 2, 2)
        plt.semilogy(iterations, kappa_errors, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Relative Error in κ")
        plt.title("Error in Recovered Conductivity")

        plt.tight_layout()
        plt.savefig("figures/inverse_problem_convergence.png")

    return fem_problem, kappa_true, kappa_recovered, rel_error


def check_boundary_conditions(fem_problem, u_exact, u_computed):
    boundary_nodes = fem_problem.dirichlet_nodes
    boundary_x = fem_problem.mesh.coordinates[0, boundary_nodes]
    boundary_y = fem_problem.mesh.coordinates[1, boundary_nodes]

    # Values at boundary
    u_exact_boundary = u_exact[boundary_nodes]
    u_computed_boundary = u_computed[boundary_nodes]

    # Compute error at boundary
    boundary_error = np.abs(u_computed_boundary - u_exact_boundary)
    max_error = np.max(boundary_error)

    print(f"Maximum boundary error: {max_error:.6e}")

    # Plot boundary values
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(boundary_x, boundary_y, c=u_exact_boundary)
    plt.colorbar(label="Exact")
    plt.title("Exact Solution at Boundary")

    plt.subplot(1, 3, 2)
    plt.scatter(boundary_x, boundary_y, c=u_computed_boundary)
    plt.colorbar(label="Computed")
    plt.title("Computed Solution at Boundary")

    plt.savefig("figures/boundary_check.png")


# test to verify the FEM solver with a simpler problem
def test_simple_solution():
    fem_problem = FEMProblem(64, rtol=1e-10)
    # Constant solution u(x,y) = 1
    u_exact = np.ones(fem_problem.num_nodes)
    u_d = np.zeros_like(u_exact)
    u_d[fem_problem.dirichlet_nodes] = u_exact[fem_problem.dirichlet_nodes]
    f = np.zeros_like(u_exact)  # No forcing for constant solution

    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.num_elements)
    )
    elem_coords = np.array(elem_coords)
    x_elem = np.mean(elem_coords[0], axis=0)
    y_elem = np.mean(elem_coords[1], axis=0)
    kappa = np.ones_like(x_elem)  # Constant conductivity

    fem_problem.set_parameters(f=f, u_d=u_d)
    u_computed = fem_problem.forward(kappa)

    error = np.linalg.norm(u_computed - u_exact) / np.linalg.norm(u_exact)
    print(f"Error for constant solution: {error:.6e}")

    return u_computed, u_exact


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inverse problem MMS test")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Noise level as fraction of signal (default: 0.01)",
    )
    parser.add_argument(
        "--resolution", type=int, default=64, help="Mesh resolution (default: 64x64)"
    )
    parser.add_argument("--noplot", action="store_true", help="Disable plotting")

    args = parser.parse_args()

    test_simple_solution()

    print("=" * 80)
    print("RUNNING METHOD OF MANUFACTURED SOLUTIONS FOR FORWARD PROBLEM...")
    fem_problem, u_exact, u_computed, kappa, error = run_manufactured_solution_test()
    print("=" * 80)

    check_boundary_conditions(fem_problem, u_exact, u_computed)

    print("=" * 80)
    print("RUNNING CONVERGENCE STUDY FOR FORWARD PROBLEM...")
    print("=" * 80)
    resolutions, errors = run_convergence_study()

    print("=" * 80)
    print("INVERSE PROBLEM TEST WITH METHOD OF MANUFACTURED SOLUTIONS")
    print("=" * 80)
    print(f"\nMesh resolution: {args.resolution}x{args.resolution}")
    print(f"Noise level: {args.noise:.2%}")
    print(f"Plotting: {'disabled' if args.noplot else 'enabled'}")

    # Run the test with the specified noise level
    fem_problem, kappa_true, kappa_recovered, error = run_inverse_problem_test(
        noise_level=args.noise, resolution=args.resolution, plot_results=not args.noplot
    )

    print("\n" + "=" * 80)
    print(f"TEST COMPLETED WITH RELATIVE ERROR: {error:.2%}")
    print("=" * 80)

    print("\nMMS tests completed successfully.")
