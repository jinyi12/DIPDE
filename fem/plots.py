"""
Plots for the paper and slides
"""
import numpy as np
import matplotlib.pyplot as plt
from fem.inverse_problem import FEMProblem, verify_gradient

""" Standard formatting for paper """
def format_for_paper():
    plt.rcParams.update({'image.cmap': 'viridis'})
    cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                                        'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                        'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
                                        'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
    plt.rcParams.update({'font.family': 'serif'})
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'mathtext.fontset': 'custom'})
    plt.rcParams.update({'mathtext.rm': 'serif'})
    plt.rcParams.update({'mathtext.it': 'serif:italic'})
    plt.rcParams.update({'mathtext.bf': 'serif:bold'})
    plt.close('all')


#   Test gradient with zero BCs and forcing
def test_gradient_1():

    fem_problem = FEMProblem(64, rtol=1e-12, atol=0.0)

    u_d = np.zeros(fem_problem.num_nodes)
    node_coords = fem_problem.mesh.coordinates
    f = 500*(node_coords[0, :] ** 2 + node_coords[1, :] ** 2)
    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.num_elements)[:]
    )
    y_vals = np.mean(np.array(elem_coords)[0, :, :], axis=0)
    kappa = 1 + 100 * y_vals
    fem_problem.set_parameters(f=f, u_d=u_d)
    u_true = fem_problem.forward(kappa)
    # fem_problem.mesh.plot(u_true)
    uhat = u_true + 0.1 * np.random.randn(u_true.size)
    uhat[fem_problem.dirichlet_nodes] = u_d[fem_problem.dirichlet_nodes]
    kappa_noise = np.random.randn(kappa.size)
    kappa = kappa + 0.5 * kappa_noise / np.linalg.norm(kappa_noise)

    # test gradient
    fem_problem.set_parameters(uhat=uhat)
    epsilon_list = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
    dJ, errors = verify_gradient(fem_problem, kappa, uhat, epsilon_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5))

    im = ax1.tricontourf(
        fem_problem.mesh.coordinates[0, :],
        fem_problem.mesh.coordinates[1, :],
        u_true,
        20,
        cmap="viridis",
    )
    # plt.colorbar(im, ax=ax1)

    # Overlay mesh
    # for e in range(fem_problem.mesh.num_elements):
    #     nodes = fem_problem.mesh.connectivity[:, e]
    #     # Close the loop
    #     nodes = np.append(nodes, nodes[0])
    #     ax1.plot(
    #         fem_problem.mesh.coordinates[0, nodes], fem_problem.mesh.coordinates[1, nodes], "k-", lw=0.5
    #     )

    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_xlim([0, 1.0])
    ax1.set_ylim([0, 1.0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.axis("equal")

    ax2.plot(epsilon_list, errors)
    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel(r"$L^2$ difference")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim([epsilon_list[-1], epsilon_list[0]])

    fig.tight_layout()
    plt.show()
    fig.savefig('./figures/test_gradient.jpg', dpi=300)

def patch_test():
    fem_problem = FEMProblem(5, rtol=1e-12, atol=0.0)
    
    node_coords = fem_problem.mesh.coordinates
    u_d = node_coords[0, :]
    f = np.zeros_like(u_d)
    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.num_elements)[:]
    )
    kappa = np.ones(fem_problem.num_elements)
    fem_problem.set_parameters(f=f, u_d=u_d)
    u_true = node_coords[0,:]
    u_fem = fem_problem.forward(kappa)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5))

    im = ax1.tricontourf(
        fem_problem.mesh.coordinates[0, :],
        fem_problem.mesh.coordinates[1, :],
        u_true,
        20,
        cmap="viridis",
    )
    # plt.colorbar(im, ax=ax1)

    # Overlay mesh
    for e in range(fem_problem.mesh.num_elements):
        nodes = fem_problem.mesh.connectivity[:, e]
        # Close the loop
        nodes = np.append(nodes, nodes[0])
        ax1.plot(
            fem_problem.mesh.coordinates[0, nodes], fem_problem.mesh.coordinates[1, nodes], "k-", lw=0.5
        )
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_xlim([0, 1.0])
    ax1.set_ylim([0, 1.0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.axis("equal")
    
    im = ax2.tricontourf(
        fem_problem.mesh.coordinates[0, :],
        fem_problem.mesh.coordinates[1, :],
        u_true,
        20,
        cmap="viridis",
    )
    # plt.colorbar(im, ax=ax1)

    # Overlay mesh
    for e in range(fem_problem.mesh.num_elements):
        nodes = fem_problem.mesh.connectivity[:, e]
        # Close the loop
        nodes = np.append(nodes, nodes[0])
        ax2.plot(
            fem_problem.mesh.coordinates[0, nodes], fem_problem.mesh.coordinates[1, nodes], "k-", lw=0.5
        )
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.set_xlim([0, 1.0])
    ax2.set_ylim([0, 1.0])
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.tight_layout()
    plt.show()
    fig.savefig('./figures/patch_test.jpg', dpi=300)
    
    l2_err = fem_problem.dx*np.linalg.norm(u_true-u_fem)
    print('l2 error:', l2_err)

def mmse_example():
    # Set the maximum angle theta (in radians)
    theta_max = np.pi / 4  # you can change this

    # Create theta values from -theta_max to +theta_max
    theta = np.linspace(-theta_max, theta_max, 500)

    # Parametrize the unit circle
    x = np.cos(theta)
    y = np.sin(theta)

    # Define the function f(theta)
    f_theta = 1 - theta**2

    # Normalize f(theta) to be used as weights for averaging
    weights = f_theta / np.trapz(f_theta, theta)  # normalize so the integral is 1
    
    # Compute the weighted average of (x, y) under f(theta)
    avg_x = np.trapz(x * weights, theta)
    avg_y = np.trapz(y * weights, theta)

    # Plotting
    plt.figure(figsize=(3.5, 2))
    plt.gca().set_aspect('equal')
    sc = plt.scatter(x, y, c=weights, cmap='viridis', s=5)
    plt.plot([avg_x], [avg_y], 'ro', label='MMSE')
    plt.plot([1], [0], 'bo', label='MAP')
    # plt.axis('equal')
    cbar = plt.colorbar(sc)
    cbar.set_ticks([])  # removes the ticks entirely
    plt.legend(loc=1)
    plt.xlim([0.5, 2.0])
    plt.ylim([-0.8, 0.8])
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.tight_layout()
    # plt.grid(True)
    # plt.show()
    plt.savefig('./figures/mmse_example.jpg', dpi=500)

if __name__ == '__main__':
    np.random.seed(42)
    format_for_paper()
    # test_gradient_1()
    # patch_test()
    mmse_example()
