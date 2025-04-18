"""
Inverse problem, including:
Adjoint solver
stiffness matrix grad w.r.t. conductivity.
"""

import numpy as np
import numpy.typing as npt
from scipy import sparse
from .mesh import Mesh
from scipy.optimize import minimize, OptimizeResult
from scipy.sparse.linalg import LinearOperator


class FEMProblem:
    """
    if apply_dirichlet_on == "boundary", take all boundary nodes to be dirichlet
    if None, no dirichlet nodes are defined
    alternatively, you can provide a function which takes the mesh and returns indices of dirichlet nodes

    f and u_d can be set here, later with set_parameters, or when using inverse_step/forward.

    rtol: relative tolerance of the linear solver
    """

    def __init__(
        self,
        num_pixels: int,
        f=None,
        u_d=None,
        uhat=None,
        u0=None,
        apply_dirichlet_on="boundary",
        rtol=1e-5,
        atol=0.0,
    ):
        self.atol = atol
        self.rtol = rtol
        self.u_d = u_d
        self.uhat = uhat
        self.u0 = u0
        self.w0 = None
        print("making mesh...")
        self.mesh = Mesh(0.0, 1.0, 0.0, 1.0, num_pixels, num_pixels)
        self.dx = self.mesh.dx
        self.dy = self.mesh.dy
        self.num_elements = self.mesh.num_elements
        self.num_nodes = self.mesh.num_nodes
        self.connectivity = self.mesh.connectivity
        if not f is None:
            self._calculate_forcing(f)
        if apply_dirichlet_on == "boundary":
            self.dirichlet_nodes = self.mesh.boundary_nodes
        elif apply_dirichlet_on == None:
            self.dirichlet_nodes = np.array([])
        else:
            self.dirichlet_nodes = apply_dirichlet_on(self.mesh)
        self.dirichlet_nodes_bool = np.zeros(self.num_nodes, dtype=bool)
        self.dirichlet_nodes_bool[self.dirichlet_nodes] = True
        self.active_nodes_bool = np.logical_not(self.dirichlet_nodes_bool)
        self.active_nodes = np.where(self.active_nodes_bool)[0]
        self.num_active_nodes = np.sum(self.active_nodes_bool)
        # self.Kprime_submatrix = np.array(
        #     [[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]],
        #     dtype=float,
        # )
        # For a uniform square mesh, self.dx = self.dy = h, so we introduce a factor 1/(6h^2)
        self.Kprime_submatrix = (1.0 / 6.0) * np.array(
            [[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]],
            dtype=float,
        )
        self._set_ij()
        self.Ka = None
        self.Kd = None

    def _set_ij(self):
        I = np.expand_dims(self.connectivity, axis=0)
        I = np.tile(I, (4, 1, 1))
        I = I.flatten()
        J = np.expand_dims(self.connectivity, axis=1)
        J = np.tile(J, (1, 4, 1))
        J = J.flatten()
        self.I = I
        self.J = J

        reduced_map = np.zeros(self.num_nodes, dtype=int)
        for i, node in enumerate(self.active_nodes):
            reduced_map[node] = i
        for i, node in enumerate(self.dirichlet_nodes):
            reduced_map[node] = i
        self.active_IJ_mask = self.active_nodes_bool[I] * self.active_nodes_bool[J]
        self.dirichlet_IJ_mask = (
            self.active_nodes_bool[I] * self.dirichlet_nodes_bool[J]
        )
        self.Ia = np.array([reduced_map[i] for i in I[self.active_IJ_mask]])
        self.Id = np.array(
            [
                # Somewhat confusingly, this array contains active indices
                reduced_map[i]
                for i in I[self.dirichlet_IJ_mask]
            ]
        )
        self.Ja = np.array([reduced_map[j] for j in J[self.active_IJ_mask]])
        self.Jd = np.array([reduced_map[j] for j in J[self.dirichlet_IJ_mask]])

    """ 
    Calculate f vector from nodal values
    """

    def _calculate_forcing(self, f):
        if callable(f):
            f_nodal = np.array([f(x) for x in self.mesh.coordinates])
        else:
            f_nodal = f
        self.Mprime_submatrix = (
            (self.dx**2)
            / (36.0)
            * np.array(
                [[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]],
                dtype=float,
            )
        )
        self.f = np.zeros(self.num_nodes)
        for e in range(self.num_elements):
            self.f[self.connectivity[:, e]] += (
                self.Mprime_submatrix @ f_nodal[self.connectivity[:, e]]
            )

    """
    set forcing, Dirichlet boundary conditions and objective u
    
    f: forcing field. This includes at boundary nodes
    u_d: vector of the same shape as u; contains boundary values at boundary nodes. It can be zero everywhere else.
    uhat: measured uhat value.
    u0: initial value for solver
    """

    def set_parameters(
        self,
        f: npt.NDArray[np.float64] = None,
        u_d: npt.NDArray[np.float64] = None,
        uhat: npt.NDArray[np.float64] = None,
        u0: npt.NDArray[np.float64] = None,
    ):
        if not f is None:
            self._calculate_forcing(f)
        self.u_d = u_d if not u_d is None else self.u_d
        self.uhat = uhat if not uhat is None else self.uhat
        self.u0 = u0 if not u0 is None else self.u0

    """
    Processing to create stiffness matrix from kappa and apply boundary conditions
    """

    def make_stiffness_and_bcs(
        self,
        kappa: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        u_d: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64] : npt.NDArray[np.float64]]:
        # k_data = (self.Kprime_submatrix.reshape(-1, 1) * kappa.reshape(1, -1)).flatten()
        # K = sparse.csr_array(
        #     (k_data, (self.I, self.J)), shape=(self.num_nodes, self.num_nodes)
        # )
        # K.sum_duplicates()
        # # cut out boundary nodes
        # u_d = u_d * self.dirichlet_nodes_bool
        # f_prob = f.copy() - K @ u_d
        # K = K[self.active_nodes_bool, :]
        # K = K[:, self.active_nodes_bool]
        # f_prob = f_prob[self.active_nodes_bool]
        # return K, f_prob
        k_data = (self.Kprime_submatrix.reshape(-1, 1) * kappa.reshape(1, -1)).flatten()
        k_data_a = k_data[self.active_IJ_mask]
        k_data_d = k_data[self.dirichlet_IJ_mask]
        self.Ka = sparse.csr_array(
            (k_data_a, (self.Ia, self.Ja)),
            shape=(self.num_active_nodes, self.num_active_nodes),
        )
        self.Kd = sparse.csr_array(
            (k_data_d, (self.Id, self.Jd)),
            shape=(self.num_active_nodes, self.num_nodes - self.num_active_nodes),
        )
        f_prob = f[self.active_nodes_bool] - self.Kd @ u_d[self.dirichlet_nodes_bool]
        return self.Ka, f_prob

    """
    Solve forward problem, adjoint problem, and take gradient of \|u0-u_kappa\|^2 w.r.t. kappa.

    kappa: conductivity field
    u0: initial guess of u
    """

    def inverse_step(
        self,
        kappa: npt.NDArray[np.float64],
        u0=None,
        w0=None,
    ):
        f = self.f
        u_d = self.u_d
        uhat = self.uhat
        assert not f is None, "f must be set with set_parameters()"
        assert not u_d is None, "u_d must be set with set_parameters()"
        assert not uhat is None, "uhat must be set with set_parameters()"
        K, f = self.make_stiffness_and_bcs(kappa, f, u_d)
        D_inv = 1.0 / K.diagonal()

        def jacobi_preconditioner(v):
            return D_inv * v

        M = LinearOperator(K.shape, matvec=jacobi_preconditioner)
        uhat = uhat[self.active_nodes_bool]
        # solve forward problem
        u_kappa, info1 = sparse.linalg.lgmres(
            K, f, x0=u0, rtol=self.rtol, atol=self.atol, M=M
        )
        # u_kappa, info1 = sparse.linalg.lgmres(K, f, rtol=self.rtol, atol=self.atol)
        if info1 > 0:
            print("Result of forward solver did not converge to tolerance")
        if info1 < 0:
            print("Illegal input or breakdown in forward solver.")
        # solve adjoint problem
        w, info2 = sparse.linalg.lgmres(
            K.T, uhat - u_kappa, rtol=self.rtol, atol=self.atol, M=M
        )
        if info2 > 0:
            print("Result of adjoint solver did not converge to tolerance")
        if info1 < 0:
            print("Illegal input or breakdown in adjoint solver.")
        # Add back boundary nodes
        u_kappa_ = np.zeros_like(u_d)
        u_kappa_[self.active_nodes_bool] = u_kappa
        u_kappa_[self.dirichlet_nodes_bool] = u_d[self.dirichlet_nodes_bool]
        # u_kappa = u_kappa_
        w_ = np.zeros_like(u_d)
        w_[self.active_nodes_bool] = w
        # w = w_
        # Take gradient
        dJ = np.zeros_like(kappa)
        for i, n in enumerate(self.connectivity.T):
            dJ[i] = np.einsum("i,ij,j", u_kappa_[n], self.Kprime_submatrix, w_[n])
        del K
        return u_kappa_, dJ, u_kappa, w

    """
    A forward problem

    kappa: conductivity field
    u0: initial guess of u
    """

    def forward(self, kappa: npt.NDArray[np.float64], u0=None):
        f = self.f
        u_d = self.u_d
        assert not f is None, "f must be set with set_parameters()"
        assert not u_d is None, "u_d must be set with set_parameters()"
        K, f = self.make_stiffness_and_bcs(kappa, f, u_d)
        D_inv = 1.0 / K.diagonal()

        def jacobi_preconditioner(v):
            return D_inv * v

        M = LinearOperator(K.shape, matvec=jacobi_preconditioner)
        u_kappa, info = sparse.linalg.lgmres(
            K, f, rtol=self.rtol, atol=self.atol, x0=u0, M=M
        )
        # Add back boundary nodes
        u_kappa_ = np.zeros_like(u_d)
        u_kappa_[self.dirichlet_nodes_bool] = u_d[self.dirichlet_nodes_bool]
        u_kappa_[self.active_nodes_bool] = u_kappa
        del K
        return u_kappa_

    """ Evaluates objective, calling forward """

    def objective(self, kappa: npt.NDArray[np.float64]):
        u_kappa_ = self.forward(kappa)
        # Data misfit term
        misfit = 0.5 * np.sum((u_kappa_ - self.uhat) ** 2)

        return misfit

    """ Evaluates gradient of objective, calling inverse_step """

    def gradient(self, kappa: npt.NDArray[np.float64]):
        # save solution to use as initial value for next solve.
        u_kappa_, dJ, u_kappa, w = self.inverse_step(kappa, self.u0, self.w0)
        self.u0 = u_kappa
        self.w0 = w
        return dJ


"""
verify gradient of functional with finite differences
"""


def verify_gradient(
    fem_problem: FEMProblem,
    kappa: npt.NDArray[np.float64],
    uhat: npt.NDArray[np.float64],
    epsilon: float | list[float],
    dirs=None,
) -> np.float64 | npt.NDArray[np.float64]:
    u_kappa, dJ, _, _ = fem_problem.inverse_step(kappa)
    J0 = 0.5 * np.sum((u_kappa - uhat) ** 2)
    if dirs == None:
        # chose 10 random unit directions
        dirs = np.random.randn(10, kappa.size)
        dirs = dirs / np.linalg.norm(dirs, axis=1).reshape(-1, 1)
    if isinstance(epsilon, (list, np.ndarray)):
        err_list = []
        for e in epsilon:
            d_errs = []
            for dir in dirs:
                u_kappa_d = fem_problem.forward(kappa + e * dir)
                Jd = 0.5 * np.sum((u_kappa_d - uhat) ** 2)
                fd_dir_grad = (Jd - J0) / e
                exact_dir_grad = np.dot(dJ, dir)
                d_errs.append((fd_dir_grad - exact_dir_grad) / (exact_dir_grad))
            err_list.append(np.sqrt(np.mean(np.array(d_errs) ** 2)))
        return dJ, np.array(err_list)
    else:
        d_errs = []
        for dir in dirs:
            u_kappa_d = fem_problem.forward(kappa + epsilon * dir)
            Jd = np.sum((u_kappa_d - uhat) ** 2)
            exact_dir_grad = np.dot(dJ, dir)
            d_errs.append((fd_dir_grad - exact_dir_grad) / (exact_dir_grad))
        return dJ, np.sqrt(np.mean(np.array(d_errs) ** 2))


#   Test gradient with zero BCs and forcing
def test_gradient_1():
    import matplotlib.pyplot as plt

    fem_problem = FEMProblem(64, rtol=1e-12, atol=0.0)

    u_d = np.zeros(fem_problem.num_nodes)
    node_coords = fem_problem.mesh.coordinates
    f = 500 * (node_coords[0, :] ** 2 + node_coords[1, :] ** 2)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

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

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
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
    
# Test gradient with zero forcing and nonzero BCs
def test_gradient_2():
    import matplotlib.pyplot as plt

    fem_problem = FEMProblem(64, rtol=1e-12)
    # define boundary condition
    u_d = fem_problem.mesh.coordinates[0, :]
    # make a conductivity field
    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.num_elements)[:]
    )
    y_vals = np.mean(np.array(elem_coords)[0, :, :], axis=0)
    kappa = 1 + 100 * y_vals
    f = np.zeros_like(u_d)
    fem_problem.set_parameters(f=f, u_d=u_d)
    u_true = fem_problem.forward(kappa)
    fem_problem.mesh.plot(u_true)
    uhat = u_true + 0.1 * np.random.randn(u_true.size)
    uhat[fem_problem.dirichlet_nodes] = u_d[fem_problem.dirichlet_nodes]
    kappa_noise = np.random.randn(kappa.size)
    kappa = kappa + 0.5 * kappa_noise / np.linalg.norm(kappa_noise)

    # test gradient
    fem_problem.set_parameters(uhat=uhat)
    epsilon_list = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
    dJ, errors = verify_gradient(fem_problem, kappa, uhat, epsilon_list)

    plt.figure()
    plt.plot(epsilon_list, errors)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"E")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.xlim([epsilon_list[-1], epsilon_list[0]])
    plt.savefig("figures/gradient_error.png")
    plt.show()


# Solving an inverse problem
def example_inverse_problem():
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize, OptimizeResult
    import time

    # problem setup is the same as test_gradient_1

    fem_problem = FEMProblem(64, rtol=1e-9)

    # u_d = np.zeros(fem_problem.num_nodes)
    # node_coords = fem_problem.mesh.coordinates
    # f = node_coords[0, :]**2 + node_coords[1, :]**2
    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.num_elements)[:]
    )
    elem_coords = np.array(elem_coords)
    x_vals = np.mean(elem_coords[1, :, :], axis=0)
    y_vals = np.mean(elem_coords[0, :, :], axis=0)
    # kappa = 1 + 100 * y_vals
    # fem_problem.set_parameters(f=f, u_d=u_d)

    # fem_problem = FEMProblem(64, rtol=1e-12)
    # define boundary condition
    u_d = fem_problem.mesh.coordinates[0, :]
    # make a conductivity field
    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.num_elements)[:]
    )
    y_vals = np.mean(np.array(elem_coords)[0, :, :], axis=0)
    kappa = 1 + 100 * y_vals
    f = np.zeros_like(u_d)
    fem_problem.set_parameters(f=f, u_d=u_d)

    u_true = fem_problem.forward(kappa)
    fem_problem.set_parameters(uhat=u_true)

    # a noisy guess of kappa
    # kappa_noise = np.random.randn(kappa.size)
    # kappa0 = kappa + 5 * kappa_noise
    # kappa0[kappa0 < 0] = 0
    kappa0 = np.full_like(kappa, 50)
    kappa0 += 20 * np.random.randn(kappa0.size)
    kappa0[kappa < 0] = 0

    options = {
        # "c1": 0.2,
        # 'c2':
        "maxiter": 100,
        # "norm": 2,
        # "return_all": True,
    }

    def callback(intermediate_result: OptimizeResult):
        kappa_current = intermediate_result.x
        J = intermediate_result.fun
        if not hasattr(callback, "count"):
            callback.count = 0
        callback.count += 1
        # J = 0.5*np.sum((u_true - uk)**2)
        # print(np.linalg.norm(kappa0 - kappa_current))
        print(
            f"finished iteration {callback.count} with distance {np.linalg.norm(kappa - kappa_current):.4f}, objective {J:.4f}"
        )

    start_time = time.perf_counter()
    max_restarts = 5
    current_guess = kappa0.copy()
    for i in range(max_restarts):
        result = minimize(
            fem_problem.objective,
            current_guess,
            method="L-BFGS-B",
            jac=fem_problem.gradient,
            callback=callback,
            options=options,
            bounds=[(0.1, None) for _ in range(kappa.size)],
        )
        current_guess = result.x
        # Check if termination criteria (e.g. gradient norm) are still above a threshold
        if np.linalg.norm(fem_problem.gradient(current_guess)) > 1e-6:
            print(f"Restarting optimization: restart {i + 1}")
        else:
            break

    end_time = time.perf_counter()
    print(f"Optimization took {(end_time - start_time):.4f} s.")
    # kappa_vals = returned["allvecs"]
    # kappa_found = returned["x"]

    # for i in range(500):
    #     uk, dJ, _, _ = fem_problem.inverse_step(kappa_found)
    #     J = 0.5*np.sum((u_true-uk)**2)
    #     print("finished iteration", i, "with distance",
    #           np.linalg.norm(kappa - kappa_found), "objective", J)
    #     kappa_found -= 0.1*dJ

    # kappa_found = result["allvecs"]
    kappa_found = result.x

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharex=True, sharey=True)
    ax1.tricontourf(
        x_vals,
        y_vals,
        kappa,
        20,
        cmap="viridis",
        vmin=0,
        vmax=100,
    )
    ax2.tricontourf(
        x_vals,
        y_vals,
        kappa0,
        20,
        cmap="viridis",
        vmin=0,
        vmax=100,
    )
    ax3.tricontourf(
        x_vals,
        y_vals,
        kappa_found,
        20,
        cmap="viridis",
        vmin=0,
        vmax=100,
    )
    ax4.tricontourf(
        x_vals,
        y_vals,
        kappa0 - kappa,
        20,
        cmap="viridis",
        vmin=-50,
        vmax=50,
    )
    ax5.tricontourf(
        x_vals,
        y_vals,
        kappa0 - kappa_found,
        20,
        cmap="viridis",
        vmin=-50,
        vmax=50,
    )
    plt.tight_layout()
    plt.savefig("figures/kappa_comparison.png")
    plt.show()

    u_kappa_found = fem_problem.forward(kappa_found)
    fem_problem.mesh.plot(u_kappa_found, title="u_kappa_found")

    # fem_problem.mesh.plot(uk)


if __name__ == "__main__":
    np.random.seed(43)
    test_gradient_1()
    # test_gradient_2()
    # example_inverse_problem()
