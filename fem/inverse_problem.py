"""
Inverse problem, including:
Adjoint solver
stiffness matrix grad w.r.t. conductivity. 
"""

import numpy as np
import numpy.typing as npt
from scipy import sparse
from .mesh import Mesh

# Kprime_submatrix = np.array([
#     [4, -1, -2, -1],
#     [-1, 4, -1, -2],
#     [-2, -1, 4, -1],
#     [-1, -2, -1, 4]
# ], dtype=np.float)

# class SubmatrixData:

#     def __init__(self, kappa, connectivity):
#         self.Kprime_submatrix = np.array([
#             [4, -1, -2, -1],
#             [-1, 4, -1, -2],
#             [-2, -1, 4, -1],
#             [-1, -2, -1, 4]
#         ], dtype=float)
#         self.kappa = kappa
#         self.connectivity = connectivity

#     """ I'll need to change this to handle B.Cs. Probably get rid of this class. """
#     def __getitem__(self, index):
#         Ki = index//self.Kprime_submatrix.size
#         j = index % self.Kprime_submatrix.size
#         return self.Kprime_matrix.flatten()[Ki]*self.kappa[j]

#     def __len__(self,index):
#         return self.Kprime_submatrix.size*self.kappa.size

#     def get_ij(self):
#         i = np.expand_dims(self.connectivity, axis = 0)
#         i = np.tile(i, (4, 1, 1))
#         i = i.flatten()
#         j = np.expand_dims(self.connectivity, axis=1)
#         j = np.tile(j, (1, 4, 1))
#         j = j.flatten()
#         return i, j


class FEMProblem:
    """
    if apply_dirichlet_on == "boundary", take all boundary nodes to be dirichlet
    if None, no dirichlet nodes are defined
    alternatively, you can provide a function which takes the mesh and returns indices of dirichlet nodes

    f and u_d can be set here, later with set_parameters, or when using inverse_step/forward.
    
    rtol: relative tolerance of the linear solver
    """

    def __init__(
        self, num_pixels: int, f=None, u_d=None, apply_dirichlet_on="boundary", rtol=1e-5
    ):
        print("making mesh...")
        self.rtol = rtol
        self.f = f
        self.u_d = u_d
        self.mesh = Mesh(0.0, 1.0, 0.0, 1.0, num_pixels, num_pixels)
        self.dx = self.mesh.dx
        self.dy = self.mesh.dy
        self.num_elements = self.mesh.num_elements
        self.num_nodes = self.mesh.num_nodes
        self.connectivity = self.mesh.connectivity
        if apply_dirichlet_on == "boundary":
            self.dirichlet_nodes = self.mesh.boundary_nodes
        elif apply_dirichlet_on == None:
            self.dirichlet_nodes = np.array([])
        else:
            self.dirichlet_nodes = apply_dirichlet_on(self.mesh)
        self.dirichlet_nodes_bool = np.zeros(self.num_nodes, dtype=bool)
        self.dirichlet_nodes_bool[self.dirichlet_nodes] = True
        self.active_nodes_bool = np.logical_not(self.dirichlet_nodes_bool)
        self.Kprime_submatrix = np.array(
            [[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]],
            dtype=float,
        )
        self.I, self.J = self._get_ij()

    def _get_ij(self):
        I = np.expand_dims(self.connectivity, axis=0)
        I = np.tile(I, (4, 1, 1))
        I = I.flatten()
        J = np.expand_dims(self.connectivity, axis=1)
        J = np.tile(J, (1, 4, 1))
        J = J.flatten()
        return I, J

    """ set forcing and Dirichlet boundary conditions """

    def set_parameters(self, f: npt.NDArray[np.float64], u_d: npt.NDArray[np.float64]):
        self.f = f
        self.u_d = u_d
        

    """
    Processing to create stiffness matrix from kappa and apply boundary conditions
    """

    def make_stiffness_and_bcs(
        self,
        kappa: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        u_d: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64] : npt.NDArray[np.float64]]:
        k_data = (self.Kprime_submatrix.reshape(-1, 1) * kappa.reshape(1, -1)).flatten()
        K = sparse.csr_array(
            (k_data, (self.I, self.J)), shape=(self.num_nodes, self.num_nodes)
        )
        K.sum_duplicates()
        # cut out boundary nodes
        u_d = u_d * self.dirichlet_nodes_bool
        f_prob = f.copy() - K @ u_d
        K = K[self.active_nodes_bool, self.active_nodes_bool]
        f_prob = f_prob[self.active_nodes_bool]
        return K, f_prob

    """
    Solve forward problem, adjoint problem, and take gradient of \|u0-u_kappa\|^2 w.r.t. kappa.
    
    kappa: conductivity field
    f: forcing field. This includes at boundary nodes
    u0: current guess of u0.
    u_d: vector of the same shape as u; contains boundary values at boundary nodes. It can be zero everywhere else.
    """

    def inverse_step(
        self,
        kappa: npt.NDArray[np.float64],
        u0: npt.NDArray[np.float64],
        f=None,
        u_d=None,
    ):
        f = f if not f is None else self.f
        u_d = u_d if not u_d is None else self.u_d
        assert not f is None, "f must be set with set_parameters()"
        assert not u_d is None, "u_d must be set with set_parameters()"
        K, f = self.make_stiffness_and_bcs(kappa, f, u_d)
        u0 = u0[self.active_nodes_bool]
        # solve forward problem
        u_kappa, info1 = sparse.linalg.lgmres(K, f, rtol=self.rtol)
        # solve adjoint problem
        w, info2 = sparse.linalg.lgmres(K.T, u0 - u_kappa, rtol=self.rtol)
        # Add back boundary nodes
        u_kappa_ = np.zeros_like(u_d)
        u_kappa_[self.dirichlet_nodes_bool] = u_d[self.dirichlet_nodes_bool]
        u_kappa_[self.active_nodes_bool] = u_kappa
        u_kappa = u_kappa_
        w_ = np.zeros_like(u_d)
        w_[self.active_nodes_bool] = w
        w = w_
        # Take gradient
        dJ = np.zeros_like(kappa)
        for i, n in enumerate(self.connectivity.T):
            dJ[i] = np.einsum(
                "i,ij,j", u_kappa[n], self.Kprime_submatrix, w[n]
            )
        del K, w, w_
        return u_kappa_, dJ

    """
    A forward problem
    
    kappa: conductivity field
    f: forcing field. This includes at boundary nodes
    u_d: vector of the same shape as u; contains boundary values at boundary nodes. It can be zero everywhere else.
    """

    def forward(self, kappa: npt.NDArray[np.float64], f=None, u_d=None):
        f = f if not f is None else self.f
        u_d = u_d if not u_d is None else self.u_d
        assert not f is None, "f must be set with set_parameters()"
        assert not u_d is None, "u_d must be set with set_parameters()"
        K, f = self.make_stiffness_and_bcs(kappa, f, u_d)
        u_kappa, info = sparse.linalg.lgmres(K, f, rtol=self.rtol)
        # Add back boundary nodes
        u_kappa_ = np.zeros_like(u_d)
        u_kappa_[self.dirichlet_nodes_bool] = u_d[self.dirichlet_nodes_bool]
        u_kappa_[self.active_nodes_bool] = u_kappa
        del K
        return u_kappa_


"""
verify gradient of functional with finite differences
"""


def verify_gradient(
    fem_problem: FEMProblem,
    kappa: npt.NDArray[np.float64],
    u0: npt.NDArray[np.float64],
    epsilon: float | list[float],
    dirs = None,
) -> np.float64 | npt.NDArray[np.float64]:
    u_kappa, dJ = fem_problem.inverse_step(kappa, u0)
    J0 = np.sum((u_kappa - u0) ** 2)
    if dirs == None:
        # chose 10 random unissst directions
        dirs = np.random.randn(10, kappa.size)
        dirs = dirs / np.linalg.norm(dirs, axis=1).reshape(-1, 1)
    if isinstance(epsilon, (list, np.ndarray)) :
        err_list = []
        for e in epsilon:
            d_errs = []
            for dir in dirs:
                u_kappa_d = fem_problem.forward(kappa + e * dir)
                Jd = np.sum((u_kappa_d - u0) ** 2)
                print(Jd)
                fd_dir_grad = (Jd - J0) / e
                d_errs.append(fd_dir_grad - np.dot(dJ, dir))
            err_list.append(np.mean(np.array(d_errs) ** 2))
        return dJ, np.array(err_list)
    else:
        d_errs = []
        for dir in dirs:
            u_kappa_d = fem_problem.forward(kappa + epsilon * dir)
            Jd = np.sum((u_kappa_d - u0) ** 2)
            fd_dir_grad = (Jd - J0) / epsilon
            d_errs.append(fd_dir_grad - np.dot(dJ, dir))
        return dJ, np.mean(np.array(d_errs) ** 2)


if __name__ == "__main__":
    fem_problem = FEMProblem(64)
    # define boundary condition
    u_d = fem_problem.mesh.coordinates[0, :]
    # make a conductivity field
    elem_coords = fem_problem.mesh.get_element_coordinates(
        np.arange(fem_problem.mesh.num_elements)[:]
    )
    y_vals = np.mean(np.array(elem_coords)[0, :, :], axis=0)
    kappa = 1 + 100 * y_vals

    f = np.zeros_like(u_d)

    fem_problem.set_parameters(f, u_d)

    u_true = fem_problem.forward(kappa)
    fem_problem.mesh.plot(u_true)

    # test gradient
    u0 = u_true + 0.1 * np.random.randn(u_true.size)
    kappa_noise = np.random.randn(kappa.size)
    kappa = kappa + 0.5 * kappa_noise/np.linalg.norm(kappa_noise)
    epsilon_list = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
    dJ, errors = verify_gradient(fem_problem, kappa, u0, epsilon_list)
    errors = errors/np.linalg.norm(dJ)

    # print(errors)
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(np.log(epsilon_list), np.log(errors))
    # plt.show()
    
    # solve for kappa
    rate = 0.1
    kappa_est = np.ones(fem_problem.mesh.num_elements)
    kappa_conv = []
    J_conv = []
    for i in range(200):
        print('iteration', i)
        u_kappa, dJ = fem_problem.inverse_step(kappa_est, u_true)
        
        J = np.sum((u_true - u_kappa)**2)
        J_conv.append(J)
        kappa_err = np.sum((kappa - kappa_est)**2)
        kappa_conv.append(kappa_err)
        kappa_est -= rate*dJ
    print(kappa_err.shape)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(J_conv)
    ax2.plot(kappa_conv)
    plt.show()
    print(u_d.shape)

    # I, J = fem_problem.I, fem_problem.J
    # Kprime_submatrix = np.array([
    #     [4, -1, -2, -1],
    #     [-1, 4, -1, -2],
    #     [-2, -1, 4, -1],
    #     [-1, -2, -1, 4]
    # ], dtype=float)
    # kappa = np.ones(fem_problem.mesh.num_elements)
    # k_data = Kprime_submatrix.reshape(-1, 1)*kappa.reshape(1, -1)
    # k_data = k_data.flatten()
    # K = sparse.csr_array((k_data, (fem_problem.I, fem_problem.J)), shape=(fem_problem.num_nodes, fem_problem.num_nodes))
    # dirichlet_nodes = fem_problem.dirichlet_nodes
    # K = K[:,dirichlet_nodes]
    # K = K[dirichlet_nodes,:]
# dJ = np.zeros_like(kappa)
# v = np.zeros()
# for i, u_ in enumerate(u):
#     v[i:self.elem_connectivity[i]] =
#     for j, w_ in enumerate(w):
#         v[]
#         dJ[]
#     v[self.elem_connectivity]
# dJ = v.T@w
# dJ[i] =

# dJ = np.zeros_like(kappa)
# += u_*self.submatrix_data.Kprime_submatrix.flatten()
