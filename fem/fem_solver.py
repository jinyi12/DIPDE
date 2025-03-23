"""
Finite Element Method solver for 2D elliptic PDEs.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, Tuple

from .mesh import Mesh
from .element import Element
from .quadrature import QuadratureRule
from .shape_functions import ShapeFunctions


class FEMSolver:
    """
    Class for solving elliptic PDEs using the finite element method.
    
    Provides methods for assembly, solving, and error estimation.
    """
    
    def __init__(self, mesh: Mesh, quadrature_order: int = 2):
        """
        Initialize the FEM solver.
        
        Args:
            mesh: The finite element mesh
            quadrature_order: Order of quadrature rule (default: 2)
        """
        self.mesh = mesh
        self.quadrature = QuadratureRule(quadrature_order)
        self.element = Element(self.quadrature)
        self.solution = None
    
    def assemble(
        self, 
        force_function: Callable[[float, float], float],
        diffusion_tensor: Optional[npt.NDArray[np.float64]] = None
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Assemble the global stiffness matrix and load vector.
        
        Args:
            force_function: Source term function f(x,y)
            diffusion_tensor: Diffusion tensor (default: identity)
            
        Returns:
            Tuple of global stiffness matrix and load vector
        """
        # Initialize global stiffness matrix and load vector
        stiffness_matrix = np.zeros((self.mesh.num_equations, self.mesh.num_equations))
        load_vector = np.zeros(self.mesh.num_equations)
        
        # Loop over all elements
        for e in range(self.mesh.num_elements):
            # Get element node indices
            element_nodes = self.mesh.connectivity[:, e]
            
            # Get element node coordinates
            node_x, node_y = self.mesh.get_element_coordinates(e)
            
            # Compute element stiffness matrix and load vector
            element_stiffness = self.element.stiffness_matrix(node_x, node_y, diffusion_tensor)
            element_load = self.element.load_vector(node_x, node_y, force_function)
            
            # Assemble into global system
            for a in range(4):  # Loop over local nodes
                A = element_nodes[a]  # Global node number
                P = self.mesh.id_array[A]  # Global equation number
                
                if P > 0:  # If not a Dirichlet boundary node
                    # Assemble load vector
                    load_vector[P-1] += element_load[a]
                    
                    # Assemble stiffness matrix
                    for b in range(4):  # Loop over local nodes
                        B = element_nodes[b]  # Global node number
                        Q = self.mesh.id_array[B]  # Global equation number
                        
                        if Q > 0:  # If not a Dirichlet boundary node
                            stiffness_matrix[P-1, Q-1] += element_stiffness[a, b]
        
        return stiffness_matrix, load_vector
    
    def solve(
        self, 
        force_function: Callable[[float, float], float],
        diffusion_tensor: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """
        Solve the PDE system.
        
        Args:
            force_function: Source term function f(x,y)
            diffusion_tensor: Diffusion tensor (default: identity)
            
        Returns:
            Solution vector at all nodes
        """
        # Assemble the system
        stiffness_matrix, load_vector = self.assemble(force_function, diffusion_tensor)
        
        # Solve the system
        u_reduced = np.linalg.solve(stiffness_matrix, load_vector)
        
        # Expand solution to include Dirichlet nodes
        self.solution = np.zeros(self.mesh.num_nodes)
        
        for i in range(self.mesh.num_nodes):
            eq_num = self.mesh.id_array[i]
            if eq_num > 0:
                self.solution[i] = u_reduced[eq_num-1]
            # else: already zero (homogeneous Dirichlet BC)
        
        return self.solution
    
    def evaluate_solution(self, xi: float, eta: float, element_index: int) -> float:
        """
        Evaluate the solution at a point within an element.
        
        Args:
            xi: Local coordinate in xi direction (-1 to 1)
            eta: Local coordinate in eta direction (-1 to 1)
            element_index: Element index
            
        Returns:
            Solution value at the specified point
        """
        if self.solution is None:
            raise ValueError("Solution not available. Call solve() first.")
        
        # Get element node indices
        element_nodes = self.mesh.connectivity[:, element_index]
        
        # Get element solution values
        element_solution = self.solution[element_nodes]
        
        # Evaluate shape functions at (xi, eta)
        N = ShapeFunctions.N(xi, eta)
        
        # Interpolate solution
        return np.dot(N, element_solution)
    
    def evaluate_error(
        self,
        exact_solution: Callable[[float, float], float],
        exact_gradient: Optional[Callable[[float, float], npt.NDArray[np.float64]]] = None
    ) -> Tuple[float, float]:
        """
        Evaluate the L2 and H1 error norms.
        
        Args:
            exact_solution: Exact solution function u(x,y)
            exact_gradient: Exact gradient function [du/dx, du/dy](x,y)
            
        Returns:
            Tuple of L2 and H1 error norms
        """
        if self.solution is None:
            raise ValueError("Solution not available. Call solve() first.")
        
        l2_error_squared = 0.0
        h1_error_squared = 0.0
        
        # Get quadrature points and weights
        q_points = self.quadrature.get_points()
        q_weights = self.quadrature.get_weights()
        
        # Loop over all elements
        for e in range(self.mesh.num_elements):
            # Get element node indices
            element_nodes = self.mesh.connectivity[:, e]
            
            # Get element node coordinates
            node_x, node_y = self.mesh.get_element_coordinates(e)
            
            # Get element solution values
            element_solution = self.solution[element_nodes]
            
            # Loop over quadrature points
            for i, xi in enumerate(q_points):
                for j, eta in enumerate(q_points):
                    # Map to physical coordinates
                    physical_point = ShapeFunctions.map_to_physical(xi, eta, node_x, node_y)
                    x, y = physical_point[0], physical_point[1]
                    
                    # Compute shape functions and derivatives
                    N = ShapeFunctions.N(xi, eta)
                    
                    # Compute Jacobian and transformation matrix
                    transform_matrix, det_J = ShapeFunctions.gradient_transformation_matrix(
                        xi, eta, node_x, node_y
                    )
                    
                    # Compute FEM solution at this point
                    u_h = np.dot(N, element_solution)
                    
                    # Compute exact solution at this point
                    u_exact = exact_solution(x, y)
                    
                    # Compute L2 error contribution
                    l2_error_squared += (u_exact - u_h)**2 * det_J * q_weights[i] * q_weights[j]
                    
                    # Compute H1 error if exact gradient is provided
                    if exact_gradient is not None:
                        # Compute FEM gradient at this point
                        dN_dxi = ShapeFunctions.dN_dxi(eta)
                        dN_deta = ShapeFunctions.dN_deta(xi)
                        grad_N_ref = np.array([dN_dxi, dN_deta]).T
                        grad_N_physical = grad_N_ref @ transform_matrix
                        grad_u_h = np.dot(grad_N_physical.T, element_solution)
                        
                        # Compute exact gradient at this point
                        grad_u_exact = exact_gradient(x, y)
                        
                        # Compute H1 error contribution
                        gradient_error = grad_u_exact - grad_u_h
                        h1_error_squared += np.dot(gradient_error, gradient_error) * det_J * q_weights[i] * q_weights[j]
        
        l2_error = np.sqrt(l2_error_squared)
        h1_error = np.sqrt(l2_error_squared + h1_error_squared)
        
        return l2_error, h1_error
