"""
Element-level computations for 2D FEM.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable, Optional

from .shape_functions import ShapeFunctions
from .quadrature import QuadratureRule


class Element:
    """
    Class for element-level computations in FEM.
    """
    
    def __init__(self, quadrature: QuadratureRule):
        """
        Initialize element with quadrature rule.
        
        Args:
            quadrature: Quadrature rule for numerical integration
        """
        self.quadrature = quadrature
        self.shape_functions = ShapeFunctions()
    
    def stiffness_matrix(
        self, 
        node_x: npt.NDArray[np.float64], 
        node_y: npt.NDArray[np.float64], 
        diffusion_tensor: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """
        Compute element stiffness matrix.
        
        Args:
            node_x: x-coordinates of the element nodes
            node_y: y-coordinates of the element nodes
            diffusion_tensor: Diffusion tensor (default: identity matrix)
            
        Returns:
            Element stiffness matrix (4x4)
        """
        if diffusion_tensor is None:
            diffusion_tensor = np.eye(2)  # Identity matrix for isotropic case
        
        # Initialize stiffness matrix
        k_element = np.zeros((4, 4))
        
        # Get quadrature points and weights
        q_points = self.quadrature.get_points()
        q_weights = self.quadrature.get_weights()
        
        # Loop over quadrature points
        for i, xi in enumerate(q_points):
            for j, eta in enumerate(q_points):
                # Compute shape function derivatives
                dN_dxi = ShapeFunctions.dN_dxi(eta)
                dN_deta = ShapeFunctions.dN_deta(xi)
                
                # Compute transformation matrix and Jacobian determinant
                transform_matrix, det_J = ShapeFunctions.gradient_transformation_matrix(
                    xi, eta, node_x, node_y
                )
                
                # Compute gradients in physical space for each shape function
                B = np.zeros((2, 4))
                for a in range(4):
                    grad_N_ref = np.array([dN_dxi[a], dN_deta[a]])
                    B[:, a] = transform_matrix @ grad_N_ref
                
                # Compute contribution to stiffness matrix
                k_element += B.T @ diffusion_tensor @ B * det_J * q_weights[i] * q_weights[j]
        
        return k_element
    
    def load_vector(
        self, 
        node_x: npt.NDArray[np.float64], 
        node_y: npt.NDArray[np.float64], 
        force_function: Callable[[float, float], float]
    ) -> npt.NDArray[np.float64]:
        """
        Compute element load vector.
        
        Args:
            node_x: x-coordinates of the element nodes
            node_y: y-coordinates of the element nodes
            force_function: Source term function f(x,y)
            
        Returns:
            Element load vector (4,)
        """
        # Initialize load vector
        f_element = np.zeros(4)
        
        # Get quadrature points and weights
        q_points = self.quadrature.get_points()
        q_weights = self.quadrature.get_weights()
        
        # Loop over quadrature points
        for i, xi in enumerate(q_points):
            for j, eta in enumerate(q_points):
                # Map to physical coordinates
                physical_point = ShapeFunctions.map_to_physical(xi, eta, node_x, node_y)
                x, y = physical_point[0], physical_point[1]
                
                # Compute shape functions
                N = ShapeFunctions.N(xi, eta)
                
                # Compute Jacobian determinant
                det_J = ShapeFunctions.jacobian(xi, eta, node_x, node_y)
                
                # Compute force at this point
                force_value = force_function(x, y)
                
                # Add contribution to load vector
                f_element += N * force_value * det_J * q_weights[i] * q_weights[j]
        
        return f_element
