"""
Shape functions for quadrilateral elements in 2D FEM.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple


class ShapeFunctions:
    """
    Implementation of shape functions for bilinear quadrilateral elements.
    
    Coordinates are assumed to be:
    (-1,-1), (1,-1), (1,1), (-1,1) in the reference element.
    """
    
    @staticmethod
    def N(xi: float, eta: float) -> npt.NDArray[np.float64]:
        """
        Evaluate shape functions at a point in the reference element.
        
        Args:
            xi: Local coordinate in xi direction (-1 to 1)
            eta: Local coordinate in eta direction (-1 to 1)
            
        Returns:
            Array of shape function values [N1, N2, N3, N4]
        """
        if not (-1 <= xi <= 1 and -1 <= eta <= 1):
            return np.zeros(4)
            
        xi_a = np.array([-1, 1, 1, -1])
        eta_a = np.array([-1, -1, 1, 1])
        
        return (1/4) * (1 + xi_a * xi) * (1 + eta_a * eta)
    
    @staticmethod
    def dN_dxi(eta: float) -> npt.NDArray[np.float64]:
        """
        Evaluate derivatives of shape functions with respect to xi.
        
        Args:
            eta: Local coordinate in eta direction (-1 to 1)
            
        Returns:
            Array of derivatives [dN1/dxi, dN2/dxi, dN3/dxi, dN4/dxi]
        """
        if not -1 <= eta <= 1:
            return np.zeros(4)
            
        xi_a = np.array([-1, 1, 1, -1])
        eta_a = np.array([-1, -1, 1, 1])
        
        return (1/4) * xi_a * (1 + eta_a * eta)
    
    @staticmethod
    def dN_deta(xi: float) -> npt.NDArray[np.float64]:
        """
        Evaluate derivatives of shape functions with respect to eta.
        
        Args:
            xi: Local coordinate in xi direction (-1 to 1)
            
        Returns:
            Array of derivatives [dN1/deta, dN2/deta, dN3/deta, dN4/deta]
        """
        if not -1 <= xi <= 1:
            return np.zeros(4)
            
        xi_a = np.array([-1, 1, 1, -1])
        eta_a = np.array([-1, -1, 1, 1])
        
        return (1/4) * eta_a * (1 + xi_a * xi)
    
    @staticmethod
    def map_to_physical(xi: float, eta: float, x_nodes: npt.NDArray[np.float64], 
                        y_nodes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Map coordinates from reference element to physical element.
        
        Args:
            xi: Local coordinate in xi direction (-1 to 1)
            eta: Local coordinate in eta direction (-1 to 1)
            x_nodes: x-coordinates of the element nodes
            y_nodes: y-coordinates of the element nodes
            
        Returns:
            Array [x, y] of physical coordinates
        """
        N = ShapeFunctions.N(xi, eta)
        x = np.dot(N, x_nodes)
        y = np.dot(N, y_nodes)
        
        return np.array([x, y])
    
    @staticmethod
    def jacobian(xi: float, eta: float, x_nodes: npt.NDArray[np.float64], 
                y_nodes: npt.NDArray[np.float64]) -> float:
        """
        Calculate the Jacobian determinant of the mapping.
        
        Args:
            xi: Local coordinate in xi direction (-1 to 1)
            eta: Local coordinate in eta direction (-1 to 1)
            x_nodes: x-coordinates of the element nodes
            y_nodes: y-coordinates of the element nodes
            
        Returns:
            Determinant of the Jacobian matrix
        """
        dN_dxi = ShapeFunctions.dN_dxi(eta)
        dN_deta = ShapeFunctions.dN_deta(xi)
        
        dx_dxi = np.dot(dN_dxi, x_nodes)
        dx_deta = np.dot(dN_deta, x_nodes)
        dy_dxi = np.dot(dN_dxi, y_nodes)
        dy_deta = np.dot(dN_deta, y_nodes)
        
        return dx_dxi * dy_deta - dx_deta * dy_dxi
    
    @staticmethod
    def gradient_transformation_matrix(xi: float, eta: float, 
                                     x_nodes: npt.NDArray[np.float64], 
                                     y_nodes: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Calculate the transformation matrix for converting gradients from reference to physical space.
        
        Args:
            xi: Local coordinate in xi direction (-1 to 1)
            eta: Local coordinate in eta direction (-1 to 1)
            x_nodes: x-coordinates of the element nodes
            y_nodes: y-coordinates of the element nodes
            
        Returns:
            Tuple containing:
                - Transformation matrix for gradients
                - Determinant of the Jacobian matrix
        """
        dN_dxi = ShapeFunctions.dN_dxi(eta)
        dN_deta = ShapeFunctions.dN_deta(xi)
        
        dx_dxi = np.dot(dN_dxi, x_nodes)
        dx_deta = np.dot(dN_deta, x_nodes)
        dy_dxi = np.dot(dN_dxi, y_nodes)
        dy_deta = np.dot(dN_deta, y_nodes)
        
        det_J = dx_dxi * dy_deta - dx_deta * dy_dxi
        
        if abs(det_J) < 1e-10:
            raise ValueError("Jacobian determinant is near zero, element may be degenerate")
        
        # Matrix relating [dxi/dx, dxi/dy; deta/dx, deta/dy] to convert gradients
        inv_J_transpose = (1.0/det_J) * np.array([
            [dy_deta, -dx_deta],
            [-dy_dxi, dx_dxi]
        ])
        
        return inv_J_transpose, det_J