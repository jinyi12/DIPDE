"""
Quadrature rules for numerical integration in FEM.
"""

import numpy as np
import numpy.typing as npt
from scipy import integrate
from typing import Tuple


class QuadratureRule:
    """
    Gaussian quadrature rule for numerical integration over the reference element [-1,1]×[-1,1].
    """

    def __init__(self, order: int = 2):
        """
        Initialize quadrature rule of given order.

        Args:
            order: Order of the quadrature rule (1, 2, or 3)
        """
        if order not in [1, 2, 3]:
            raise ValueError("Quadrature order must be 1, 2, or 3")

        self.order = order

        # Precomputed Gauss points and weights
        gauss_rules = {
            1: (np.array([0.0]), np.array([2.0])),
            2: (np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]), np.array([1.0, 1.0])),
            3: (
                np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]),
                np.array([5 / 9, 8 / 9, 5 / 9]),
            ),
        }

        self.points, self.weights = gauss_rules[order]

    def get_points(self) -> npt.NDArray[np.float64]:
        """
        Get quadrature points.

        Returns:
            Array of quadrature points in [-1,1]
        """
        return self.points

    def get_weights(self) -> npt.NDArray[np.float64]:
        """
        Get quadrature weights.

        Returns:
            Array of quadrature weights
        """
        return self.weights

    def integrate_1d(self, f, a: float = -1.0, b: float = 1.0) -> float:
        """
        Integrate 1D function over [a,b] using this quadrature rule.

        Args:
            f: Function to integrate
            a: Lower integration bound
            b: Upper integration bound

        Returns:
            Approximated integral value
        """
        # Transform quadrature points from [-1,1] to [a,b]
        transformed_points = 0.5 * (b - a) * self.points + 0.5 * (a + b)

        # Scale weights
        scaled_weights = 0.5 * (b - a) * self.weights

        # Evaluate function at transformed points and compute weighted sum
        return np.sum(f(transformed_points) * scaled_weights)

    def integrate_2d(self, f) -> float:
        """
        Integrate 2D function over reference element [-1,1]×[-1,1].

        Args:
            f: Function taking (xi, eta) and returning scalar

        Returns:
            Approximated integral value
        """
        result = 0.0

        for i, xi in enumerate(self.points):
            for j, eta in enumerate(self.points):
                result += f(xi, eta) * self.weights[i] * self.weights[j]

        return result

    @staticmethod
    def integrate_with_scipy(f, a: float = -1.0, b: float = 1.0, **kwargs) -> float:
        """
        Integrate 1D function using scipy's quadrature.

        Args:
            f: Function to integrate
            a: Lower integration bound
            b: Upper integration bound
            **kwargs: Additional arguments to pass to scipy.integrate.quad

        Returns:
            Approximated integral value
        """
        result, _ = integrate.quad(f, a, b, **kwargs)
        return result

    @staticmethod
    def integrate_2d_with_scipy(
        f, ranges: Tuple[Tuple[float, float], Tuple[float, float]], **kwargs
    ) -> float:
        """
        Integrate 2D function using scipy's dblquad.

        Args:
            f: Function taking (x, y) and returning scalar
            ranges: ((x_min, x_max), (y_min, y_max)) integration bounds
            **kwargs: Additional arguments to pass to scipy.integrate.dblquad

        Returns:
            Approximated integral value
        """
        (x_min, x_max), (y_min, y_max) = ranges

        def y_bounds(x):
            return y_min, y_max

        result, _ = integrate.dblquad(
            lambda y, x: f(x, y), x_min, x_max, y_bounds, **kwargs
        )
        return result
