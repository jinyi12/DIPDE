"""
Mesh generation and management for 2D FEM.
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import List, Optional


class Mesh:
    """
    Class for managing finite element mesh.

    Provides methods for mesh generation, connectivity, and boundary conditions.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        num_elements_x: int,
        num_elements_y: int,
    ):
        """
        Initialize a rectangular mesh.

        Args:
            x_min: Left boundary of the domain
            x_max: Right boundary of the domain
            y_min: Bottom boundary of the domain
            y_max: Top boundary of the domain
            num_elements_x: Number of elements in x direction
            num_elements_y: Number of elements in y direction
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y

        # Mesh spacing
        self.dx = (x_max - x_min) / num_elements_x
        self.dy = (y_max - y_min) / num_elements_y

        # Generate coordinates and connectivity
        self.coordinates = self._generate_coordinates()
        self.connectivity = self._generate_connectivity()

        # Total number of nodes and elements
        self.num_nodes = self.coordinates.shape[1]
        self.num_elements = self.connectivity.shape[1]

        # Find boundary nodes
        self.boundary_nodes = self._find_boundary_nodes()
        self.dirichlet_nodes = (
            self.boundary_nodes
        )  # By default, all boundary nodes are Dirichlet nodes

        # Generate ID array for assembly
        self.id_array = self._generate_id_array()
        self.num_equations = len(self.id_array[self.id_array != 0])

    def _generate_coordinates(self) -> npt.NDArray[np.float64]:
        """
        Generate nodal coordinates for the mesh.

        Returns:
            Array of shape (2, num_nodes) containing x,y coordinates
        """
        num_nodes_x = self.num_elements_x + 1
        num_nodes_y = self.num_elements_y + 1
        total_nodes = num_nodes_x * num_nodes_y

        coordinates = np.zeros((2, total_nodes))

        for j in range(num_nodes_y):
            for i in range(num_nodes_x):
                node_index = j * num_nodes_x + i
                coordinates[0, node_index] = self.x_min + i * self.dx
                coordinates[1, node_index] = self.y_min + j * self.dy

        return coordinates

    def _generate_connectivity(self) -> npt.NDArray[np.int64]:
        """
        Generate element connectivity matrix.

        Returns:
            Array of shape (4, num_elements) containing node indices for each element
        """
        num_nodes_x = self.num_elements_x + 1
        num_elements = self.num_elements_x * self.num_elements_y

        connectivity = np.zeros((4, num_elements), dtype=np.int64)

        for j in range(self.num_elements_y):
            for i in range(self.num_elements_x):
                elem_index = j * self.num_elements_x + i
                connectivity[0, elem_index] = j * num_nodes_x + i  # Bottom left
                connectivity[1, elem_index] = j * num_nodes_x + i + 1  # Bottom right
                connectivity[2, elem_index] = (j + 1) * num_nodes_x + i + 1  # Top right
                connectivity[3, elem_index] = (j + 1) * num_nodes_x + i  # Top left

        return connectivity

    def _find_boundary_nodes(self) -> List[int]:
        """
        Find all boundary nodes in the mesh.

        Returns:
            List of boundary node indices
        """
        boundary_nodes = []

        for i in range(self.num_nodes):
            x, y = self.coordinates[0, i], self.coordinates[1, i]
            if (
                np.isclose(x, self.x_min)
                or np.isclose(x, self.x_max)
                or np.isclose(y, self.y_min)
                or np.isclose(y, self.y_max)
            ):
                boundary_nodes.append(i)

        return boundary_nodes

    def _generate_id_array(self) -> npt.NDArray[np.int64]:
        """
        Generate ID array for mapping local to global degrees of freedom.

        For Dirichlet boundary conditions, the ID is set to 0.
        For interior nodes, the ID is set to a sequential positive integer.

        Returns:
            Array of shape (num_nodes,) containing mapping indices
        """
        id_array = np.zeros(self.num_nodes, dtype=np.int64)
        equation_counter = 0

        for i in range(self.num_nodes):
            if i in self.dirichlet_nodes:
                id_array[i] = 0
            else:
                equation_counter += 1
                id_array[i] = equation_counter

        return id_array

    def get_element_nodes(self, element_index: int) -> npt.NDArray[np.int64]:
        """
        Get the node indices of an element.

        Args:
            element_index: Element index

        Returns:
            Array of node indices
        """
        return self.connectivity[:, element_index]

    def get_element_coordinates(self, element_index: int) -> tuple:
        """
        Get the coordinates of the nodes of an element.

        Args:
            element_index: Element index

        Returns:
            Tuple (x_coords, y_coords) of node coordinates
        """
        nodes = self.get_element_nodes(element_index)
        x_coords = self.coordinates[0, nodes]
        y_coords = self.coordinates[1, nodes]

        return x_coords, y_coords

    def plot(
        self, values: Optional[npt.NDArray[np.float64]] = None, title: str = "Mesh"
    ) -> None:
        """
        Plot the mesh and optionally nodal values.

        Args:
            values: Nodal values to plot (colormap)
            title: Plot title
        """
        plt.figure(figsize=(10, 8))

        # Plot nodes
        if values is None:
            plt.scatter(
                self.coordinates[0, :],
                self.coordinates[1, :],
                c="blue",
                label="Interior Nodes",
            )
            plt.scatter(
                self.coordinates[0, self.boundary_nodes],
                self.coordinates[1, self.boundary_nodes],
                c="red",
                label="Boundary Nodes",
            )
        else:
            plt.tricontourf(
                self.coordinates[0, :],
                self.coordinates[1, :],
                values,
                20,
                cmap="viridis",
            )
            plt.colorbar(label="Value")

            # Overlay mesh
            for e in range(self.num_elements):
                nodes = self.connectivity[:, e]
                # Close the loop
                nodes = np.append(nodes, nodes[0])
                plt.plot(
                    self.coordinates[0, nodes], self.coordinates[1, nodes], "k-", lw=0.5
                )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        # plt.grid(True)
        if values is None:
            plt.legend()
        plt.axis("equal")
        plt.savefig(f"figures/{title}.png")
        plt.show()
