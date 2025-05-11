import warnings
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Callable, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt  # TODO: remove matplotlib
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix  # type: ignore
from scipy.sparse.linalg import spsolve  # type: ignore

matplotlib.use("TkAgg")


@dataclass(frozen=True)
class DiscretePlanePartition:
    """Represents a discrete partition of a 2D plane."""

    x0: float  # Starting value of x
    h: float  # Step size along x-axis
    n: int  # Number of steps along x-axis

    y0: float  # Starting value of y
    k: float  # Step size along y-axis
    m: int  # Number of steps along y-axis


BoundaryCondition2D = Callable[[float, float], float]
InternalCondition2D = Callable[[float, float], Tuple[float, bool]]


class BoundaryConditionType(str, Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class BoundaryOrientation(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class BoundaryConditionData:
    """Stores boundary condition type and associated function."""

    condition_type: BoundaryConditionType
    condition: BoundaryCondition2D


class LaplaceSolver:
    """Solves Laplace's differential equation using **Finite difference method** (FDM)"""

    def __init__(self, partition: DiscretePlanePartition):
        self.partition = partition
        self.boundary_conditions: Dict[BoundaryOrientation, List[BoundaryConditionData]] = {
            BoundaryOrientation.LEFT: [],
            BoundaryOrientation.RIGHT: [],
            BoundaryOrientation.TOP: [],
            BoundaryOrientation.BOTTOM: [],
        }
        self.internal_conditions: List[InternalCondition2D] = []

    def add_boundary_condition(self, orientation: BoundaryOrientation, cond: BoundaryConditionData):
        self.boundary_conditions[orientation].append(cond)

    def add_internal_condition(self, cond: InternalCondition2D):
        self.internal_conditions.append(cond)

    @cache
    def __check_internal_conditions(self, x: float, y: float) -> Tuple[float, bool]:
        for cond in self.internal_conditions:
            val, ok = cond(x, y)
            if ok:
                return val, True
        return 0, False

    # TODO: tweak max_iterations and tolerance
    def solve(self, max_iterations: int = 15000, tolerance: float = 1e-6) -> np.ndarray:
        """
        Solves Laplace's equation using the finite difference method with given boundary conditions.

        Args:
            max_iterations: Maximum number of iterations for the solver
            tolerance: Convergence tolerance for the solution

        Returns:
            A 2D numpy array containing the solution values at each grid point
        """
        # Initialize the solution grid
        n, m = self.partition.n, self.partition.m
        u = np.zeros((n + 1, m + 1))  # +1 to include endpoints

        # Apply internal conditions (override any boundary conditions if they conflict)
        print("Applying internal")
        self._apply_internal_conditions(u)
        self._plot_solution(u, "internal")

        # Apply boundary conditions
        print("Applying boundary")
        self._apply_boundary_conditions(u)
        self._plot_solution(u, "boundary")

        # Iterative solution using the finite difference method
        for iteration in range(max_iterations):
            max_diff = 0.0

            # Update interior points using the 5-point stencil
            for i in range(1, n):
                for j in range(1, m):
                    # Skip points that are fixed by internal conditions
                    x = self.partition.x0 + i * self.partition.h
                    y = self.partition.y0 + j * self.partition.k
                    _, is_fixed = self.__check_internal_conditions(x, y)
                    if is_fixed:
                        continue

                    # Store old value before updating
                    old_val = u[i, j]

                    # Standard 5-point stencil for Laplace's equation
                    u[i, j] = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1])

                    # Calculate maximum difference for convergence check
                    current_diff = abs(u[i, j] - old_val)
                    if current_diff > max_diff:
                        max_diff = current_diff

            # Check for convergence
            if max_diff < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

            if iteration % 100 == 0:
                print(f"Iteration {iteration} has max_diff={max_diff} (needed {tolerance})")

        self._plot_solution(u, "final")
        return u

    def solve_sle(self) -> np.ndarray:
        """Solves Laplace's equation using a sparse linear system with singularity handling"""
        n, m = self.partition.n, self.partition.m
        total_points = (n + 1) * (m + 1)

        # Initialize sparse matrix and right-hand side vector
        A = lil_matrix((total_points, total_points))
        b = np.zeros(total_points)

        # Create mapping from grid indices to equation numbers
        index_map = {}
        eq_num = 0
        fixed_values = {}
        has_dirichlet = False

        # First pass: identify fixed points and check for Dirichlet conditions
        for i in range(n + 1):
            for j in range(m + 1):
                x = self.partition.x0 + i * self.partition.h
                y = self.partition.y0 + j * self.partition.k

                # Check internal conditions first (they override boundaries)
                val, is_fixed = self.__check_internal_conditions(x, y)
                if is_fixed:
                    fixed_values[(i, j)] = val
                    continue

                # Check boundary conditions
                is_boundary = False
                if i == 0:  # Left boundary
                    for cond in self.boundary_conditions[BoundaryOrientation.LEFT]:
                        if cond.condition_type == BoundaryConditionType.DIRICHLET:
                            fixed_values[(i, j)] = cond.condition(x, y)
                            has_dirichlet = True
                            is_boundary = True
                            break
                elif i == n:  # Right boundary
                    for cond in self.boundary_conditions[BoundaryOrientation.RIGHT]:
                        if cond.condition_type == BoundaryConditionType.DIRICHLET:
                            fixed_values[(i, j)] = cond.condition(x, y)
                            has_dirichlet = True
                            is_boundary = True
                            break
                if j == 0:  # Bottom boundary
                    for cond in self.boundary_conditions[BoundaryOrientation.BOTTOM]:
                        if cond.condition_type == BoundaryConditionType.DIRICHLET:
                            fixed_values[(i, j)] = cond.condition(x, y)
                            has_dirichlet = True
                            is_boundary = True
                            break
                elif j == m:  # Top boundary
                    for cond in self.boundary_conditions[BoundaryOrientation.TOP]:
                        if cond.condition_type == BoundaryConditionType.DIRICHLET:
                            fixed_values[(i, j)] = cond.condition(x, y)
                            has_dirichlet = True
                            is_boundary = True
                            break

                if not is_boundary:
                    index_map[(i, j)] = eq_num
                    eq_num += 1

        # If no Dirichlet conditions found, add one to fix the solution
        if not has_dirichlet and len(index_map) > 0:
            # Choose an arbitrary point to fix (e.g., first interior point)
            first_point = next(iter(index_map))
            fixed_values[first_point] = 0.0  # Arbitrary value
            del index_map[first_point]
            eq_num -= 1
            warnings.warn("No Dirichlet conditions found. Fixing one point arbitrarily.")

        # Second pass: build equations
        for (i, j), eq in index_map.items():
            A[eq, eq] = 4.0  # Diagonal element

            # Handle neighbors
            for (di, dj), coeff in [((1, 0), -1), ((-1, 0), -1), ((0, 1), -1), ((0, -1), -1)]:
                ni, nj = i + di, j + dj

                if (ni, nj) in fixed_values:
                    b[eq] -= coeff * fixed_values[(ni, nj)]
                elif (ni, nj) in index_map:
                    A[eq, index_map[(ni, nj)]] = coeff
                else:
                    # Neumann boundary condition handling
                    x = self.partition.x0 + i * self.partition.h
                    y = self.partition.y0 + j * self.partition.k
                    h = self.partition.h
                    k = self.partition.k

                    if ni < 0:  # Left Neumann
                        for cond in self.boundary_conditions[BoundaryOrientation.LEFT]:
                            if cond.condition_type == BoundaryConditionType.NEUMANN:
                                b[eq] += h * cond.condition(x, y)
                    elif ni > n:  # Right Neumann
                        for cond in self.boundary_conditions[BoundaryOrientation.RIGHT]:
                            if cond.condition_type == BoundaryConditionType.NEUMANN:
                                b[eq] += h * cond.condition(x, y)
                    elif nj < 0:  # Bottom Neumann
                        for cond in self.boundary_conditions[BoundaryOrientation.BOTTOM]:
                            if cond.condition_type == BoundaryConditionType.NEUMANN:
                                b[eq] += k * cond.condition(x, y)
                    elif nj > m:  # Top Neumann
                        for cond in self.boundary_conditions[BoundaryOrientation.TOP]:
                            if cond.condition_type == BoundaryConditionType.NEUMANN:
                                b[eq] += k * cond.condition(x, y)

        # Convert to CSR format for efficient solving
        A_csr = csr_matrix(A)
        # plot_matrix(A_csr)

        # Solve the system with handling of singular matrices
        try:
            from scipy.sparse.linalg import lsqr

            solution, *_ = lsqr(A_csr, b)

            # solution = spsolve(A_csr, b)
        except Exception as e:
            warnings.warn(f"Linear solver failed: {str(e)}. Trying least squares solution.")
            from scipy.sparse.linalg import lsqr

            solution, *_ = lsqr(A_csr, b)

        # Reconstruct the solution grid
        u = np.zeros((n + 1, m + 1))

        # Fill fixed values first
        for (i, j), val in fixed_values.items():
            u[i, j] = val

        # Fill solved values
        for (i, j), eq in index_map.items():
            u[i, j] = solution[eq]

        return u

    def _apply_boundary_conditions(self, u: np.ndarray):
        """Apply all boundary conditions to the solution grid."""
        n, m = self.partition.n, self.partition.m
        h, k = self.partition.h, self.partition.k

        # Apply conditions for each boundary
        for orientation, conditions in self.boundary_conditions.items():
            for cond_data in conditions:
                if orientation == BoundaryOrientation.LEFT:
                    # Left boundary (x = x0)
                    x = self.partition.x0
                    for j in range(m + 1):
                        y = self.partition.y0 + j * k
                        if cond_data.condition_type == BoundaryConditionType.DIRICHLET:
                            u[0, j] = cond_data.condition(x, y)
                        elif cond_data.condition_type == BoundaryConditionType.NEUMANN:
                            # Approximate Neumann condition using forward difference
                            if m > 0:  # Need at least one interior point
                                u[0, j] = u[1, j] - h * cond_data.condition(x, y)

                elif orientation == BoundaryOrientation.RIGHT:
                    # Right boundary (x = x0 + n*h)
                    x = self.partition.x0 + n * h
                    for j in range(m + 1):
                        y = self.partition.y0 + j * k
                        if cond_data.condition_type == BoundaryConditionType.DIRICHLET:
                            u[n, j] = cond_data.condition(x, y)
                        elif cond_data.condition_type == BoundaryConditionType.NEUMANN:
                            # Approximate Neumann condition using backward difference
                            if n > 0:  # Need at least one interior point
                                u[n, j] = u[n - 1, j] + h * cond_data.condition(x, y)

                elif orientation == BoundaryOrientation.BOTTOM:
                    # Bottom boundary (y = y0)
                    y = self.partition.y0
                    for i in range(n + 1):
                        x = self.partition.x0 + i * h
                        if cond_data.condition_type == BoundaryConditionType.DIRICHLET:
                            u[i, 0] = cond_data.condition(x, y)
                        elif cond_data.condition_type == BoundaryConditionType.NEUMANN:
                            # Approximate Neumann condition using forward difference
                            if m > 0:  # Need at least one interior point
                                u[i, 0] = u[i, 1] - k * cond_data.condition(x, y)

                elif orientation == BoundaryOrientation.TOP:
                    # Top boundary (y = y0 + m*k)
                    y = self.partition.y0 + m * k
                    for i in range(n + 1):
                        x = self.partition.x0 + i * h
                        if cond_data.condition_type == BoundaryConditionType.DIRICHLET:
                            u[i, m] = cond_data.condition(x, y)
                        elif cond_data.condition_type == BoundaryConditionType.NEUMANN:
                            # Approximate Neumann condition using backward difference
                            if m > 0:  # Need at least one interior point
                                u[i, m] = u[i, m - 1] + k * cond_data.condition(x, y)

    def _apply_internal_conditions(self, u: np.ndarray):
        """Apply internal conditions to the solution grid."""
        n, m = self.partition.n, self.partition.m
        h, k = self.partition.h, self.partition.k

        for i in range(n + 1):
            for j in range(m + 1):
                x = self.partition.x0 + i * h
                y = self.partition.y0 + j * k
                val, is_fixed = self.__check_internal_conditions(x, y)
                if is_fixed:
                    u[i, j] = val

    def _plot_solution(self, u: np.ndarray, title: str):
        """Create a heat map visualization of the solution."""
        plt.figure(figsize=(10, 8))

        # Create grid coordinates
        x = np.linspace(
            self.partition.x0, self.partition.x0 + self.partition.n * self.partition.h, self.partition.n + 1
        )
        y = np.linspace(
            self.partition.y0, self.partition.y0 + self.partition.m * self.partition.k, self.partition.m + 1
        )
        X, Y = np.meshgrid(x, y)

        # Plot heat map
        heatmap = plt.pcolormesh(X, Y, u.T, shading="auto")
        plt.colorbar(heatmap, label="Solution Value")

        # Add title and labels
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")  # Одинаковый масштаб по осям

        # Save to file
        plt.savefig(f"tmp/{title}.png")  # Saves in current directory
        plt.close()  # Close the figure to free memory

        # plt.show(block=False)


def plot_matrix(A_csr):
    """Visualizes the sparse matrix structure using matplotlib"""
    plt.figure(figsize=(10, 10))

    # Convert to COO format for plotting
    A_coo = A_csr.tocoo()

    # Plot non-zero entries
    plt.scatter(A_coo.col, A_coo.row, c=A_coo.data, marker="s", s=5, cmap="coolwarm")

    plt.colorbar(label="Value")
    plt.title("Sparse Matrix Structure (Non-zero entries)")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.gca().invert_yaxis()  # To match matrix orientation
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
