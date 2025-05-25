import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from libs.computations.gradient import gradient_vectors


@dataclass(frozen=True)
class DiscretePlanePartition:
    """Represents a discrete partition of a 2D plane."""

    Lx: float  # Length along x-axis
    Nx: int  # Number of points along x-axis
    Ly: float  # Length along y-axis
    Ny: int  # Number of points along y-axis

    @property
    def dx(self) -> float:
        return self.Lx / (self.Nx - 1)

    @property
    def dy(self) -> float:
        return self.Ly / (self.Ny - 1)


BoundaryCondition1D = Callable[[float], float]
InternalCondition2D = Callable[[float, float], Tuple[float, bool]]


class BoundaryConditionType(str, Enum):
    DIRICHLET = "dirichlet"


class BoundaryOrientation(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class BoundaryConditionData:
    """Stores boundary condition type and associated function."""

    type_: BoundaryConditionType
    function: BoundaryCondition1D


@dataclass(frozen=True)
class Conductor:
    """Represents a conductor with unknown but uniform potential."""

    mask: NDArray[np.bool_]  # Boolean mask (True where conductor exists)
    initial_guess: float = 0.0  # Initial guess for potential


class LaplaceSolver:
    """Solves Laplace's equation with conductors using matrix method."""

    def __init__(self, partition: DiscretePlanePartition):
        self.partition = partition
        self.boundary_conditions: Dict[BoundaryOrientation, List[BoundaryConditionData]] = {
            BoundaryOrientation.LEFT: [],
            BoundaryOrientation.RIGHT: [],
            BoundaryOrientation.TOP: [],
            BoundaryOrientation.BOTTOM: [],
        }
        self.internal_conditions: List[InternalCondition2D] = []
        self.conductors: List[Conductor] = []

    def add_boundary_condition(
        self, orientation: BoundaryOrientation, cond: BoundaryConditionData
    ) -> None:
        self.boundary_conditions[orientation].append(cond)

    def add_internal_condition(self, cond: InternalCondition2D) -> None:
        self.internal_conditions.append(cond)

    def add_conductor(self, conductor: Conductor) -> None:
        self.conductors.append(conductor)

    def _apply_boundary_conditions(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply all registered boundary conditions to the solution array."""
        Ny, Nx = u.shape
        y = np.linspace(0, self.partition.Ly, Ny)
        x = np.linspace(0, self.partition.Lx, Nx)

        # Apply left boundary conditions
        for cond in self.boundary_conditions[BoundaryOrientation.LEFT]:
            if cond.type_ == BoundaryConditionType.DIRICHLET:
                f = np.vectorize(cond.function)
                u[:, 0] = f(y)

        # Apply right boundary conditions
        for cond in self.boundary_conditions[BoundaryOrientation.RIGHT]:
            if cond.type_ == BoundaryConditionType.DIRICHLET:
                f = np.vectorize(cond.function)
                u[:, -1] = f(y)

        # Apply top boundary conditions
        for cond in self.boundary_conditions[BoundaryOrientation.TOP]:
            if cond.type_ == BoundaryConditionType.DIRICHLET:
                f = np.vectorize(cond.function)
                u[-1, :] = f(x)

        # Apply bottom boundary conditions
        for cond in self.boundary_conditions[BoundaryOrientation.BOTTOM]:
            if cond.type_ == BoundaryConditionType.DIRICHLET:
                f = np.vectorize(cond.function)
                u[0, :] = f(x)

        return u

    def _apply_internal_conditions(
        self, u: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Apply all registered internal conditions and return the solution array and fixed mask."""
        Ny, Nx = u.shape
        fixed_mask = np.zeros((Ny, Nx), dtype=bool)
        y = np.linspace(0, self.partition.Ly, Ny)
        x = np.linspace(0, self.partition.Lx, Nx)

        for cond in self.internal_conditions:
            for i in range(Ny):
                for j in range(Nx):
                    value, is_hit = cond(x[j], y[i])
                    if is_hit:
                        u[i, j] = value
                        fixed_mask[i, j] = True

        return u, fixed_mask

    def solve(self) -> NDArray[np.float64]:
        """Solve Laplace's equation with conductors using matrix method."""
        Nx = self.partition.Nx
        Ny = self.partition.Ny
        dx = self.partition.dx
        dy = self.partition.dy

        # Initialize solution array
        u: NDArray[np.float64] = np.zeros((Ny, Nx))

        # Apply boundary conditions
        u = self._apply_boundary_conditions(u)

        # Apply internal conditions and get fixed mask
        u, fixed_mask = self._apply_internal_conditions(u)

        # Get conductor masks
        conductor_masks = [cond.mask for cond in self.conductors]
        num_conductors = len(self.conductors)

        # Combine all fixed points (internal conditions + conductors)
        for cond_mask in conductor_masks:
            fixed_mask = np.logical_or(fixed_mask, cond_mask)

        # Create index mapping:
        # - Regular points get unique indices
        # - Conductor points reference their phi_cond variables
        index_map = np.full((Ny, Nx), -1, dtype=int)
        current_idx = 0

        # Index regular points (not fixed, not in conductors)
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                if not fixed_mask[i, j]:
                    index_map[i, j] = current_idx
                    current_idx += 1

        # Index conductor points (each conductor gets one phi_cond variable)
        conductor_indices = []
        phi_cond_start_idx = current_idx
        for cond_idx, cond_mask in enumerate(conductor_masks):
            cond_points = np.argwhere(cond_mask)
            for i, j in cond_points:
                index_map[i, j] = phi_cond_start_idx + cond_idx
            conductor_indices.append(phi_cond_start_idx + cond_idx)

        # Total number of unknowns:
        # - Regular points
        # - Conductor potentials (one per conductor)
        total_unknowns = current_idx + num_conductors

        # Build sparse matrix and right-hand side
        A = lil_matrix((total_unknowns, total_unknowns))
        b = np.zeros(total_unknowns)

        # 1. Laplace equations for regular points
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                row = index_map[i, j]
                if row < 0:
                    continue  # Fixed point (boundary or internal condition)

                # Skip if this is a conductor potential variable
                if row >= phi_cond_start_idx:
                    continue

                # Laplace operator: (u[i+1,j] + u[i-1,j] - 2u[i,j])/dy² + (u[i,j+1] + u[i,j-1] - 2u[i,j])/dx² = 0
                A[row, row] = -2 / dx**2 - 2 / dy**2

                # x-direction neighbors
                for di, dj in [(0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    neighbor_idx = index_map[ni, nj]
                    if neighbor_idx >= 0:
                        A[row, neighbor_idx] = 1 / dx**2
                    else:  # Fixed value (boundary or internal condition)
                        b[row] -= u[ni, nj] / dx**2

                # y-direction neighbors
                for di, dj in [(1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    neighbor_idx = index_map[ni, nj]
                    if neighbor_idx >= 0:
                        A[row, neighbor_idx] = 1 / dy**2
                    else:  # Fixed value
                        b[row] -= u[ni, nj] / dy**2

        # 2. Conductor equations: all points in conductor have same potential
        for cond_idx, cond_mask in enumerate(conductor_masks):
            cond_points = np.argwhere(cond_mask)
            phi_cond_var = phi_cond_start_idx + cond_idx

            for i, j in cond_points:
                row = index_map[i, j]
                A[row, row] = 1
                A[row, phi_cond_var] = -1
                b[row] = 0

        # 3. Zero net charge condition for each conductor
        for cond_idx, cond_mask in enumerate(conductor_masks):
            cond_points = np.argwhere(cond_mask)
            phi_cond_var = phi_cond_start_idx + cond_idx
            row = phi_cond_start_idx + cond_idx

            # Approximate surface charge density (normal derivative)
            for i, j in cond_points:
                # Check all 4 neighbors
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= Ny or nj < 0 or nj >= Nx:
                        continue  # Skip out-of-bounds

                    if not cond_mask[ni, nj]:  # Point outside conductor
                        neighbor_idx = index_map[ni, nj]
                        if neighbor_idx >= 0:
                            A[row, neighbor_idx] += 1
                        A[row, phi_cond_var] -= 1

        # Solve the system
        solution = spsolve(A.tocsr(), b)

        # Map solution back to grid
        u = np.zeros((Ny, Nx))
        for i in range(Ny):
            for j in range(Nx):
                idx = index_map[i, j]
                if idx >= 0:
                    u[i, j] = solution[idx]

        return u


def __uniform_condition(
    x_max: float,
    potential_low: float,
    potential_high: float,
) -> BoundaryCondition1D:
    def cond(x: float) -> float:
        return potential_low + (x / x_max) * (potential_high - potential_low)

    return cond


def __save_electric_lines_plot(
    potential: NDArray[np.float64],
    width: float,
    height: float,
    filename: str,
) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    NX = 250
    NY = 250

    Ex, Ey = gradient_vectors(potential, width / NX, height / NY)

    x_grid = np.linspace(0, width, NX)
    y_grid = np.linspace(0, height, NY)

    X, Y = np.meshgrid(x_grid, y_grid)

    fig, ax = plt.subplots(figsize=(15, 10))

    # Original electric field and potential plots
    ax.streamplot(
        X,
        Y,
        -Ex,
        -Ey,
        color="red",
        density=(0.25, 0.8),
        linewidth=1,
        broken_streamlines=False,
    )
    ax.contour(X, Y, potential, levels=20, colors="gray", alpha=0.5)

    # Remove axes and borders
    ax.set_axis_off()
    ax.set_position((0, 0, 1, 1))

    # Save with high quality
    plt.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True)
    plt.close()


def add_circular_conductor(solver, center_x, center_y, radius):  # type: ignore
    """
    Добавляет круглый проводник без предположения о потенциале
    Потенциал будет определен автоматически из решения
    """
    partition = solver.partition
    x = np.linspace(0, partition.Lx, partition.Nx)
    y = np.linspace(0, partition.Ly, partition.Ny)
    X, Y = np.meshgrid(x, y)

    # Создаем маску проводника
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    cond_mask = dist <= radius

    # Добавляем проводник без указания потенциала
    solver.add_conductor(Conductor(cond_mask))


# Example usage
if __name__ == "__main__":
    # Create grid
    partition = DiscretePlanePartition(Lx=30.0, Nx=250, Ly=20.0, Ny=250)

    # Initialize solver
    solver = LaplaceSolver(partition)

    left_potential = 15
    right_potential = 25

    # Set boundary conditions (0 on left, 1 on right)
    solver.add_boundary_condition(
        BoundaryOrientation.LEFT,
        BoundaryConditionData(BoundaryConditionType.DIRICHLET, lambda y: left_potential),
    )
    solver.add_boundary_condition(
        BoundaryOrientation.RIGHT,
        BoundaryConditionData(BoundaryConditionType.DIRICHLET, lambda y: right_potential),
    )
    solver.add_boundary_condition(
        BoundaryOrientation.TOP,
        BoundaryConditionData(
            BoundaryConditionType.DIRICHLET,
            __uniform_condition(partition.Lx, left_potential, right_potential),
        ),
    )
    solver.add_boundary_condition(
        BoundaryOrientation.BOTTOM,
        BoundaryConditionData(
            BoundaryConditionType.DIRICHLET,
            __uniform_condition(partition.Lx, left_potential, right_potential),
        ),
    )

    # # Add a conductor (square in the center)
    # cond_mask = np.zeros((partition.Ny, partition.Nx), dtype=bool)
    # cond_mask[100:150, 100:150] = True
    # solver.add_conductor(Conductor(cond_mask, initial_guess=7))

    # Add a conductor (circle in the center)
    add_circular_conductor(solver, center_x=partition.Lx / 2, center_y=partition.Ly / 2, radius=4)

    # Solve and plot
    solution = solver.solve()

    __save_electric_lines_plot(
        potential=solution,
        width=partition.Lx,
        height=partition.Ly,
        filename="tmp/test.png",
    )

    plt.imshow(solution, cmap="viridis", origin="lower")
    plt.colorbar(label="Potential")
    plt.title("Potential distribution with conductor")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
