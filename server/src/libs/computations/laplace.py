from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import lil_matrix  # type: ignore
from scipy.sparse.linalg import spsolve  # type: ignore


@dataclass(frozen=True)
class DiscretePlanePartition:
    """Represents a discrete partition of a 2D plane."""

    Lx: float  # Starting value of x
    Nx: int  # Step size along x-axis

    Ly: float  # Starting value of y
    Ny: int  # Step size along y-axis


BoundaryCondition1D = Callable[[float], float]  # Accept 1 agrument, since second dimension is fixed
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

    type_: BoundaryConditionType
    function: BoundaryCondition1D


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

    def _apply_boundary_conditions(self, u: np.ndarray) -> np.ndarray:
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

    def _apply_internal_conditions(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def solve(self) -> np.ndarray:
        """Solve Laplace's equation with the given boundary and internal conditions."""
        Ny = self.partition.Ny
        Nx = self.partition.Nx
        dx = self.partition.Lx / (Nx - 1)
        dy = self.partition.Ly / (Ny - 1)

        # Initialize solution array
        u = np.zeros((Ny, Nx))

        # Apply boundary conditions
        u = self._apply_boundary_conditions(u)

        # Apply internal conditions and get fixed mask
        u, fixed_mask = self._apply_internal_conditions(u)

        # Count number of unknown points (all internal points minus fixed ones)
        N = (Ny - 2) * (Nx - 2) - np.sum(fixed_mask[1:-1, 1:-1])

        # Create mapping from grid indices to matrix indices
        index_map = np.zeros((Ny, Nx), dtype=int)
        current_idx = 0
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                if not fixed_mask[i, j]:
                    index_map[i, j] = current_idx
                    current_idx += 1
                else:
                    index_map[i, j] = -1

        # Build sparse matrix and right-hand side
        A = lil_matrix((N, N))
        rhs = np.zeros(N)

        current_idx = 0
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                if fixed_mask[i, j]:
                    continue

                # Main diagonal element
                A[current_idx, current_idx] = -2 / dx**2 - 2 / dy**2

                # x-direction neighbors
                for di, dj in [(0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if fixed_mask[ni, nj]:
                        rhs[current_idx] -= u[ni, nj] / dx**2
                    elif nj == 0 or nj == Nx - 1:
                        rhs[current_idx] -= u[ni, nj] / dx**2
                    else:
                        neighbor_idx = index_map[ni, nj]
                        A[current_idx, neighbor_idx] = 1 / dx**2

                # y-direction neighbors
                for di, dj in [(1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if fixed_mask[ni, nj]:
                        rhs[current_idx] -= u[ni, nj] / dy**2
                    elif ni == 0 or ni == Ny - 1:
                        rhs[current_idx] -= u[ni, nj] / dy**2
                    else:
                        neighbor_idx = index_map[ni, nj]
                        A[current_idx, neighbor_idx] = 1 / dy**2

                current_idx += 1

        # Solve the system
        solution = spsolve(A.tocsr(), rhs)

        # Map solution back to grid
        current_idx = 0
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                if not fixed_mask[i, j]:
                    u[i, j] = solution[current_idx]
                    current_idx += 1

        return u
