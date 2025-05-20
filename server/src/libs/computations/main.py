import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from libs.computations.gradient import gradient_magnitude, gradient_vectors
from libs.computations.laplace import (
    BoundaryConditionData,
    BoundaryConditionType,
    BoundaryOrientation,
    DiscretePlanePartition,
    InternalCondition2D,
    LaplaceSolver,
)
from libs.shapes.arrow import Arrow
from libs.shapes.core.shape import Shape
from libs.shapes.rect import Rect
from libs.shapes.ring import Ring


def generate_internal_condition(shape: Shape, potential: float) -> InternalCondition2D:
    def cond(x: float, y: float) -> Tuple[float, bool]:
        is_inside_shape = shape.check_surface(x, y)
        if is_inside_shape:
            value = potential
        else:
            value = 0

        return value, is_inside_shape

    return cond


def left_electrode_condition(y: float) -> float:
    return 0


def right_electrode_condition(y: float) -> float:
    ELECTRODE_Y_LOWER = 3
    ELECTRODE_Y_UPPER = 17

    potential = 14  # Из методички - 14 [В]

    if ELECTRODE_Y_LOWER <= y <= ELECTRODE_Y_UPPER:
        return potential

    return 0


def zero_dirichlet(_: float) -> float:
    return 0


def apply_electrodes_potential(x: float, y: float, shape: Shape) -> float:
    if shape.check_surface(x, y):
        return 0

    return 0 + (x / 100) * 47.62


def _plot_field(potential, width, height):
    Ex, Ey = gradient_vectors(potential, width / NX, height / NY)
    x_grid = np.linspace(0, width, NX)
    y_grid = np.linspace(0, height, NY)

    X, Y = np.meshgrid(x_grid, y_grid)
    plt.figure(figsize=(25, 17.5))

    plt.streamplot(X, Y, Ex, Ey, color="red", density=2, linewidth=1)
    plt.contour(X, Y, potential, levels=20, colors="gray")
    plt.savefig("tmp/electric_lines.png")


def _plot_solution(self: LaplaceSolver, u: np.ndarray, title: str) -> None:
    """Create a heat map visualization of the solution."""
    plt.figure(figsize=(10, 8))

    # Create grid coordinatesc
    x = np.linspace(0, self.partition.Lx, self.partition.Nx)
    y = np.linspace(0, self.partition.Ly, self.partition.Ny)
    X, Y = np.meshgrid(x, y)

    # Plot heat map
    # clipped = np.clip(u.T, 0, 30)
    # heatmap = plt.pcolormesh(X, Y, clipped, shading="auto")
    heatmap = plt.pcolormesh(X, Y, u, shading="auto")
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


if __name__ == "__main__":
    # Partition
    WIDTH = 30  # cm
    HEIGHT = 20  # cm

    NX = 500
    NY = 500

    partition = DiscretePlanePartition(WIDTH, NX, HEIGHT, NY)

    # Setup solver
    solver = LaplaceSolver(partition)

    # Internal conditions
    arrow = Arrow(WIDTH / 2, HEIGHT / 2, height=6, length=8, angle=math.pi / 4)
    ring = Ring(WIDTH / 2, HEIGHT / 2, inner_radius=0, outer_radius=6)
    rect = Rect(WIDTH / 2, HEIGHT / 2, a=20, b=2)
    shape_potential = 7.35

    shape_condition_ring = generate_internal_condition(ring, shape_potential)
    solver.add_internal_condition(shape_condition_ring)

    # shape_condition_arrow = generate_internal_condition/(arrow, shape_potential)
    # solver.add_internal_condition(shape_condition_arrow)

    # shape_condition_rect = generate_internal_condition(rect, shape_potential)
    # solver.add_internal_condition(shape_condition_rect)

    # Boudndary conditions
    solver.add_boundary_condition(
        BoundaryOrientation.TOP,
        BoundaryConditionData(
            BoundaryConditionType.DIRICHLET,
            zero_dirichlet,
        ),
    )

    solver.add_boundary_condition(
        BoundaryOrientation.BOTTOM,
        BoundaryConditionData(
            BoundaryConditionType.DIRICHLET,
            zero_dirichlet,
        ),
    )

    solver.add_boundary_condition(
        BoundaryOrientation.LEFT,
        BoundaryConditionData(
            BoundaryConditionType.DIRICHLET,
            left_electrode_condition,
        ),
    )

    solver.add_boundary_condition(
        BoundaryOrientation.RIGHT,
        BoundaryConditionData(
            BoundaryConditionType.DIRICHLET,
            right_electrode_condition,
        ),
    )

    # Solve
    # potential = solver.solve(tolerance=1e-5)
    potential = solver.solve()
    print(f"Potential: min={np.min(potential)}, max={np.max(potential)}")
    _plot_solution(solver, potential, title="potential")

    # x = np.linspace(0, partition.Lx, partition.Nx)
    # y = np.linspace(0, partition.Ly, partition.Ny)
    # X, Y = np.meshgrid(x, y)
    # elctrode_potential_fn = np.vectorize(apply_electrodes_potential)
    # elctrode_potential = elctrode_potential_fn(X, Y, shape=rect)
    # _plot_solution(solver, elctrode_potential, title="electrode_potential")

    # potential = potential + elctrode_potential

    _plot_field(potential, WIDTH, HEIGHT)

    electric_field = gradient_magnitude(potential, WIDTH / NX, WIDTH / NY)
    print(f"Electric field: min={np.min(electric_field)}, max={np.max(electric_field)}")
    _plot_solution(solver, np.log1p(electric_field), title="electric_field")

    # # Альтернатива: цветные заливки между линиями уровня
    # contour = plt.contour(X, Y, np.log1p(electric_field), levels=20, cmap="viridis")
    # plt.clabel(contour, inline=True, fontsize=8)  # подписи уровней
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.grid(True)
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
    # plt.show()
