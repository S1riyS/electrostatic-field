import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from libs.computations.gradient import gradient_magnitude
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
from libs.shapes.ring import Ring


def ring_condition(x: float, y: float) -> Tuple[float, bool]:
    # Center of ring is at (x, y) = (15, 10)
    center_x = 15
    center_y = 10

    inner_radius = 3
    outer_radius = 6
    potential = 7.35

    distance = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    if distance >= inner_radius and distance <= outer_radius:
        return potential, True
    return 0, False


def arrow_condition(x: float, y: float) -> Tuple[float, bool]:
    # Center of arrow is at (x, y) = (15, 10)
    center_x = 15
    center_y = 10

    h = 6
    l = 8
    assert l > h
    potential = 7.35

    relative_x = x - center_x
    relative_y = y - center_y

    # Central rect
    central_rect_length = l - h
    in_central_rect_x = -(central_rect_length / 2) <= relative_x <= (central_rect_length / 2)
    in_central_rect_y = -(h / 2) <= relative_y <= (h / 2)
    in_central_rect = in_central_rect_x and in_central_rect_y

    # Left triangle
    in_left_x = -(l / 2) <= relative_x <= -(central_rect_length / 2)
    in_left_upper = relative_y <= (l / 2) + relative_x
    in_left_lower = relative_y >= -(l / 2) - relative_x
    in_left = in_left_x and in_left_upper and in_left_lower

    # Right triangle
    in_right_x = (central_rect_length / 2) <= relative_x <= (l / 2)
    in_right_y = -(h / 2) <= relative_y <= (h / 2)
    in_right_upper = relative_y >= -(central_rect_length) / 2 + relative_x
    in_right_lower = relative_y <= (central_rect_length) / 2 - relative_x
    in_right = in_right_x and in_right_y and (in_right_upper or in_right_lower)

    in_arrow = in_central_rect or in_left or in_right
    if in_arrow:
        return potential, True

    return 0, False


def generate_internal_condition(shape: Shape, potential: float) -> InternalCondition2D:
    def cond(x: float, y: float) -> Tuple[float, bool]:
        is_inside_shape = shape.check_point(x, y)
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


def zero_neumann(x: float, y: float) -> float:
    return 0.0  # The derivative value we want (∂f/∂n = 0)


def zero_dirichlet(_: float) -> float:
    return 0


def _plot_solution(self: LaplaceSolver, u: np.ndarray, title: str):
    """Create a heat map visualization of the solution."""
    plt.figure(figsize=(10, 8))

    # Create grid coordinates
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

    NX = 750
    NY = 750

    partition = DiscretePlanePartition(WIDTH, NX, HEIGHT, NY)

    # Setup solver
    solver = LaplaceSolver(partition)

    # Internal conditions
    arrow = Arrow(WIDTH / 2, HEIGHT / 2, height=6, length=8, angle=math.pi / 4)
    ring = Ring(WIDTH / 2, HEIGHT / 2, inner_radius=3, outer_radius=6)
    shape_potential = 7.35

    # shape_condition_ring = generate_internal_condition(ring, shape_potential)
    # solver.add_internal_condition(shape_condition_ring)

    shape_condition_arrow = generate_internal_condition(arrow, shape_potential)
    solver.add_internal_condition(shape_condition_arrow)

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

    electric_field = gradient_magnitude(potential, WIDTH / NX, WIDTH / NY)
    print(f"Electric field: min={np.min(electric_field)}, max={np.max(electric_field)}")
    _plot_solution(solver, np.log1p(electric_field), title="electric_field")
