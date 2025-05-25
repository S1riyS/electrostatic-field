import io
from typing import List, Tuple

import numpy as np
from fastapi import HTTPException
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from libs.computations.gradient import gradient_vectors
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
from schemas.simulation import (
    SimulationArrowShape,
    SimulationRequest,
    SimulationRingShape,
)

# TODO: move to config or retrieve from request
NX = 500
NY = 500


class SimulationService:
    async def run(self, data: SimulationRequest) -> io.BytesIO:
        # Retrieve shape
        shape = self._retrieve_shape(data)
        if shape is None:
            raise HTTPException(
                status_code=400,
                detail="Couldn't create shape",
            )

        # Setup Laplce equation solver
        plane_partition = self._setup_plane_partition(data, NX, NY)
        solver = LaplaceSolver(plane_partition)

        # Setup conditions
        for orientation, cond in self._setup_boundary_conditions(data):
            solver.add_boundary_condition(orientation, cond)

        internal_condition = self._setup_internal_condition(shape, data.conductor.potential)
        solver.add_internal_condition(internal_condition)

        u = solver.solve()

        return self._render_solution(u, plane_partition, shape)

    def _retrieve_shape(self, data: SimulationRequest) -> Shape | None:
        if isinstance(data.conductor.shape, SimulationRingShape):
            return Ring(
                x=data.conductor.x,
                y=data.conductor.y,
                inner_radius=0,  # * Inner radius is 0 by default (doesn't affect physics)
                outer_radius=data.conductor.shape.outer_radius,
            )
        elif isinstance(data.conductor.shape, SimulationArrowShape):
            return Arrow(
                x=data.conductor.x,
                y=data.conductor.y,
                height=data.conductor.shape.height,
                length=data.conductor.shape.length,
                angle=data.conductor.shape.angle,
            )

        return None

    def _setup_plane_partition(
        self,
        data: SimulationRequest,
        Nx: int,
        Ny: int,
    ) -> DiscretePlanePartition:
        return DiscretePlanePartition(
            data.bath.x_boundary,
            Nx,
            data.bath.y_boundary,
            Ny,
        )

    def _setup_boundary_conditions(
        self,
        data: SimulationRequest,
    ) -> List[Tuple[BoundaryOrientation, BoundaryConditionData]]:
        def uniform_distribution_boundary(x: float) -> float:
            x_max = data.bath.x_boundary
            potential_low = data.electrodes.left_potential
            potential_high = data.electrodes.right_potential
            return potential_low + (x / x_max) * (potential_high - potential_low)

        def left_electrode_boundary(_: float) -> float:
            return data.electrodes.left_potential

        def right_electrode_boundary(_: float) -> float:
            return data.electrodes.right_potential

        return [
            (
                BoundaryOrientation.TOP,
                BoundaryConditionData(
                    BoundaryConditionType.DIRICHLET,
                    uniform_distribution_boundary,
                ),
            ),
            (
                BoundaryOrientation.BOTTOM,
                BoundaryConditionData(
                    BoundaryConditionType.DIRICHLET,
                    uniform_distribution_boundary,
                ),
            ),
            (
                BoundaryOrientation.LEFT,
                BoundaryConditionData(
                    BoundaryConditionType.DIRICHLET,
                    left_electrode_boundary,
                ),
            ),
            (
                BoundaryOrientation.RIGHT,
                BoundaryConditionData(
                    BoundaryConditionType.DIRICHLET,
                    right_electrode_boundary,
                ),
            ),
        ]

    def _setup_internal_condition(self, shape: Shape, potential: float) -> InternalCondition2D:
        def cond(x: float, y: float) -> Tuple[float, bool]:
            is_inside_shape = shape.check_surface(x, y)
            if shape.check_surface(x, y):
                value = potential
            else:
                value = 0

            return value, is_inside_shape

        return cond

    def _render_solution(
        self,
        potential: NDArray[np.float64],
        partition: DiscretePlanePartition,
        shape: Shape,
    ) -> io.BytesIO:
        """Renders the solution to an image file and returns it as bytes."""
        # Get X and Y components of the electric field vector field
        Ex, Ey = gradient_vectors(potential, partition.dx, partition.dy)
        x_grid = np.linspace(0, partition.Lx, NX)
        y_grid = np.linspace(0, partition.Ly, NY)

        # Create figure
        k = 20 / partition.Lx
        _, ax = plt.subplots(figsize=(partition.Lx * k, partition.Ly * k))

        # Meshgrid
        X, Y = np.meshgrid(x_grid, y_grid)

        # Electric field and potential plots
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
        ax.contour(X, Y, potential, levels=20, colors="gray")

        # Figure settings
        ax.set_axis_off()  # Remove all axes and borders
        ax.set_position((0, 0, 1, 1))  # Fill the entire figure

        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=300, transparent=True)
        plt.close()
        buf.seek(0)

        return buf
