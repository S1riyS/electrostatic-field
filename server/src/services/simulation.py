from typing import List, Tuple

from fastapi import HTTPException
import numpy as np
from numpy.typing import NDArray

from libs.computations.gradient import gradient
from libs.computations.laplace import (
    BoundaryConditionData,
    BoundaryConditionType,
    BoundaryOrientation,
    DiscretePlanePartition,
    InternalCondition2D,
    LaplaceSolver,
)
from libs.computations.potential import get_electrodes_potential_field
from libs.shapes.arrow import Arrow
from libs.shapes.core.enums import ShapeType
from libs.shapes.core.shape import Shape
from libs.shapes.ring import Ring
from schemas.simulation import SimulationArrowShape, SimulationRequest, SimulationResponse, SimulationRingShape


class SimulationService:
    async def run(self, data: SimulationRequest) -> SimulationResponse:
        # Retrieve shape
        shape = self.__retrieve_shape(data)
        if shape is None:
            raise HTTPException(
                status_code=400,
                detail="Couldn't create shape",
            )

        # Setup Laplce equation solver
        plane_partition = self.__setup_plane_partition(data)
        solver = LaplaceSolver(plane_partition)

        # Setup conditions
        for orientation, cond in self.__setup_boundary_conditions(data):
            solver.add_boundary_condition(orientation, cond)

        internal_condition = self.__setup_internal_condition(shape, data.conductor.potential)
        solver.add_internal_condition(internal_condition)

        u = solver.solve()
        v = get_electrodes_potential_field(data.electrodes, plane_partition)

        u += v

        potential: List[List[float]] = u.tolist()
        electric_field: List[List[Tuple[float, float]]] = (
            -gradient(u, plane_partition.dx, plane_partition.dy)
        ).tolist()

        return SimulationResponse(
            potential=potential,
            electric_field=electric_field,
        )

    def __retrieve_shape(self, data: SimulationRequest) -> Shape | None:
        if isinstance(data.conductor.shape, SimulationRingShape):
            return Ring(
                data.conductor.x,
                data.conductor.y,
                data.conductor.shape.inner_radius,
                data.conductor.shape.outer_radius,
            )
        elif isinstance(data.conductor.shape, SimulationArrowShape):
            return Arrow(
                data.conductor.x,
                data.conductor.y,
                data.conductor.shape.height,
                data.conductor.shape.length,
                data.conductor.shape.angle,
            )

        return None

    def __setup_plane_partition(self, data: SimulationRequest) -> DiscretePlanePartition:
        Nx = 500
        Ny = 500
        return DiscretePlanePartition(
            data.bath.x_boundary,
            Nx,
            data.bath.y_boundary,
            Ny,
        )

    def __setup_boundary_conditions(
        self,
        data: SimulationRequest,
    ) -> List[Tuple[BoundaryOrientation, BoundaryConditionData]]:
        def zero_boundary(_: float) -> float:
            return 0

        def positive_electrode_boundary(y: float) -> float:
            if data.electrodes.y_lower <= y <= data.electrodes.y_upper:
                return data.electrodes.potential
            return 0

        return [
            (
                BoundaryOrientation.TOP,
                BoundaryConditionData(BoundaryConditionType.DIRICHLET, zero_boundary),
            ),
            (
                BoundaryOrientation.BOTTOM,
                BoundaryConditionData(BoundaryConditionType.DIRICHLET, zero_boundary),
            ),
            (
                BoundaryOrientation.LEFT,
                BoundaryConditionData(BoundaryConditionType.DIRICHLET, zero_boundary),
            ),
            # (
            #     BoundaryOrientation.RIGHT,
            #     BoundaryConditionData(BoundaryConditionType.DIRICHLET, positive_electrode_boundary),
            # ),
            (
                BoundaryOrientation.RIGHT,
                BoundaryConditionData(BoundaryConditionType.DIRICHLET, zero_boundary),
            ),
        ]

    def __setup_internal_condition(self, shape: Shape, potential: float) -> InternalCondition2D:
        def cond(x: float, y: float) -> Tuple[float, bool]:
            is_inside_shape = shape.check_point(x, y)
            if shape.check_point(x, y):
                value = potential
            else:
                value = 0

            return value, is_inside_shape

        return cond
