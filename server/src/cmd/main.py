import os
from typing import Callable, List, Tuple

import numpy as np
from libs.computations.gradient import gradient, gradient_magnitude
from libs.computations.laplace import (
    BoundaryCondition1D,
    BoundaryConditionData,
    BoundaryConditionType,
    BoundaryOrientation,
    DiscretePlanePartition,
    LaplaceSolver,
)
from libs.shapes.core.shape import Shape
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from schemas.simulation import (
    SimulationBath,
    SimulationConductor,
    SimulationElectrode,
    SimulationRequest,
    SimulationRingShape,
)
from services.simulation import SimulationService

ConditionsGeneratorFunction = Callable[[SimulationRequest], List[Tuple[BoundaryOrientation, BoundaryConditionData]]]

NX = 500
NY = 500
CALCULATED_ELECTRIC_FIELD = 47.62
IMAGES_DIR = "images"


def __get_request() -> SimulationRequest:
    return SimulationRequest(
        bath=SimulationBath(
            x_boundary=30,
            y_boundary=20,
        ),
        conductor=SimulationConductor(
            x=15,
            y=10,
            potential=7.35,
            shape=SimulationRingShape(inner_radius=3, outer_radius=6),
        ),
        electrodes=SimulationElectrode(
            y_lower=3,
            y_upper=17,
            potential=14,
        ),
    )


def __zero_boundary(_: float) -> float:
    return 0


def __generate_electrode_condition(y_lower: float, y_upper: float, potential: float) -> BoundaryCondition1D:
    def cond(x: float) -> float:
        is_inside_shape = x >= y_lower and x <= y_upper
        if is_inside_shape:
            return potential
        return 0

    return cond


def __one_electrode_conditions(request: SimulationRequest) -> List[Tuple[BoundaryOrientation, BoundaryConditionData]]:
    """
    Boundary conditions for simulation when potential of positive electrode has potential specified in request
    and zero in negative electrode
    """
    return [
        (
            BoundaryOrientation.TOP,
            BoundaryConditionData(BoundaryConditionType.DIRICHLET, __zero_boundary),
        ),
        (
            BoundaryOrientation.BOTTOM,
            BoundaryConditionData(BoundaryConditionType.DIRICHLET, __zero_boundary),
        ),
        (
            BoundaryOrientation.LEFT,
            BoundaryConditionData(BoundaryConditionType.DIRICHLET, __zero_boundary),
        ),
        (
            BoundaryOrientation.RIGHT,
            BoundaryConditionData(
                BoundaryConditionType.DIRICHLET,
                __generate_electrode_condition(
                    request.electrodes.y_lower,
                    request.electrodes.y_upper,
                    request.electrodes.potential,
                ),
            ),
        ),
    ]


def __two_electrodes_conditions(request: SimulationRequest) -> List[Tuple[BoundaryOrientation, BoundaryConditionData]]:
    """
    Boundary conditions for simulation when potential of positive electrode has potential specified in request divied by 2
    and negative electrode has the same potential, but with opposite sign
    """
    return [
        (
            BoundaryOrientation.TOP,
            BoundaryConditionData(BoundaryConditionType.DIRICHLET, __zero_boundary),
        ),
        (
            BoundaryOrientation.BOTTOM,
            BoundaryConditionData(BoundaryConditionType.DIRICHLET, __zero_boundary),
        ),
        (
            BoundaryOrientation.LEFT,
            BoundaryConditionData(
                BoundaryConditionType.DIRICHLET,
                __generate_electrode_condition(
                    request.electrodes.y_lower,
                    request.electrodes.y_upper,
                    -request.electrodes.potential / 2,
                ),
            ),
        ),
        (
            BoundaryOrientation.RIGHT,
            BoundaryConditionData(
                BoundaryConditionType.DIRICHLET,
                __generate_electrode_condition(
                    request.electrodes.y_lower,
                    request.electrodes.y_upper,
                    request.electrodes.potential / 2,
                ),
            ),
        ),
    ]


def __separate_potentials_conditions(
    request: SimulationRequest,
) -> List[Tuple[BoundaryOrientation, BoundaryConditionData]]:
    """
    Boundary conditions for simulation when potential of conductor and potential of electrodes are calculated separately
    """
    return []


def __apply_electrodes_potential(x: float, y: float, shape: Shape, electric_field_value: float) -> float:
    if shape.check_inside(x, y):
        return 0

    return 0 + (x / 100) * electric_field_value


def __save_heatmap(data: NDArray[np.float64], partition: DiscretePlanePartition, filename: str):
    """
    Save a 2D numpy array as a heatmap image.

    Parameters:
    -----------
    data : 2D numpy.ndarray
        The data array to visualize
    filename : str
        Output filename (with extension, e.g., 'heatmap.png')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(10, 8))

    # Create grid coordinatesc
    x = np.linspace(0, partition.Lx, partition.Nx)
    y = np.linspace(0, partition.Ly, partition.Ny)
    X, Y = np.meshgrid(x, y)

    heatmap = plt.pcolormesh(X, Y, data, shading="auto")
    plt.colorbar(heatmap)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def main() -> None:
    # Different approaches to setup boundary conditions
    approaches: List[Tuple[str, ConditionsGeneratorFunction]] = [
        ("one_electrode", __one_electrode_conditions),
        ("two_electrodes", __two_electrodes_conditions),
        ("separate_potentials", __separate_potentials_conditions),
    ]

    # Setup core components
    request = __get_request()
    service = SimulationService()

    # Run simulation with different approaches
    for approach_name, conditions_gen_fn in approaches:
        print(f"Running simulation with {approach_name} approach")

        # Retrieve shape
        shape = service._retrieve_shape(request)
        assert shape is not None

        # Setup Laplce equation solver
        partition = service._setup_plane_partition(request)
        solver = LaplaceSolver(partition)

        # Setup internal condition
        internal_condition = service._setup_internal_condition(shape, request.conductor.potential)
        solver.add_internal_condition(internal_condition)

        conditions = conditions_gen_fn(request)
        for orientation, cond in conditions:
            solver.add_boundary_condition(orientation, cond)

        # Calcilate potential
        u = solver.solve()

        if approach_name == "separate_potentials":
            x = np.linspace(0, partition.Lx, partition.Nx)
            y = np.linspace(0, partition.Ly, partition.Ny)
            X, Y = np.meshgrid(x, y)
            elctrode_potential_fn = np.vectorize(__apply_electrodes_potential)
            elctrode_potential = elctrode_potential_fn(
                X,
                Y,
                shape=shape,
                electric_field_value=CALCULATED_ELECTRIC_FIELD,
            )

            u += elctrode_potential

        __save_heatmap(
            u,
            partition=partition,
            filename=f"{IMAGES_DIR}/{approach_name}/potential.png",
        )

        # Calculate electric field
        electric_field = np.log1p(gradient_magnitude(u, partition.dx, partition.dy))
        __save_heatmap(
            electric_field,
            partition=partition,
            filename=f"{IMAGES_DIR}/{approach_name}/electric_field.png",
        )


if __name__ == "__main__":
    main()
